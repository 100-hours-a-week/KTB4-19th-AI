import asyncio
import json

import pytest

from zipsai.api import indexing
from zipsai.contracts.indexing import IndexingJobRequest
from zipsai.main import app

# 자동 fixture가 대체하기 전의 원본. 실제 실행 경로를 보는 테스트가 쓴다.
REAL_RUN_JOB = indexing._run_job


@pytest.fixture(autouse=True)
def scheduled_jobs(monkeypatch: pytest.MonkeyPatch) -> list[IndexingJobRequest]:
    # 배경 작업이 실제 S3·Embedding에 붙지 않게 막고 예약 여부만 기록한다.
    runs: list[IndexingJobRequest] = []
    monkeypatch.setattr(indexing, "_run_job", runs.append)
    return runs


def post(path: str, payload: dict | None = None) -> tuple[int, bytes]:
    body = json.dumps(payload).encode() if payload is not None else b""
    messages = []
    received = False

    async def receive() -> dict:
        nonlocal received
        if received:
            return {"type": "http.disconnect"}
        received = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: dict) -> None:
        messages.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
        "root_path": "",
    }
    asyncio.run(app(scope, receive, send))

    response_body = b"".join(
        message.get("body", b"")
        for message in messages
        if message["type"] == "http.response.body"
    )
    status_code = next(
        message["status"]
        for message in messages
        if message["type"] == "http.response.start"
    )
    return status_code, response_body


def valid_payload() -> dict:
    return {
        "building_id": 101,
        "doc_id": "notice-001",
        "title": "Water tank cleaning",
        "file_key": "s3://zipsai-files/buildings/101/notice-001.pdf",
    }


def test_accepted_job_returns_202_with_an_empty_body() -> None:
    status_code, body = post("/api/v3/ai/indexing/jobs", valid_payload())

    assert status_code == 202
    assert body == b""


def test_job_status_lookup_is_gone() -> None:
    # v1에서 조회 경로를 없앴다. 백엔드가 폴링하면 404를 받아야 한다.
    status_code, _ = post("/api/v3/ai/indexing/jobs/any-id")

    assert status_code == 404


def test_building_id_is_the_only_always_required_field() -> None:
    status_code, _ = post("/api/v3/ai/indexing/jobs", {"doc_id": "d"})

    assert status_code == 400


@pytest.mark.parametrize("missing", ["doc_id", "title", "file_key"])
def test_document_fields_must_be_sent_together(missing: str) -> None:
    # 셋 중 하나만 빠지면 무엇을 색인하라는 요청인지 알 수 없다.
    payload = valid_payload()
    del payload[missing]

    status_code, _ = post("/api/v3/ai/indexing/jobs", payload)

    assert status_code == 422


def test_request_without_a_document_or_a_list_is_rejected() -> None:
    status_code, _ = post("/api/v3/ai/indexing/jobs", {"building_id": 101})

    assert status_code == 422


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("building_id", "101"),
        # 빈 doc_id는 같은 building의 다른 문서와 교체 필터가 겹친다.
        ("doc_id", ""),
        ("file_key", 3),
    ],
)
def test_invalid_field_returns_422(field: str, value: object) -> None:
    payload = valid_payload()
    payload[field] = value

    status_code, _ = post("/api/v3/ai/indexing/jobs", payload)

    assert status_code == 422


def test_accepted_job_schedules_the_indexing_pipeline(
    scheduled_jobs: list[IndexingJobRequest],
) -> None:
    post("/api/v3/ai/indexing/jobs", valid_payload())

    assert [job.doc_id for job in scheduled_jobs] == ["notice-001"]


def test_rejected_payload_schedules_nothing(
    scheduled_jobs: list[IndexingJobRequest],
) -> None:
    payload = valid_payload()
    del payload["title"]

    post("/api/v3/ai/indexing/jobs", payload)

    assert scheduled_jobs == []


def test_dependency_failure_does_not_break_the_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 202는 이미 나갔다. 배경 작업이 터져도 예외가 응답으로 새면 안 된다.
    def broken() -> tuple[object, object]:
        raise RuntimeError("qdrant unreachable")

    monkeypatch.setattr(indexing, "_run_job", REAL_RUN_JOB)
    monkeypatch.setattr(indexing, "_dependencies", broken)

    status_code, body = post("/api/v3/ai/indexing/jobs", valid_payload())

    assert status_code == 202
    assert body == b""


def test_cleanup_only_request_carries_no_document(
    scheduled_jobs: list[IndexingJobRequest],
) -> None:
    status_code, body = post(
        "/api/v3/ai/indexing/jobs",
        {"building_id": 101, "valid_doc_ids": ["doc-a", "doc-b"]},
    )

    assert status_code == 202
    assert body == b""
    job = scheduled_jobs[0]
    assert job.has_document is False
    assert job.valid_doc_ids == ["doc-a", "doc-b"]


def test_index_and_cleanup_ride_on_one_request(
    scheduled_jobs: list[IndexingJobRequest],
) -> None:
    payload = valid_payload() | {"valid_doc_ids": ["notice-001", "rule-2026"]}

    post("/api/v3/ai/indexing/jobs", payload)

    job = scheduled_jobs[0]
    assert job.has_document is True
    assert job.valid_doc_ids == ["notice-001", "rule-2026"]


def test_empty_valid_doc_ids_returns_422(
    scheduled_jobs: list[IndexingJobRequest],
) -> None:
    # 빈 목록을 받아주면 백엔드 실수 한 번에 건물 문서가 전멸한다.
    status_code, _ = post(
        "/api/v3/ai/indexing/jobs", {"building_id": 101, "valid_doc_ids": []}
    )

    assert status_code == 422
    assert scheduled_jobs == []


def test_removed_fields_are_ignored_not_rejected(
    scheduled_jobs: list[IndexingJobRequest],
) -> None:
    # 엄격하게 거절하면 백엔드가 우리보다 먼저 배포되는 순간 깨진다.
    payload = valid_payload() | {
        "source_type": "notice",
        "published_at": "2026-09-15T00:00:00Z",
        "trace_id": "trace-001",
    }

    status_code, _ = post("/api/v3/ai/indexing/jobs", payload)

    assert status_code == 202
    assert scheduled_jobs[0].doc_id == "notice-001"
