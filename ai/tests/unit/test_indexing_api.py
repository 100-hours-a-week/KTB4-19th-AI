import asyncio
import json
import threading

import pytest

from zipsai.api import indexing
from zipsai.api.indexing import JobOutcome
from zipsai.contracts.indexing import IndexingJobRequest, JobStatus
from zipsai.indexing.admission import reset as reset_admission
from zipsai.main import app

# 자동 fixture가 대체하기 전의 원본. 실제 실행 경로를 보는 테스트가 쓴다.
REAL_RUN_JOB = indexing._run_job


@pytest.fixture(autouse=True)
def scheduled_jobs(monkeypatch: pytest.MonkeyPatch) -> list[IndexingJobRequest]:
    # 배경 작업이 실제 S3·Embedding에 붙지 않게 막고 예약 여부만 기록한다.
    reset_admission()
    runs: list[IndexingJobRequest] = []
    def fake(payload: IndexingJobRequest, _job_id: str) -> JobOutcome:
        runs.append(payload)
        return JobOutcome(results=[])

    monkeypatch.setattr(indexing, "_run_job", fake)
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
    start = next(
        message for message in messages if message["type"] == "http.response.start"
    )
    post.headers = {
        key.decode().lower(): value.decode() for key, value in start["headers"]
    }
    return start["status"], response_body


post.headers = {}


def valid_payload() -> dict:
    return {
        "building_id": 101,
        "trace_id": "trace-001",
        "doc_id": "notice-001",
        "title": "Water tank cleaning",
        "file_key": "s3://zipsai-files/buildings/101/notice-001.pdf",
    }


def test_accepted_job_returns_200_with_trace_id() -> None:
    status_code, body = post("/api/v3/ai/indexing/jobs", valid_payload())

    assert status_code == 200
    assert json.loads(body)["trace_id"] == "trace-001"


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
    status_code, _ = post(
        "/api/v3/ai/indexing/jobs", {"building_id": 101, "trace_id": "trace-001"}
    )

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


def test_dependency_failure_returns_setup_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken() -> tuple[object, object]:
        raise RuntimeError("qdrant unreachable")

    monkeypatch.setattr(indexing, "_run_job", REAL_RUN_JOB)
    monkeypatch.setattr(indexing, "_dependencies", broken)

    status_code, body = post("/api/v3/ai/indexing/jobs", valid_payload())
    result = json.loads(body)["results"][0]

    assert status_code == 200
    assert result["status"] == "failed"
    assert result["reason_code"] == "SETUP_FAILED"
    assert result["retryable"] is True


def test_cleanup_only_request_carries_no_document(
    scheduled_jobs: list[IndexingJobRequest],
) -> None:
    status_code, body = post(
        "/api/v3/ai/indexing/jobs",
        {
            "building_id": 101,
            "trace_id": "trace-001",
            "valid_doc_ids": ["doc-a", "doc-b"],
        },
    )

    assert status_code == 200
    assert json.loads(body)["trace_id"] == "trace-001"
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


def test_empty_valid_doc_ids_is_accepted_as_building_wipe(
    scheduled_jobs: list[IndexingJobRequest],
) -> None:
    # 마지막 문서를 지우면 백엔드가 빈 목록을 보낸다 — 정상 요청으로 받는다.
    status_code, _ = post(
        "/api/v3/ai/indexing/jobs",
        {"building_id": 101, "trace_id": "trace-001", "valid_doc_ids": []},
    )

    assert status_code == 200
    assert scheduled_jobs[0].valid_doc_ids == []


def test_removed_fields_are_ignored_not_rejected(
    scheduled_jobs: list[IndexingJobRequest],
) -> None:
    # 엄격하게 거절하면 백엔드가 우리보다 먼저 배포되는 순간 깨진다.
    payload = valid_payload() | {
        "source_type": "notice",
        "published_at": "2026-09-15T00:00:00Z",
        "turn_id": "turn-001",
    }

    status_code, _ = post("/api/v3/ai/indexing/jobs", payload)

    assert status_code == 200
    assert scheduled_jobs[0].doc_id == "notice-001"


def _install_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    *,
    index_status: JobStatus = JobStatus.SUCCEEDED,
    reasons: tuple[str, ...] = (),
    cleanup_error: bool = False,
) -> None:
    monkeypatch.setattr(indexing, "_run_job", REAL_RUN_JOB)
    monkeypatch.setattr(indexing, "_dependencies", lambda: (object(), object()))

    def fake_run(request, *, on_fail, on_stats=None, **_kwargs):
        for reason in reasons:
            on_fail(reason)
        if on_stats is not None:
            on_stats(2, 3, 3)
        return index_status

    monkeypatch.setattr(indexing, "run_indexing_job", fake_run)

    def cleanup(*_args, **_kwargs):
        if cleanup_error:
            raise RuntimeError("qdrant down")

    monkeypatch.setattr(indexing, "delete_missing_documents", cleanup)


def test_success_returns_index_and_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_pipeline(monkeypatch)
    payload = valid_payload() | {"valid_doc_ids": ["notice-001"]}

    status_code, body = post("/api/v3/ai/indexing/jobs", payload)
    results = json.loads(body)["results"]

    assert status_code == 200
    assert len(results) == 2
    assert results[0]["reason_code"] is None
    assert "reason_code" in results[0]
    assert results[0]["kind"] == "document_index"
    assert results[0]["retryable"] is False
    assert results[1]["kind"] == "document_cleanup"
    assert "doc_id" not in results[1]


def test_parse_failure_skips_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_pipeline(
        monkeypatch, index_status=JobStatus.FAILED, reasons=("PARSE_FAILED",)
    )
    payload = valid_payload() | {"valid_doc_ids": ["notice-001"]}

    status_code, body = post("/api/v3/ai/indexing/jobs", payload)
    results = json.loads(body)["results"]

    assert status_code == 200
    assert results[0]["status"] == "failed"
    assert results[0]["reason_code"] == "PARSE_FAILED"
    assert results[0]["retryable"] is False
    assert results[1] == {
        "kind": "document_cleanup",
        "status": "skipped",
        "reason_code": "INDEX_FAILED",
        "retryable": False,
    }


def test_setup_failure_fails_index_and_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken() -> tuple[object, object]:
        raise RuntimeError("qdrant unreachable")

    monkeypatch.setattr(indexing, "_run_job", REAL_RUN_JOB)
    monkeypatch.setattr(indexing, "_dependencies", broken)
    payload = valid_payload() | {"valid_doc_ids": ["notice-001"]}

    status_code, body = post("/api/v3/ai/indexing/jobs", payload)
    results = json.loads(body)["results"]

    assert status_code == 200
    assert [(item["kind"], item["reason_code"], item["retryable"]) for item in results] == [
        ("document_index", "SETUP_FAILED", True),
        ("document_cleanup", "SETUP_FAILED", True),
    ]


def test_cleanup_only_returns_one_result(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_pipeline(monkeypatch)

    status_code, body = post(
        "/api/v3/ai/indexing/jobs",
        {
            "building_id": 101,
            "trace_id": "trace-001",
            "valid_doc_ids": ["doc-a"],
        },
    )
    results = json.loads(body)["results"]

    assert status_code == 200
    assert results == [
        {
            "kind": "document_cleanup",
            "status": "succeeded",
            "reason_code": None,
            "retryable": False,
        }
    ]


def _hold_job(monkeypatch: pytest.MonkeyPatch):
    started = threading.Event()
    let_go = threading.Event()
    calls = {"n": 0}

    def blocking(payload: IndexingJobRequest, _job_id: str) -> JobOutcome:
        calls["n"] += 1
        if calls["n"] == 1:
            started.set()
            assert let_go.wait(2)
        return JobOutcome(results=[])

    monkeypatch.setattr(indexing, "_run_job", blocking)
    return started, let_go


def test_same_doc_id_returns_429_until_the_job_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started, let_go = _hold_job(monkeypatch)
    holder = threading.Thread(
        target=lambda: post("/api/v3/ai/indexing/jobs", valid_payload())
    )
    holder.start()
    assert started.wait(2)

    status_code, body = post("/api/v3/ai/indexing/jobs", valid_payload())

    assert status_code == 429
    assert post.headers["retry-after"] == "30"
    assert json.loads(body) == {"trace_id": "trace-001", "error_code": "IN_PROGRESS"}
    let_go.set()
    holder.join(2)
    status_code, _ = post("/api/v3/ai/indexing/jobs", valid_payload())
    assert status_code == 200


def test_cleanup_without_doc_id_locks_the_building(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started, let_go = _hold_job(monkeypatch)
    payload = {
        "building_id": 101,
        "trace_id": "trace-001",
        "valid_doc_ids": ["doc-a"],
    }
    holder = threading.Thread(target=lambda: post("/api/v3/ai/indexing/jobs", payload))
    holder.start()
    assert started.wait(2)

    status_code, body = post("/api/v3/ai/indexing/jobs", payload)

    assert status_code == 429
    assert json.loads(body)["error_code"] == "IN_PROGRESS"
    let_go.set()
    holder.join(2)


def test_exception_releases_the_document_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(payload: IndexingJobRequest, _job_id: str) -> JobOutcome:
        raise RuntimeError("boom")

    monkeypatch.setattr(indexing, "_run_job", boom)

    for _ in range(2):
        try:
            status_code, _ = post("/api/v3/ai/indexing/jobs", valid_payload())
        except RuntimeError:
            continue
        assert status_code != 429
