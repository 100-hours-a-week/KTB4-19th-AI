import asyncio
import json

import pytest

from zipsai.api import indexing
from zipsai.main import app

# 자동 fixture가 대체하기 전의 원본. 실제 실행 경로를 보는 테스트가 쓴다.
REAL_RUN_JOB = indexing._run_job


@pytest.fixture(autouse=True)
def scheduled_jobs(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    # 배경 작업이 실제 S3·Embedding에 붙지 않게 막고 예약 여부만 기록한다.
    runs: list[str] = []

    def record(job_id: str, payload: object) -> None:
        runs.append(job_id)

    monkeypatch.setattr(indexing, "_run_job", record)
    return runs


def request(method: str, path: str, payload: dict | None = None) -> tuple[int, dict]:
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
        "method": method,
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
    return next(
        message["status"]
        for message in messages
        if message["type"] == "http.response.start"
    ), json.loads(response_body)


def valid_payload() -> dict:
    return {
        "building_id": 101,
        "doc_id": "notice-001",
        "source_type": "notice",
        "title": "Water tank cleaning",
        "published_at": "2026-09-15T00:00:00Z",
        "file_key": "buildings/101/notice-001.pdf",
        "trace_id": "trace-001",
    }


def test_creates_accepted_job_and_returns_current_status() -> None:
    status_code, created = request("POST", "/api/v3/ai/indexing/jobs", valid_payload())

    assert status_code == 202
    assert created["status"] == "accepted"
    assert created["job_id"]

    status_code, current = request(
        "GET", f"/api/v3/ai/indexing/jobs/{created['job_id']}"
    )

    assert status_code == 200
    assert current == created


def test_missing_required_field_returns_422() -> None:
    payload = valid_payload()
    del payload["title"]

    status_code, _ = request("POST", "/api/v3/ai/indexing/jobs", payload)

    assert status_code == 422


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("building_id", "101"),
        ("source_type", "other"),
        ("published_at", "not-a-datetime"),
        # 빈 doc_id는 같은 building의 다른 문서와 교체 필터가 겹친다.
        ("doc_id", ""),
    ],
)
def test_invalid_field_returns_422(field: str, value: str) -> None:
    payload = valid_payload()
    payload[field] = value

    status_code, _ = request("POST", "/api/v3/ai/indexing/jobs", payload)

    assert status_code == 422


def test_unknown_job_returns_404() -> None:
    status_code, _ = request(
        "GET", "/api/v3/ai/indexing/jobs/00000000-0000-0000-0000-000000000000"
    )

    assert status_code == 404


def test_accepted_job_schedules_the_indexing_pipeline(
    scheduled_jobs: list[str],
) -> None:
    _, created = request("POST", "/api/v3/ai/indexing/jobs", valid_payload())

    assert scheduled_jobs == [created["job_id"]]


def test_rejected_payload_schedules_nothing(scheduled_jobs: list[str]) -> None:
    payload = valid_payload()
    del payload["title"]

    request("POST", "/api/v3/ai/indexing/jobs", payload)

    assert scheduled_jobs == []


def test_dependency_failure_marks_the_job_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken() -> tuple[object, object]:
        raise RuntimeError("qdrant unreachable")

    monkeypatch.setattr(indexing, "_run_job", REAL_RUN_JOB)
    monkeypatch.setattr(indexing, "_dependencies", broken)

    _, created = request("POST", "/api/v3/ai/indexing/jobs", valid_payload())
    _, current = request("GET", f"/api/v3/ai/indexing/jobs/{created['job_id']}")

    assert current["status"] == "failed"
