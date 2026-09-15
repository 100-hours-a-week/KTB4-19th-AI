import asyncio
import json

import pytest

from zipsai.main import app


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
