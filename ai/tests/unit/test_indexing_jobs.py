import json
import logging
import threading
import time

import pytest
from test_indexing_api import post, valid_payload

from zipsai.api import indexing
from zipsai.api.indexing import JobOutcome
from zipsai.contracts.notification import ResultStatus
from zipsai.contracts.sync_response import DocumentResult
from zipsai.errors import JobCancelled
from zipsai.indexing.admission import reset as reset_admission
from zipsai.indexing.registry import current_hooks, get_registry, reset_registry
from zipsai.metrics import INDEXING_RESULTS


@pytest.fixture(autouse=True)
def _fresh_registry():
    reset_admission()
    reset_registry()
    yield
    reset_registry()
    reset_admission()


def _result(reason: str | None, *, retryable: bool, doc_id: str | None):
    return DocumentResult(
        kind="document_index",
        status=ResultStatus.SUCCEEDED if reason is None else ResultStatus.FAILED,
        reason_code=reason,
        retryable=retryable,
        doc_id=doc_id,
    )


def _count(reason: str) -> float:
    return INDEXING_RESULTS.labels(
        kind="document_index", status="failed", reason_code=reason
    )._value.get()


def _hold(monkeypatch: pytest.MonkeyPatch, seen: list[str]):
    started = threading.Event()
    let_go = threading.Event()

    def fake(payload, _job_id: str) -> JobOutcome:
        if payload.doc_id == "notice-001":
            started.set()
            assert let_go.wait(3)
        seen.append(payload.file_key or "")
        return JobOutcome(results=[_result(None, retryable=False, doc_id=payload.doc_id)])

    monkeypatch.setattr(indexing, "_run_job", fake)
    return started, let_go


def test_same_request_runs_the_pipeline_once(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []
    started, let_go = _hold(monkeypatch, seen)
    holder = threading.Thread(target=lambda: post("/api/v3/ai/indexing/jobs", valid_payload()))
    holder.start()
    assert started.wait(2)
    try:
        status_code, body = post("/api/v3/ai/indexing/jobs", valid_payload())
        parsed = json.loads(body)
        assert status_code == 429
        assert parsed["status"] == "IN_PROGRESS"
        assert parsed["error_code"] == "IN_PROGRESS"
        assert parsed["progress"] == {"done": 0, "total": 0}
        assert seen == []
    finally:
        let_go.set()
        holder.join(2)
    assert seen == ["s3://zipsai-files/buildings/101/notice-001.pdf"]


def test_fast_job_returns_the_existing_success_body(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(payload, _job_id: str) -> JobOutcome:
        return JobOutcome(results=[_result(None, retryable=False, doc_id=payload.doc_id)])

    monkeypatch.setattr(indexing, "_run_job", fake)
    status_code, body = post("/api/v3/ai/indexing/jobs", valid_payload())
    parsed = json.loads(body)

    assert status_code == 200
    assert set(parsed) == {"trace_id", "results"}
    assert parsed["results"][0]["status"] == "succeeded"
    assert parsed["results"][0]["reason_code"] is None


def test_other_document_waits_in_the_queue(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []
    started, let_go = _hold(monkeypatch, seen)
    holder = threading.Thread(target=lambda: post("/api/v3/ai/indexing/jobs", valid_payload()))
    holder.start()
    assert started.wait(2)
    monkeypatch.setattr("zipsai.settings.INDEXING_FIRST_WAIT_SECONDS", 0)
    try:
        other = valid_payload() | {"doc_id": "notice-002", "trace_id": "trace-002"}
        status_code, body = post("/api/v3/ai/indexing/jobs", other)
        parsed = json.loads(body)
        assert status_code == 429
        assert parsed["status"] == "QUEUED"
        assert parsed["error_code"] == "QUEUED"
        assert parsed["doc_id"] == "notice-002"
        assert parsed["progress"] == {"done": 0, "total": 0}
        assert post.headers["retry-after"] == "30"
    finally:
        let_go.set()
        holder.join(2)


def test_queued_replace_never_starts_the_old_file(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="zipsai.api.indexing")
    seen: list[str] = []
    started, let_go = _hold(monkeypatch, seen)
    holder = threading.Thread(target=lambda: post("/api/v3/ai/indexing/jobs", valid_payload()))
    holder.start()
    assert started.wait(2)
    monkeypatch.setattr("zipsai.settings.INDEXING_FIRST_WAIT_SECONDS", 0)
    old_key = "s3://zipsai-files/buildings/101/old.pdf"
    new_key = "s3://zipsai-files/buildings/101/new.pdf"
    first = valid_payload() | {
        "doc_id": "notice-002",
        "trace_id": "trace-old",
        "file_key": old_key,
    }
    try:
        status_code, _body = post("/api/v3/ai/indexing/jobs", first)
        assert status_code == 429
        replaced = first | {"trace_id": "trace-new", "file_key": new_key}
        status_code, _body = post("/api/v3/ai/indexing/jobs", replaced)
        assert status_code == 429
        assert old_key not in seen
        cancelled = [
            record
            for record in caplog.records
            if record.getMessage() == "job_cancelled" and record.reason == "replaced"
        ]
        assert len(cancelled) == 1
        assert cancelled[0].file_key_path == old_key
    finally:
        let_go.set()
        holder.join(2)
    deadline = time.monotonic() + 2
    while new_key not in seen and time.monotonic() < deadline:
        time.sleep(0.01)
    assert old_key not in seen
    assert new_key in seen


def test_queued_delete_never_starts_the_pipeline(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="zipsai.api.indexing")
    seen: list[str] = []
    started, let_go = _hold(monkeypatch, seen)
    holder = threading.Thread(target=lambda: post("/api/v3/ai/indexing/jobs", valid_payload()))
    holder.start()
    assert started.wait(2)
    monkeypatch.setattr("zipsai.settings.INDEXING_FIRST_WAIT_SECONDS", 0)
    queued_key = "s3://zipsai-files/buildings/101/queued.pdf"
    queued = valid_payload() | {
        "doc_id": "notice-002",
        "trace_id": "trace-queued",
        "file_key": queued_key,
    }
    try:
        assert post("/api/v3/ai/indexing/jobs", queued)[0] == 429
        cleanup = {
            "building_id": 101,
            "trace_id": "trace-cleanup",
            "valid_doc_ids": ["notice-001"],
        }
        status_code, body = post("/api/v3/ai/indexing/jobs", cleanup)
        assert status_code == 200
        assert "error_code" not in json.loads(body)
        assert queued_key not in seen
        deleted = [
            record
            for record in caplog.records
            if record.getMessage() == "job_cancelled" and record.reason == "deleted"
        ]
        assert len(deleted) == 1
        assert deleted[0].doc_id == "notice-002"
    finally:
        let_go.set()
        holder.join(2)
    assert queued_key not in seen


def test_running_delete_does_not_save(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="zipsai.api.indexing")
    started = threading.Event()
    let_go = threading.Event()
    saved: list[str] = []

    def fake(payload, _job_id: str) -> JobOutcome:
        if payload.doc_id is None:
            return JobOutcome(results=[])
        started.set()
        assert let_go.wait(3)
        hooks = current_hooks()
        if hooks is not None and hooks.should_cancel():
            raise JobCancelled()
        saved.append(payload.doc_id or "")
        return JobOutcome(results=[_result(None, retryable=False, doc_id=payload.doc_id)])

    monkeypatch.setattr(indexing, "_run_job", fake)
    outcome: dict[str, tuple[int, bytes]] = {}

    def run() -> None:
        outcome["pair"] = post("/api/v3/ai/indexing/jobs", valid_payload())

    holder = threading.Thread(target=run)
    holder.start()
    assert started.wait(2)
    try:
        cleanup = {"building_id": 101, "trace_id": "trace-del", "valid_doc_ids": []}
        assert post("/api/v3/ai/indexing/jobs", cleanup)[0] == 200
    finally:
        let_go.set()
        holder.join(3)
    status_code, body = outcome["pair"]
    parsed = json.loads(body)
    assert status_code == 200
    assert parsed["results"][0]["status"] == "failed"
    assert parsed["results"][0]["reason_code"] is None
    assert parsed["results"][0]["retryable"] is False
    assert saved == []
    assert any(
        record.getMessage() == "job_cancelled" and record.reason == "deleted"
        for record in caplog.records
    )
    again, again_body = post(
        "/api/v3/ai/indexing/jobs",
        valid_payload() | {"trace_id": "trace-again"},
    )
    assert again == 200
    assert json.loads(again_body)["trace_id"] == "trace-again"


def test_terminal_failure_is_replayed(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake(payload, _job_id: str) -> JobOutcome:
        calls.append(payload.trace_id)
        return JobOutcome(
            results=[_result("PARSE_FAILED", retryable=False, doc_id=payload.doc_id)]
        )

    monkeypatch.setattr(indexing, "_run_job", fake)
    before = _count("PARSE_FAILED")
    first_status, first_body = post("/api/v3/ai/indexing/jobs", valid_payload())
    second_status, second_body = post(
        "/api/v3/ai/indexing/jobs",
        valid_payload() | {"trace_id": "trace-replay"},
    )

    assert first_status == 200
    assert second_status == 200
    assert json.loads(second_body) == json.loads(first_body)
    assert json.loads(second_body)["trace_id"] == "trace-001"
    assert calls == ["trace-001"]
    assert _count("PARSE_FAILED") == before + 1


def test_retryable_failure_starts_again(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="zipsai.api.indexing")
    calls: list[str] = []

    def fake(payload, _job_id: str) -> JobOutcome:
        calls.append(payload.trace_id)
        return JobOutcome(
            results=[_result("EMBEDDING_FAILED", retryable=True, doc_id=payload.doc_id)]
        )

    monkeypatch.setattr(indexing, "_run_job", fake)
    assert post("/api/v3/ai/indexing/jobs", valid_payload())[0] == 200
    caplog.clear()
    status_code, body = post(
        "/api/v3/ai/indexing/jobs",
        valid_payload() | {"trace_id": "trace-retry"},
    )

    assert status_code == 200
    assert json.loads(body)["trace_id"] == "trace-retry"
    assert calls == ["trace-001", "trace-retry"]
    assert not any(record.getMessage() == "job_record_hit" for record in caplog.records)


def test_expired_record_starts_again(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = {"now": 1_000.0}
    get_registry().set_clock(lambda: clock["now"])
    monkeypatch.setattr("zipsai.settings.INDEXING_RECORD_TTL_SECONDS", 10)
    calls: list[str] = []

    def fake(payload, _job_id: str) -> JobOutcome:
        calls.append(payload.trace_id)
        return JobOutcome(results=[_result(None, retryable=False, doc_id=payload.doc_id)])

    monkeypatch.setattr(indexing, "_run_job", fake)
    assert post("/api/v3/ai/indexing/jobs", valid_payload())[0] == 200
    clock["now"] = 1_010.0
    status_code, body = post(
        "/api/v3/ai/indexing/jobs",
        valid_payload() | {"trace_id": "trace-after-ttl"},
    )

    assert status_code == 200
    assert json.loads(body)["trace_id"] == "trace-after-ttl"
    assert calls == ["trace-001", "trace-after-ttl"]
