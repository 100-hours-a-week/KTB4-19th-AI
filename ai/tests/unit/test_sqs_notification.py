import json

from zipsai.contracts.notification import (
    ItemResult,
    Notification,
    NotificationKind,
    ResultStatus,
)
from zipsai.integrations import sqs


class FakeSqs:
    def __init__(self):
        self.sent = []

    def send_message(self, **kwargs):
        self.sent.append(kwargs)


def comment_notification():
    return Notification(
        kind=NotificationKind.COMMENT_UPSERT,
        trace_id="t",
        results=[
            ItemResult(status=ResultStatus.SUCCEEDED, complaint_detail_id=9001),
            ItemResult(
                status=ResultStatus.FAILED,
                reason_code="EMBEDDING_FAILED",
                complaint_detail_id=9002,
            ),
        ],
    )


def test_message_keeps_null_reason_and_drops_unused_keys():
    message = comment_notification().to_message()

    assert message == {
        "kind": "comment_upsert",
        "trace_id": "t",
        "results": [
            {"status": "succeeded", "reason_code": None, "complaint_detail_id": 9001},
            {
                "status": "failed",
                "reason_code": "EMBEDDING_FAILED",
                "complaint_detail_id": 9002,
            },
        ],
    }


def test_sends_to_configured_queue(monkeypatch):
    monkeypatch.setattr(sqs, "NOTIFICATION_QUEUE_URL", "https://sqs/queue")
    fake = FakeSqs()

    sqs.send_notification(comment_notification(), client=fake)

    assert fake.sent[0]["QueueUrl"] == "https://sqs/queue"
    assert json.loads(fake.sent[0]["MessageBody"])["kind"] == "comment_upsert"


def test_no_queue_means_no_send(monkeypatch):
    monkeypatch.setattr(sqs, "NOTIFICATION_QUEUE_URL", None)
    fake = FakeSqs()

    sqs.send_notification(comment_notification(), client=fake)

    assert fake.sent == []


def _run_document_job(monkeypatch, payload, *, index_status, reasons=()):
    from zipsai.api import indexing
    from zipsai.contracts.indexing import IndexingJobRequest

    monkeypatch.setattr(indexing, "_dependencies", lambda: (object(), object()))

    def fake_run(request, *, on_fail, on_stats=None, **_kwargs):
        for reason in reasons:
            on_fail(reason)
        return index_status

    monkeypatch.setattr(indexing, "run_indexing_job", fake_run)
    monkeypatch.setattr(indexing, "delete_missing_documents", lambda *a: None)
    outcome = indexing._run_job(IndexingJobRequest.model_validate(payload), "job")
    return [item.to_dict() for item in outcome.results]


DOC = {
    "building_id": 101,
    "trace_id": "t",
    "doc_id": "notice-1",
    "title": "공지",
    "file_key": "k",
    "valid_doc_ids": ["notice-1"],
}


def test_document_success_returns_index_and_cleanup(monkeypatch):
    from zipsai.contracts.indexing import JobStatus

    results = _run_document_job(monkeypatch, DOC, index_status=JobStatus.SUCCEEDED)

    assert [item["kind"] for item in results] == ["document_index", "document_cleanup"]
    assert results[0]["reason_code"] is None
    assert results[0]["doc_id"] == "notice-1"
    assert "doc_id" not in results[1]


def test_index_failure_skips_cleanup(monkeypatch):
    from zipsai.contracts.indexing import JobStatus

    results = _run_document_job(
        monkeypatch, DOC, index_status=JobStatus.FAILED, reasons=["PARSE_FAILED"]
    )

    assert results[0]["reason_code"] == "PARSE_FAILED"
    assert results[0]["retryable"] is False
    assert results[1]["status"] == "skipped"
    assert results[1]["reason_code"] == "INDEX_FAILED"


def test_missing_queue_is_logged_as_a_warning(monkeypatch, caplog):
    monkeypatch.setattr(sqs, "NOTIFICATION_QUEUE_URL", None)

    sqs.send_notification(comment_notification(), client=FakeSqs())

    assert [
        r.levelname for r in caplog.records if r.message == "notification_skipped"
    ] == ["WARNING"]


def test_send_failure_is_logged_not_raised(monkeypatch, caplog):
    from botocore.exceptions import ClientError

    class Broken:
        def send_message(self, **_):
            raise ClientError({"Error": {"Code": "AccessDenied"}}, "SendMessage")

    monkeypatch.setattr(sqs, "NOTIFICATION_QUEUE_URL", "https://sqs/queue")

    sqs.send_notification(comment_notification(), client=Broken())

    assert any(r.message == "notification_failed" for r in caplog.records)


def test_pipeline_reports_the_failed_stage_as_reason_code(tmp_path):
    from qdrant_client import QdrantClient

    from zipsai.contracts.indexing import IndexingJobRequest, JobStatus
    from zipsai.errors import DocumentFetchError
    from zipsai.indexing.pipeline import run_indexing_job

    def broken_download(_key, _dir):
        raise DocumentFetchError("no object")

    reasons = []
    status = run_indexing_job(
        IndexingJobRequest.model_validate(DOC),
        download=broken_download,
        encoder=None,
        client=QdrantClient(":memory:"),
        on_fail=reasons.append,
    )

    assert status is JobStatus.FAILED
    assert reasons == ["FILE_DOWNLOAD_FAILED"]
