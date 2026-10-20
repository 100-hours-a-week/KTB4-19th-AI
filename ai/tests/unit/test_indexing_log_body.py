import json
import logging

import pytest
from test_indexing_api import post, valid_payload

from zipsai.api import comment as comment_api
from zipsai.api import indexing
from zipsai.api.indexing import JobOutcome
from zipsai.contracts.notification import ItemResult, ResultStatus
from zipsai.contracts.sync_response import DocumentResult
from zipsai.indexing import admission
from zipsai.indexing.admission import acquire, release
from zipsai.indexing.admission import reset as reset_admission
from zipsai.observability import JsonFormatter
from zipsai.settings import COMMENT_API_PREFIX

SECRET_TITLE = "SECRETTITLE"
SECRET_QUERY = "X-Amz-Signature=SECRETQUERY"
SECRET_COMMENT = "SECRETCOMMENTBODY"
SECRET_LOCATION = "SECRETLOCATION"
SECRET_SYMPTOM = "SECRETSYMPTOM"
RESULT_EVENTS = {
    "index_result",
    "comment_item_done",
    "comment_done",
    "indexing_rejected",
}


def _comment(**overrides):
    item = {
        "complaint_detail_id": 9001,
        "building_id": 101,
        "complaint_type": "COMPLAINT",
        "location": SECRET_LOCATION,
        "issue_type": "leak",
        "symptom": SECRET_SYMPTOM,
        "comment": SECRET_COMMENT,
        "updated_at": "2026-10-15T00:00:12+09:00",
    }
    return {**item, **overrides}


@pytest.fixture
def _quiet_backends(monkeypatch: pytest.MonkeyPatch):
    reset_admission()

    def fake_job(payload, _job_id):
        return JobOutcome(
            results=[
                DocumentResult(
                    kind="document_index",
                    status=ResultStatus.SUCCEEDED,
                    reason_code=None,
                    retryable=False,
                    doc_id=payload.doc_id,
                )
            ],
            pages=1,
            chunks=1,
            points=1,
        )

    def fake_upsert(request, *, encoder, client, stats=None):
        if stats is not None:
            stats["embed_batches"] = 1
            stats["embed_ms"] = 4
            stats["item_ms"] = [3] * len(request.items)
        return [
            ItemResult(
                status=ResultStatus.SUCCEEDED,
                complaint_detail_id=item.complaint_detail_id,
                updated_at=item.updated_at_raw,
            )
            for item in request.items
        ]

    monkeypatch.setattr(indexing, "_run_job", fake_job)
    monkeypatch.setattr(comment_api, "upsert_comments", fake_upsert)
    monkeypatch.setattr(comment_api, "_dependencies", lambda: (object(), object()))
    yield
    reset_admission()


def test_result_logs_omit_source_text(
    _quiet_backends, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="zipsai.api.indexing")
    caplog.set_level(logging.INFO, logger="zipsai.api.comment")
    payload = valid_payload() | {
        "title": SECRET_TITLE,
        "file_key": f"s3://bucket/buildings/101/notice.pdf?{SECRET_QUERY}",
    }
    bad = _comment(complaint_detail_id=9004, complaint_type="OTHER")

    assert post("/api/v3/ai/indexing/jobs", payload)[0] == 200
    assert (
        post(
            f"{COMMENT_API_PREFIX}/comment",
            {"trace_id": "trace-1", "items": [_comment(), bad]},
        )[0]
        == 200
    )
    monkeypatch.setattr(admission, "INDEXING_MAX_CONCURRENCY", 1)
    reset_admission()
    acquire("doc:held")
    try:
        status, _body = post(
            f"{COMMENT_API_PREFIX}/comment",
            {"trace_id": "trace-busy", "items": [_comment()]},
        )
        assert status == 503
    finally:
        release("doc:held")

    formatter = JsonFormatter()
    lines = [formatter.format(record) for record in caplog.records]
    events = {json.loads(line)["event"] for line in lines}
    assert RESULT_EVENTS <= events

    for line in lines:
        event = json.loads(line)["event"]
        assert SECRET_QUERY not in line
        if event == "job_accepted":
            assert json.loads(line)["file_key_path"].endswith("notice.pdf")
            continue
        if event not in RESULT_EVENTS:
            continue
        assert SECRET_TITLE not in line
        assert SECRET_COMMENT not in line
        assert SECRET_LOCATION not in line
        assert SECRET_SYMPTOM not in line
