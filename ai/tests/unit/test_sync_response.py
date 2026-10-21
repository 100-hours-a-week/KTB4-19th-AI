from zipsai.contracts.notification import ResultStatus
from zipsai.contracts.sync_response import CommentResult, DocumentResult, SyncBody


def test_success_keeps_null_reason_code():
    body = SyncBody(
        trace_id="trace-001",
        results=[
            DocumentResult(
                kind="document_index",
                status=ResultStatus.SUCCEEDED,
                doc_id="notice-1",
            ),
            DocumentResult(
                kind="document_cleanup",
                status=ResultStatus.SUCCEEDED,
            ),
        ],
    ).to_body()

    assert body["results"][0]["reason_code"] is None
    assert "reason_code" in body["results"][0]
    assert body["results"][0]["doc_id"] == "notice-1"
    assert "doc_id" not in body["results"][1]
    assert body["results"][1]["retryable"] is False


def test_comment_result_keeps_null_target_keys():
    raw = CommentResult(status=ResultStatus.SKIPPED, reason_code="VALIDATION_FAILED")

    assert raw.to_dict() == {
        "status": "skipped",
        "reason_code": "VALIDATION_FAILED",
        "retryable": False,
        "complaint_detail_id": None,
        "updated_at": None,
    }
