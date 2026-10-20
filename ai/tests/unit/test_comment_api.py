import json
import logging

import pytest
from pydantic import ValidationError
from test_indexing_api import post

from zipsai.api import comment as comment_api
from zipsai.contracts.comment import MAX_ITEMS, CommentDeleteRequest, CommentRequest
from zipsai.contracts.notification import ItemResult, ResultStatus
from zipsai.indexing import admission
from zipsai.indexing.admission import acquire, in_flight, release
from zipsai.indexing.admission import reset as reset_admission
from zipsai.settings import COMMENT_API_PREFIX


def _item(**overrides):
    item = {
        "complaint_detail_id": 9001,
        "building_id": 101,
        "complaint_type": "COMPLAINT",
        "location": "지하주차장 B2",
        "issue_type": "leak",
        "symptom": "천장에서 물이 떨어집니다",
        "comment": "배관 누수를 확인했고 10/16 보수 예정입니다.",
        "updated_at": "2026-10-15T00:00:12+09:00",
    }
    return {**item, **overrides}


def _request(*items):
    return {"trace_id": "trace-1", "items": list(items) or [_item()]}


def _ok(request):
    return [
        ItemResult(
            status=ResultStatus.SUCCEEDED,
            complaint_detail_id=item.complaint_detail_id,
            updated_at=item.updated_at_raw,
        )
        for item in request.items
    ]


@pytest.fixture(autouse=True)
def _clear_admission():
    reset_admission()
    yield
    reset_admission()


@pytest.fixture
def scheduled(monkeypatch):
    runs = []

    def fake_upsert(request, *, encoder, client, stats=None):
        runs.append(("upsert", request))
        if stats is not None:
            stats["embed_batches"] = 0
            stats["embed_ms"] = 0
            stats["item_ms"] = [1] * len(request.items)
        return _ok(request)

    def fake_delete(request, *, client, stats=None):
        runs.append(("delete", request))
        if stats is not None:
            stats["embed_batches"] = 0
            stats["embed_ms"] = 0
            stats["item_ms"] = [1] * len(request.items)
        return _ok(request)

    monkeypatch.setattr(comment_api, "upsert_comments", fake_upsert)
    monkeypatch.setattr(comment_api, "delete_comments", fake_delete)
    monkeypatch.setattr(comment_api, "_dependencies", lambda: (object(), object()))
    return runs


def test_complaint_and_qa_items_pass():
    qa = _item(
        complaint_detail_id=9002, complaint_type="QA", location=None, issue_type=None
    )

    request = CommentRequest.model_validate(_request(_item(), qa))

    assert [item.complaint_type for item in request.items] == ["COMPLAINT", "QA"]
    assert request.items[0].updated_at.utcoffset() is not None


@pytest.mark.parametrize("field", ["location", "issue_type"])
def test_nullable_fields_must_still_be_present(field):
    item = _item()
    del item[field]

    with pytest.raises(ValidationError):
        CommentRequest.model_validate(_request(item))


@pytest.mark.parametrize(
    "overrides",
    [
        {"updated_at": "2026-10-15T00:00:12"},
        {"complaint_type": "OTHER"},
        {"issue_type": "unknown"},
        {"symptom": "   "},
        {"comment": ""},
        {"complaint_detail_id": "9001"},
    ],
)
def test_invalid_item_is_rejected(overrides):
    with pytest.raises(ValidationError):
        CommentRequest.model_validate(_request(_item(**overrides)))


def test_item_count_must_be_between_1_and_100():
    CommentRequest.model_validate(_request(*[_item() for _ in range(MAX_ITEMS)]))

    for count in (0, MAX_ITEMS + 1):
        body = {"trace_id": "trace-1", "items": [_item() for _ in range(count)]}
        with pytest.raises(ValidationError):
            CommentRequest.model_validate(body)


def _delete_item(**kw):
    return {
        "building_id": 101,
        "complaint_detail_id": 9001,
        "updated_at": "2026-10-15T10:05:00+09:00",
        **kw,
    }


def test_delete_request_needs_building_and_id():
    body = {"trace_id": "trace-1", "items": [_delete_item()]}
    assert CommentDeleteRequest.model_validate(body).items[0].building_id == 101

    body["items"][0].pop("building_id")
    with pytest.raises(ValidationError):
        CommentDeleteRequest.model_validate(body)


@pytest.mark.parametrize(
    "updated_at", [None, "2026-10-15T10:05:00"], ids=["missing", "no-offset"]
)
def test_delete_request_needs_an_aware_updated_at(updated_at):
    entry = _delete_item()
    if updated_at is None:
        del entry["updated_at"]
    else:
        entry["updated_at"] = updated_at

    with pytest.raises(ValidationError):
        CommentDeleteRequest.model_validate({"trace_id": "t", "items": [entry]})


def test_delete_keeps_the_raw_updated_at():
    raw = "2026-10-15T10:05:00.120+09:00"
    body = {"trace_id": "t", "items": [_delete_item(updated_at=raw)]}

    assert CommentDeleteRequest.model_validate(body).items[0].updated_at_raw == raw


def test_returns_200_with_item_result(scheduled):
    status, body = post(f"{COMMENT_API_PREFIX}/comment", _request())

    assert status == 200
    assert json.loads(body) == {
        "trace_id": "trace-1",
        "results": [
            {
                "status": "succeeded",
                "reason_code": None,
                "retryable": False,
                "complaint_detail_id": 9001,
                "updated_at": "2026-10-15T00:00:12+09:00",
            }
        ],
    }
    assert len(scheduled) == 1


def test_delete_returns_200(scheduled):
    payload = {"trace_id": "t", "items": [_delete_item()]}

    status, body = post(f"{COMMENT_API_PREFIX}/comment/delete", payload)

    assert status == 200
    assert json.loads(body)["results"][0]["status"] == "succeeded"
    assert [kind for kind, _request in scheduled] == ["delete"]


def test_delete_response_keeps_raw_updated_at(scheduled):
    raw = "2026-10-15T10:05:00.120+09:00"
    payload = {"trace_id": "t", "items": [_delete_item(updated_at=raw)]}

    status, body = post(f"{COMMENT_API_PREFIX}/comment/delete", payload)

    result = json.loads(body)["results"][0]
    assert status == 200
    assert result["updated_at"] == raw
    assert result["reason_code"] is None
    assert result["retryable"] is False


def test_invalid_items_are_skipped_in_request_order(scheduled):
    bad_type = _item(complaint_detail_id=9004, complaint_type="OTHER")
    missing = _item(complaint_detail_id=9061)
    del missing["location"]

    status, body = post(
        f"{COMMENT_API_PREFIX}/comment", _request(_item(), bad_type, missing)
    )

    results = json.loads(body)["results"]
    assert status == 200
    assert [item["complaint_detail_id"] for item in results] == [9001, 9004, 9061]
    assert [(item["status"], item["reason_code"]) for item in results] == [
        ("succeeded", None),
        ("skipped", "VALIDATION_FAILED"),
        ("skipped", "VALIDATION_FAILED"),
    ]
    assert results[1]["retryable"] is False
    assert results[1]["updated_at"] == "2026-10-15T00:00:12+09:00"
    assert len(scheduled[0][1].items) == 1


def test_non_object_item_has_null_ids(scheduled):
    status, body = post(
        f"{COMMENT_API_PREFIX}/comment", {"trace_id": "t", "items": [True]}
    )

    result = json.loads(body)["results"][0]
    assert status == 200
    assert result["status"] == "skipped"
    assert result["reason_code"] == "VALIDATION_FAILED"
    assert result["complaint_detail_id"] is None
    assert result["updated_at"] is None
    assert scheduled == []


def test_bool_id_is_not_echoed(scheduled):
    raw = _item(complaint_detail_id=True)

    status, body = post(f"{COMMENT_API_PREFIX}/comment", _request(raw))

    result = json.loads(body)["results"][0]
    assert status == 200
    assert result["complaint_detail_id"] is None
    assert result["reason_code"] == "VALIDATION_FAILED"


def test_over_100_items_stays_422():
    body = {"trace_id": "trace-1", "items": [_item() for _ in range(MAX_ITEMS + 1)]}

    status, _payload = post(f"{COMMENT_API_PREFIX}/comment", body)

    assert status == 422


def test_empty_items_stays_422():
    status, _payload = post(
        f"{COMMENT_API_PREFIX}/comment", {"trace_id": "trace-1", "items": []}
    )

    assert status == 422


def test_setup_failure_keeps_validation_skips(monkeypatch):
    def boom():
        raise RuntimeError("qdrant down")

    monkeypatch.setattr(comment_api, "_dependencies", boom)
    bad = _item(complaint_detail_id=9004, complaint_type="OTHER")

    status, body = post(f"{COMMENT_API_PREFIX}/comment", _request(_item(), bad))

    results = json.loads(body)["results"]
    assert status == 200
    assert (results[0]["status"], results[0]["reason_code"], results[0]["retryable"]) == (
        "failed",
        "SETUP_FAILED",
        True,
    )
    assert (results[1]["status"], results[1]["reason_code"], results[1]["retryable"]) == (
        "skipped",
        "VALIDATION_FAILED",
        False,
    )
    assert results[1]["complaint_detail_id"] == 9004


def test_over_capacity_returns_503(scheduled, monkeypatch):
    monkeypatch.setattr(admission, "INDEXING_MAX_CONCURRENCY", 1)
    reset_admission()
    acquire("doc:held")
    try:
        status, body = post(f"{COMMENT_API_PREFIX}/comment", _request())

        assert status == 503
        assert json.loads(body) == {"trace_id": "trace-1", "error_code": "BUSY"}
        assert post.headers["retry-after"] == "10"
        assert scheduled == []
        assert in_flight() == 1
    finally:
        release("doc:held")


def test_comment_done_includes_embed_stats(scheduled, caplog):
    caplog.set_level(logging.INFO, logger="zipsai.api.comment")

    post(f"{COMMENT_API_PREFIX}/comment", _request())

    done = next(record for record in caplog.records if record.getMessage() == "comment_done")
    assert done.kind == "upsert"
    assert done.items == 1
    assert done.succeeded == 1
    assert done.failed == 0
    assert done.skipped == 0
    assert done.http_status == 200
    assert done.embed_batches == 0
    assert done.embed_ms == 0
    item_log = next(
        record for record in caplog.records if record.getMessage() == "comment_item_done"
    )
    assert item_log.comment_chars == len(_item()["comment"])
    assert item_log.symptom_chars == len(_item()["symptom"])
    assert item_log.validation_fields is None
    assert "배관" not in str(item_log.__dict__)


def test_validation_failure_logs_field_names_only(scheduled, caplog):
    caplog.set_level(logging.INFO, logger="zipsai.api.comment")
    bad = _item(complaint_detail_id=9004, complaint_type="OTHER", comment="비밀 문장입니다")

    post(f"{COMMENT_API_PREFIX}/comment", _request(bad))

    item_log = next(
        record for record in caplog.records if record.getMessage() == "comment_item_done"
    )
    assert item_log.reason_code == "VALIDATION_FAILED"
    assert item_log.validation_fields == "complaint_type"
    assert item_log.comment_chars is None
    assert item_log.symptom_chars is None
    assert "비밀" not in str(item_log.__dict__)
    accepted = next(
        record for record in caplog.records if record.getMessage() == "comment_accepted"
    )
    assert accepted.kind == "upsert"
    assert accepted.building_ids == "101"
    assert accepted.complaint_detail_ids == "9004"


def test_comment_results_are_counted(scheduled):
    from zipsai.metrics import INDEXING_RESULTS

    def count(kind: str, status: str, reason: str) -> float:
        return INDEXING_RESULTS.labels(
            kind=kind, status=status, reason_code=reason
        )._value.get()

    before_ok = count("comment_upsert", "succeeded", "none")
    before_skip = count("comment_upsert", "skipped", "VALIDATION_FAILED")
    before_delete = count("comment_delete", "succeeded", "none")
    bad = _item(complaint_detail_id=9004, complaint_type="OTHER")

    post(f"{COMMENT_API_PREFIX}/comment", _request(_item(), bad))
    post(
        f"{COMMENT_API_PREFIX}/comment/delete",
        {"trace_id": "t", "items": [_delete_item()]},
    )

    assert count("comment_upsert", "succeeded", "none") == before_ok + 1
    assert count("comment_upsert", "skipped", "VALIDATION_FAILED") == before_skip + 1
    assert count("comment_delete", "succeeded", "none") == before_delete + 1


def test_422_omits_invalid_items_for_batch_level_error(scheduled):
    status, body = post(f"{COMMENT_API_PREFIX}/comment", {"items": [_item()]})

    assert status == 422
    assert "invalid_items" not in json.loads(body)["error"]
    assert scheduled == []
