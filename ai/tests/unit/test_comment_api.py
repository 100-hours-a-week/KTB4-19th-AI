import json

import pytest
from pydantic import ValidationError
from test_indexing_api import post

from zipsai.api import comment as comment_api
from zipsai.contracts.comment import MAX_ITEMS, CommentDeleteRequest, CommentRequest
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


@pytest.fixture
def scheduled(monkeypatch):
    runs = []
    monkeypatch.setattr(
        comment_api, "_run", lambda kind, payload, _job: runs.append((kind, payload))
    )
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


def test_delete_request_needs_building_and_id():
    body = {
        "trace_id": "trace-1",
        "items": [{"building_id": 101, "complaint_detail_id": 9001}],
    }
    assert CommentDeleteRequest.model_validate(body).items[0].building_id == 101

    body["items"][0].pop("building_id")
    with pytest.raises(ValidationError):
        CommentDeleteRequest.model_validate(body)


def test_accepts_with_empty_body(scheduled):
    assert post(f"{COMMENT_API_PREFIX}/comment", _request()) == (202, b"")
    assert len(scheduled) == 1


def test_delete_accepts_with_empty_body(scheduled):
    payload = {
        "trace_id": "t",
        "items": [{"building_id": 101, "complaint_detail_id": 9001}],
    }
    assert post(f"{COMMENT_API_PREFIX}/comment/delete", payload) == (202, b"")


def test_422_lists_every_invalid_item(scheduled):
    bad_type = _item(complaint_detail_id=9004, complaint_type="OTHER")
    missing = _item(complaint_detail_id=9061)
    del missing["location"]

    status, body = post(
        f"{COMMENT_API_PREFIX}/comment", _request(_item(), bad_type, missing)
    )

    error = json.loads(body)["error"]
    assert status == 422
    assert error["code"] == "VALIDATION_ERROR"
    assert [
        (i["index"], i["complaint_detail_id"], i["field"])
        for i in error["invalid_items"]
    ] == [(1, 9004, "complaint_type"), (2, 9061, "location")]
    assert scheduled == []


def test_422_omits_invalid_items_for_batch_level_error(scheduled):
    status, body = post(f"{COMMENT_API_PREFIX}/comment", {"items": [_item()]})

    assert status == 422
    assert "invalid_items" not in json.loads(body)["error"]
