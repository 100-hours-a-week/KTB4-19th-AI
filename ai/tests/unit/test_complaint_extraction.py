import pytest

import zipsai.complaint.node as node_module
from zipsai.complaint.node import extract_complaint_fields, handle_complaint
from zipsai.contracts.converse import ComplaintDraft, ConverseRequest
from zipsai.errors import ComplaintExtractionError


def _make_request(text: str) -> ConverseRequest:
    return ConverseRequest.model_validate(
        {
            "building_id": 1,
            "room_no": "301",
            "resident_id": "linda",
            "conversation_id": "conv-001",
            "trace_id": "trace-001",
            "current_route": "complaint",
            "current_complaint_state": "collecting",
            "message": {"message_id": "msg-001", "text": text, "image_urls": []},
            "conversation_history": [],
            "complaint_draft": None,
        }
    )


def test_extract_complaint_fields_parses_llm_json(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        node_module,
        "generate_text",
        lambda system_prompt, user_prompt: (
            '{"issue_type": "leak", "location": "화장실", "symptom": "천장에서 물이 떨어져요"}'
        ),
    )

    result = extract_complaint_fields(
        _make_request("화장실 천장에서 물이 계속 떨어져요")
    )

    assert result == ComplaintDraft(
        issue_type="leak", location="화장실", symptom="천장에서 물이 떨어져요"
    )


def test_extract_complaint_fields_defaults_missing_keys_to_none(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        node_module,
        "generate_text",
        lambda system_prompt, user_prompt: (
            '{"issue_type": null, "location": null, "symptom": null}'
        ),
    )

    result = extract_complaint_fields(_make_request("음.."))

    assert result == ComplaintDraft()


def test_extract_complaint_fields_raises_on_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        node_module, "generate_text", lambda system_prompt, user_prompt: "not json"
    )

    with pytest.raises(ComplaintExtractionError):
        extract_complaint_fields(_make_request("아무 말"))


def test_extract_complaint_fields_raises_on_invalid_issue_type(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        node_module,
        "generate_text",
        lambda system_prompt, user_prompt: (
            '{"issue_type": "bogus", "location": null, "symptom": null}'
        ),
    )

    with pytest.raises(ComplaintExtractionError):
        extract_complaint_fields(_make_request("아무 말"))


def test_handle_complaint_merges_new_values_without_erasing_existing_fields(
    monkeypatch: pytest.MonkeyPatch,
):
    request = _make_request("화장실로 정정할게요.")
    request.complaint_draft = ComplaintDraft(
        issue_type="water_supply",
        location="주방",
        symptom="온수가 나오지 않음",
    )
    monkeypatch.setattr(
        node_module,
        "extract_complaint_fields",
        lambda _: ComplaintDraft(location="화장실"),
    )

    result = handle_complaint(request)["result"]

    assert result.complaint_draft == ComplaintDraft(
        issue_type="water_supply",
        location="화장실",
        symptom="온수가 나오지 않음",
    )
    assert result.missing_fields == []
