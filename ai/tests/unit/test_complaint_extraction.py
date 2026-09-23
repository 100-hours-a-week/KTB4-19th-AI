import pytest

import zipsai.complaint.node as node_module
from zipsai.complaint.node import extract_complaint_fields, handle_complaint
from zipsai.contracts.converse import (
    ComplaintDraft,
    ConverseRequest,
    ImageAnalysis,
    ImageObservation,
)
from zipsai.errors import ComplaintExtractionError, ImageAnalysisError


def _make_request(text: str, image_urls: list[str] | None = None) -> ConverseRequest:
    return ConverseRequest.model_validate(
        {
            "building_id": 1,
            "room_no": "301",
            "resident_id": "linda",
            "conversation_id": "conv-001",
            "trace_id": "trace-001",
            "current_route": "complaint",
            "current_complaint_state": "collecting",
            "message": {
                "message_id": "msg-001",
                "text": text,
                "image_urls": image_urls or [],
            },
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


def test_extract_complaint_fields_accepts_json_code_fence(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        node_module,
        "generate_text",
        lambda system_prompt, user_prompt: (
            '```json\n{"issue_type": "leak", "location": "화장실", '
            '"symptom": "천장에서 물이 떨어져요"}\n```'
        ),
    )

    result = extract_complaint_fields(
        _make_request("화장실 천장에서 물이 계속 떨어져요")
    )

    assert result.issue_type == "leak"


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


def test_handle_complaint_returns_image_analysis_without_changing_text_fields(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        node_module,
        "extract_complaint_fields",
        lambda _: ComplaintDraft(location="욕실"),
    )
    monkeypatch.setattr(
        node_module,
        "analyze_images",
        lambda _, __: ImageAnalysis(
            images=[
                ImageObservation(
                    url="https://example.com/leak.jpg",
                    summary="바닥에 물이 고여 있음",
                    ocr_text="E1",
                )
            ]
        ),
        raising=False,
    )

    result = handle_complaint(
        _make_request("욕실 바닥이 젖었어요", ["https://example.com/leak.jpg"])
    )["result"]

    assert result.complaint_draft.location == "욕실"
    assert result.complaint_draft.image_urls == ["https://example.com/leak.jpg"]
    assert result.image_analysis.images[0].ocr_text == "E1"


def test_handle_complaint_keeps_text_flow_when_vlm_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    def raise_image_analysis_error(_: list[str], __: str) -> ImageAnalysis:
        raise ImageAnalysisError("VLM returned an invalid image analysis")

    monkeypatch.setattr(
        node_module,
        "extract_complaint_fields",
        lambda _: ComplaintDraft(location="욕실"),
    )
    monkeypatch.setattr(
        node_module,
        "analyze_images",
        raise_image_analysis_error,
        raising=False,
    )

    result = handle_complaint(
        _make_request("욕실 바닥이 젖었어요", ["https://example.com/leak.jpg"])
    )["result"]

    assert result.image_analysis is None
    assert result.complaint_draft.image_urls == ["https://example.com/leak.jpg"]


def test_handle_complaint_acknowledges_photo_when_fields_still_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        node_module, "extract_complaint_fields", lambda _: ComplaintDraft()
    )
    monkeypatch.setattr(
        node_module,
        "analyze_images",
        lambda _, __: ImageAnalysis(
            images=[
                ImageObservation(
                    url="https://example.com/leak.jpg",
                    summary="천장에서 물이 흐르는 흔적",
                    ocr_text=None,
                )
            ]
        ),
        raising=False,
    )

    reply = handle_complaint(
        _make_request("이거 보세요", ["https://example.com/leak.jpg"])
    )["reply"]

    assert (
        reply
        == "사진은 확인했습니다. 정확한 접수를 위해 위치와 증상을 간단히 말씀해 주시겠어요?"
    )


def test_handle_complaint_uses_generic_reply_without_photo(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        node_module, "extract_complaint_fields", lambda _: ComplaintDraft()
    )

    reply = handle_complaint(_make_request("음.."))["reply"]

    assert reply == "민원 접수를 위해 발생 위치와 불편 증상을 알려주세요."
