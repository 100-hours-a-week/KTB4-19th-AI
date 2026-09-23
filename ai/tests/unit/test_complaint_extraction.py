from datetime import datetime

import pytest

import zipsai.complaint.node as node_module
from zipsai.complaint.node import (
    _extract_complaint_fields_and_reply,
    extract_complaint_fields,
    handle_complaint,
)
from zipsai.contracts.converse import (
    ComplaintDraft,
    ComplaintState,
    ConverseRequest,
    ImageAnalysis,
    ImageObservation,
)
from zipsai.errors import ComplaintExtractionError, ImageAnalysisError


def _make_request(
    text: str,
    image_urls: list[str] | None = None,
    current_complaint_state: str = "collecting",
) -> ConverseRequest:
    return ConverseRequest.model_validate(
        {
            "building_id": 1,
            "room_no": "301",
            "resident_id": "linda",
            "conversation_id": "conv-001",
            "trace_id": "trace-001",
            "current_route": "complaint",
            "current_complaint_state": current_complaint_state,
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


def test_extract_complaint_fields_retries_once_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[int] = []

    def flaky_generate_text(system_prompt: str, user_prompt: str) -> str:
        calls.append(1)
        if len(calls) == 1:
            return "not json"
        return '{"issue_type": "leak", "location": "화장실", "symptom": "물이 새요"}'

    monkeypatch.setattr(node_module, "generate_text", flaky_generate_text)

    result = extract_complaint_fields(_make_request("화장실에서 물이 새요"))

    assert len(calls) == 2
    assert result == ComplaintDraft(
        issue_type="leak", location="화장실", symptom="물이 새요"
    )


def test_extract_complaint_fields_gives_up_after_exhausting_retries(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[int] = []

    def always_broken(system_prompt: str, user_prompt: str) -> str:
        calls.append(1)
        return "not json"

    monkeypatch.setattr(node_module, "generate_text", always_broken)

    with pytest.raises(ComplaintExtractionError):
        extract_complaint_fields(_make_request("아무 말"))

    assert len(calls) == node_module._EXTRACTION_ATTEMPTS


def test_extract_complaint_fields_parses_occurred_at(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        node_module,
        "generate_text",
        lambda system_prompt, user_prompt: (
            '{"issue_type": null, "location": null, "symptom": null, "occurred_at": "2026-09-22"}'
        ),
    )

    result = extract_complaint_fields(_make_request("어제부터 그랬어요"))

    assert result.occurred_at == datetime(2026, 9, 22)


def test_extract_complaint_fields_passes_todays_kst_date_to_prompt(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, str] = {}

    def fake_generate_text(system_prompt: str, user_prompt: str) -> str:
        captured["user_prompt"] = user_prompt
        return '{"issue_type": null, "location": null, "symptom": null, "occurred_at": null}'

    monkeypatch.setattr(node_module, "generate_text", fake_generate_text)

    extract_complaint_fields(_make_request("어제부터 그랬어요"))

    today = datetime.now(node_module._KST).date().isoformat()
    assert f"오늘 날짜(Asia/Seoul): {today}" in captured["user_prompt"]


def test_extract_complaint_fields_and_reply_parses_reply_and_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        node_module,
        "generate_text",
        lambda system_prompt, user_prompt: (
            '{"issue_type": null, "location": "화장실", "symptom": null, "occurred_at": null, '
            '"missing": ["symptom"], "reply": "어떤 증상인지 알려주시겠어요?"}'
        ),
    )

    draft, reply, missing = _extract_complaint_fields_and_reply(
        _make_request("화장실이 이상해요")
    )

    assert draft == ComplaintDraft(location="화장실")
    assert reply == "어떤 증상인지 알려주시겠어요?"
    assert missing == {"symptom"}


def test_extract_complaint_fields_and_reply_defaults_missing_keys_to_empty(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        node_module,
        "generate_text",
        lambda system_prompt, user_prompt: (
            '{"issue_type": null, "location": null, "symptom": null}'
        ),
    )

    _, reply, missing = _extract_complaint_fields_and_reply(_make_request("음.."))

    assert reply == ""
    assert missing == set()


def test_extract_complaint_fields_and_reply_ignores_invalid_missing_entries(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        node_module,
        "generate_text",
        lambda system_prompt, user_prompt: (
            '{"issue_type": null, "location": null, "symptom": null, '
            '"missing": ["location", "issue_type", "bogus"], "reply": "위치를 알려주세요"}'
        ),
    )

    _, _reply, missing = _extract_complaint_fields_and_reply(_make_request("음.."))

    assert missing == {"location"}


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
        "_extract_complaint_fields_and_reply",
        lambda _: (ComplaintDraft(location="화장실"), "", set()),
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
        "_extract_complaint_fields_and_reply",
        lambda _: (ComplaintDraft(location="욕실"), "", set()),
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
        "_extract_complaint_fields_and_reply",
        lambda _: (ComplaintDraft(location="욕실"), "", set()),
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


def test_handle_complaint_distinguishes_photo_analysis_failure_from_no_photo(
    monkeypatch: pytest.MonkeyPatch,
):
    def raise_image_analysis_error(_: list[str], __: str) -> ImageAnalysis:
        raise ImageAnalysisError("VLM returned an invalid image analysis")

    monkeypatch.setattr(
        node_module,
        "_extract_complaint_fields_and_reply",
        lambda _: (ComplaintDraft(location="욕실"), "", set()),
    )
    monkeypatch.setattr(
        node_module, "analyze_images", raise_image_analysis_error, raising=False
    )

    reply = handle_complaint(
        _make_request("욕실 바닥이 젖었어요", ["https://example.com/leak.jpg"])
    )["reply"]

    assert reply == "사진을 받았지만 분석에 실패했어요. 어떤 불편 증상인지 알려주세요."


def test_handle_complaint_acknowledges_photo_when_fields_still_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        node_module,
        "_extract_complaint_fields_and_reply",
        lambda _: (ComplaintDraft(), "", set()),
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
        == "사진은 확인했습니다. 민원 접수를 위해 발생 위치와 불편 증상을 알려주세요."
    )


def test_handle_complaint_uses_generic_reply_without_photo(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        node_module,
        "_extract_complaint_fields_and_reply",
        lambda _: (ComplaintDraft(), "", set()),
    )

    reply = handle_complaint(_make_request("음.."))["reply"]

    assert reply == "민원 접수를 위해 발생 위치와 불편 증상을 알려주세요."


def test_handle_complaint_uses_llm_generated_reply_when_missing_matches(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        node_module,
        "_extract_complaint_fields_and_reply",
        lambda _: (
            ComplaintDraft(),
            "화장실 세면대인지 변기 쪽인지, 어떤 증상인지 알려주시겠어요?",
            {"location", "symptom"},
        ),
    )

    reply = handle_complaint(_make_request("화장실이 좀 이상해요"))["reply"]

    assert reply == "화장실 세면대인지 변기 쪽인지, 어떤 증상인지 알려주시겠어요?"


def test_handle_complaint_falls_back_when_llm_missing_disagrees_with_actual(
    monkeypatch: pytest.MonkeyPatch,
):
    # LLM은 location만 빠졌다고(잘못) 생각하지만, 실제 missing_fields는 symptom이다.
    monkeypatch.setattr(
        node_module,
        "_extract_complaint_fields_and_reply",
        lambda _: (
            ComplaintDraft(location="화장실"),
            "정확한 위치를 알려주시겠어요?",
            {"location"},
        ),
    )

    reply = handle_complaint(_make_request("화장실이 이상해요"))["reply"]

    assert reply == "어떤 불편 증상인지 알려주세요."


def test_handle_complaint_prefixes_llm_reply_when_photo_analyzed(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        node_module,
        "_extract_complaint_fields_and_reply",
        lambda _: (
            ComplaintDraft(),
            "정확히 어디쯤인지 알려주시겠어요?",
            {"location", "symptom"},
        ),
    )
    monkeypatch.setattr(
        node_module,
        "analyze_images",
        lambda _, __: ImageAnalysis(
            images=[
                ImageObservation(
                    url="https://example.com/leak.jpg",
                    summary="바닥에 물이 고여 있음",
                    ocr_text=None,
                )
            ]
        ),
        raising=False,
    )

    reply = handle_complaint(
        _make_request("이거 보세요", ["https://example.com/leak.jpg"])
    )["reply"]

    assert reply == "사진은 확인했습니다. 정확히 어디쯤인지 알려주시겠어요?"


def test_handle_complaint_sets_ready_to_confirm_when_fields_complete(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        node_module,
        "_extract_complaint_fields_and_reply",
        lambda _: (
            ComplaintDraft(issue_type="leak", location="화장실", symptom="물이 새요"),
            "",
            set(),
        ),
    )

    result = handle_complaint(_make_request("화장실에서 물이 새요"))

    assert result["complaint_state"] == ComplaintState.READY_TO_CONFIRM


def test_handle_complaint_keeps_collecting_when_fields_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        node_module,
        "_extract_complaint_fields_and_reply",
        lambda _: (ComplaintDraft(location="화장실"), "", set()),
    )

    result = handle_complaint(_make_request("화장실이 이상해요"))

    assert result["complaint_state"] == ComplaintState.COLLECTING


def test_handle_complaint_merges_edit_after_ready_to_confirm_without_erasing_fields(
    monkeypatch: pytest.MonkeyPatch,
):
    request = _make_request(
        "화장실로 정정할게요.", current_complaint_state="ready_to_confirm"
    )
    request.complaint_draft = ComplaintDraft(
        issue_type="water_supply",
        location="주방",
        symptom="온수가 나오지 않음",
    )
    monkeypatch.setattr(
        node_module,
        "_extract_complaint_fields_and_reply",
        lambda _: (ComplaintDraft(location="화장실"), "", set()),
    )

    result = handle_complaint(request)

    assert result["result"].complaint_draft == ComplaintDraft(
        issue_type="water_supply",
        location="화장실",
        symptom="온수가 나오지 않음",
    )
    assert result["complaint_state"] == ComplaintState.READY_TO_CONFIRM


def test_handle_complaint_asks_only_about_missing_symptom(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        node_module,
        "_extract_complaint_fields_and_reply",
        lambda _: (ComplaintDraft(location="화장실"), "", set()),
    )

    reply = handle_complaint(_make_request("화장실이 이상해요"))["reply"]

    assert reply == "어떤 불편 증상인지 알려주세요."


def test_handle_complaint_asks_only_about_missing_location(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        node_module,
        "_extract_complaint_fields_and_reply",
        lambda _: (ComplaintDraft(symptom="물이 새요"), "", set()),
    )

    reply = handle_complaint(_make_request("물이 새요"))["reply"]

    assert reply == "정확한 발생 위치를 알려주세요."


def test_handle_complaint_defaults_issue_type_to_other_when_unclassified(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        node_module,
        "_extract_complaint_fields_and_reply",
        lambda _: (ComplaintDraft(location="화장실", symptom="이상해요"), "", set()),
    )

    result = handle_complaint(_make_request("화장실이 이상해요"))

    assert result["complaint_state"] == ComplaintState.READY_TO_CONFIRM
    assert result["result"].complaint_draft.issue_type == "other"


def test_handle_complaint_merges_occurred_at_without_erasing_existing_value(
    monkeypatch: pytest.MonkeyPatch,
):
    request = _make_request("화장실로 정정할게요.")
    request.complaint_draft = ComplaintDraft(
        issue_type="water_supply",
        location="주방",
        symptom="온수가 나오지 않음",
        occurred_at=datetime(2026, 9, 20),
    )
    monkeypatch.setattr(
        node_module,
        "_extract_complaint_fields_and_reply",
        lambda _: (ComplaintDraft(location="화장실"), "", set()),
    )

    result = handle_complaint(request)["result"]

    assert result.complaint_draft.occurred_at == datetime(2026, 9, 20)
