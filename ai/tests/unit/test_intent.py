import pytest

import zipsai.orchestration.intent as intent_module
from zipsai.contracts.converse import (
    ConverseRequest,
    ImageAnalysis,
    ImageAttachment,
    ImageObservation,
    Route,
)
from zipsai.errors import (
    ImageAnalysisError,
    IntentClassificationError,
    LlmUnavailableError,
)
from zipsai.orchestration.intent import classify_intent
from zipsai.orchestration.state import AgentState


def _build_state(
    text: str | None,
    *,
    images: list[dict[str, object]] | None = None,
    history: list[dict[str, object]] | None = None,
    current_route: Route | None = None,
) -> AgentState:
    return {
        "request": ConverseRequest.model_validate(
            {
                "building_id": 1,
                "room_no": "301",
                "resident_id": "linda",
                "conversation_id": "conv-001",
                "turn_id": "turn-001",
                "trace_id": "trace-001",
                "current_route": current_route,
                "current_complaint_state": None,
                "message": {
                    "message_id": "msg-001",
                    "text": text,
                    "images": images or [],
                },
                "conversation_history": history or [],
                "complaint_draft": None,
            }
        ),
        "route": None,
        "complaint_state": None,
        "reply": None,
        "result": None,
        "image_analysis": None,
    }


def test_classify_intent_analyzes_new_images_and_returns_route(monkeypatch):
    calls: list[tuple[list[ImageAttachment], str, str]] = []
    observation = ImageObservation(
        attachmentId=123, summary="세탁기 아래 물이 고여 있음", ocrText=None
    )

    def classify(images, *, system_prompt, user_prompt):
        calls.append((images, system_prompt, user_prompt))
        return Route.COMPLAINT, ImageAnalysis(images=[observation])

    monkeypatch.setattr(intent_module, "classify_and_analyze", classify)
    state = _build_state(
        "세탁기 밑으로 물이 새요",
        images=[{"attachmentId": 123, "url": "https://example.com/leak.jpg"}],
    )

    result = classify_intent(state)

    assert result["route"] is Route.COMPLAINT
    assert result["image_analysis"].images == [observation]
    assert calls[0][0][0].attachment_id == 123
    assert "세탁기 밑으로 물이 새요" in calls[0][2]
    assert "현재 진행 중인 route" in calls[0][2]


def test_classify_intent_still_uses_vlm_without_new_images(monkeypatch):
    calls: list[list[ImageAttachment]] = []

    def classify(images, **_kwargs):
        calls.append(images)
        return Route.KNOWLEDGE, ImageAnalysis(images=[])

    monkeypatch.setattr(intent_module, "classify_and_analyze", classify)

    result = classify_intent(_build_state("쓰레기 배출일이 언제예요?"))

    assert calls == [[]]
    assert result == {"route": Route.KNOWLEDGE, "image_analysis": None}


def test_classify_intent_wraps_invalid_vlm_response(monkeypatch):
    def fail(*_args, **_kwargs):
        raise ImageAnalysisError("invalid structured response")

    monkeypatch.setattr(intent_module, "classify_and_analyze", fail)

    with pytest.raises(IntentClassificationError):
        classify_intent(_build_state("화장실에서 물이 새요"))


def test_classify_intent_propagates_provider_failure(monkeypatch):
    def fail(*_args, **_kwargs):
        raise LlmUnavailableError("provider unavailable")

    monkeypatch.setattr(intent_module, "classify_and_analyze", fail)

    with pytest.raises(LlmUnavailableError):
        classify_intent(_build_state("화장실에서 물이 새요"))
