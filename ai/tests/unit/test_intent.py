import pytest

import zipsai.orchestration.intent as intent_module
from zipsai.contracts.converse import ConverseRequest, Route
from zipsai.errors import IntentClassificationError, LlmUnavailableError
from zipsai.orchestration.intent import _RouteResponse, classify_intent
from zipsai.orchestration.state import AgentState


def _build_state(text: str, image_urls: list[str] | None = None) -> AgentState:
    return {
        "request": ConverseRequest.model_validate(
            {
                "building_id": 1,
                "room_no": "301",
                "resident_id": "linda",
                "conversation_id": "conv-001",
                "turn_id": "turn-001",
                "trace_id": "trace-001",
                "current_route": None,
                "current_complaint_state": None,
                "message": {
                    "message_id": "msg-001",
                    "text": text,
                    "image_urls": image_urls or [],
                },
                "conversation_history": [],
                "complaint_draft": None,
            }
        ),
        "route": None,
        "complaint_state": None,
        "reply": None,
    }


def test_classify_intent_returns_route_from_llm(monkeypatch: pytest.MonkeyPatch):
    received_prompts: list[tuple[str, str]] = []

    def fake_generate_structured(system_prompt: str, user_prompt: str, **kwargs):
        received_prompts.append((system_prompt, user_prompt))
        return _RouteResponse(route=Route.KNOWLEDGE)

    monkeypatch.setattr(intent_module, "generate_structured", fake_generate_structured)

    result = classify_intent(_build_state("쓰레기 언제 버려요?"))

    assert result == {"route": Route.KNOWLEDGE}
    assert "쓰레기 언제 버려요?" in received_prompts[0][1]


def test_classify_intent_raises_when_llm_returns_nothing_usable(
    monkeypatch: pytest.MonkeyPatch,
):
    """refusal이나 길이 제한으로 파싱할 내용이 없으면 generate_structured가 None을 준다."""
    monkeypatch.setattr(
        intent_module, "generate_structured", lambda *_, **__: None
    )

    with pytest.raises(IntentClassificationError):
        classify_intent(_build_state("화장실에서 물이 새요"))


def test_classify_intent_propagates_llm_failure(monkeypatch: pytest.MonkeyPatch):
    def failing_generate_structured(system_prompt: str, user_prompt: str, **kwargs):
        raise LlmUnavailableError("LLM request failed")

    monkeypatch.setattr(intent_module, "generate_structured", failing_generate_structured)

    with pytest.raises(LlmUnavailableError):
        classify_intent(_build_state("화장실에서 물이 새요"))


def test_classify_intent_normalizes_blank_text_with_an_image(monkeypatch):
    received_prompts: list[tuple[str, str]] = []

    def fake_generate_structured(system_prompt: str, user_prompt: str, **kwargs):
        received_prompts.append((system_prompt, user_prompt))
        return _RouteResponse(route=Route.COMPLAINT)

    monkeypatch.setattr(intent_module, "generate_structured", fake_generate_structured)

    classify_intent(_build_state("   ", ["https://example.com/leak.jpg"]))

    assert "현재 발화: 없음" in received_prompts[0][1]
