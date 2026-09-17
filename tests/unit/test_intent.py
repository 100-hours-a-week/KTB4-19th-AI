import pytest

import zipsai.orchestration.intent as intent_module
from zipsai.contracts.converse import ConverseRequest, Route
from zipsai.errors import IntentClassificationError, LlmUnavailableError
from zipsai.orchestration.intent import classify_intent, parse_route
from zipsai.orchestration.state import AgentState


def _build_state(text: str) -> AgentState:
    return {
        "request": ConverseRequest.model_validate(
            {
                "building_id": 1,
                "resident_context": {"unit_id": 57, "resident_id": "linda"},
                "conversation_id": "conv-001",
                "trace_id": "trace-001",
                "message": {"message_id": "msg-001", "text": text},
            }
        ),
        "route": None,
        "conversation_state": None,
        "response": None,
        "reply": None,
    }


def test_parse_route_accepts_knowledge():
    assert parse_route("knowledge") is Route.KNOWLEDGE


def test_parse_route_rejects_unknown_response():
    with pytest.raises(IntentClassificationError):
        parse_route("anything_else")


def test_classify_intent_returns_route_from_llm(monkeypatch: pytest.MonkeyPatch):
    received_prompts: list[tuple[str, str]] = []

    def fake_generate_text(system_prompt: str, user_prompt: str) -> str:
        received_prompts.append((system_prompt, user_prompt))
        return "knowledge"

    monkeypatch.setattr(
        intent_module,
        "generate_text",
        fake_generate_text,
        raising=False,
    )

    result = classify_intent(_build_state("쓰레기 언제 버려요?"))

    assert result == {"route": Route.KNOWLEDGE}
    assert "쓰레기 언제 버려요?" in received_prompts[0][1]


def test_classify_intent_propagates_llm_failure(monkeypatch: pytest.MonkeyPatch):
    def failing_generate_text(system_prompt: str, user_prompt: str) -> str:
        raise LlmUnavailableError("LLM request failed")

    monkeypatch.setattr(
        intent_module,
        "generate_text",
        failing_generate_text,
        raising=False,
    )

    with pytest.raises(LlmUnavailableError):
        classify_intent(_build_state("화장실에서 물이 새요"))
