import zipsai.orchestration.clarify as clarify_module
from zipsai.contracts.converse import ComplaintState, ConverseRequest, Route
from zipsai.orchestration.clarify import CLARIFY_REPLY, handle_clarify
from zipsai.orchestration.state import AgentState


def _make_state(current_route: Route | None = None) -> AgentState:
    request = ConverseRequest.model_validate(
        {
            "building_id": 1,
            "room_no": "301",
            "resident_id": "linda",
            "conversation_id": "conv-001",
            "trace_id": "trace-001",
            "current_route": current_route,
            "current_complaint_state": None,
            "message": {
                "message_id": "msg-001",
                "text": "그거요",
                "image_urls": [],
            },
            "conversation_history": [
                {
                    "message_id": "msg-000",
                    "role": "assistant",
                    "text": CLARIFY_REPLY,
                    "image_urls": [],
                }
            ],
            "complaint_draft": None,
        }
    )
    return {
        "request": request,
        "route": Route.CLARIFY,
        "complaint_state": ComplaintState.COLLECTING,
        "reply": None,
    }


def test_handle_clarify_returns_fixed_question_for_first_clarify():
    result = handle_clarify(_make_state())

    assert result == {"reply": CLARIFY_REPLY, "complaint_state": None}


def test_handle_clarify_uses_llm_after_prior_clarify(
    monkeypatch,
):
    received_prompts: list[tuple[str, str]] = []

    def fake_generate_text(system_prompt: str, user_prompt: str) -> str:
        received_prompts.append((system_prompt, user_prompt))
        return "민원이라면 어떤 시설에 문제가 있나요?"

    monkeypatch.setattr(
        clarify_module,
        "generate_text",
        fake_generate_text,
    )

    result = handle_clarify(_make_state(Route.CLARIFY))

    assert result == {
        "reply": "민원이라면 어떤 시설에 문제가 있나요?",
        "complaint_state": None,
    }
    assert "그거요" in received_prompts[0][1]
