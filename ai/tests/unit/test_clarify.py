import zipsai.orchestration.clarify as clarify_module
from zipsai.contracts.converse import (
    ComplaintDraft,
    ComplaintState,
    ConverseRequest,
    Route,
    RouteResult,
)
from zipsai.orchestration.clarify import CLARIFY_REPLY, handle_clarify
from zipsai.orchestration.state import AgentState


def _make_state(
    current_route: Route | None = None,
    complaint_draft: dict[str, object] | None = None,
) -> AgentState:
    request = ConverseRequest.model_validate(
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
            "complaint_draft": complaint_draft,
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

    assert result == {
        "reply": CLARIFY_REPLY,
        "complaint_state": None,
        "result": RouteResult(),
    }


def test_handle_clarify_keeps_the_complaint_draft_it_was_given():
    # 수집 중에 경로가 바뀌어도 초안을 돌려줘야 한다. 빼면 백엔드가 받는
    # complaint_draft가 null이 되어 모아둔 값이 사라진다.
    draft = {"issue_type": "leak", "location": "화장실", "symptom": "물이 새요"}

    result = handle_clarify(_make_state(complaint_draft=draft))

    assert result["result"].complaint_draft == ComplaintDraft(**draft)


def test_handle_clarify_uses_llm_after_prior_clarify(
    monkeypatch,
):
    received_prompts: list[tuple[str, str]] = []

    def fake_generate_text(system_prompt: str, user_prompt: str, **_: object) -> str:
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
        "result": RouteResult(),
    }
    assert "그거요" in received_prompts[0][1]
