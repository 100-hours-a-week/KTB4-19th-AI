import pytest
from pydantic import ValidationError

from zipsai.contracts.converse import (
    ComplaintDraft,
    ConversationState,
    ConverseRequest,
    Route,
)
from zipsai.orchestration.state import AgentState


def test_converse_request_accepts_real_example():
    payload = {
        "building_id": 1,
        "resident_context": {"unit_id": 57, "resident_id": "linda"},
        "conversation_id": "conv-001",
        "trace_id": "conv-001-003",
        "current_route": "complaint",
        "conversation_state": "action_selection",
        "message": {
            "message_id": "msg-003",
            "text": None,
            "image_urls": [],
            "user_action": "REQUEST_REGISTER",
        },
        "conversation_history": [],
        "complaint_draft": None,
    }
    req = ConverseRequest.model_validate(payload)
    assert req.current_route is Route.COMPLAINT
    assert req.conversation_state is ConversationState.ACTION_SELECTION


def test_complaint_draft_rejects_invalid_issue_type():
    with pytest.raises(ValidationError):
        ComplaintDraft(issue_type="아무거나")


def test_agent_state_keeps_request_route_response_and_conversation_state():
    request = ConverseRequest.model_validate(
        {
            "building_id": 1,
            "resident_context": {"unit_id": 57, "resident_id": "linda"},
            "conversation_id": "conv-001",
            "trace_id": "trace-001",
            "message": {"message_id": "msg-001"},
        }
    )
    state: AgentState = {
        "request": request,
        "route": None,
        "conversation_state": ConversationState.COLLECTING,
        "response": None,
        "reply": None,
    }

    assert state["request"] is request
    assert state["route"] is None
    assert state["conversation_state"] is ConversationState.COLLECTING
    assert state["response"] is None
    assert state["reply"] is None
    assert "conversation_state" in AgentState.__annotations__
