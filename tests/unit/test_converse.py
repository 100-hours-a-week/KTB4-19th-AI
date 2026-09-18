import pytest
from pydantic import ValidationError

from zipsai.contracts.converse import (
    Citation,
    ComplaintDraft,
    ComplaintState,
    ConverseData,
    ConverseRequest,
    IncomingMessage,
    Meta,
    Route,
    RouteResult,
)
from zipsai.orchestration.state import AgentState


def test_converse_request_accepts_real_example():
    payload = {
        "building_id": 1,
        "room_no": "301",
        "resident_id": "linda",
        "conversation_id": "conv-001",
        "trace_id": "conv-001-003",
        "current_route": "complaint",
        "current_complaint_state": "collecting",
        "message": {
            "message_id": "msg-003",
            "text": None,
            "image_urls": ["https://example.com/leak.jpg"],
        },
        "conversation_history": [],
        "complaint_draft": None,
    }
    req = ConverseRequest.model_validate(payload)
    assert req.current_route is Route.COMPLAINT
    assert req.current_complaint_state is ComplaintState.COLLECTING


def test_converse_request_rejects_complaint_state_for_non_complaint_route():
    with pytest.raises(ValidationError):
        ConverseRequest.model_validate(
            {
                "building_id": 1,
                "room_no": "301",
                "resident_id": "linda",
                "conversation_id": "conv-001",
                "trace_id": "trace-001",
                "current_route": "knowledge",
                "current_complaint_state": "collecting",
                "message": {
                    "message_id": "msg-001",
                    "text": "주차는요?",
                    "image_urls": [],
                },
                "conversation_history": [],
                "complaint_draft": None,
            }
        )


def test_converse_request_rejects_legacy_conversation_state_field():
    with pytest.raises(ValidationError):
        ConverseRequest.model_validate(
            {
                "building_id": 1,
                "room_no": "301",
                "resident_id": "linda",
                "conversation_id": "conv-001",
                "trace_id": "trace-001",
                "current_route": None,
                "current_complaint_state": None,
                "conversation_state": "collecting",
                "message": {
                    "message_id": "msg-001",
                    "text": "주차는요?",
                    "image_urls": [],
                },
                "conversation_history": [],
                "complaint_draft": None,
            }
        )


def test_incoming_message_rejects_legacy_user_action_field():
    with pytest.raises(ValidationError):
        IncomingMessage.model_validate(
            {
                "message_id": "msg-001",
                "text": "민원 접수해주세요",
                "image_urls": [],
                "user_action": "REQUEST_REGISTER",
            }
        )


def test_complaint_state_rejects_removed_action_selection_state():
    with pytest.raises(ValueError):
        ComplaintState("action_selection")


def test_converse_data_rejects_complaint_state_for_non_complaint_route():
    with pytest.raises(ValidationError):
        ConverseData(
            route=Route.KNOWLEDGE,
            next_complaint_state=ComplaintState.COLLECTING,
            reply="주차는 지하 주차장을 이용해 주세요.",
            result=RouteResult(),
            meta=Meta(model="test-model", timing_ms=1),
        )


def test_complaint_draft_rejects_invalid_issue_type():
    with pytest.raises(ValidationError):
        ComplaintDraft(issue_type="아무거나")


def test_complaint_draft_rejects_qa_card_issue_type():
    with pytest.raises(ValidationError):
        ComplaintDraft(issue_type="qa_card")


def test_history_and_citation_follow_api_contract():
    request = ConverseRequest.model_validate(
        {
            "building_id": 1,
            "room_no": "301",
            "resident_id": "linda",
            "conversation_id": "conv-001",
            "trace_id": "trace-001",
            "current_route": None,
            "current_complaint_state": None,
            "message": {
                "message_id": "msg-002",
                "text": "점검 일정은요?",
                "image_urls": [],
            },
            "conversation_history": [
                {
                    "message_id": "msg-001",
                    "role": "assistant",
                    "text": None,
                    "image_urls": ["https://example.com/image.jpg"],
                }
            ],
            "complaint_draft": None,
        }
    )
    citation = Citation(
        source_type="building_document",
        source_id="doc-1",
        title="정기 점검 안내",
        snippet=None,
        url=None,
    )

    assert request.conversation_history[0].message_id == "msg-001"
    assert citation.source_type == "building_document"


def test_agent_state_keeps_request_route_and_complaint_state():
    request = ConverseRequest.model_validate(
        {
            "building_id": 1,
            "room_no": "301",
            "resident_id": "linda",
            "conversation_id": "conv-001",
            "trace_id": "trace-001",
            "current_route": None,
            "current_complaint_state": None,
            "message": {
                "message_id": "msg-001",
                "text": None,
                "image_urls": ["https://example.com/image.jpg"],
            },
            "conversation_history": [],
            "complaint_draft": None,
        }
    )
    state: AgentState = {
        "request": request,
        "route": None,
        "complaint_state": ComplaintState.COLLECTING,
        "reply": None,
    }

    assert state["request"] is request
    assert state["route"] is None
    assert state["complaint_state"] is ComplaintState.COLLECTING
    assert state["reply"] is None
    assert "complaint_state" in AgentState.__annotations__
