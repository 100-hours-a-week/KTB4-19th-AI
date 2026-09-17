import pytest

import zipsai.orchestration.graph as graph_module
from zipsai.contracts.converse import ConversationState, ConverseRequest, Route
from zipsai.errors import LlmUnavailableError
from zipsai.orchestration.graph import build_graph, select_next_node
from zipsai.orchestration.state import AgentState


def _make_request() -> ConverseRequest:
    return ConverseRequest.model_validate(
        {
            "building_id": 1,
            "resident_context": {"unit_id": 57, "resident_id": "linda"},
            "conversation_id": "conv-001",
            "trace_id": "trace-001",
            "message": {"message_id": "msg-001", "text": "도와주세요"},
        }
    )


def _make_state(route: Route | None) -> AgentState:
    return {
        "request": _make_request(),
        "route": route,
        "conversation_state": ConversationState.COLLECTING,
        "response": None,
    }


def _stub_classify_intent(state: AgentState) -> dict[str, Route]:
    return {"route": state["route"] or Route.CLARIFY}


@pytest.mark.parametrize(
    ("route", "expected_node"),
    [
        (Route.COMPLAINT, "complaint"),
        (Route.KNOWLEDGE, "knowledge"),
        (Route.CLARIFY, "clarify"),
        (None, "clarify"),
    ],
)
def test_select_next_node_returns_route_node(route: Route | None, expected_node: str):
    assert select_next_node(_make_state(route)) == expected_node


@pytest.mark.parametrize(
    ("route", "handler_name"),
    [
        (Route.COMPLAINT, "handle_complaint"),
        (Route.KNOWLEDGE, "handle_knowledge"),
    ],
)
def test_graph_invokes_feature_handler_for_selected_route(
    monkeypatch: pytest.MonkeyPatch,
    route: Route,
    handler_name: str,
):
    state = _make_state(route)
    handled_requests: list[ConverseRequest] = []

    def handler(request: ConverseRequest) -> None:
        handled_requests.append(request)

    monkeypatch.setattr(graph_module, handler_name, handler)
    monkeypatch.setattr(graph_module, "classify_intent", _stub_classify_intent)

    build_graph().invoke(state)

    assert handled_requests == [state["request"]]


def test_graph_finishes_on_clarify_route(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(graph_module, "classify_intent", _stub_classify_intent)

    result = build_graph().invoke(_make_state(None))

    assert result["route"] is Route.CLARIFY


def test_graph_propagates_intent_classification_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    def failing_classify_intent(state: AgentState) -> dict[str, Route]:
        raise LlmUnavailableError("boom")

    monkeypatch.setattr(graph_module, "classify_intent", failing_classify_intent)

    with pytest.raises(LlmUnavailableError):
        build_graph().invoke(_make_state(None))
