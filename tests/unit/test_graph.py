import pytest

import zipsai.orchestration.graph as graph_module
from zipsai.contracts.converse import (
    ComplaintState,
    ConverseRequest,
    Route,
    RouteResult,
)
from zipsai.errors import LlmUnavailableError
from zipsai.orchestration.graph import build_graph, select_entry_node, select_next_node
from zipsai.orchestration.state import AgentState


def _make_request(
    current_route: Route | None = None,
    current_complaint_state: ComplaintState | None = None,
) -> ConverseRequest:
    return ConverseRequest.model_validate(
        {
            "building_id": 1,
            "room_no": "301",
            "resident_id": "linda",
            "conversation_id": "conv-001",
            "trace_id": "trace-001",
            "current_route": current_route,
            "current_complaint_state": current_complaint_state,
            "message": {
                "message_id": "msg-001",
                "text": "도와주세요",
                "image_urls": [],
            },
            "conversation_history": [],
            "complaint_draft": None,
        }
    )


def _make_state(route: Route | None) -> AgentState:
    return {
        "request": _make_request(),
        "route": route,
        "complaint_state": ComplaintState.COLLECTING,
        "reply": None,
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
    ("current_route", "current_complaint_state", "expected_node"),
    [
        (Route.COMPLAINT, ComplaintState.COLLECTING, "complaint"),
        (Route.COMPLAINT, None, "classify_intent"),
        (Route.KNOWLEDGE, None, "classify_intent"),
        (None, None, "classify_intent"),
    ],
)
def test_select_entry_node_skips_classify_intent_when_complaint_in_progress(
    current_route: Route | None,
    current_complaint_state: ComplaintState | None,
    expected_node: str,
):
    state: AgentState = {
        "request": _make_request(current_route, current_complaint_state),
        "route": None,
        "complaint_state": current_complaint_state,
        "reply": None,
        "result": RouteResult(),
    }
    assert select_entry_node(state) == expected_node


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


def test_graph_skips_classify_intent_when_complaint_in_progress(
    monkeypatch: pytest.MonkeyPatch,
):
    request = _make_request(Route.COMPLAINT, ComplaintState.COLLECTING)
    state: AgentState = {
        "request": request,
        "route": None,
        "complaint_state": request.current_complaint_state,
        "reply": None,
        "result": RouteResult(),
    }
    handled_requests: list[ConverseRequest] = []

    def handler(req: ConverseRequest) -> dict[str, object]:
        handled_requests.append(req)
        return {
            "complaint_state": ComplaintState.COLLECTING,
            "reply": "ok",
            "result": RouteResult(),
        }

    def failing_classify_intent(_state: AgentState) -> dict[str, Route]:
        raise AssertionError("classify_intent must not run mid-complaint")

    monkeypatch.setattr(graph_module, "handle_complaint", handler)
    monkeypatch.setattr(graph_module, "classify_intent", failing_classify_intent)

    build_graph().invoke(state)

    assert handled_requests == [request]


def test_graph_finishes_on_clarify_route(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(graph_module, "classify_intent", _stub_classify_intent)

    result = build_graph().invoke(_make_state(None))

    assert result["route"] is Route.CLARIFY
    assert result["complaint_state"] is None
    assert result["reply"] is not None


@pytest.mark.parametrize("route", [Route.KNOWLEDGE, Route.CLARIFY])
def test_graph_clears_complaint_state_for_non_complaint_route(
    monkeypatch: pytest.MonkeyPatch,
    route: Route,
):
    monkeypatch.setattr(graph_module, "classify_intent", _stub_classify_intent)

    result = build_graph().invoke(_make_state(route))

    assert result["complaint_state"] is None


def test_graph_propagates_intent_classification_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    def failing_classify_intent(state: AgentState) -> dict[str, Route]:
        raise LlmUnavailableError("boom")

    monkeypatch.setattr(graph_module, "classify_intent", failing_classify_intent)

    with pytest.raises(LlmUnavailableError):
        build_graph().invoke(_make_state(None))
