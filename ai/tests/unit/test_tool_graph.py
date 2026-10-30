from types import SimpleNamespace

import pytest

import zipsai.orchestration.graph as graph_module
from zipsai.contracts.converse import ConverseRequest, Route, RouteResult
from zipsai.integrations.llm import ToolCall, ToolDecision
from zipsai.orchestration.state import AgentState
from zipsai.orchestration.tool_contracts import (
    BuildingDocsResult,
    ComplaintTurnArgs,
    ComplaintTurnResult,
)


def _request() -> ConverseRequest:
    return ConverseRequest.model_validate(
        {
            "building_id": 1,
            "room_no": "301",
            "resident_id": "linda",
            "conversation_id": "conv-001",
            "turn_id": "turn-001",
            "trace_id": "trace-001",
            "current_route": None,
            "current_complaint_state": None,
            "message": {"message_id": "msg-001", "text": "주차는 어디에 하나요?"},
            "conversation_history": [],
            "complaint_draft": None,
        }
    )


def _state() -> AgentState:
    return {
        "request": _request(),
        "route": None,
        "complaint_state": None,
        "reply": None,
        "result": RouteResult(),
    }


def _decision(*, content: str | None = None, calls: tuple[ToolCall, ...] = ()):
    return ToolDecision(message=SimpleNamespace(), content=content, tool_calls=calls)


def test_graph_returns_to_agent_after_building_search(monkeypatch: pytest.MonkeyPatch):
    events: list[dict[str, str]] = []
    calls = iter(
        [
            _decision(calls=(ToolCall("call-1", "search_building_docs", {"query": "주차"}),)),
            _decision(content="지하 주차장을 이용해 주세요."),
        ]
    )
    monkeypatch.setattr(
        graph_module,
        "classify_intent",
        lambda _state: {"route": Route.KNOWLEDGE, "image_analysis": None},
    )
    monkeypatch.setattr(graph_module, "generate_tool_decision", lambda *_a, **_kw: next(calls))
    monkeypatch.setattr(graph_module, "get_stream_writer", lambda: events.append)
    monkeypatch.setattr(
        graph_module,
        "search_building_docs",
        lambda _args, _request: BuildingDocsResult(status="no_results"),
    )

    result = graph_module.build_graph().invoke(_state())

    assert result["route"] is Route.KNOWLEDGE
    assert result["reply"] == "지하 주차장을 이용해 주세요."
    assert result["tool_call_count"] == 1
    assert result["agent_messages"][-2]["role"] == "tool"
    assert result["agent_messages"][-2]["tool_call_id"] == "call-1"
    assert events == [{"type": "status", "status": "건물 문서를 확인하고 있습니다."}]


def test_graph_ends_immediately_after_complaint_tool(monkeypatch: pytest.MonkeyPatch):
    decision = _decision(
        calls=(ToolCall("call-1", "process_complaint_turn", {}),)
    )
    tool_result = ComplaintTurnResult(
        reply="어느 장소에서 발생했나요?",
        complaint_state="collecting",
        result=RouteResult(),
    )
    monkeypatch.setattr(
        graph_module,
        "classify_intent",
        lambda _state: {"route": Route.COMPLAINT, "image_analysis": None},
    )
    monkeypatch.setattr(graph_module, "generate_tool_decision", lambda *_a, **_kw: decision)
    monkeypatch.setattr(
        graph_module,
        "process_complaint_turn",
        lambda args, request, **kwargs: (
            args == ComplaintTurnArgs(), request, kwargs, tool_result
        )[-1],
    )

    result = graph_module.build_graph().invoke(_state())

    assert result["route"] is Route.COMPLAINT
    assert result["reply"] == tool_result.reply
    assert result["complaint_state"] == tool_result.complaint_state
    assert result["tool_call_count"] == 1


def test_graph_rejects_parallel_tool_calls_without_executing_them(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = iter(
        [
            _decision(
                calls=(
                    ToolCall("call-search", "search_building_docs", {"query": "주차"}),
                    ToolCall("call-complaint", "process_complaint_turn", {}),
                )
            ),
            _decision(content="한 가지 요청을 알려주세요."),
        ]
    )
    monkeypatch.setattr(
        graph_module,
        "classify_intent",
        lambda _state: {"route": Route.CLARIFY, "image_analysis": None},
    )
    monkeypatch.setattr(graph_module, "generate_tool_decision", lambda *_a, **_kw: next(calls))
    monkeypatch.setattr(
        graph_module,
        "search_building_docs",
        lambda *_args: pytest.fail("parallel tool call must not run"),
    )
    monkeypatch.setattr(
        graph_module,
        "process_complaint_turn",
        lambda *_args, **_kwargs: pytest.fail("parallel tool call must not run"),
    )

    result = graph_module.build_graph().invoke(_state())

    assert result["reply"] == "한 가지 요청을 알려주세요."
    assert result["tool_call_count"] == 0
    assert [message["tool_call_id"] for message in result["agent_messages"][-3:-1]] == [
        "call-search",
        "call-complaint",
    ]


def test_graph_returns_invalid_arguments_to_agent_without_running_search(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = iter(
        [
            _decision(calls=(ToolCall("call-1", "search_building_docs", {"query": " "}),)),
            _decision(content="질문을 조금 더 구체적으로 알려주세요."),
        ]
    )
    monkeypatch.setattr(
        graph_module,
        "classify_intent",
        lambda _state: {"route": Route.KNOWLEDGE, "image_analysis": None},
    )
    monkeypatch.setattr(graph_module, "generate_tool_decision", lambda *_a, **_kw: next(calls))
    monkeypatch.setattr(
        graph_module,
        "search_building_docs",
        lambda *_args: pytest.fail("invalid query must not reach search"),
    )

    result = graph_module.build_graph().invoke(_state())

    assert result["reply"] == "질문을 조금 더 구체적으로 알려주세요."
    assert '"code": "invalid_arguments"' in result["agent_messages"][-2]["content"]
