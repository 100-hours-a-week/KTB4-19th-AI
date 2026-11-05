from types import SimpleNamespace

import pytest

import zipsai.orchestration.graph as graph_module
from zipsai.contracts.converse import (
    ComplaintDraft,
    ComplaintState,
    ConverseRequest,
    HistoryTurn,
    Route,
    RouteResult,
)
from zipsai.integrations.llm import ToolCall, ToolDecision
from zipsai.orchestration.state import AgentState
from zipsai.orchestration.tool_contracts import (
    BuildingDocsResult,
    ComplaintTurnArgs,
    ComplaintTurnResult,
    ToolFailure,
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
    tool_events: list[tuple[str, str, float]] = []
    calls = iter(
        [
            _decision(calls=(ToolCall("call-1", "search_building_docs", {"query": "주차"}),)),
            _decision(content="지하 주차장을 이용해 주세요."),
        ]
    )
    monkeypatch.setattr(
        graph_module,
        "classify_intent",
        lambda _state: {
            "route": Route.KNOWLEDGE,
            "image_analysis": None,
        },
    )
    monkeypatch.setattr(graph_module, "generate_tool_decision", lambda *_a, **_kw: next(calls))
    monkeypatch.setattr(graph_module, "get_stream_writer", lambda: events.append)
    monkeypatch.setattr(
        graph_module,
        "observe_tool_call",
        lambda tool, outcome, duration: tool_events.append((tool, outcome, duration)),
    )
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
    assert tool_events[0][0:2] == ("search_building_docs", "success")
    assert tool_events[0][2] >= 0


def test_action_context_includes_active_complaint_draft_missing_fields_and_last_question():
    request = _request().model_copy(
        update={
            "current_route": Route.COMPLAINT,
            "current_complaint_state": ComplaintState.COLLECTING,
            "complaint_draft": ComplaintDraft(issue_type="leak", symptom="물이 새요"),
            "conversation_history": [
                HistoryTurn(message_id="a1", role="assistant", text="어느 장소에서 발생했나요?")
            ],
        }
    )

    messages = graph_module._initial_messages({**_state(), "request": request})
    context = messages[-1]["content"]

    assert "현재 민원 상태: collecting" in context
    assert '"issue_type":"leak"' in context
    assert "미수집 민원 항목: location" in context
    assert "assistant: 어느 장소에서 발생했나요?" in context


def test_action_prompt_treats_complaint_context_as_hint_not_route_lock():
    system = graph_module._AGENT_SYSTEM

    assert "answers the last complaint question" in system
    assert "A clear new question may take another route" in system
    assert "ask for clarification when intent is ambiguous" in system
    assert "previous route is context and does not force a tool call" in system


def test_action_receives_full_active_catalog_regardless_of_route(monkeypatch: pytest.MonkeyPatch):
    offered: list[str] = []

    monkeypatch.setattr(
        graph_module,
        "classify_intent",
        lambda _state: {"route": Route.COMPLAINT, "image_analysis": None},
    )

    def decide(_messages, tools, **_kwargs):
        offered.extend(tool["function"]["name"] for tool in tools)
        return _decision(content="요청을 확인하겠습니다.")

    monkeypatch.setattr(graph_module, "generate_tool_decision", decide)

    graph_module.build_graph().invoke(_state())

    assert offered == ["search_building_docs", "process_complaint_turn"]


def test_prepare_does_not_create_tool_plan_in_graph_state(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        graph_module,
        "classify_intent",
        lambda _state: {
            "route": Route.COMPLAINT,
            "image_analysis": None,
        },
    )
    monkeypatch.setattr(
        graph_module,
        "generate_tool_decision",
        lambda *_args, **_kwargs: _decision(content="두 요청을 확인하겠습니다."),
    )

    result = graph_module.build_graph().invoke(_state())

    assert "required_capabilities" not in result


def test_agent_prompt_allows_a_natural_reply_or_one_tool(
    monkeypatch: pytest.MonkeyPatch,
):
    seen_system_messages: list[str] = []

    monkeypatch.setattr(
        graph_module,
        "classify_intent",
        lambda _state: {
            "route": Route.COMPLAINT,
            "image_analysis": None,
        },
    )

    def decide(messages, _tools, **_kwargs):
        seen_system_messages.extend(
            message["content"]
            for message in messages
            if isinstance(message, dict) and message.get("role") == "system"
        )
        return _decision(content="확인하겠습니다.")

    monkeypatch.setattr(graph_module, "generate_tool_decision", decide)

    graph_module.build_graph().invoke(_state())

    assert any("Use an offered tool when needed" in message for message in seen_system_messages)
    assert any("reply naturally in Korean" in message for message in seen_system_messages)


def test_graph_runs_required_search_then_complaint_before_final_agent_reply(
    monkeypatch: pytest.MonkeyPatch,
):
    decisions = iter(
        [
            _decision(calls=(ToolCall("call-search", "search_building_docs", {"query": "누수 처리 기한"}),)),
            _decision(calls=(ToolCall("call-complaint", "process_complaint_turn", {}),)),
            _decision(content="처리 기한은 확인 중이며 민원 초안을 정리했습니다."),
        ]
    )
    tool_result = ComplaintTurnResult(
        reply="어느 장소에서 발생했나요?",
        complaint_state="collecting",
        result=RouteResult(
            complaint_draft=ComplaintDraft(issue_type="leak", location="욕실"),
            missing_fields=["symptom"],
        ),
    )
    monkeypatch.setattr(
        graph_module,
        "classify_intent",
        lambda _state: {
            "route": Route.COMPLAINT,
            "image_analysis": None,
        },
    )
    monkeypatch.setattr(
        graph_module, "generate_tool_decision", lambda *_a, **_kw: next(decisions)
    )
    monkeypatch.setattr(
        graph_module,
        "search_building_docs",
        lambda _args, _request: BuildingDocsResult(status="no_results"),
    )
    monkeypatch.setattr(
        graph_module,
        "process_complaint_turn",
        lambda args, request, **kwargs: (
            args == ComplaintTurnArgs(), request, kwargs, tool_result
        )[-1],
    )

    result = graph_module.build_graph().invoke(_state())

    assert result["route"] is Route.COMPLAINT
    assert result["reply"] == "처리 기한은 확인 중이며 민원 초안을 정리했습니다."
    assert result["complaint_result"] == tool_result
    assert result["complaint_state"] == tool_result.complaint_state
    assert result["result"].qa_card_draft.question == "누수 처리 기한"
    assert result["result"].complaint_draft == tool_result.result.complaint_draft
    assert result["result"].missing_fields == ["symptom"]
    assert result["result"].has_sufficient_evidence is False
    assert result["tool_call_count"] == 2
    assert result["completed_tools"] == (
        "search_building_docs",
        "process_complaint_turn",
    )


def test_graph_uses_final_agent_reply_after_second_tool_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    decisions = iter(
        [
            _decision(calls=(ToolCall("call-search", "search_building_docs", {"query": "누수 처리 기한"}),)),
            _decision(calls=(ToolCall("call-complaint", "process_complaint_turn", {}),)),
            _decision(content="민원 처리는 현재 실패했지만, 처리 기한은 관리자에게 확인해 주세요."),
        ]
    )
    monkeypatch.setattr(
        graph_module,
        "classify_intent",
        lambda _state: {
            "route": Route.COMPLAINT,
            "image_analysis": None,
        },
    )
    monkeypatch.setattr(graph_module, "generate_tool_decision", lambda *_a, **_kw: next(decisions))
    monkeypatch.setattr(
        graph_module,
        "search_building_docs",
        lambda _args, _request: BuildingDocsResult(status="no_results"),
    )
    monkeypatch.setattr(
        graph_module,
        "process_complaint_turn",
        lambda *_args, **_kwargs: ToolFailure(code="timeout", message="민원 도구 시간 초과"),
    )

    result = graph_module.build_graph().invoke(_state())

    assert result["tool_call_count"] == 2
    assert result["reply"] == "민원 처리는 현재 실패했지만, 처리 기한은 관리자에게 확인해 주세요."


def test_graph_rejects_repeated_completed_tool_without_running_it_again(
    monkeypatch: pytest.MonkeyPatch,
):
    decisions = iter(
        [
            _decision(calls=(ToolCall("call-search", "search_building_docs", {"query": "주차"}),)),
            _decision(calls=(ToolCall("call-repeat", "search_building_docs", {"query": "주차"}),)),
            _decision(content="검색 결과를 다시 요청할 수 없습니다."),
        ]
    )
    search_calls = []
    monkeypatch.setattr(
        graph_module,
        "classify_intent",
        lambda _state: {
            "route": Route.KNOWLEDGE,
            "image_analysis": None,
        },
    )
    monkeypatch.setattr(
        graph_module, "generate_tool_decision", lambda *_args, **_kwargs: next(decisions)
    )
    monkeypatch.setattr(
        graph_module,
        "search_building_docs",
        lambda args, _request: (
            search_calls.append(args.query),
            BuildingDocsResult(status="no_results"),
        )[-1],
    )

    result = graph_module.build_graph().invoke(_state())

    assert search_calls == ["주차"]
    assert result["completed_tools"] == ("search_building_docs",)
    assert any(
        isinstance(message, dict) and '"code": "invalid_arguments"' in message.get("content", "")
        for message in result["agent_messages"]
    )
    assert result["reply"] == "검색 결과를 다시 요청할 수 없습니다."


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
    assert result["tool_call_count"] == 1
    assert [message["tool_call_id"] for message in result["agent_messages"][-3:-1]] == [
        "call-search",
        "call-complaint",
    ]


def test_graph_records_parallel_tool_call_rejection(
    monkeypatch: pytest.MonkeyPatch,
):
    rejection_reasons: list[str] = []
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
        "observe_tool_rejection",
        lambda reason: rejection_reasons.append(reason),
    )

    graph_module.build_graph().invoke(_state())

    assert rejection_reasons == ["parallel_calls"]


def test_graph_stops_offering_tools_after_repeated_parallel_requests(
    monkeypatch: pytest.MonkeyPatch,
):
    decisions = iter(
        [
            _decision(calls=(ToolCall("call-1", "search_building_docs", {"query": "주차"}), ToolCall("call-2", "process_complaint_turn", {}))),
            _decision(calls=(ToolCall("call-3", "search_building_docs", {"query": "주차"}), ToolCall("call-4", "process_complaint_turn", {}))),
            _decision(content="한 가지 요청을 알려주세요."),
        ]
    )
    offered_tools: list[list[object]] = []

    monkeypatch.setattr(
        graph_module,
        "classify_intent",
        lambda _state: {"route": Route.CLARIFY, "image_analysis": None},
    )

    def decide(_messages, tools, **_kwargs):
        offered_tools.append(tools)
        return next(decisions)

    monkeypatch.setattr(graph_module, "generate_tool_decision", decide)

    result = graph_module.build_graph().invoke(_state())

    assert result["reply"] == "한 가지 요청을 알려주세요."
    assert result["tool_call_count"] == 2
    assert offered_tools[-1] == []


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
        lambda _state: {
            "route": Route.KNOWLEDGE,
            "image_analysis": None,
        },
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
