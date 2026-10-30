"""Tool Calling 그래프가 어떤 메시지를 주고받는지 출력하는 오프라인 데모.

실행:
    cd ai && uv run python scripts/demo_tool_calling.py

실제 모델·Qdrant·민원 처리 LLM은 호출하지 않는다. 대신 Agent의 도구 선택과
도구 결과를 고정해, 그래프의 제어 흐름과 OpenAI 호환 ToolMessage 형태를 보여 준다.
"""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import patch

import zipsai.orchestration.graph as graph_module
from zipsai.contracts.converse import (
    Citation,
    ComplaintDraft,
    ComplaintState,
    ConverseRequest,
    Route,
    RouteResult,
)
from zipsai.integrations.llm import ToolCall, ToolDecision
from zipsai.orchestration.tool_contracts import (
    BuildingDocsResult,
    ComplaintTurnResult,
    SearchEvidence,
)


def _request(text: str) -> ConverseRequest:
    return ConverseRequest.model_validate(
        {
            "building_id": 101,
            "room_no": "301",
            "resident_id": "demo-resident",
            "conversation_id": "demo-conversation",
            "turn_id": "demo-turn",
            "trace_id": "demo-trace",
            "current_route": None,
            "current_complaint_state": None,
            "message": {"message_id": "demo-message", "text": text},
            "conversation_history": [],
            "complaint_draft": None,
        }
    )


def _initial_state(request: ConverseRequest) -> dict[str, object]:
    return {
        "request": request,
        "route": None,
        "complaint_state": None,
        "reply": None,
        "result": RouteResult(),
        "image_analysis": None,
    }


def _assistant_message(calls: tuple[ToolCall, ...]) -> dict[str, object]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": call.arguments}}
            for call in calls
        ],
    }


def _decisions(values: list[ToolDecision]):
    pending = iter(values)
    return lambda *_args, **_kwargs: next(pending)


def _print_event(node: str, update: dict[str, object]) -> None:
    print(f"\n[{node}]")
    calls = update.get("pending_tool_calls")
    if calls:
        for call in calls:
            print(f"  tool_calls: {call.name}({json.dumps(call.arguments, ensure_ascii=False)})")

    messages = update.get("agent_messages")
    if isinstance(messages, list) and messages:
        message = messages[-1]
        if isinstance(message, dict) and message.get("role") == "tool":
            print(f"  ToolMessage.tool_call_id: {message['tool_call_id']}")
            print(f"  ToolMessage.content: {message['content']}")

    if "reply" in update:
        print(f"  final reply: {update['reply']}")
    if "tool_call_count" in update:
        print(f"  executed tool calls: {update['tool_call_count']}")


def _run(title: str, request: ConverseRequest) -> None:
    print("\n" + "=" * 72)
    print(title)
    print(f"resident request: {request.message.text}")
    print("=" * 72)
    result: dict[str, object] = {}
    for update in graph_module.build_graph().stream(_initial_state(request), stream_mode="updates"):
        for node, node_update in update.items():
            _print_event(node, node_update)
            result.update(node_update)
    print("\n[END]")
    print(f"  route: {result['route'].value}")
    print(f"  reply: {result['reply']}")
    print(f"  tool_call_count: {result['tool_call_count']}")
    route_result = result["result"]
    print(f"  citations: {len(route_result.citations)}")
    print(f"  complaint_state: {result['complaint_state']}")


@contextmanager
def _patched_search_flow() -> Iterator[None]:
    call = ToolCall("call-search-1", "search_building_docs", {"query": "주차 위치"})
    evidence = SearchEvidence(
        text="입주민은 지하 1층 지정 구역을 이용합니다.",
        citation=Citation(
            source_type="building_document",
            source_id="parking-guide",
            title="주차 이용 안내",
            snippet="입주민은 지하 1층 지정 구역을 이용합니다.",
            url=None,
        ),
    )
    with (
        patch.object(graph_module, "classify_intent", lambda _state: {"route": Route.KNOWLEDGE}),
        patch.object(
            graph_module,
            "generate_tool_decision",
            _decisions(
                [
                    ToolDecision(_assistant_message((call,)), None, (call,)),
                    ToolDecision(
                        {"role": "assistant", "content": "입주민 주차는 지하 1층 지정 구역을 이용해 주세요."},
                        "입주민 주차는 지하 1층 지정 구역을 이용해 주세요.",
                        (),
                    ),
                ]
            ),
        ),
        patch.object(
            graph_module,
            "search_building_docs",
            lambda _args, _request: BuildingDocsResult(status="ok", evidence=[evidence]),
        ),
    ):
        yield


@contextmanager
def _patched_complaint_flow() -> Iterator[None]:
    call = ToolCall("call-complaint-1", "process_complaint_turn", {})
    complaint = ComplaintTurnResult(
        reply="어느 장소에서 물이 새는지 알려주세요.",
        complaint_state=ComplaintState.COLLECTING,
        result=RouteResult(
            complaint_draft=ComplaintDraft(issue_type="leak"),
            missing_fields=["location", "symptom"],
        ),
    )
    with (
        patch.object(graph_module, "classify_intent", lambda _state: {"route": Route.COMPLAINT}),
        patch.object(
            graph_module,
            "generate_tool_decision",
            _decisions([ToolDecision(_assistant_message((call,)), None, (call,))]),
        ),
        patch.object(
            graph_module,
            "process_complaint_turn",
            lambda _args, _request, **_kwargs: complaint,
        ),
    ):
        yield


def main() -> None:
    with _patched_search_flow():
        _run("1. 건물 정보 질문: 검색 후 Agent가 근거를 보고 답변", _request("주차는 어디에 하나요?"))
    with _patched_complaint_flow():
        _run("2. 민원: 민원 도구의 완성 결과로 즉시 종료", _request("화장실에서 물이 새요."))


if __name__ == "__main__":
    main()
