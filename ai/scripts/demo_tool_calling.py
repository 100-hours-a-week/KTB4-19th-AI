"""Tool Calling 그래프를 오프라인 또는 실제 AI API로 실행한다.

실행:
    cd ai && uv run python scripts/demo_tool_calling.py
    cd ai && uv run python scripts/demo_tool_calling.py --live

기본 실행은 고정 결과를 사용한다. --live는 로컬 Converse API를 호출해 실제 모델과
도구 실행 결과를 확인한다.
"""

import argparse
import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import ClassVar
from unittest.mock import patch
from uuid import uuid4

import httpx

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


class _DemoLogHandler(logging.Handler):
    """Print only bounded fields from the graph's structured log events."""

    _FIELDS: ClassVar[dict[str, tuple[str, ...]]] = {
        "agent_decision": ("route", "outcome", "tool_call_count"),
        "tool_call_selected": ("tool", "tool_call_count"),
        "tool_call_completed": ("tool", "outcome", "duration_ms"),
        "tool_call_retry": ("tool", "reason", "attempt"),
        "tool_call_protocol_failure": ("reason",),
        "tool_call_rejected": ("reason", "tool"),
    }

    def emit(self, record: logging.LogRecord) -> None:
        event = record.getMessage()
        fields = self._FIELDS.get(event)
        if fields is None:
            return
        values = " ".join(
            f"{name}={getattr(record, name)}"
            for name in fields
            if hasattr(record, name)
        )
        print(f"[LOG] {event} {values}")


def _request(text: str, *, request_id: str = "demo") -> ConverseRequest:
    return ConverseRequest.model_validate(
        {
            "building_id": 101,
            "room_no": "301",
            "resident_id": "demo-resident",
            "conversation_id": f"{request_id}-conversation",
            "turn_id": f"{request_id}-turn",
            "trace_id": request_id,
            "current_route": None,
            "current_complaint_state": None,
            "message": {"message_id": f"{request_id}-message", "text": text},
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
    print(f"\n[{node.upper()}]")
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
        if node != "tool":
            label = "assistant reply"
        elif len(update.get("completed_tools", ())) == 1:
            label = "pipeline reply (final)"
        else:
            label = "pipeline reply (composer continues)"
        print(f"  {label}: {update['reply']}")
    if "tool_call_count" in update:
        print(f"  executed tool calls: {update['tool_call_count']}")


def _run(title: str, request: ConverseRequest) -> dict[str, object]:
    print("\n" + "=" * 72)
    print(title)
    print(f"resident request: {request.message.text}")
    print("=" * 72)
    result: dict[str, object] = {}
    logger = graph_module.logger
    previous_level = logger.level
    handler = _DemoLogHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        for update in graph_module.build_graph().stream(
            _initial_state(request), stream_mode="updates"
        ):
            for node, node_update in update.items():
                _print_event(node, node_update)
                result.update(node_update)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
    print("\n[END]")
    print(f"  route: {result['route'].value}")
    print(f"  reply: {result['reply']}")
    print(f"  tool_call_count: {result['tool_call_count']}")
    print(f"  attempted_tools: {result.get('attempted_tools', ())}")
    print(f"  completed_tools: {result.get('completed_tools', ())}")
    route_result = result["result"]
    print(f"  complaint_state: {result['complaint_state']}")
    print("  structured result:")
    print(json.dumps(route_result.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return result


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
        patch.object(graph_module, "classify_intent", lambda _state: {"route": Route.KNOWLEDGE, "planned_tools": ["search_building_docs"]}),
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
            complaint_draft=ComplaintDraft(issue_type="plumbing"),
            missing_fields=["location", "symptom"],
        ),
    )
    with (
        patch.object(graph_module, "classify_intent", lambda _state: {"route": Route.COMPLAINT, "planned_tools": ["process_complaint_turn"]}),
        patch.object(
            graph_module,
            "generate_tool_decision",
            _decisions(
                [
                    ToolDecision(_assistant_message((call,)), None, (call,)),
                    ToolDecision(
                        {"role": "assistant", "content": "추가 도구 없이 답변할 수 있습니다."},
                        "추가 도구 없이 답변할 수 있습니다.",
                        (),
                    ),
                ]
            ),
        ),
        patch.object(
            graph_module,
            "process_complaint_turn",
            lambda _args, _request, **_kwargs: complaint,
        ),
    ):
        yield


@contextmanager
def _patched_no_results_complaint_flow() -> Iterator[None]:
    search_call = ToolCall(
        "call-search-1", "search_building_docs", {"query": "누수 처리 기한"}
    )
    complaint_call = ToolCall("call-complaint-1", "process_complaint_turn", {})
    complaint = ComplaintTurnResult(
        reply="어느 장소에서 물이 새는지 알려주세요.",
        complaint_state=ComplaintState.COLLECTING,
        result=RouteResult(
            complaint_draft=ComplaintDraft(issue_type="plumbing", location="욕실"),
            missing_fields=["symptom"],
        ),
    )
    with (
        patch.object(graph_module, "classify_intent", lambda _state: {"route": Route.COMPLAINT, "planned_tools": ["search_building_docs", "process_complaint_turn"]}),
        patch.object(
            graph_module,
            "generate_tool_decision",
            _decisions(
                [
                    ToolDecision(_assistant_message((search_call,)), None, (search_call,)),
                    ToolDecision(_assistant_message((complaint_call,)), None, (complaint_call,)),
                    ToolDecision(
                        {"role": "assistant", "content": "민원 초안을 만들었습니다. 처리 기한은 확인할 자료가 없습니다."},
                        "민원 초안을 만들었습니다. 처리 기한은 확인할 자료가 없습니다.",
                        (),
                    ),
                ]
            ),
        ),
        patch.object(
            graph_module,
            "search_building_docs",
            lambda _args, _request: BuildingDocsResult(status="no_results"),
        ),
        patch.object(
            graph_module,
            "process_complaint_turn",
            lambda _args, _request, **_kwargs: complaint,
        ),
    ):
        yield


@contextmanager
def _patched_protocol_retry_flow() -> Iterator[None]:
    invalid_text = 'search_building_docs {"query":"이 문자열을 로그에 남기지 마세요"}'
    with (
        patch.object(
            graph_module,
            "classify_intent",
            lambda _state: {"route": Route.KNOWLEDGE, "planned_tools": ["search_building_docs"]},
        ),
        patch.object(
            graph_module,
            "generate_tool_decision",
            _decisions(
                [
                    ToolDecision(
                        {"role": "assistant", "content": invalid_text},
                        invalid_text,
                        (),
                    ),
                    ToolDecision(
                        {"role": "assistant", "content": "주차는 지하 1층에서 하실 수 있습니다."},
                        "주차는 지하 1층에서 하실 수 있습니다.",
                        (),
                    ),
                ]
            ),
        ),
    ):
        yield


def _run_live() -> int:
    trace_id = f"tool-call-live-{uuid4().hex[:12]}"
    request = _request(
        "누수 처리 기한도 알려주고, 욕실 누수 민원도 작성해 주세요.",
        request_id=trace_id,
    )
    url = os.getenv(
        "AI_CONVERSE_URL",
        "http://127.0.0.1:8000/api/v3/ai/converse",
    )
    response = httpx.post(
        url,
        json=request.model_dump(mode="json"),
        timeout=150,
    )
    body = response.json()
    model = body.get("data", {}).get("meta", {}).get("model")
    print(f"API: {url}")
    print(f"trace_id: {trace_id}")
    print(f"HTTP: {response.status_code}")
    if model:
        print(f"model: {model}")
    print(json.dumps(body, ensure_ascii=False, indent=2))
    print(
        "tool logs: docker compose -f ../compose.yaml logs --since=5m zipsai-api 2>&1 "
        f"| rg '{trace_id}'"
    )
    return 0 if response.is_success else 1


def main(*, live: bool = False) -> int:
    if live:
        return _run_live()

    with _patched_search_flow():
        search_result = _run(
            "1. 건물 정보 질문: Action → 검색 → Action/Composer",
            _request("주차는 어디에 하나요?"),
        )
    assert search_result["route"] is Route.KNOWLEDGE
    assert len(search_result["result"].citations) == 1
    print("  시나리오 검증: PASS (검색 citation 1개)")

    with _patched_complaint_flow():
        complaint_result = _run(
            "2. 민원: Action → 민원 pipeline → 추가 도구 확인 → workflow reply",
            _request("화장실에서 물이 새요."),
        )
    assert complaint_result["reply"] == "어느 장소에서 물이 새는지 알려주세요."
    assert complaint_result["tool_call_count"] == 1
    print("  시나리오 검증: PASS (composer 합성 없이 workflow reply 사용)")

    with _patched_no_results_complaint_flow():
        combined_result = _run(
            "3. 검색 no_results + 민원 초안: 대화 하나에 카드 하나라 민원 우선",
            _request("누수 처리 기한도 알려주고, 욕실 누수 민원도 작성해 주세요."),
        )
    assert combined_result["result"].qa_card_draft is None
    assert combined_result["result"].complaint_draft.location == "욕실"
    assert combined_result["completed_tools"] == (
        "search_building_docs",
        "process_complaint_turn",
    )
    print("  시나리오 검증: PASS (민원 초안 보존, QA 카드 없음)")

    with _patched_protocol_retry_flow():
        protocol_result = _run(
            "4. Action 형식 오류: text → retry → 안전한 종료",
            _request("주차는 어디에 하나요?"),
        )
    assert protocol_result["tool_call_count"] == 0
    assert protocol_result["reply"] == "요청한 작업을 처리하지 못했습니다. 다시 말씀해 주세요."
    print("  시나리오 검증: PASS (도구 실행 없이 실패 닫힘)")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="로컬 Converse API를 실제 호출")
    raise SystemExit(main(live=parser.parse_args().live))
