"""실제 Qwen Pod와 기존 LangGraph Tool Calling 흐름을 한 건으로 관찰한다.

실행: cd ai && uv run python scripts/test_tool_calling_pod_smoke.py
POD_BASE_URL(…/v1), POD_API_KEY, POD_MODEL 설정이 필요하다. 검색/민원 실행부는
가짜 결과를 사용하고, 공개 Web 검색은 설정된 Tavily API에 실제 요청한다.
"""

import asyncio
import json
import os
import time
from unittest.mock import patch

# 이 스크립트는 다른 provider로 조용히 전환하지 않고 반드시 Pod를 확인한다.
os.environ["ACTIVE_MODEL_PROVIDER"] = "pod"

import zipsai.integrations.llm as llm_module
import zipsai.orchestration.graph as graph_module
from zipsai.contracts.converse import (
    Citation,
    ComplaintDraft,
    ComplaintState,
    ConverseRequest,
    RouteResult,
)
from zipsai.integrations.runpod import get_locator
from zipsai.orchestration.tool_contracts import (
    BuildingDocsResult,
    ComplaintTurnResult,
    SearchEvidence,
)
from zipsai.settings import TAVILY_API_KEY, get_settings


def _pod_settings():
    if os.getenv("POD_BASE_URL"):
        return get_settings("pod")
    locator = get_locator()
    if locator is None:
        raise SystemExit("POD_BASE_URL 또는 RunPod locator 환경변수를 설정하세요.")

    async def resolve():
        try:
            await locator.refresh()
            return get_settings("pod")
        finally:
            await locator.aclose()

    return asyncio.run(resolve())


def _request() -> ConverseRequest:
    return ConverseRequest.model_validate(
        {
            "building_id": 101,
            "room_no": "301",
            "resident_id": "synthetic-test-resident",
            "conversation_id": "pod-smoke",
            "turn_id": "pod-smoke-turn",
            "trace_id": "pod-smoke",
            "current_route": None,
            "current_complaint_state": None,
            "message": {
                "message_id": "pod-smoke-message",
                "text": "올겨울 수도관 동파를 예방하는 방법을 공개 정보로 찾아 요약해 주세요.",
            },
            "conversation_history": [],
            "complaint_draft": None,
        }
    )


def _stub_results():
    evidence = SearchEvidence(
        text="입주민 주차 등록은 관리사무소에서 신청합니다.",
        citation=Citation(
            source_type="building_document",
            source_id="demo-parking",
            title="주차 안내",
            snippet="관리사무소에서 신청",
            url=None,
        ),
    )
    complaint = ComplaintTurnResult(
        reply="누수 위치와 증상을 확인했습니다.",
        complaint_state=ComplaintState.COLLECTING,
        result=RouteResult(
            complaint_draft=ComplaintDraft(
                issue_type="plumbing", location="욕실", symptom="천장에서 물이 샘"
            ),
            missing_fields=[],
        ),
    )
    return evidence, complaint


def main() -> None:
    settings = _pod_settings()
    if (
        settings.provider != "pod"
        or "qwen" not in settings.llm_model.lower()
        or not settings.llm_base_url
        or not settings.llm_api_key
    ):
        raise SystemExit(
            f"Qwen Pod 설정을 확인할 수 없습니다 (provider={settings.provider}, model={settings.llm_model})."
        )
    if not TAVILY_API_KEY:
        raise SystemExit("실제 Web 검색 확인을 위해 TAVILY_API_KEY를 설정하세요.")
    print(
        f"provider: {settings.provider} | model: {settings.llm_model} | base_url: {settings.llm_base_url}"
    )
    print("요청: 올겨울 수도관 동파를 예방하는 방법을 공개 정보로 찾아 요약해 주세요.")
    evidence, complaint = _stub_results()
    decisions, tools = [], []
    real_decide = llm_module.generate_tool_decision
    real_classify = graph_module.classify_intent

    def classify(state):
        started = time.perf_counter()
        prepared = real_classify(state)
        decisions.append(
            {
                "stage": "intent_classification",
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                "route": getattr(prepared.get("route"), "value", prepared.get("route")),
            }
        )
        return prepared

    def pod_only(selected, request):
        if selected.provider != "pod" or not selected.llm_base_url:
            raise RuntimeError("실제 Pod 설정이 아니어서 Qwen 요청을 중단했습니다.")
        return request(selected), selected

    def decide(*args, **kwargs):
        started = time.perf_counter()
        decision = real_decide(*args, **kwargs)
        decisions.append(
            {
                "stage": "tool_decision",
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                "tool_calls": [
                    {"name": call.name, "arguments": call.arguments}
                    for call in decision.tool_calls
                ],
                "content": decision.content,
            }
        )
        return decision

    def measured(name, result):
        def run(*_args, **_kwargs):
            started = time.perf_counter()
            try:
                return result
            finally:
                tools.append(
                    {
                        "name": name,
                        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                        "stub": True,
                    }
                )

        return run

    real_web_search = graph_module.search_public_web

    def measured_web(*args, **kwargs):
        started = time.perf_counter()
        result = real_web_search(*args, **kwargs)
        tools.append(
            {
                "name": "search_public_web",
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                "status": getattr(result, "status", "error"),
                "error_code": getattr(result, "code", None),
                "result_count": len(getattr(result, "evidence", ())),
                "citation_urls": [
                    item.citation.url for item in getattr(result, "evidence", ())
                ],
                "stub": False,
            }
        )
        return result

    started = time.perf_counter()
    with (
        patch.object(graph_module, "classify_intent", classify),
        patch.object(graph_module, "generate_tool_decision", decide),
        patch.object(llm_module, "_call_with_fallback", pod_only),
        patch.object(
            graph_module,
            "search_building_docs",
            measured(
                "search_building_docs",
                BuildingDocsResult(status="ok", evidence=[evidence]),
            ),
        ),
        patch.object(
            graph_module,
            "search_public_web",
            measured_web,
        ),
        patch.object(
            graph_module,
            "process_complaint_turn",
            measured("process_complaint_turn", complaint),
        ),
    ):
        state = {
            "request": _request(),
            "route": None,
            "complaint_state": None,
            "reply": None,
            "result": RouteResult(),
            "image_analysis": None,
        }
        result = {}
        for event in graph_module.build_graph().stream(state, stream_mode="updates"):
            for node, update in event.items():
                print(f"\n[{node}]")
                if update.get("pending_tool_calls"):
                    for call in update["pending_tool_calls"]:
                        print(
                            f"선택 도구: {call.name}({json.dumps(call.arguments, ensure_ascii=False)})"
                        )
                messages = update.get("agent_messages", [])
                if (
                    messages
                    and isinstance(messages[-1], dict)
                    and messages[-1].get("role") == "tool"
                ):
                    print(f"도구 결과 메시지: {messages[-1].get('content')}")
                if "reply" in update:
                    print(f"응답: {update['reply']}")
                result.update(update)
    print("\n[호출 지연]")
    for index, decision in enumerate(decisions, start=1):
        summary = decision.get("tool_calls", decision.get("route"))
        print(
            f"Qwen {decision['stage']} {index}: {decision['latency_ms']} ms | {summary}"
        )
    for call in tools:
        print(f"도구 실행: {call}")
    print(f"전체 그래프: {(time.perf_counter() - started) * 1000:.1f} ms")
    print(f"완료 도구: {result.get('completed_tools', ())}")
    print(f"최종 답변: {result.get('reply')}")
    route_result = result.get("result")
    if route_result:
        print(
            f"Web citations: {[item.url for item in route_result.citations if item.source_type == 'web']}"
        )
    print(
        "\n공개 Web 검색은 실제 Tavily를 호출했습니다. 건물 검색/민원 처리는 가짜 결과입니다."
    )


if __name__ == "__main__":
    main()
