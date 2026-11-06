"""Qwen Pod tool-routing 관찰 사례 36건을 JSONL/CSV로 기록한다.

실행: cd ai && uv run python scripts/test_tool_calling_pod_batch.py
POD_BASE_URL(…/v1), POD_API_KEY, POD_MODEL 설정이 필요하다. 결과는
ai/.local-test-results/ 아래에 저장한다. 공개 Web 검색은 실제 Tavily를 호출하고,
건물 검색과 민원 처리는 가짜 결과를 사용한다.
"""

import asyncio
import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

os.environ["ACTIVE_MODEL_PROVIDER"] = "pod"

import zipsai.integrations.llm as llm_module
import zipsai.orchestration.graph as graph_module
from zipsai.contracts.converse import (
    Citation,
    ComplaintDraft,
    ComplaintState,
    ConverseRequest,
    Route,
    RouteResult,
)
from zipsai.integrations.runpod import get_locator
from zipsai.orchestration.tool_contracts import (
    ComplaintTurnResult,
    SearchEvidence,
)
from zipsai.settings import TAVILY_API_KEY, get_settings

TOOLS = ("search_building_docs", "search_public_web", "process_complaint_turn")


@dataclass(frozen=True)
class Case:
    case_id: str
    category: str
    prompt: str
    expected_tools: tuple[str, ...]
    complaint_context: bool = False


CASES = tuple(
    Case(f"{category}-{index:02}", category, prompt, expected, context)
    for category, examples, expected, context in (
        (
            "building",
            (
                "우리 단지 방문 차량 등록 절차를 알려줘.",
                "택배 보관 장소와 찾아가는 방법은?",
                "재활용 분리배출은 어디서 하나요?",
                "관리사무소 운영 시간을 알려주세요.",
                "주차 등록을 해지하려면 어떻게 해야 해?",
                "공용 세탁실 이용 안내가 궁금해요.",
            ),
            ("search_building_docs",),
            False,
        ),
        (
            "public_web",
            (
                "오늘 서울 미세먼지 대응 요령을 알려줘.",
                "겨울철 수도관 동파 예방법을 알려주세요.",
                "최근 전기차 화재 예방 수칙을 찾아줘.",
                "태풍이 올 때 창문을 안전하게 관리하는 법은?",
                "반려견 분리불안 완화 방법을 알려줘.",
                "실내 곰팡이 예방에 일반적으로 좋은 방법은?",
            ),
            ("search_public_web",),
            False,
        ),
        (
            "complaint",
            (
                "욕실 천장에서 물이 떨어져요. 민원 초안 부탁해요.",
                "주방 싱크대 배수가 막혔습니다. 접수하고 싶어요.",
                "거실 보일러가 작동하지 않아요.",
                "복도 조명이 계속 깜빡입니다. 민원 작성해 주세요.",
                "엘리베이터 문이 닫힐 때 소음이 납니다.",
                "현관 도어락이 고장 났어요. 신고할게요.",
            ),
            ("process_complaint_turn",),
            False,
        ),
        (
            "building_complaint",
            (
                "방문 주차 방법을 알려주고 지하 주차장 조명 고장 민원도 작성해줘.",
                "분리수거 위치와 배출 방법을 알려주고 복도 쓰레기 민원도 넣어줘.",
                "택배 수령 장소를 알려주고 택배함 고장 신고도 작성해 주세요.",
                "관리사무소 운영 시간을 안내하고 욕실 누수 민원도 접수해줘.",
                "공용시설 이용 안내와 엘리베이터 소음 민원을 같이 부탁해요.",
                "주차 등록 절차도 알려주고 차단기 고장 민원도 만들어줘.",
            ),
            ("search_building_docs", "process_complaint_turn"),
            False,
        ),
        (
            "no_tool",
            (
                "안녕하세요!",
                "고마워요.",
                "좋은 하루 보내세요.",
                "네, 확인했습니다.",
                "도움이 됐어요.",
                "잠깐 인사하러 왔어요.",
            ),
            (),
            False,
        ),
        (
            "complaint_followup",
            (
                "욕실에서 발생했어요.",
                "물이 계속 새고 있습니다.",
                "어제 저녁부터 시작됐어요.",
                "주방 쪽 벽면입니다.",
                "천장에서 물방울이 떨어져요.",
                "가능하면 오늘 확인 부탁드려요.",
            ),
            ("process_complaint_turn",),
            True,
        ),
    )
    for index, prompt in enumerate(examples, start=1)
)

assert len(CASES) == 36 and len({case.case_id for case in CASES}) == len(CASES)
assert all(set(case.expected_tools) <= set(TOOLS) for case in CASES)


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


def _request(case: Case) -> ConverseRequest:
    return ConverseRequest.model_validate(
        {
            "building_id": 101,
            "room_no": "301",
            "resident_id": "synthetic-test-resident",
            "conversation_id": case.case_id,
            "turn_id": f"{case.case_id}-turn",
            "trace_id": case.case_id,
            "current_route": Route.COMPLAINT if case.complaint_context else None,
            "current_complaint_state": ComplaintState.COLLECTING
            if case.complaint_context
            else None,
            "message": {"message_id": f"{case.case_id}-message", "text": case.prompt},
            "conversation_history": [],
            "complaint_draft": ComplaintDraft(issue_type="plumbing", location="욕실")
            if case.complaint_context
            else None,
        }
    )


def _stub_results():
    building = SearchEvidence(
        text="관리사무소에 문의해 주세요.",
        citation=Citation(
            source_type="building_document",
            source_id="demo",
            title="건물 안내",
            snippet="관리사무소에 문의",
            url=None,
        ),
    )
    complaint = ComplaintTurnResult(
        reply="문제 위치와 증상을 확인했습니다.",
        complaint_state=ComplaintState.COLLECTING,
        result=RouteResult(
            complaint_draft=ComplaintDraft(
                issue_type="plumbing", location="욕실", symptom="시설 문제"
            ),
            missing_fields=["symptom"],
        ),
    )
    return building, complaint


def _run_case(case: Case, stubs) -> dict[str, object]:
    decisions, tool_events = [], []
    real_decide = llm_module.generate_tool_decision
    real_classify = graph_module.classify_intent
    real_web_search = graph_module.search_public_web

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
                tool_events.append(
                    {
                        "name": name,
                        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                        "stub": True,
                    }
                )

        return run

    def measured_web(*args, **kwargs):
        started = time.perf_counter()
        result = real_web_search(*args, **kwargs)
        tool_events.append(
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
            graph_module, "search_building_docs", measured(TOOLS[0], stubs[0])
        ),
        patch.object(graph_module, "search_public_web", measured_web),
        patch.object(
            graph_module, "process_complaint_turn", measured(TOOLS[2], stubs[1])
        ),
    ):
        request = _request(case)
        state = {
            "request": request,
            "route": None,
            "complaint_state": request.current_complaint_state,
            "reply": None,
            "result": RouteResult(),
            "image_analysis": None,
        }
        result = {}
        for event in graph_module.build_graph().stream(state, stream_mode="updates"):
            for update in event.values():
                result.update(update)
    actual = list(result.get("completed_tools", ()))
    return {
        "case_id": case.case_id,
        "category": case.category,
        "prompt": case.prompt,
        "expected_tools": list(case.expected_tools),
        "actual_tools": actual,
        "tools_match": actual == list(case.expected_tools),
        "model_decisions": decisions,
        "tool_events": tool_events,
        "total_latency_ms": round((time.perf_counter() - started) * 1000, 1),
        "route": getattr(result.get("route"), "value", result.get("route")),
        "reply": result.get("reply"),
        "tool_call_count": result.get("tool_call_count", 0),
    }


def _error_chain(error: Exception) -> str:
    details, seen = [], set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen and len(details) < 4:
        seen.add(id(current))
        details.append(f"{type(current).__name__}: {str(current)[:500]}")
        current = current.__cause__ or current.__context__
    return " <- ".join(details)


def _has_client_error(error: Exception) -> bool:
    current: BaseException | None = error
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        status_code = getattr(current, "status_code", None)
        if isinstance(status_code, int) and 400 <= status_code < 500:
            return True
        current = current.__cause__ or current.__context__
    return False


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
    output_dir = Path(".local-test-results")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 1_000_000:06d}"
    jsonl_path = output_dir / f"tool-calling-pod-{run_id}.jsonl"
    csv_path = output_dir / f"tool-calling-pod-{run_id}.csv"
    stubs = _stub_results()
    rows = []
    fields = (
        "case_id",
        "category",
        "prompt",
        "expected_tools",
        "actual_tools",
        "tools_match",
        "total_latency_ms",
        "model_decisions",
        "tool_events",
        "route",
        "reply",
        "tool_call_count",
        "error",
    )
    print(f"provider={settings.provider} model={settings.llm_model} cases={len(CASES)}")
    with (
        jsonl_path.open("w", encoding="utf-8") as jsonl_output,
        csv_path.open("w", encoding="utf-8-sig", newline="") as csv_output,
    ):
        writer = csv.DictWriter(csv_output, fieldnames=fields)
        writer.writeheader()
        for index, case in enumerate(CASES, start=1):
            try:
                row = _run_case(case, stubs)
                row["error"] = ""
                status = "MATCH" if row["tools_match"] else "CHECK"
                stop_after_row = any(
                    event["name"] == "search_public_web"
                    and event.get("status") == "error"
                    for event in row["tool_events"]
                )
            except Exception as error:  # noqa: BLE001 keep later cases useful after a failed call
                stop_after_row = _has_client_error(error)
                row = {
                    "case_id": case.case_id,
                    "category": case.category,
                    "prompt": case.prompt,
                    "expected_tools": list(case.expected_tools),
                    "actual_tools": [],
                    "tools_match": False,
                    "total_latency_ms": None,
                    "model_decisions": [],
                    "tool_events": [],
                    "route": None,
                    "reply": None,
                    "tool_call_count": 0,
                    "error": _error_chain(error),
                }
                status = "ERROR"
            rows.append(row)
            jsonl_output.write(json.dumps(row, ensure_ascii=False) + "\n")
            jsonl_output.flush()
            writer.writerow(
                {
                    key: json.dumps(row[key], ensure_ascii=False)
                    if isinstance(row[key], (list, dict))
                    else row[key]
                    for key in fields
                }
            )
            csv_output.flush()
            print(
                f"[{index:02}/{len(CASES)}] {status} {case.case_id} expected={row['expected_tools']} actual={row['actual_tools']} latency={row['total_latency_ms']}ms"
            )
            if row["error"]:
                print(f"    원인: {row['error']}")
            if stop_after_row:
                print("    Pod 또는 Tavily 요청 오류라 남은 사례는 실행하지 않습니다.")
                break
            for decision_index, decision in enumerate(row["model_decisions"], start=1):
                summary = [call["name"] for call in decision.get("tool_calls", ())]
                print(
                    f"    Qwen {decision['stage']}#{decision_index} {decision['latency_ms']}ms -> {summary or decision.get('route', 'final answer')}"
                )
    matches = sum(bool(row["tools_match"]) for row in rows)
    failures = sum(bool(row["error"]) for row in rows)
    print(
        f"\n예상 도구 시퀀스 일치: {matches}/{len(rows)} 실행 사례 | 실행 실패: {failures}"
    )
    if len(rows) < len(CASES):
        print(f"미실행 사례: {len(CASES) - len(rows)}")
    print(f"JSONL: {jsonl_path.resolve()}\nCSV:   {csv_path.resolve()}")
    print(
        "공개 Web 검색은 실제 Tavily를 호출합니다. 건물 검색/민원은 가짜 결과이며, 정답률 평가셋은 아닙니다."
    )


if __name__ == "__main__":
    main()
