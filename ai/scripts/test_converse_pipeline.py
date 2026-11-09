"""실제 로컬 Converse API에 요청을 보내 #188 tool pipeline을 확인한다.

실행 예시:
    cd ai
    COMPOSE_PROJECT_NAME=issue188-local-smoke \\
    AI_CONVERSE_URL=http://127.0.0.1:18001/api/v3/ai/converse \\
    uv run python scripts/test_converse_pipeline.py

각 요청은 설정된 LLM provider를 호출한다. zipsai-api 컨테이너의 trace 로그도 읽는다.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from argparse import ArgumentParser
from pathlib import Path
from uuid import uuid4

import httpx

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URL = "http://127.0.0.1:8000/api/v3/ai/converse"
_LOG_EVENTS = {
    "stage_done",
    "agent_decision",
    "tool_call_selected",
    "tool_call_message",
    "tool_result_message",
    "tool_call_completed",
    "tool_call_rejected",
    "tool_call_retry",
    "tool_call_protocol_failure",
}
_LOG_JSON = re.compile(r"\|\s*(\{.*\})\s*$")


def _request(
    text: str,
    *,
    trace_id: str,
    current_route: str | None = None,
    complaint_state: str | None = None,
    complaint_draft: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "building_id": 101,
        "room_no": "301",
        "resident_id": "synthetic-test-resident",
        "conversation_id": f"pipeline-{trace_id}",
        "turn_id": f"{trace_id}-turn",
        "trace_id": trace_id,
        "current_route": current_route,
        "current_complaint_state": complaint_state,
        "message": {"message_id": f"{trace_id}-message", "text": text},
        "conversation_history": [],
        "complaint_draft": complaint_draft,
    }


def _docker_logs(since: int, trace_id: str) -> list[dict[str, object]]:
    completed = subprocess.run(
        ["docker", "compose", "logs", "--no-color", f"--since={since}", "zipsai-api"],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=15,
    )
    if completed.returncode:
        raise RuntimeError(
            "Compose 로그를 읽지 못했습니다. zipsai-api가 이 저장소의 compose로 실행 중인지 확인하세요.\n"
            + completed.stderr.strip()
        )

    events = []
    for line in completed.stdout.splitlines():
        match = _LOG_JSON.search(line)
        if not match:
            continue
        try:
            event = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if event.get("trace_id") == trace_id and event.get("event") in _LOG_EVENTS:
            events.append(event)
    return events


def _show_logs(events: list[dict[str, object]]) -> tuple[set[str], set[str], set[str]]:
    planned: set[str] = set()
    selected: set[str] = set()
    failed: set[str] = set()
    for event in events:
        name = event.get("event")
        if name == "stage_done" and event.get("stage") == "intent":
            planned.update(event.get("planned_tools", []))
            print(
                f"  INTENT plan={json.dumps(event.get('planned_tools', []), ensure_ascii=False)} "
                f"route={event.get('intent_route')}"
            )
        elif name == "agent_decision":
            print(
                f"  ACTION outcome={event.get('outcome')} route={event.get('route')} "
                f"tool_call_count={event.get('tool_call_count')}"
            )
        elif name == "tool_call_selected":
            tool = str(event.get("tool"))
            selected.add(tool)
            print(f"  TOOL selected={tool} call={event.get('tool_call_count')}")
        elif name in {"tool_call_message", "tool_result_message"}:
            print(f"  {name.upper()} {event.get('tool_message')}")
        elif name == "tool_call_completed":
            if event.get("outcome") == "failure":
                failed.add(str(event.get("tool")))
            print(
                f"  TOOL result={event.get('tool')} outcome={event.get('outcome')} "
                f"duration_ms={event.get('duration_ms')}"
            )
        elif name in {"tool_call_rejected", "tool_call_retry", "tool_call_protocol_failure"}:
            details = {
                key: value
                for key, value in event.items()
                if key in {"reason", "tool", "attempt", "requested_count"}
            }
            print(f"  TOOL guard event={name} details={json.dumps(details, ensure_ascii=False)}")
    return planned, selected, failed


def _run_with_messages(
    request: dict[str, object],
) -> tuple[int, dict[str, object], set[str], set[str], set[str]] | None:
    """Run the same API handler in the API container and print its in-memory LLM/tool messages."""
    payload = json.dumps(request, ensure_ascii=False)
    source = f'''\
import os
os.environ["AI_LOG_TOOL_MESSAGES"] = "true"
import json
from fastapi.testclient import TestClient
from zipsai.main import app

with TestClient(app) as client:
    response = client.post("/api/v3/ai/converse", json=json.loads({payload!r}))
print("HTTP", response.status_code)
print("RESPONSE", response.text)
'''
    completed = subprocess.run(
        ["docker", "compose", "-f", "compose.yaml", "exec", "-T", "zipsai-api", "python", "-"],
        cwd=ROOT,
        input=source,
        capture_output=True,
        check=False,
        text=True,
        timeout=240,
    )
    output = f"{completed.stdout}\n{completed.stderr}"
    if completed.returncode:
        print(output)
        return None
    match = re.search(r"^HTTP (\d+)\s*$", output, re.MULTILINE)
    response_match = re.search(r"^RESPONSE (.+)$", output, re.MULTILINE)
    if not match or not response_match:
        return None
    try:
        body = json.loads(response_match.group(1))
    except json.JSONDecodeError:
        return None
    events = []
    for line in output.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") in _LOG_EVENTS:
            events.append(event)
    print("PIPELINE LOGS:")
    planned, selected, failed = _show_logs(events)
    return int(match.group(1)), body, planned, selected, failed


def _run_case(
    *,
    title: str,
    request: dict[str, object],
    expected_tools: set[str],
    expected_route: str,
    preserve_complaint: bool = False,
    show_messages: bool = False,
) -> bool:
    url = os.getenv("AI_CONVERSE_URL", DEFAULT_URL)
    trace_id = str(request["trace_id"])
    print(f"\n## {title}\ntrace_id={trace_id}")
    print("REQUEST:")
    print(json.dumps(request, ensure_ascii=False, indent=2))
    if show_messages:
        result = _run_with_messages(request)
        if result is None:
            print("In-container message capture failed; check that the Compose service is running.")
            return False
        status_code, body, planned_tools, selected_tools, failed_tools = result
        print("RESPONSE JSON:")
        print(json.dumps(body, ensure_ascii=False, indent=2))
    else:
        started_at = int(time.time())
        try:
            response = httpx.post(url, json=request, timeout=180)
        except httpx.HTTPError as error:
            print(f"HTTP request failed: {type(error).__name__}: {error}")
            return False
        status_code = response.status_code
        try:
            body = response.json()
        except ValueError:
            print(f"HTTP {status_code}\n{response.text[:2000]}")
            return False
        print(f"HTTP {status_code}\nRESPONSE:")
        print(json.dumps(body, ensure_ascii=False, indent=2))
        try:
            events = _docker_logs(started_at, trace_id)
        except (OSError, subprocess.SubprocessError, RuntimeError) as error:
            print(f"\nLOGS unavailable: {error}")
            return False
        print("PIPELINE LOGS:")
        planned_tools, selected_tools, failed_tools = _show_logs(events)

    data = body.get("data") if isinstance(body, dict) else None
    result = data.get("result") if isinstance(data, dict) else None
    route = data.get("route") if isinstance(data, dict) else None
    next_state = data.get("next_complaint_state") if isinstance(data, dict) else None
    passed = (
        status_code < 400
        and route == expected_route
        and planned_tools == expected_tools
        and selected_tools == expected_tools
        and not failed_tools
    )
    if preserve_complaint:
        sent_state = request["current_complaint_state"]
        sent_draft = request["complaint_draft"]
        got_draft = result.get("complaint_draft") if isinstance(result, dict) else None
        draft_preserved = isinstance(got_draft, dict) and all(
            got_draft.get(key) == sent_draft.get(key)
            for key in ("issue_type", "location", "symptom")
        )
        passed = passed and next_state == sent_state and draft_preserved
        print(
            f"STATE check expected={sent_state}, got={next_state}; "
            f"draft_preserved={draft_preserved}"
        )
    if planned_tools != expected_tools:
        print(f"FAIL plan expected={sorted(expected_tools)} got={sorted(planned_tools)}")
    if selected_tools != expected_tools:
        print(f"FAIL action expected={sorted(expected_tools)} got={sorted(selected_tools)}")
    if failed_tools:
        print(f"FAIL tool execution failed: {sorted(failed_tools)}")
    print(f"ROUTE expected={expected_route}, got={route}; state={next_state}")
    print("CASE PASS" if passed else "CASE FAIL")
    return passed


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show-messages",
        action="store_true",
        help="API 컨테이너 안에서 동일 핸들러를 실행해 LLM 메시지와 도구 인자를 표시",
    )
    args = parser.parse_args()
    complaint = {
        "issue_type": "plumbing",
        "location": "욕실",
        "symptom": "천장에서 물이 떨어짐",
        "attachment_ids": [],
    }
    incomplete_complaint = {
        "issue_type": "plumbing",
        "location": None,
        "symptom": "욕실 천장에서 물이 떨어짐",
        "attachment_ids": [],
    }
    cases = [
        {
            "title": "새 민원 단독",
            "request": _request(
                "욕실 천장에서 물이 떨어집니다. 민원 초안을 작성해 주세요.",
                trace_id=f"pipeline-complaint-{uuid4().hex[:8]}",
            ),
            "expected_tools": {"process_complaint_turn"},
            "expected_route": "complaint",
        },
        {
            "title": "진행 중 민원에서 새 건물 질문: 상태와 초안 보존",
            "request": _request(
                "우리 아파트 방문 차량 등록 절차를 알려주세요.",
                trace_id=f"pipeline-active-complaint-{uuid4().hex[:8]}",
                current_route="complaint",
                complaint_state="collecting",
                complaint_draft=complaint,
            ),
            "expected_tools": {"search_building_docs"},
            "expected_route": "knowledge",
            "preserve_complaint": True,
        },
        {
            "title": "건물 질문과 새 민원 복합 요청",
            "request": _request(
                "욕실 천장에서 물이 떨어지는 민원 초안을 만들고, 방문 차량 등록 방법도 알려주세요.",
                trace_id=f"pipeline-composite-{uuid4().hex[:8]}",
            ),
            "expected_tools": {"process_complaint_turn", "search_building_docs"},
            "expected_route": "complaint",
        },
        {
            "title": "건물 질문 단독",
            "request": _request(
                "우리 아파트 분리수거 배출 요일을 알려주세요.",
                trace_id=f"pipeline-building-only-{uuid4().hex[:8]}",
            ),
            "expected_tools": {"search_building_docs"},
            "expected_route": "knowledge",
        },
        {
            "title": "일반 공개 정보 질문 단독",
            "request": _request(
                "오늘 서울 미세먼지 상황을 알려주세요.",
                trace_id=f"pipeline-public-web-{uuid4().hex[:8]}",
            ),
            "expected_tools": {"search_public_web"},
            "expected_route": "knowledge",
        },
        {
            "title": "진행 중 민원에서 무관한 공개 정보 질문",
            "request": _request(
                "오늘 서울 미세먼지 상황도 알려주세요.",
                trace_id=f"pipeline-active-public-{uuid4().hex[:8]}",
                current_route="complaint",
                complaint_state="collecting",
                complaint_draft=complaint,
            ),
            "expected_tools": {"search_public_web"},
            "expected_route": "knowledge",
            "preserve_complaint": True,
        },
        {
            "title": "진행 중 민원에 빠진 위치 정보 제공",
            "request": _request(
                "301호 욕실 천장에서 물이 떨어지고 있어요.",
                trace_id=f"pipeline-complaint-followup-{uuid4().hex[:8]}",
                current_route="complaint",
                complaint_state="collecting",
                complaint_draft=incomplete_complaint,
            ),
            "expected_tools": {"process_complaint_turn"},
            "expected_route": "complaint",
        },
        {
            "title": "문맥 없는 모호한 질문",
            "request": _request(
                "그거 어떻게 하면 되나요?",
                trace_id=f"pipeline-ambiguous-{uuid4().hex[:8]}",
            ),
            "expected_tools": set(),
            "expected_route": "clarify",
        },
    ]
    print(f"Converse endpoint: {os.getenv('AI_CONVERSE_URL', DEFAULT_URL)}")
    print("LLM provider/model은 API 설정을 사용합니다. 요청 본문은 synthetic resident 데이터입니다.")
    if args.show_messages:
        print("메시지 캡처는 API 컨테이너의 동일 Converse 핸들러에서 실행됩니다.")
    outcomes = [_run_case(**case, show_messages=args.show_messages) for case in cases]
    passed = sum(outcomes)
    print(f"\nSUMMARY: {passed}/{len(outcomes)} cases passed")
    return 0 if all(outcomes) else 1


if __name__ == "__main__":
    sys.exit(main())
