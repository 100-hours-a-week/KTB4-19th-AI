"""Tool Calling의 SSE 공개 경계를 확인하는 오프라인 데모.

실행:
    cd ai && uv run python scripts/demo_tool_calling_stream.py

Graph가 status와 내부 ToolMessage를 모두 내보낸 상황을 흉내 낸다. API는 status만
전달하고 ToolMessage를 제거한 뒤, 최종 답변과 complete 이벤트를 보낸다.
"""

import json
from unittest.mock import patch

from fastapi.testclient import TestClient

import zipsai.api.converse as converse_module
from zipsai.contracts.converse import Route
from zipsai.main import app
from zipsai.settings import Settings


class _DemoGraph:
    async def astream(self, state: dict[str, object], *, stream_mode: list[str]):
        # 실제 Tool 노드가 보내는 사용자용 진행 상태다.
        yield "custom", {"type": "status", "status": "건물 문서를 확인하고 있습니다."}
        # 실제로는 Agent 재판단을 위해 남기는 내부 프로토콜 메시지다.
        # TC-7의 API 필터가 이 이벤트를 SSE 응답에서 제거해야 한다.
        yield "custom", {
            "role": "tool",
            "tool_call_id": "call-search-1",
            "content": '{"status":"ok","evidence":[...]}' ,
        }
        yield "values", {
            **state,
            "route": Route.KNOWLEDGE,
            "complaint_state": None,
            "reply": "입주민 주차는 지하 1층 지정 구역을 이용해 주세요.",
        }


def _payload() -> dict[str, object]:
    return {
        "building_id": 101,
        "room_no": "301",
        "resident_id": "demo-resident",
        "conversation_id": "demo-conversation",
        "turn_id": "demo-turn",
        "trace_id": "demo-trace",
        "current_route": None,
        "current_complaint_state": None,
        "message": {"message_id": "demo-message", "text": "주차는 어디에 하나요?"},
        "conversation_history": [],
        "complaint_draft": None,
    }


def main() -> None:
    with (
        patch.object(converse_module, "build_graph", lambda streaming=False: _DemoGraph()),
        patch.object(
            converse_module,
            "get_settings",
            lambda: Settings("demo-key", None, "demo-model", 30),
        ),
    ):
        response = TestClient(app).post("/api/v3/ai/converse/stream", json=_payload())

    response.raise_for_status()
    events = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    print("Graph 내부에는 ToolMessage가 있었지만, SSE에는 아래 이벤트만 전달됩니다.\n")
    for event in events:
        print(json.dumps(event, ensure_ascii=False, indent=2))

    assert not any("tool_call_id" in event for event in events)
    assert events[1] == {"type": "status", "status": "건물 문서를 확인하고 있습니다."}
    assert events[-1] == {"type": "done"}
    print("\n확인 완료: ToolMessage는 SSE에 노출되지 않았습니다.")


if __name__ == "__main__":
    main()
