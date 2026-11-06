"""Qwen/Pod 없이 실제 Tavily 공개 Web 검색 도구를 단독 점검한다.

실행: cd ai && uv run python scripts/test_tavily_search.py
TAVILY_API_KEY만 필요하며 결과에는 검색 지연, 파싱된 근거와 citation URL이 출력된다.
"""

import time

from zipsai.contracts.converse import ConverseRequest
from zipsai.knowledge.web import search_public_web
from zipsai.orchestration.tool_contracts import ToolFailure, WebSearchArgs
from zipsai.settings import TAVILY_API_KEY

QUERY = "겨울철 수도관 동파 예방 방법"


def main() -> int:
    if not TAVILY_API_KEY:
        print("실패: TAVILY_API_KEY가 설정되지 않았습니다.")
        return 1

    request = ConverseRequest.model_validate(
        {
            "building_id": 101,
            "room_no": "301",
            "resident_id": "synthetic-tavily-test",
            "conversation_id": "tavily-smoke",
            "turn_id": "tavily-smoke-turn",
            "trace_id": "tavily-smoke",
            "current_route": None,
            "current_complaint_state": None,
            "message": {"message_id": "tavily-smoke-message", "text": QUERY},
            "conversation_history": [],
            "complaint_draft": None,
        }
    )

    started = time.perf_counter()
    result = search_public_web(WebSearchArgs(query=QUERY), request)
    latency_ms = (time.perf_counter() - started) * 1000
    print(f"query: {QUERY}")
    print(f"latency: {latency_ms:.1f} ms")

    if isinstance(result, ToolFailure):
        print(f"Tavily tool failure: code={result.code} message={result.message}")
        return 1
    if result.status == "no_results":
        print(
            "Tavily 요청은 성공했지만 유효한 URL/content가 있는 검색 결과가 없습니다."
        )
        return 0

    print(f"status: {result.status} | parsed results: {len(result.evidence)}")
    for index, evidence in enumerate(result.evidence, start=1):
        citation = evidence.citation
        print(f"\n[{index}] {citation.title}")
        print(f"URL: {citation.url}")
        print(f"내용: {evidence.text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
