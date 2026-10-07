"""민원 20개와 되묻기 20개의 길만 classify_intent로 채점한다.

검색하지 않는다. 235문항은 읽지 않는다.
운영 주소와 컬렉션 documents는 받더라도 거절한다.

실행 (KTB4-19th-AI/ai 에서):

    uv run python ../evals/query/run_ext_route.py --check
    uv run python ../evals/query/run_ext_route.py
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from run_t1 import collection_name, qdrant_target
from score import REPRODUCE
from score_ext import RouteRow, aggregate_routes, format_route_report, load_jsonl

COLLECTION = "documents_eval"


@dataclass(frozen=True)
class Item:
    id: str
    question: str
    expected: str


def _die(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def load_items() -> list[Item]:
    complaints = [
        row
        for row in load_jsonl(REPRODUCE / "complaints" / "complaints.jsonl")
        if row["c2"]
    ]
    clarify = load_jsonl(REPRODUCE / "extensions" / "clarify.jsonl")
    if len(complaints) != 20 or len(clarify) != 20:
        _die(f"민원 {len(complaints)}건, 되묻기 {len(clarify)}건입니다. 각 20건이어야 합니다.")
    items = [
        Item(id=row["id"], question=row["turns"][0]["text"], expected="complaint")
        for row in complaints
    ]
    items.extend(
        Item(id=row["id"], question=row["question"], expected="clarify")
        for row in clarify
    )
    return items


def _request(item: Item):
    from zipsai.contracts.converse import ConverseRequest, IncomingMessage

    return ConverseRequest(
        building_id=1,
        room_no="101",
        resident_id="eval",
        conversation_id=item.id,
        turn_id=item.id,
        trace_id=item.id,
        current_route=None,
        current_complaint_state=None,
        message=IncomingMessage(message_id=item.id, text=item.question, image_urls=[]),
        conversation_history=[],
        complaint_draft=None,
    )


def score_all(items: list[Item]) -> list[RouteRow]:
    from zipsai.errors import (
        IntentClassificationError,
        LlmRateLimitedError,
        LlmTimeoutError,
        LlmUnavailableError,
        LlmUpstreamError,
    )
    from zipsai.orchestration.intent import classify_intent

    caught = (
        IntentClassificationError,
        LlmUnavailableError,
        LlmRateLimitedError,
        LlmTimeoutError,
        LlmUpstreamError,
    )
    rows: list[RouteRow] = []
    for index, item in enumerate(items, start=1):
        route = None
        failed = False
        try:
            result = classify_intent({"request": _request(item)})
            route = None if result["route"] is None else result["route"].value
            if route is None:
                failed = True
        except caught as error:
            failed = True
            print(f"실패  {item.id}  {type(error).__name__}: {error}", file=sys.stderr)
        rows.append(RouteRow(item.id, item.expected, route, failed))
        label = "실패" if failed else route
        print(f"채점 {index}/{len(items)}  {item.id}  {label}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="민원·되묻기 길 고르기 채점")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--qdrant", default="http://127.0.0.1:6333")
    parser.add_argument("--collection", default=COLLECTION)
    args = parser.parse_args()
    items = load_items()
    if args.check:
        print(f"확인  문항 {len(items)}  컬렉션은 보지 않았습니다")
        return
    qdrant_target(args.qdrant, allow_remote=False)
    collection_name(args.collection)
    print(f"길 고르기 추가 시작  문항 {len(items)}", flush=True)
    print(format_route_report(aggregate_routes(score_all(items))), end="")


if __name__ == "__main__":
    main()
