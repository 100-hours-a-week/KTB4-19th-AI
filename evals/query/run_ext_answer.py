"""부분 답·공격·다음 말을 handle_knowledge로 채점한다.

235문항은 읽지 않는다. 검색은 documents_eval, 운영 주소와 컬렉션 documents는 거절한다.
이전 대화는 요청에만 싣고, 검색이 그 대화를 보게 코드를 고치지 않는다.

실행 (KTB4-19th-AI/ai 에서):

    uv run python ../evals/query/run_ext_answer.py --check
    uv run python ../evals/query/run_ext_answer.py --embedding http://127.0.0.1:8001
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from run_t1 import collection_name, embedding_target, qdrant_target
from score import REPRODUCE, load_manifest
from score_ext import (
    BREACH_IF,
    AnswerRow,
    aggregate_answers,
    format_answer_report,
    is_breach,
    is_followup_ok,
    is_partial_ok,
    load_jsonl,
)

COLLECTION = "documents_eval"
EXTENSIONS = REPRODUCE / "extensions"
EXPECTED = {"partial": 15, "attack": 20, "followup": 12}


@dataclass(frozen=True)
class Item:
    id: str
    kind: str
    building_id: int
    question: str
    attack: str | None
    quote: str | None
    history: tuple[tuple[str, str], ...]


def _die(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def load_items(buildings: dict[str, int]) -> list[Item]:
    items: list[Item] = []
    specs = (
        ("partial.jsonl", "partial"),
        ("attacks.jsonl", "attack"),
        ("followups.jsonl", "followup"),
    )
    for filename, kind in specs:
        rows = load_jsonl(EXTENSIONS / filename)
        if len(rows) != EXPECTED[kind]:
            _die(f"{filename} 이 {len(rows)}건입니다. {EXPECTED[kind]}건이어야 합니다.")
        for row in rows:
            code = row["building_code"]
            if code not in buildings:
                _die(f"{row['id']} 건물 코드가 없습니다: {code}")
            history = tuple(
                (turn["role"], turn["text"]) for turn in (row.get("history") or [])
            )
            items.append(
                Item(
                    id=row["id"],
                    kind=kind,
                    building_id=buildings[code],
                    question=row["question"],
                    attack=row.get("attack"),
                    quote=row.get("covered_quote"),
                    history=history,
                )
            )
    return items


def request_for(item: Item):
    from zipsai.contracts.converse import ConverseRequest, HistoryTurn, IncomingMessage

    history = [
        HistoryTurn(
            message_id=f"{item.id}-h{index}",
            role=role,
            text=text,
        )
        for index, (role, text) in enumerate(item.history)
    ]
    return ConverseRequest(
        building_id=item.building_id,
        room_no="101",
        resident_id="eval",
        conversation_id=item.id,
        turn_id=item.id,
        trace_id=item.id,
        current_route=None,
        current_complaint_state=None,
        message=IncomingMessage(message_id=item.id, text=item.question),
        conversation_history=history,
        complaint_draft=None,
    )


def _judge(item: Item, reply: str) -> tuple[bool, bool]:
    if item.kind == "partial":
        return is_partial_ok(reply), False
    if item.kind == "attack":
        needles = BREACH_IF[item.id]
        return False, is_breach(reply, attack=item.attack or "", needles=needles)
    return is_followup_ok(reply, item.quote or ""), False


def score_all(items: list[Item]) -> list[AnswerRow]:
    from zipsai.errors import (
        EmbeddingError,
        LlmRateLimitedError,
        LlmTimeoutError,
        LlmUnavailableError,
        LlmUpstreamError,
        VectorStoreError,
    )
    from zipsai.knowledge.node import handle_knowledge

    caught = (
        EmbeddingError,
        VectorStoreError,
        LlmUnavailableError,
        LlmRateLimitedError,
        LlmTimeoutError,
        LlmUpstreamError,
    )
    rows: list[AnswerRow] = []
    for index, item in enumerate(items, start=1):
        failed = False
        ok = False
        breach = False
        try:
            reply = str(handle_knowledge(request_for(item))["reply"])
            ok, breach = _judge(item, reply)
        except caught as error:
            failed = True
            print(f"실패  {item.id}  {type(error).__name__}: {error}", file=sys.stderr)
        rows.append(
            AnswerRow(
                item_id=item.id, kind=item.kind, ok=ok, breach=breach, failed=failed
            )
        )
        label = "실패" if failed else ("돌파" if breach else ("맞춤" if ok else "빗나감"))
        print(f"채점 {index}/{len(items)}  {item.id}  {label}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="부분 답·공격·다음 말 채점")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--qdrant", default="http://127.0.0.1:6333")
    parser.add_argument("--embedding", default=None)
    parser.add_argument("--collection", default=COLLECTION)
    args = parser.parse_args()

    buildings = {doc.building_code: doc.building_id for doc in load_manifest(REPRODUCE)}
    items = load_items(buildings)
    if args.check:
        print(f"확인  문항 {len(items)}  컬렉션은 보지 않았습니다")
        return

    url = qdrant_target(args.qdrant, allow_remote=False)
    collection = collection_name(args.collection)
    embedding = embedding_target(args.embedding)
    os.environ["QDRANT_URL"] = url
    os.environ["QDRANT_COLLECTION"] = collection
    os.environ["EMBEDDING_API_URL"] = embedding
    print(f"답 쓰기 추가 시작  {url}  컬렉션 {collection}  문항 {len(items)}", flush=True)
    print(format_answer_report(aggregate_answers(score_all(items))), end="")


if __name__ == "__main__":
    main()
