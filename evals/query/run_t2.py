"""합성문서 T2. 표본 100문항에 답을 쓰게 하고 G2·G5·G6과 잠정 G1을 낸다.

intent는 건너뛴다. 검색은 documents_eval, 답변은 handle_knowledge와 같은 순서로 만든다.
운영 Qdrant와 컬렉션 documents 는 거절한다. 컬렉션은 지우지 않는다.

실행 (KTB4-19th-AI/ai 에서):

    uv run python ../evals/query/run_t2.py --check
    uv run python ../evals/query/run_t2.py --embedding http://127.0.0.1:8001

답변 100회와, 답을 쓴 문항의 환각 판정이 더 나간다. 답변은 LLM_MODEL을 쓰고,
G1 판정은 LLM_JUDGE_MODEL이 설정된 경우에만 별도 모델로 실행한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import BaseModel, Field
from run_t1 import collection_name, embedding_target, qdrant_target
from score import BUILD, REPRODUCE, check_inputs, load_gold, load_manifest
from score_t2 import AnswerRecord, aggregate, format_report, select_sample

COLLECTION = "documents_eval"


class _Verdict(BaseModel):
    unsupported: list[str] = Field(default_factory=list)


_JUDGE_SYSTEM = (
    "답변에 적힌 숫자, 시각, 날짜, 연락처, 조건이 근거 본문에 그대로 있는지만 본다. "
    "그럴듯한지는 보지 않는다. 근거에 없는 항목만 unsupported에 적는다. "
    "없으면 빈 목록이다. 회피 문장과 질문을 다시 말한 것은 항목이 아니다."
)


def _die(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def _answer(
    item, client, encoder, collection: str
) -> tuple[str, str, int | None, tuple[str, ...], tuple[str, ...], str]:
    from zipsai.integrations.llm import generate_text
    from zipsai.knowledge.node import NO_EVIDENCE_REPLY, _citations
    from zipsai.knowledge.prompts import KNOWLEDGE_PROMPT, NO_EVIDENCE, format_context
    from zipsai.knowledge.retrieve import encode_question, search_chunks

    vector = encode_question(item.question, encoder=encoder, trace_id=item.id)
    chunks = search_chunks(vector, item.building_id, client=client, collection=collection)
    retrieved = tuple(str((chunk.payload or {}).get("doc_id") or "") for chunk in chunks)
    evidence = _chunk_text(chunks)
    if not chunks:
        return "no_hit", NO_EVIDENCE_REPLY, None, (), (), ""

    messages = KNOWLEDGE_PROMPT.format_messages(
        question=item.question, context=format_context(chunks)
    )
    raw = generate_text(
        system_prompt=str(messages[0].content),
        user_prompt=str(messages[1].content),
    )
    if raw.strip() == NO_EVIDENCE:
        return "declined", NO_EVIDENCE_REPLY, len(raw), (), retrieved, evidence
    citations = tuple(citation.source_id or "" for citation in _citations(chunks))
    return "answered", raw, len(raw), citations, retrieved, evidence


def _judge(question: str, reply: str, chunks_text: str) -> tuple[tuple[str, ...] | None, bool]:
    from zipsai.settings import get_settings

    judge_model = get_settings().llm_judge_model
    if judge_model is None:
        return None, False

    from zipsai.integrations.llm import generate_structured

    verdict = generate_structured(
        _JUDGE_SYSTEM,
        f"근거:\n{chunks_text}\n\n질문: {question}\n답변: {reply}",
        _Verdict,
        model=judge_model,
    )
    if verdict is None:
        return None, True
    claims = tuple(item.strip() for item in verdict.unsupported if item.strip())
    return claims, False


def _chunk_text(chunks) -> str:
    parts = []
    for chunk in chunks:
        text = (chunk.payload or {}).get("text") or ""
        if text:
            parts.append(text)
    return "\n".join(parts)


def score_sample(client, encoder, items, collection: str) -> list[AnswerRecord]:
    rows: list[AnswerRecord] = []
    for index, item in enumerate(items, start=1):
        outcome, reply, chars, citations, retrieved, evidence = _answer(
            item, client, encoder, collection
        )
        unsupported: tuple[str, ...] | None = None
        judge_failed = False
        if outcome == "answered":
            unsupported, judge_failed = _judge(item.question, reply, evidence)
        rows.append(
            AnswerRecord(
                item_id=item.id,
                question=item.question,
                answerable=item.answerable,
                scope=item.scope,
                outcome=outcome,
                reply=reply,
                answer_chars=chars,
                citation_ids=citations,
                retrieved_ids=retrieved,
                unsupported=unsupported,
                judge_failed=judge_failed,
                evidence=evidence,
            )
        )
        print(f"채점 {index}/{len(items)}  {item.id}  {outcome}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="합성문서 T2 답변 채점")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--qdrant", default="http://127.0.0.1:6333")
    parser.add_argument("--allow-remote", action="store_true")
    parser.add_argument("--embedding", default=None)
    parser.add_argument("--collection", default=COLLECTION)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    errors = check_inputs(REPRODUCE, BUILD)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        raise SystemExit(1)
    docs = load_manifest(REPRODUCE)
    items, _gold_errors = load_gold(REPRODUCE, docs)
    sample = select_sample(items)
    if args.check:
        print(f"확인  표본 {len(sample)}  표 B 20  컬렉션은 보지 않았습니다")
        return

    from qdrant_client import QdrantClient

    from zipsai.integrations.embedding_client import HttpEncoder
    from zipsai.knowledge.retrieve import QUERY_TIMEOUT_SECONDS
    from zipsai.settings import get_settings

    get_settings()
    url = qdrant_target(args.qdrant, args.allow_remote)
    collection = collection_name(args.collection)
    client = QdrantClient(url=url)
    if not client.collection_exists(collection_name=collection):
        _die(f"컬렉션 {collection} 이 없습니다. T1 색인을 먼저 실행하세요.")
    encoder = HttpEncoder(
        base_url=embedding_target(args.embedding),
        timeout=QUERY_TIMEOUT_SECONDS,
        attempts=1,
        lock_wait_seconds=3.5,
    )
    print(f"답변 시작  {url}  컬렉션 {collection}  표본 {len(sample)}", flush=True)
    rows = score_sample(client, encoder, sample, collection)
    metrics = aggregate(rows)
    print(format_report(metrics, rows), end="")
    if args.out:
        payload = {
            "metrics": metrics,
            "rows": [row.__dict__ for row in rows],
        }
        args.out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"기록 {args.out}")


if __name__ == "__main__":
    main()
