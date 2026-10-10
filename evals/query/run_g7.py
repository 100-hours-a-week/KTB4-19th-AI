"""T2 답변 충실도(G7). 근거에 있는 질문 관련 정보를 답변이 빠뜨렸는지 judge로 본다.

답변은 새로 만들지 않는다 — run_t2.py --out 결과를 읽는다. 근거 본문(evidence)이 없는
옛 결과는 --restore 로 같은 컬렉션에서 검색만 다시 해 근거를 되살리고, 검색된 문서가
저장된 retrieved_ids와 같은 묶음일 때만 채점한다(동점 순위는 뒤바뀔 수 있다). 다르면 그 문항은 판정 실패로 남긴다.

실행 (KTB4-19th-AI/ai 에서):

    uv run python ../evals/query/run_g7.py --from results/run_t2.json --out results/run_g7.json
    uv run python ../evals/query/run_g7.py --from results/run_t2.json --restore \\
        --collection documents_eval_aa18cc0 --embedding http://127.0.0.1:8001
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import BaseModel, Field
from run_t1 import collection_name, embedding_target, qdrant_target
from run_t2 import COLLECTION, _chunk_text
from score import REPRODUCE, load_gold, load_manifest
from score_t2 import CompletenessRecord, aggregate_completeness


class _Verdict(BaseModel):
    missing: list[str] = Field(default_factory=list)


_JUDGE_SYSTEM = (
    "질문에 답하는 데 필요한 정보가 근거 본문에 있는데 답변에서 빠졌는지만 본다. "
    "질문과 직접 관련된 조건, 제한, 주의 사항, 일정, 장소가 대상이다. "
    "질문과 관련 없는 정보는 적지 않는다. 답변이 같은 뜻으로 이미 말한 것은 빠진 것이 아니다. "
    "길게 설명했다고 더 좋게 보지 않는다. 빠진 정보만 missing에 짧게 적고, 없으면 빈 목록이다."
)


def _die(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def _judge(
    question: str, reply: str, evidence: str, model: str
) -> tuple[str, ...] | None:
    from zipsai.integrations.llm import generate_structured

    verdict = generate_structured(
        _JUDGE_SYSTEM,
        f"근거:\n{evidence}\n\n질문: {question}\n답변: {reply}",
        _Verdict,
        model=model,
    )
    if verdict is None:
        return None
    return tuple(item.strip() for item in verdict.missing if item.strip())


def _restore(item, client, encoder, collection: str) -> tuple[tuple[str, ...], str]:
    from zipsai.knowledge.retrieve import encode_question, search_chunks

    vector = encode_question(item.question, encoder=encoder, trace_id=item.id)
    chunks = search_chunks(
        vector, item.building_id, client=client, collection=collection
    )
    retrieved = tuple(
        str((chunk.payload or {}).get("doc_id") or "") for chunk in chunks
    )
    return retrieved, _chunk_text(chunks)


def format_report(metrics: dict[str, object], rows: list[CompletenessRecord]) -> str:
    value = metrics["G7"]
    shown = "—" if value is None else f"{value:.3f}"
    lines = [
        "G7 답변 충실도",
        (
            f"G7 {shown}  ({metrics['G7_complete']}/{metrics['G7_judged']})  "
            f"빠진 정보 {metrics['G7_missing']}건  판정 실패 {metrics['G7_failed']}"
        ),
    ]
    for row in rows:
        if row.failed:
            lines.append(f"실패 {row.item_id}  {row.reason}")
        elif row.missing:
            lines.append(f"빠짐 {row.item_id}  " + " / ".join(row.missing))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="T2 답변 충실도(G7) 채점")
    parser.add_argument("--from", dest="source", type=Path, required=True)
    parser.add_argument("--restore", action="store_true")
    parser.add_argument("--qdrant", default="http://127.0.0.1:6333")
    parser.add_argument("--allow-remote", action="store_true")
    parser.add_argument("--embedding", default=None)
    parser.add_argument("--collection", default=COLLECTION)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    from zipsai.settings import get_settings

    judge_model = get_settings().llm_judge_model
    if judge_model is None:
        _die("LLM_JUDGE_MODEL 이 없습니다. G7은 judge로만 채점합니다.")

    rows = [
        row
        for row in json.loads(args.source.read_text(encoding="utf-8"))["rows"]
        if row["outcome"] == "answered"
    ]
    needs_restore = any(not row.get("evidence") for row in rows)
    if needs_restore and not args.restore:
        _die(
            "근거 본문이 없는 결과입니다. --restore 로 검색만 다시 해 근거를 되살리세요."
        )

    items: dict[str, object] = {}
    client = encoder = None
    collection = collection_name(args.collection)
    if needs_restore:
        from qdrant_client import QdrantClient

        from zipsai.integrations.embedding_client import HttpEncoder
        from zipsai.knowledge.retrieve import QUERY_TIMEOUT_SECONDS

        gold, _errors = load_gold(REPRODUCE, load_manifest(REPRODUCE))
        items = {item.id: item for item in gold}
        client = QdrantClient(url=qdrant_target(args.qdrant, args.allow_remote))
        if not client.collection_exists(collection_name=collection):
            _die(f"컬렉션 {collection} 이 없습니다.")
        encoder = HttpEncoder(
            base_url=embedding_target(args.embedding),
            timeout=QUERY_TIMEOUT_SECONDS,
            attempts=1,
            lock_wait_seconds=3.5,
        )

    records: list[CompletenessRecord] = []
    for index, row in enumerate(rows, start=1):
        evidence = row.get("evidence") or ""
        if not evidence:
            retrieved, evidence = _restore(
                items[row["item_id"]], client, encoder, collection
            )
            # 동점 순위는 검색할 때마다 뒤바뀔 수 있어 순서가 아니라 문서 묶음으로 비교한다.
            if sorted(retrieved) != sorted(row["retrieved_ids"]):
                records.append(
                    CompletenessRecord(row["item_id"], None, True, "restore_mismatch")
                )
                print(
                    f"채점 {index}/{len(rows)}  {row['item_id']}  복원 불일치",
                    flush=True,
                )
                continue
        missing = _judge(row["question"], row["reply"], evidence, judge_model)
        failed = missing is None
        records.append(
            CompletenessRecord(
                row["item_id"], missing, failed, "judge_failed" if failed else None
            )
        )
        print(f"채점 {index}/{len(rows)}  {row['item_id']}", flush=True)

    metrics = aggregate_completeness(records)
    print(format_report(metrics, records), end="")
    if args.out:
        payload = {"metrics": metrics, "rows": [record.__dict__ for record in records]}
        args.out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"기록 {args.out}")


if __name__ == "__main__":
    main()
