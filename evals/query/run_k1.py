"""색인 원본 51건에서 문서당 첫 청크를 Qdrant 실제 색인 결과로 뽑아 원문과 같은 뜻인지 LLM Judge로 본다.

실제 색인 결과(documents_eval 컬렉션)를 읽는다 — 로컬에서 chunk_pages()를 다시 돌리지 않는다.
원문 쪽은 parse_document 결과(마스킹·정제 전)를 쓴다. T1을 먼저 돌려 51건이 색인돼
있어야 한다. LLM_JUDGE_MODEL이 없으면 돌리지 않는다.

실행 (KTB4-19th-AI/ai 에서):

    uv run python ../evals/query/run_k1.py --check
    uv run python ../evals/query/run_k1.py --qdrant http://127.0.0.1:6333
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from pydantic import BaseModel, Field
from run_t1 import collection_name, qdrant_target
from score import BUILD, REPRODUCE, build_files, load_manifest

_JUDGE_SYSTEM = (
    "원문 문단과 색인된 청크 텍스트를 비교한다.\n"
    "청크가 원문의 뜻을 온전히 담고 있으면 통과다. 표현이 달라도 뜻이 같으면 통과다.\n"
    "실패라면 아래 중 가장 가까운 유형을 하나 골라 reason에 적는다 — 다른 중간 산출물을 "
    "안 남겨도 이 분류로 어느 단계가 원인인지 짐작할 수 있다.\n"
    "  - ocr_garbled: 글자가 비정상적으로 띄어 쓰이거나 깨짐(파싱 단계 의심)\n"
    "  - truncated: 문장이나 사실이 중간에 잘림(청킹 단계 의심)\n"
    "  - over_masked: 지우면 안 될 내용이 마스킹으로 사라짐(마스킹 단계 의심)\n"
    "  - boilerplate_leftover: 반복되는 머리말·목차·페이지 번호가 안 지워지고 남음(정제 단계 의심)\n"
    "  - other: 위 네 가지에 안 맞는 다른 문제"
)

_REASONS = ("ocr_garbled", "truncated", "over_masked", "boilerplate_leftover", "other")


class _Verdict(BaseModel):
    passed: bool
    reason: str | None = Field(default=None)


@dataclass(frozen=True)
class K1Row:
    doc_id: str
    page: int | None
    passed: bool
    reason: str | None
    failed: bool
    failure_kind: str | None = None


def _die(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def fetch_indexed_chunk(client, collection: str, building_id: int, doc_id: str) -> tuple[int, str] | None:
    """Qdrant에서 이 문서의 실제 색인 포인트를 찾아 (page, text) 하나를 결정적으로 고른다.

    point id가 uuid라 청크 순번이 없다 — page·section·text 순으로 정렬해 첫 번째를 고른다.
    """
    from qdrant_client import models

    from zipsai.integrations.qdrant import building_condition

    points, _ = client.scroll(
        collection_name=collection,
        scroll_filter=models.Filter(
            must=[
                building_condition(building_id),
                models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)),
            ]
        ),
        with_payload=True,
        limit=200,
    )
    if not points:
        return None
    chosen = min(
        points,
        key=lambda p: (
            p.payload.get("page", 0),
            p.payload.get("section") or "",
            p.payload.get("text", ""),
        ),
    )
    return chosen.payload.get("page", 0), str(chosen.payload.get("text", ""))


def narrow_original(raw_page_text: str, cleaned_page_text: str, chunk_text: str) -> str:
    """청크가 정제 텍스트의 어디서 왔는지 찾아, 원문 쪽도 비슷한 구간으로 좁힌다.

    마스킹·정제로 길이가 달라지므로 정확한 글자 단위 매핑은 못 한다 — 정제 텍스트 안에서
    청크의 상대 위치(비율)를 구해 원문에도 같은 비율로 적용하는 근사치다. 청크가 정제
    텍스트에 없거나 두 번 이상 나오면(모호함) 페이지 전체를 그대로 쓴다.
    """
    if not cleaned_page_text or cleaned_page_text.count(chunk_text) != 1:
        return raw_page_text
    start = cleaned_page_text.find(chunk_text)
    end = start + len(chunk_text)
    cleaned_len = len(cleaned_page_text)
    raw_len = len(raw_page_text)
    raw_start = int(start / cleaned_len * raw_len)
    raw_end = max(int(end / cleaned_len * raw_len), raw_start + 1)
    narrowed = raw_page_text[raw_start:raw_end]
    return narrowed or raw_page_text


def build_samples(
    client, collection: str
) -> list[tuple[str, int | None, str, str, bool, str | None]]:
    """(doc_id, page, 원문, 청크, failed, failure_kind) 튜플을 문서마다 하나씩 모은다.

    파일 없음·파싱 실패·색인 결과 없음도 실패 행으로 남긴다 — 분모에서 빼면
    가장 심각한 실패(아예 색인이 안 됨)가 통계에서 사라진다.
    """
    from zipsai.errors import EmptyDocumentError, PdfParseError
    from zipsai.indexing.clean import apply_cleaning
    from zipsai.indexing.mask import apply_masking
    from zipsai.indexing.parse import parse_document

    docs = load_manifest(REPRODUCE)
    names = build_files(BUILD)
    samples: list[tuple[str, int | None, str, str, bool, str | None]] = []
    for doc in docs:
        path = names.get(doc.build_name)
        if path is None:
            print(f"실패  {doc.doc_id}  file_missing", file=sys.stderr)
            samples.append((doc.doc_id, None, "", "", True, "file_missing"))
            continue
        try:
            pages = parse_document(path)
            masking = apply_masking(pages)
            cleaning = apply_cleaning(masking.pages)
        except (PdfParseError, EmptyDocumentError) as error:
            print(f"실패  {doc.doc_id}  {type(error).__name__}", file=sys.stderr)
            samples.append((doc.doc_id, None, "", "", True, type(error).__name__))
            continue

        indexed = fetch_indexed_chunk(client, collection, doc.building_id, doc.doc_id)
        if indexed is None:
            print(f"실패  {doc.doc_id}  no_index(T1을 먼저 돌렸는지 확인)", file=sys.stderr)
            samples.append((doc.doc_id, None, "", "", True, "no_index"))
            continue
        page, chunk_text = indexed

        raw_page_text = str(next((p["text"] for p in pages if p["page"] == page), ""))
        cleaned_page_text = str(
            next((p["text"] for p in cleaning.pages if p["page"] == page), "")
        )
        original = narrow_original(raw_page_text, cleaned_page_text, chunk_text)
        samples.append((doc.doc_id, page, original, chunk_text, False, None))
    return samples


def _judge(original: str, chunk: str) -> tuple[bool, str | None, bool]:
    from zipsai.integrations.llm import generate_structured
    from zipsai.settings import get_settings

    judge_model = get_settings().llm_judge_model
    if judge_model is None:
        return False, None, True

    verdict = generate_structured(
        _JUDGE_SYSTEM,
        f"원문:\n{original}\n\n청크:\n{chunk}",
        _Verdict,
        model=judge_model,
    )
    if verdict is None:
        return False, None, True
    reason = verdict.reason if verdict.reason in _REASONS else "other"
    return verdict.passed, reason, False


def score_all(
    samples: list[tuple[str, int | None, str, str, bool, str | None]]
) -> list[K1Row]:
    rows: list[K1Row] = []
    for index, (doc_id, page, original, chunk, pre_failed, failure_kind) in enumerate(
        samples, start=1
    ):
        if pre_failed:
            rows.append(K1Row(doc_id, page, False, None, True, failure_kind))
            print(f"채점 {index}/{len(samples)}  {doc_id}  실패({failure_kind})", flush=True)
            continue
        passed, reason, failed = _judge(original, chunk)
        rows.append(
            K1Row(doc_id, page, passed, reason, failed, "judge_error" if failed else None)
        )
        label = "실패" if failed else ("통과" if passed else f"미달({reason})")
        print(f"채점 {index}/{len(samples)}  {doc_id}  {label}", flush=True)
    return rows


def aggregate(rows: list[K1Row]) -> dict[str, object]:
    n = len(rows)
    passed = sum(1 for row in rows if row.passed and not row.failed)
    reasons: dict[str, int] = {}
    for row in rows:
        if not row.passed and not row.failed and row.reason:
            reasons[row.reason] = reasons.get(row.reason, 0) + 1
    return {
        "n": n,
        "K1": (passed / n) if n else None,
        "passed": passed,
        "reasons": reasons,
        "failed": sum(1 for row in rows if row.failed),
    }


def format_report(metrics: dict[str, object]) -> str:
    lines = [
        "색인 청크 적합성",
        f"K1 {metrics['K1']:.3f}  ({metrics['passed']}/{metrics['n']})"
        if metrics["K1"] is not None
        else "K1 —  (표본 0)",
    ]
    for reason, count in metrics["reasons"].items():
        lines.append(f"  미달 사유 {reason}: {count}건")
    lines.append(f"실패 {metrics['failed']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="색인 청크 적합성(K1) 채점")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--qdrant", default="http://127.0.0.1:6333")
    parser.add_argument("--collection", default="documents_eval")
    args = parser.parse_args()
    docs = load_manifest(REPRODUCE)
    if args.check:
        print(f"확인  문서 {len(docs)}건  모델은 부르지 않았습니다")
        return

    from zipsai.settings import get_settings

    if get_settings().llm_judge_model is None:
        _die("LLM_JUDGE_MODEL이 설정되지 않았습니다. K1은 judge 전용 모델이 필요합니다.")

    from qdrant_client import QdrantClient

    url = qdrant_target(args.qdrant, allow_remote=False)
    collection = collection_name(args.collection)
    client = QdrantClient(url=url)
    if not client.collection_exists(collection_name=collection):
        _die(f"컬렉션 {collection}이 없습니다. T1을 먼저 돌려 색인하세요.")

    print(f"K1 시작  문서 {len(docs)}건", flush=True)
    samples = build_samples(client, collection)
    rows = score_all(samples)
    print(format_report(aggregate(rows)), end="")


if __name__ == "__main__":
    main()
