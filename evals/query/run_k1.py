"""색인 원본 51건에서 문서당 첫 청크를 뽑아 원문과 같은 뜻인지 LLM Judge로 본다.

Qdrant·임베딩을 쓰지 않는다 — download→parse→mask→clean→chunk 네 단계를
로컬로 그대로 밟아서 나온 청크를, 같은 쪽(page)의 원문(마스킹·정제 전
parse_document 결과)과 비교한다. LLM_JUDGE_MODEL이 없으면 돌리지 않는다.

실행 (KTB4-19th-AI/ai 에서):

    uv run python ../evals/query/run_k1.py --check
    uv run python ../evals/query/run_k1.py
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field
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
    page: int
    passed: bool
    reason: str | None
    failed: bool


def _die(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def first_chunk_per_doc() -> list[tuple[str, int, str, str]]:
    """각 문서를 로컬로 파싱·마스킹·정제·청킹까지만 돌려 (doc_id, page, 원문, 청크) 1건씩 모은다."""
    from zipsai.errors import EmptyDocumentError, PdfParseError
    from zipsai.indexing.chunk import chunk_pages
    from zipsai.indexing.clean import apply_cleaning
    from zipsai.indexing.mask import apply_masking
    from zipsai.indexing.parse import parse_document

    docs = load_manifest(REPRODUCE)
    names = build_files(BUILD)
    samples: list[tuple[str, int, str, str]] = []
    for doc in docs:
        path = names.get(doc.build_name)
        if path is None:
            print(f"건너뜀  {doc.doc_id}  파일 없음", file=sys.stderr)
            continue
        try:
            pages = parse_document(path)
            masking = apply_masking(pages)
            cleaning = apply_cleaning(masking.pages)
            chunks = chunk_pages(cleaning.pages)
        except (PdfParseError, EmptyDocumentError) as error:
            print(f"건너뜀  {doc.doc_id}  {type(error).__name__}", file=sys.stderr)
            continue
        if not chunks:
            continue
        chunk = chunks[0]
        original = next((p["text"] for p in pages if p["page"] == chunk.page), "")
        samples.append((doc.doc_id, chunk.page, str(original), chunk.text))
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
    reason = verdict.reason if verdict.reason in _REASONS else None
    return verdict.passed, reason, False


def score_all(samples: list[tuple[str, int, str, str]]) -> list[K1Row]:
    rows: list[K1Row] = []
    for index, (doc_id, page, original, chunk) in enumerate(samples, start=1):
        passed, reason, failed = _judge(original, chunk)
        rows.append(K1Row(doc_id, page, passed, reason, failed))
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
    args = parser.parse_args()
    docs = load_manifest(REPRODUCE)
    if args.check:
        print(f"확인  문서 {len(docs)}건  모델은 부르지 않았습니다")
        return

    from zipsai.settings import get_settings

    if get_settings().llm_judge_model is None:
        _die("LLM_JUDGE_MODEL이 설정되지 않았습니다. K1은 judge 전용 모델이 필요합니다.")

    print(f"K1 시작  문서 {len(docs)}건", flush=True)
    samples = first_chunk_per_doc()
    rows = score_all(samples)
    print(format_report(aggregate(rows)), end="")


if __name__ == "__main__":
    main()
