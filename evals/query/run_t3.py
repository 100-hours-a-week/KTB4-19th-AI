"""합성문서 235문항 + 확장 47문항이 knowledge로 분류되는 비율(C1, 모집단 282)을 낸다.

classify_intent만 호출한다. 검색과 답변 생성은 하지 않는다.
운영 Qdrant와 컬렉션 documents 는 거절한다. Qdrant와 embedding에는 접속하지 않는다.

확장 47문항(partial·attacks·followups)은 run_ext_answer.py가 생성 품질만
보려고 라우터를 안 거치고 handle_knowledge를 바로 부르는데, 그 문항들이
실제로 knowledge로 분류되는지는 여기서 본다(A-100-00 §4).

실행 (KTB4-19th-AI/ai 에서):

    uv run python ../evals/query/run_t3.py --check
    uv run python ../evals/query/run_t3.py

LLM_API_KEY와 LLM_MODEL이 필요하다. 질문 문장은 설정된 LLM으로만 간다.
"""

from __future__ import annotations

import argparse
import sys

from run_t1 import collection_name, qdrant_target
from score import REPRODUCE, GoldItem, load_gold, load_manifest
from score_t3 import RouteRecord, aggregate, format_report

COLLECTION = "documents_eval"
EXPECTED = 235


def _die(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def request_for(item: GoldItem):
    from zipsai.contracts.converse import ConverseRequest, IncomingMessage

    return ConverseRequest(
        building_id=item.building_id,
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


def classify_one(item: GoldItem) -> str:
    from zipsai.orchestration.intent import classify_intent

    result = classify_intent({"request": request_for(item)})
    return result["route"].value


def score_all(items: list[GoldItem], classify) -> list[RouteRecord]:
    from zipsai.errors import (
        IntentClassificationError,
        LlmRateLimitedError,
        LlmTimeoutError,
        LlmUnavailableError,
        LlmUpstreamError,
    )

    rows: list[RouteRecord] = []
    for index, item in enumerate(items, start=1):
        route: str | None = None
        failed = False
        try:
            route = classify(item)
            if route is None:
                failed = True
        except (
            IntentClassificationError,
            LlmUnavailableError,
            LlmRateLimitedError,
            LlmTimeoutError,
            LlmUpstreamError,
        ) as error:
            failed = True
            route = None
            print(f"실패  {item.id}  {type(error).__name__}: {error}", file=sys.stderr)
        rows.append(RouteRecord(item_id=item.id, route=route, failed=failed))
        label = "실패" if failed else route
        print(f"채점 {index}/{len(items)}  {item.id}  {label}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="합성문서 T3 길 고르기 채점")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--qdrant", default="http://127.0.0.1:6333")
    parser.add_argument("--collection", default=COLLECTION)
    args = parser.parse_args()

    docs = load_manifest(REPRODUCE)
    items, _errors = load_gold(REPRODUCE, docs)
    if len(items) != EXPECTED:
        _die(f"골드셋이 {len(items)}건입니다. {EXPECTED}건이어야 합니다.")

    from run_ext_answer import load_items as load_extension_items

    buildings = {doc.building_code: doc.building_id for doc in docs}
    extension_items = load_extension_items(buildings)
    total = len(items) + len(extension_items)

    if args.check:
        print(f"확인  문항 {total}({len(items)}+{len(extension_items)})  컬렉션은 보지 않았습니다")
        return

    qdrant_target(args.qdrant, allow_remote=False)
    collection_name(args.collection)

    from zipsai.settings import get_settings

    get_settings()
    print(f"분류 시작  문항 {total}  Qdrant와 embedding에는 접속하지 않습니다", flush=True)
    rows = score_all(items, classify_one) + score_all(extension_items, classify_one)
    print(format_report(aggregate(rows)), end="")


if __name__ == "__main__":
    main()
