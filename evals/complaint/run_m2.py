"""사진 120장과 실패 2장을 handle_complaint로 채점한다.

사진은 data URL로 넣는다. 정답 글자는 photos.jsonl에 적힌 것만 쓴다.
운영 주소로 보내지 않는다.

실행 (KTB4-19th-AI/ai 에서):

    uv run python ../evals/complaint/run_m2.py --check
    uv run python ../evals/complaint/run_m2.py
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
from pathlib import Path

from pydantic import BaseModel

# 운영 기본값은 꺼짐이다. 이 러너만 data URL 사진을 쓰므로 여기서 켠다.
os.environ.setdefault("ALLOW_DATA_URL_IMAGES", "true")

COMPLAINT = Path(__file__).resolve().parent
QUERY = COMPLAINT.parent / "query"
sys.path.insert(0, str(COMPLAINT))
sys.path.insert(0, str(QUERY))

from score_ext import load_jsonl
from score_m import (
    FailRow,
    PhotoRow,
    aggregate_photos,
    format_photo_report,
    ocr_same,
)

REPRODUCE = COMPLAINT.parent / "synthetic" / "2-reproduce"
PHOTOS = REPRODUCE / "complaints" / "photos.jsonl"
FAILURES = REPRODUCE / "complaints" / "failures"
DEFAULT_DIR = REPRODUCE / "complaints" / "photos"
EXPECTED = 120
FAIL_PREFIX = "사진을 받았지만 분석에 실패했어요."


def _die(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _request(item_id: str, url: str):
    from zipsai.contracts.converse import (
        ConverseRequest,
        ImageAttachment,
        IncomingMessage,
    )

    return ConverseRequest(
        building_id=1,
        room_no="101",
        resident_id="eval",
        conversation_id=item_id,
        turn_id=item_id,
        trace_id=item_id,
        current_route=None,
        current_complaint_state=None,
        message=IncomingMessage(
            message_id=item_id,
            text="이 사진 봐 주세요",
            images=[ImageAttachment(attachment_id=1, url=url)],
        ),
        conversation_history=[],
        complaint_draft=None,
    )


def _observe(outcome: dict) -> tuple[object, object]:
    analysis = outcome["result"].image_analysis
    if analysis is None or not analysis.images:
        return None, None
    image = analysis.images[0]
    return image.ocr_text, image.summary


class _SummaryVerdict(BaseModel):
    passed: bool


_M5_SYSTEM = (
    "사진 속 상황 설명과 VLM 요약을 비교한다. "
    "포함돼야 할 사실이 다른 단어로 쓰였어도 같은 뜻이면 포함된 것으로 본다. "
    "포함되면 안 될 사실이 의미상 암시만 돼도 위반으로 본다."
)


def _judge_m5(
    must_include: list[str], must_not: list[str], summary: str
) -> tuple[bool, bool]:
    """통과 여부, 판정 실패 여부. 단어가 있는지로 되돌리지 않는다."""
    from zipsai.integrations.llm import generate_structured
    from zipsai.settings import get_settings

    judge_model = get_settings().llm_judge_model
    if judge_model is None:
        return False, True
    verdict = generate_structured(
        _M5_SYSTEM,
        "들어가야 할 말: "
        + ", ".join(must_include)
        + "\n들어가면 안 되는 말: "
        + ", ".join(must_not)
        + f"\n요약: {summary}",
        _SummaryVerdict,
        model=judge_model,
    )
    if verdict is None:
        return False, True
    return verdict.passed, False


def score_all(items: list[dict], photo_dir: Path) -> tuple[list[PhotoRow], list[FailRow]]:
    from zipsai.complaint.node import handle_complaint
    from zipsai.errors import (
        ComplaintExtractionError,
        ImageAnalysisError,
        LlmRateLimitedError,
        LlmTimeoutError,
        LlmUnavailableError,
        LlmUpstreamError,
    )
    from zipsai.tracing import record_judge_score, trace_eval_item

    caught = (
        ComplaintExtractionError,
        ImageAnalysisError,
        LlmUnavailableError,
        LlmRateLimitedError,
        LlmTimeoutError,
        LlmUpstreamError,
    )
    photos: list[PhotoRow] = []
    for index, item in enumerate(items, start=1):
        path = photo_dir / item["file"]
        failed = False
        ocr_ok = False
        fitted = False
        try:
            request = _request(item["id"], data_url(path))
            from zipsai.orchestration.intent import classify_intent

            classified = classify_intent({"request": request})
            outcome = handle_complaint(
                request, image_analysis=classified.get("image_analysis")
            )
            ocr, summary = _observe(outcome)
            if ocr is None and summary is None and outcome["result"].image_analysis is None:
                failed = True
            else:
                ocr_ok = ocr_same(ocr, item["ocr_text"])
                if summary is None:
                    fitted = False
                else:
                    with trace_eval_item("M5", item["id"]):
                        fitted, judge_failed = _judge_m5(
                            item["must_include"],
                            item["must_not_include"],
                            str(summary),
                        )
                        if not judge_failed:
                            record_judge_score("M5", fitted, item_id=item["id"])
                    if judge_failed:
                        failed = True
        except caught as error:
            failed = True
            print(f"실패  {item['id']}  {type(error).__name__}: {error}", file=sys.stderr)
        photos.append(PhotoRow(item["id"], ocr_ok, fitted, failed))
        print(f"채점 {index}/{len(items)}  {item['id']}", flush=True)

    fails: list[FailRow] = []
    for path in sorted(FAILURES.glob("*.jpg")):
        failed = False
        notified = False
        try:
            outcome = handle_complaint(_request(path.stem, data_url(path)))
            notified = str(outcome["reply"]).startswith(FAIL_PREFIX)
        except caught as error:
            failed = True
            print(f"실패  {path.name}  {type(error).__name__}: {error}", file=sys.stderr)
        fails.append(FailRow(path.stem, notified, failed))
        print(f"채점 실패유도  {path.name}", flush=True)
    return photos, fails


def main() -> None:
    parser = argparse.ArgumentParser(description="민원 사진 채점")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--photos-dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    if not PHOTOS.exists():
        _die(f"사진 정답이 없습니다: {PHOTOS}")
    items = load_jsonl(PHOTOS)
    failures = sorted(FAILURES.glob("*.jpg"))
    if len(items) != EXPECTED or len(failures) != 2:
        _die(f"사진 {len(items)}장, 실패 파일 {len(failures)}개입니다. 120장과 2개여야 합니다.")
    if args.check:
        print(f"확인  사진 {len(items)}  실패 {len(failures)}  모델은 부르지 않았습니다")
        return
    print(f"사진 읽기 시작  사진 {len(items)}", flush=True)
    photos, fails = score_all(items, args.photos_dir)
    print(format_photo_report(aggregate_photos(photos, fails)), end="")
    from zipsai.tracing import get_tracing_client

    tracing = get_tracing_client()
    if tracing is not None:
        tracing.flush()


if __name__ == "__main__":
    main()
