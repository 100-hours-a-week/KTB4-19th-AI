"""Measure handle_complaint wall-clock latency for text+photo turns.

Compares the current code (parallel text+VLM when no existing symptom can be
switched away from) against whatever is checked out when this script runs —
run it once on this branch and once on HEAD before Task 3 to get a before/after
pair. Run from the repository root:

PYTHONPATH=.:ai/src ai/.venv/bin/python ai/scripts/bench_task3_parallel_latency.py

This is an offline probe, not a replacement for the production handler or the
evals/complaint accuracy harness (which has zero text+photo gold cases today).
"""

import base64
import mimetypes
import statistics
import time
from pathlib import Path
from unittest.mock import patch

from zipsai.complaint import node
from zipsai.contracts.converse import ComplaintDraft, ConverseRequest

ROOT = Path(__file__).resolve().parents[2]
PHOTOS = ROOT / "evals" / "complaint" / "photos"
REPS = 3


def _data_url(image_path: Path) -> str:
    mime = mimetypes.guess_type(image_path)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(image_path.read_bytes()).decode('ascii')}"


def _request(text: str, photo: str, draft: ComplaintDraft | None) -> ConverseRequest:
    # image_urls의 확장자 검증기는 http(s) URL을 가정해 data: URI를 거부한다.
    # 계약 통과용 더미 URL을 넣고, 실제 분석 대상은 _timed_run이 analyze_images를
    # 감싸 로컬 파일의 data URL로 바꿔치기한다.
    return ConverseRequest.model_validate(
        {
            "building_id": 1,
            "room_no": "301",
            "resident_id": "bench",
            "conversation_id": "bench-conv",
            "turn_id": "bench-turn",
            "trace_id": "bench-trace",
            "current_route": "complaint",
            "current_complaint_state": "collecting",
            "message": {
                "message_id": "bench-msg",
                "text": text,
                "image_urls": [f"https://example.invalid/{photo}"],
            },
            "conversation_history": [],
            "complaint_draft": draft.model_dump(mode="json") if draft else None,
        }
    )


CASES: list[tuple[str, ConverseRequest, str]] = [
    (
        "new_complaint (no existing symptom -> eligible for parallel)",
        _request("세탁기에서 LE 오류가 떠요", "washer_error.png", None),
        "washer_error.png",
    ),
    (
        "existing_symptom (switch possible -> stays sequential)",
        _request(
            "보일러에서도 E4 오류가 떠요",
            "boiler_error.png",
            ComplaintDraft(symptom="세탁기가 작동하지 않음"),
        ),
        "boiler_error.png",
    ),
    (
        "photo_only (no text -> text LLM must not be called)",
        _request("", "ac_error.png", None),
        "ac_error.png",
    ),
]


def _timed_run(request: ConverseRequest, photo: str) -> tuple[float, int]:
    """소요 시간과 텍스트 해석(_interpret_turn) 호출 횟수를 함께 잰다.

    턴 완성 단계(_generate_turn_finalization)도 generate_structured를 쓰므로,
    "사진만 있으면 텍스트 LLM 호출이 0회"를 확인하려면 해석 호출만 따로 세야 한다.
    """
    interpret_calls = {"n": 0}
    original_interpret = node._interpret_turn
    original_analyze = node.analyze_images
    real_url = _data_url(PHOTOS / photo)

    def counting_interpret(*args, **kwargs):
        interpret_calls["n"] += 1
        return original_interpret(*args, **kwargs)

    def redirecting_analyze(_urls, prompt):
        return original_analyze([real_url], prompt)

    with (
        patch.object(node, "_interpret_turn", counting_interpret),
        patch.object(node, "analyze_images", redirecting_analyze),
    ):
        started = time.perf_counter()
        node.handle_complaint(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
    return elapsed_ms, interpret_calls["n"]


def main() -> None:
    for label, request, photo in CASES:
        durations: list[float] = []
        call_counts: list[int] = []
        for _ in range(REPS):
            elapsed_ms, calls = _timed_run(request, photo)
            durations.append(elapsed_ms)
            call_counts.append(calls)
        median = statistics.median(durations)
        p95 = (
            statistics.quantiles(durations, n=20)[-1]
            if len(durations) > 1
            else durations[0]
        )
        print(f"{label}")
        print(
            f"  durations_ms={[round(d) for d in durations]} median={median:.0f} p95={p95:.0f}"
        )
        print(f"  text_llm_calls_per_run={call_counts}")
        print()


if __name__ == "__main__":
    main()
