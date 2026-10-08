"""민원 글 38개를 handle_complaint로 채점한다.

검색하지 않는다. 두 턴 문항은 첫 결과의 초안을 다음 요청에 넘긴다.

실행 (KTB4-19th-AI/ai 에서):

    uv run python ../evals/complaint/run_m1.py --check
    uv run python ../evals/complaint/run_m1.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

COMPLAINT = Path(__file__).resolve().parent
QUERY = COMPLAINT.parent / "query"
sys.path.insert(0, str(COMPLAINT))
sys.path.insert(0, str(QUERY))

from score_ext import load_jsonl
from score_m import ComplaintRow, aggregate_complaints, blank, format_complaint_report, same

REPRODUCE = COMPLAINT.parent / "synthetic" / "2-reproduce"
EXPECTED = 38


def _die(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)


class _RetryCapture(logging.Handler):
    """M7(추출 재시도율) 집계용. aa18cc0(v1)는 extraction_retry, PR#90 리팩토링
    이후는 interpretation_retry로 로그 이름이 바뀌어 둘 다 받는다 — 이 러너가
    v1 재현 워크트리로도 이식되기 때문이다."""

    _MESSAGES = frozenset({"extraction_retry", "interpretation_retry"})

    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        if record.getMessage() in self._MESSAGES:
            self.count += 1


def _request(item_id: str, index: int, text: str, history: list, draft, state):
    from zipsai.contracts.converse import ConverseRequest, HistoryTurn, IncomingMessage

    turns = [
        HistoryTurn(
            message_id=f"{item_id}-h{turn_index}",
            role=role,
            text=turn_text,
            image_urls=[],
        )
        for turn_index, (role, turn_text) in enumerate(history)
    ]
    return ConverseRequest(
        building_id=1,
        room_no="101",
        resident_id="eval",
        conversation_id=item_id,
        turn_id=f"{item_id}-{index}",
        trace_id=item_id,
        current_route="complaint" if index else None,
        current_complaint_state=state if index else None,
        message=IncomingMessage(
            message_id=f"{item_id}-{index}", text=text, image_urls=[]
        ),
        conversation_history=turns,
        complaint_draft=draft,
    )


def _row(item_id: str, expected: dict, outcome: dict | None, *, failed: bool) -> ComplaintRow:
    if failed or outcome is None:
        return ComplaintRow(item_id, False, False, False, False, False, None, True)
    draft = outcome["result"].complaint_draft
    location = None if draft is None else draft.location
    symptom = None if draft is None else draft.symptom
    issue_type = None if draft is None else draft.issue_type
    return ComplaintRow(
        item_id=item_id,
        location_ok=same(location, expected["location"]),
        symptom_ok=same(symptom, expected["symptom"]),
        type_ok=same(issue_type, expected["issue_type"]),
        location_invented=blank(expected["location"]) is None and blank(location) is not None,
        symptom_invented=blank(expected["symptom"]) is None and blank(symptom) is not None,
        issue_type=blank(issue_type),
        failed=False,
    )


def score_all(items: list[dict]) -> tuple[list[ComplaintRow], int, int, int]:
    from zipsai.complaint.node import handle_complaint
    from zipsai.errors import (
        ComplaintExtractionError,
        LlmRateLimitedError,
        LlmTimeoutError,
        LlmUnavailableError,
        LlmUpstreamError,
    )

    caught = (
        ComplaintExtractionError,
        LlmUnavailableError,
        LlmRateLimitedError,
        LlmTimeoutError,
        LlmUpstreamError,
    )
    retry_capture = _RetryCapture()
    node_logger = logging.getLogger("zipsai.complaint.node")
    node_logger.addHandler(retry_capture)
    total_turns = 0
    retry_turns = 0
    try:
        rows: list[ComplaintRow] = []
        for index, item in enumerate(items, start=1):
            history: list[tuple[str, str]] = []
            draft = None
            state = None
            outcome = None
            failed = False
            try:
                for turn_index, turn in enumerate(item["turns"]):
                    total_turns += 1
                    before = retry_capture.count
                    try:
                        outcome = handle_complaint(
                            _request(
                                item["id"], turn_index, turn["text"], history, draft, state
                            )
                        )
                    finally:
                        if retry_capture.count > before:
                            retry_turns += 1
                    draft = outcome["result"].complaint_draft
                    state = outcome["complaint_state"]
                    history.append(("user", turn["text"]))
                    history.append(("assistant", str(outcome["reply"])))
            except caught as error:
                failed = True
                print(f"실패  {item['id']}  {type(error).__name__}: {error}", file=sys.stderr)
            rows.append(_row(item["id"], item["expected"], outcome, failed=failed))
            print(f"채점 {index}/{len(items)}  {item['id']}", flush=True)
    finally:
        node_logger.removeHandler(retry_capture)
    return rows, total_turns, retry_turns, retry_capture.count


def main() -> None:
    parser = argparse.ArgumentParser(description="민원 글 채점")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    items = load_jsonl(REPRODUCE / "complaints" / "complaints.jsonl")
    if len(items) != EXPECTED:
        _die(f"민원 문항이 {len(items)}건입니다. {EXPECTED}건이어야 합니다.")
    if args.check:
        print(f"확인  문항 {len(items)}  컬렉션은 보지 않았습니다")
        return
    print(f"민원 글 시작  문항 {len(items)}", flush=True)
    rows, total_turns, retry_turns, retry_events = score_all(items)
    metrics = aggregate_complaints(
        rows, total_turns=total_turns, retry_turns=retry_turns, retry_events=retry_events
    )
    print(format_complaint_report(metrics), end="")


if __name__ == "__main__":
    main()
