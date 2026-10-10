import logging
import sys
from pathlib import Path

COMPLAINT_DIR = Path(__file__).resolve().parents[3] / "evals" / "complaint"
QUERY_DIR = Path(__file__).resolve().parents[3] / "evals" / "query"
sys.path.insert(0, str(COMPLAINT_DIR))
sys.path.insert(0, str(QUERY_DIR))

import run_m1
from score_m import ComplaintRow, aggregate_complaints


def _row(item_id: str = "c1") -> ComplaintRow:
    return ComplaintRow(
        item_id=item_id,
        location_ok=True,
        symptom_ok=True,
        type_ok=True,
        location_invented=False,
        symptom_invented=False,
        issue_type="water_supply",
        failed=False,
    )


def test_m7_one_retry_turn_over_44_total_turns() -> None:
    metrics = aggregate_complaints(
        [_row()], total_turns=44, retry_turns=1, retry_events=1
    )
    assert metrics["M7"] == 1 / 44
    assert metrics["retry_turns"] == 1
    assert metrics["retry_events"] == 1
    assert metrics["total_turns"] == 44


def test_m7_dedupes_two_retry_events_in_one_turn() -> None:
    metrics = aggregate_complaints(
        [_row()], total_turns=1, retry_turns=1, retry_events=2
    )
    assert metrics["M7"] == 1 / 1
    assert metrics["retry_turns"] == 1
    assert metrics["retry_events"] == 2


def test_retry_capture_counts_both_pre_and_post_refactor_messages() -> None:
    capture = run_m1._RetryCapture()
    logger = logging.getLogger("zipsai.complaint.node")
    logger.addHandler(capture)
    try:
        logger.warning("extraction_retry")
        logger.warning("interpretation_retry")
        logger.warning("complaint_turn")
    finally:
        logger.removeHandler(capture)
    assert capture.count == 2
