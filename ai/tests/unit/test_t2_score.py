import sys
from pathlib import Path

import pytest

QUERY = Path(__file__).resolve().parents[3] / "evals" / "query"
sys.path.insert(0, str(QUERY))

import score
import score_t2


def _item(
    item_id: str,
    *,
    answerable: bool = True,
    scope: str = "building",
) -> score.GoldItem:
    return score.GoldItem(
        id=item_id,
        building_code="b001",
        building_id=1,
        question=item_id,
        answerable=answerable,
        scope=scope,
        expected_doc_id="b001-doc" if answerable else None,
        kind="body" if answerable else None,
    )


def _row(
    item_id: str = "q",
    *,
    answerable: bool = True,
    scope: str = "building",
    outcome: str = "answered",
    answer_chars: int | None = 20,
    citation_ids: tuple[str, ...] = ("b001-doc",),
    retrieved_ids: tuple[str, ...] = ("b001-doc",),
    unsupported: tuple[str, ...] | None = (),
    judge_failed: bool = False,
) -> score_t2.AnswerRecord:
    return score_t2.AnswerRecord(
        item_id=item_id,
        question=item_id,
        answerable=answerable,
        scope=scope,
        outcome=outcome,
        reply="답",
        answer_chars=answer_chars,
        citation_ids=citation_ids,
        retrieved_ids=retrieved_ids,
        unsupported=unsupported,
        judge_failed=judge_failed,
    )


def test_sample_keeps_table_b_and_the_strata() -> None:
    items, _errors = score.load_gold(
        score.REPRODUCE, score.load_manifest(score.REPRODUCE)
    )
    sample = score_t2.select_sample(items)
    ids = {item.id for item in sample}

    assert len(sample) == 100
    assert score_t2.TABLE_B[0] in ids and len(score_t2.TABLE_B) == 20
    assert set(score_t2.TABLE_B) <= ids
    assert sum(item.answerable and item.scope == "building" for item in sample) == 45
    assert sum(item.answerable and item.scope == "zone" for item in sample) == 15
    assert sum(not item.answerable for item in sample) == 40


def test_g2_counts_only_model_declines_among_answerable() -> None:
    metrics = score_t2.aggregate(
        [
            _row("hit"),
            _row(
                "blocked",
                outcome="no_hit",
                answer_chars=None,
                citation_ids=(),
                retrieved_ids=(),
                unsupported=None,
            ),
            _row(
                "declined",
                outcome="declined",
                citation_ids=(),
                retrieved_ids=("b001-doc",),
                unsupported=None,
            ),
            _row(
                "absent",
                answerable=False,
                outcome="no_hit",
                answer_chars=None,
                citation_ids=(),
                retrieved_ids=(),
                unsupported=None,
            ),
        ]
    )

    assert metrics["G2"] == pytest.approx(1 / 3)
    assert metrics["G6"] == 20


def test_g5_requires_the_same_documents_and_g1_sums_claims() -> None:
    metrics = score_t2.aggregate(
        [
            _row("ok", unsupported=("12면",)),
            _row("miss", citation_ids=("other",), unsupported=()),
            _row("blind", judge_failed=True, unsupported=None),
        ]
    )

    assert metrics["G5"] == pytest.approx(2 / 3)
    assert metrics["G1"] == 1
    assert metrics["G1_provisional"] is True


def test_report_lists_items_with_unsupported_claims() -> None:
    rows = [
        _row("custom-1", unsupported=("12면",)),
        _row("custom-2", unsupported=()),
    ]
    text = score_t2.format_report(score_t2.aggregate(rows), rows)

    assert "G1 문항 custom-1  1건" in text
    assert "G1 문항 custom-2" not in text


def test_report_names_the_twenty_rows() -> None:
    rows = [
        _row(
            item_id,
            outcome="no_hit",
            answer_chars=None,
            citation_ids=(),
            retrieved_ids=(),
            unsupported=None,
        )
        for item_id in score_t2.TABLE_B
    ]
    text = score_t2.format_report(score_t2.aggregate(rows), rows)

    assert "합격으로 판정하지 않습니다" in text
    assert "b001-q08" in text
    assert "b006-q02" in text
