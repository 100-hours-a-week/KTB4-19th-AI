import sys
from pathlib import Path

import pytest

from zipsai.errors import IntentClassificationError

QUERY = Path(__file__).resolve().parents[3] / "evals" / "query"
sys.path.insert(0, str(QUERY))

import score
import score_t3


def _row(
    item_id: str, route: str | None, *, failed: bool = False
) -> score_t3.RouteRecord:
    return score_t3.RouteRecord(item_id=item_id, route=route, failed=failed)


def _item(item_id: str) -> score.GoldItem:
    return score.GoldItem(
        id=item_id,
        building_code="b001",
        building_id=1,
        question="주차는 몇 면인가요?",
        answerable=True,
        scope="building",
        expected_doc_id="b001-doc",
        kind="body",
    )


def test_c1_is_knowledge_count_over_all_rows() -> None:
    metrics = score_t3.aggregate(
        [
            _row("a", "knowledge"),
            _row("b", "knowledge"),
            _row("c", "knowledge"),
            _row("d", "knowledge"),
            _row("e", "complaint"),
        ]
    )

    assert metrics["n"] == 5
    assert metrics["knowledge"] == 4
    assert metrics["complaint"] == 1
    assert metrics["clarify"] == 0
    assert metrics["failed"] == 0
    assert metrics["C1"] == pytest.approx(4 / 5)


def test_failed_row_stays_in_the_denominator() -> None:
    metrics = score_t3.aggregate(
        [
            _row("a", "knowledge"),
            _row("b", "clarify"),
            _row("c", None, failed=True),
            _row("d", "knowledge", failed=True),
        ]
    )

    assert metrics["n"] == 4
    assert metrics["knowledge"] == 1
    assert metrics["complaint"] == 0
    assert metrics["clarify"] == 1
    assert metrics["failed"] == 2
    assert metrics["C1"] == pytest.approx(1 / 4)


def test_empty_rows_have_no_rate() -> None:
    metrics = score_t3.aggregate([])

    assert metrics["n"] == 0
    assert metrics["C1"] is None


def test_report_shows_rate_and_counts() -> None:
    text = score_t3.format_report(
        score_t3.aggregate(
            [
                _row("a", "knowledge"),
                _row("b", "clarify"),
                _row("c", None, failed=True),
            ]
        )
    )

    assert "0.333" in text
    assert "knowledge 1" in text
    assert "complaint 0" in text
    assert "clarify 1" in text
    assert "실패 1" in text


def test_gold_set_has_235_questions() -> None:
    items, _errors = score.load_gold(
        score.REPRODUCE, score.load_manifest(score.REPRODUCE)
    )

    assert len(items) == 235


def test_request_sends_only_the_question() -> None:
    import run_t3

    item = _item("b001-q08")
    request = run_t3.request_for(item)

    assert request.building_id == 1
    assert request.room_no == "101"
    assert request.resident_id == "eval"
    assert request.turn_id == item.id
    assert request.trace_id == item.id
    assert request.current_route is None
    assert request.complaint_draft is None
    assert request.message.text == item.question
    assert request.message.images == []
    assert request.conversation_history == []


def test_extension_classifier_sends_real_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import run_ext_answer
    import run_t3

    item = run_ext_answer.Item(
        id="f-01",
        kind="followup",
        building_id=1,
        question="그래서 언제야?",
        attack=None,
        quote=None,
        history=(
            ("user", "물탱크 청소는 언제 하나요?"),
            ("assistant", "9월 20일입니다"),
        ),
    )

    captured: dict[str, object] = {}

    def fake_classify_intent(payload: dict[str, object]) -> dict[str, object]:
        captured["request"] = payload["request"]

        class _Route:
            value = "knowledge"

        return {"route": _Route()}

    monkeypatch.setattr(
        "zipsai.orchestration.intent.classify_intent", fake_classify_intent
    )

    run_t3.classify_one_extension(item)

    request = captured["request"]
    assert len(request.conversation_history) == 2
    assert request.conversation_history[0].text == "물탱크 청소는 언제 하나요?"


def test_score_all_keeps_a_failed_call_in_the_rows() -> None:
    import run_t3

    def classify(item: score.GoldItem) -> str:
        raise IntentClassificationError(item.id)

    rows = run_t3.score_all([_item("b001-q08")], classify)
    metrics = score_t3.aggregate(rows)

    assert rows[0].route is None
    assert rows[0].failed is True
    assert metrics["n"] == 1
    assert metrics["knowledge"] == 0
    assert metrics["C1"] == pytest.approx(0)
