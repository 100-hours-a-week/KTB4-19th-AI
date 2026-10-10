import sys
import unicodedata
from pathlib import Path

import pytest

QUERY = Path(__file__).resolve().parents[3] / "evals" / "query"
sys.path.insert(0, str(QUERY))

import run_t1
import score


def _row(
    item_id: str = "q",
    answerable: bool = True,
    scope: str = "building",
    kind: str | None = "body",
    passed: bool = True,
    top_score: float | None = 0.8,
    expected_doc_id: str | None = "b001-doc",
    requested_building_id: str = "1",
    hit_doc_ids: tuple[str, ...] = ("b001-doc",),
    hit_building_ids: tuple[str, ...] = ("1",),
) -> score.Scored:
    return score.Scored(
        item_id=item_id,
        answerable=answerable,
        scope=scope,
        kind=kind,
        passed=passed,
        top_score=top_score,
        expected_doc_id=expected_doc_id,
        requested_building_id=requested_building_id,
        hit_doc_ids=hit_doc_ids,
        hit_building_ids=hit_building_ids,
    )


def test_r1_counts_foreign_chunks_only_on_answerable_building_questions() -> None:
    metrics = score.aggregate(
        [
            _row(item_id="own", hit_building_ids=("2", "1")),
            _row(item_id="zone", scope="zone", hit_building_ids=("9",)),
            _row(
                item_id="none",
                answerable=False,
                passed=False,
                expected_doc_id=None,
                hit_doc_ids=(),
                hit_building_ids=("9",),
            ),
        ]
    )

    assert metrics["R1"] == 1
    assert metrics["building"] == 1


def test_recall_keeps_gate_misses_in_r4_and_drops_them_from_r5() -> None:
    metrics = score.aggregate(
        [
            _row(item_id="hit", hit_doc_ids=("b001-doc", "other")),
            _row(
                item_id="blocked",
                passed=False,
                top_score=0.2,
                hit_doc_ids=(),
                hit_building_ids=(),
            ),
            _row(
                item_id="second",
                expected_doc_id="wanted",
                hit_doc_ids=("other", "wanted"),
            ),
        ]
    )

    assert metrics["R2"] == pytest.approx(2 / 3)
    assert metrics["R4"] == pytest.approx(2 / 3)
    assert metrics["R5"] == pytest.approx(1)
    assert metrics["R6"] == pytest.approx((1 + 0.5) / 2)
    assert metrics["R7"] == 2


def test_r3_is_the_share_of_unanswerable_questions_blocked() -> None:
    metrics = score.aggregate(
        [
            _row(
                item_id="blocked",
                answerable=False,
                passed=False,
                expected_doc_id=None,
                hit_doc_ids=(),
                hit_building_ids=(),
            ),
            _row(
                item_id="leaked",
                answerable=False,
                expected_doc_id=None,
                hit_doc_ids=("x",),
            ),
        ]
    )

    assert metrics["R3"] == pytest.approx(0.5)
    assert metrics["R5"] is None


def test_doc_id_follows_markdown_name_and_scan_file_uses_title() -> None:
    md = "documents/notice-004-소방시설점검.md"

    assert score.doc_id_of("b001", md) == "b001-notice-004-소방시설점검"
    assert (
        score.build_name("b001", md, "소방시설 점검", "PDF", "scan")
        == "b001-소방시설_점검-스캔본.pdf"
    )
    assert score.kind_of(["CHUNK_TABLE_WHOLE"]) == "table"
    assert score.kind_of(["PARSE_IMAGE_ONLY"]) == "photo"


def test_rank_ignores_nfd_nfc_difference() -> None:
    nfd = unicodedata.normalize("NFD", "세탁실")
    metrics = score.aggregate(
        [
            _row(
                expected_doc_id=f"b001-{nfd}",
                hit_doc_ids=(f"b001-{score.nfc('세탁실')}",),
            )
        ]
    )

    assert metrics["R4"] == 1


def test_r1_failure_stops_judgement_of_later_metrics() -> None:
    text = score.format_report(score.aggregate([_row(hit_building_ids=("9",))]))

    assert "판정하지 않습니다" in text
    assert "판정 안 함" in text


def test_gold_files_match_the_sheet_counts() -> None:
    assert score.check_inputs(score.REPRODUCE, score.BUILD) == []
    docs = score.load_manifest(score.REPRODUCE)
    scan = next(doc for doc in docs if doc.doc_id == "b001-notice-004-소방시설점검")

    assert scan.building_id == 1
    assert scan.kind == "scan"
    assert scan.build_name == "b001-소방시설_점검-스캔본.pdf"


def test_refuses_production_qdrant_and_the_live_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EMBEDDING_URL", raising=False)
    with pytest.raises(SystemExit):
        run_t1.qdrant_target("http://qdrant.example:6333", allow_remote=False)
    with pytest.raises(SystemExit):
        run_t1.collection_name("documents")
    with pytest.raises(SystemExit):
        run_t1.embedding_target("http://embedding:8000")
    with pytest.raises(SystemExit):
        run_t1.embedding_target(None)

    assert run_t1.qdrant_target("http://127.0.0.1:6333", allow_remote=False).endswith(
        "6333"
    )
    assert run_t1.collection_name("documents_eval") == "documents_eval"
