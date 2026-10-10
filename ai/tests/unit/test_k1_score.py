import sys
from pathlib import Path

import pytest

import zipsai.settings as settings_module
from zipsai.settings import Settings

QUERY_DIR = Path(__file__).resolve().parents[3] / "evals" / "query"
sys.path.insert(0, str(QUERY_DIR))

import run_k1


def test_judge_skips_structured_call_without_judge_model(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        settings_module,
        "get_settings",
        lambda: Settings("key", None, "answer-model", 30),
    )

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("judge 호출이 없어야 합니다")

    monkeypatch.setattr("zipsai.integrations.llm.generate_structured", fail_if_called)

    assert run_k1._judge("원문", "청크")[2] is True


def test_judge_normalizes_unknown_reason_to_other(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        settings_module,
        "get_settings",
        lambda: Settings("key", None, "answer-model", 30, "judge-model"),
    )
    verdict = run_k1._Verdict(passed=False, reason="뭔가 다른 말")
    monkeypatch.setattr(
        "zipsai.integrations.llm.generate_structured", lambda *a, **k: verdict
    )

    passed, reason, failed = run_k1._judge("원문", "청크")

    assert passed is False
    assert reason == "other"
    assert failed is False


def test_aggregate_counts_pass_rate_and_reasons() -> None:
    metrics = run_k1.aggregate(
        [
            run_k1.K1Row("b001-notice-001", 1, True, None, False),
            run_k1.K1Row("b001-notice-002", 1, False, "truncated", False),
            run_k1.K1Row("b001-notice-003", 1, False, None, True),
        ]
    )

    assert metrics["n"] == 3
    assert metrics["K1"] == pytest.approx(1 / 3)
    assert metrics["passed"] == 1
    assert metrics["reasons"] == {"truncated": 1}
    assert metrics["failed"] == 1


def test_aggregate_counts_pre_judge_failures_in_denominator() -> None:
    metrics = run_k1.aggregate(
        [
            run_k1.K1Row("b001-notice-001", 1, True, None, False),
            run_k1.K1Row("b002-notice-001", None, False, None, True, "file_missing"),
            run_k1.K1Row("b003-notice-001", None, False, None, True, "no_index"),
        ]
    )

    assert metrics["n"] == 3
    assert metrics["failed"] == 2
    assert metrics["K1"] == pytest.approx(1 / 3)


def test_narrow_original_shrinks_to_the_chunks_relative_span() -> None:
    cleaned = "머리말\n실제 내용 문단입니다\n꼬리말"
    raw = "머리말(원문)\n실제 내용 문단입니다(원문)\n꼬리말(원문)"

    narrowed = run_k1.narrow_original(raw, cleaned, "실제 내용 문단입니다")

    assert "실제 내용 문단입니다" in narrowed
    assert narrowed != raw


def test_narrow_original_falls_back_to_whole_page_when_ambiguous() -> None:
    cleaned = "같은 줄 같은 줄"
    raw = "원문 전체"

    narrowed = run_k1.narrow_original(raw, cleaned, "같은 줄")

    assert narrowed == raw


def test_fetch_indexed_chunk_picks_deterministically_among_duplicates() -> None:
    class _Point:
        def __init__(self, payload: dict[str, object]) -> None:
            self.payload = payload

    class _StubClient:
        def scroll(self, **kwargs: object):
            return (
                [
                    _Point({"page": 2, "section": "b", "text": "둘째"}),
                    _Point({"page": 1, "section": "a", "text": "첫째"}),
                ],
                None,
            )

    result = run_k1.fetch_indexed_chunk(_StubClient(), "documents_eval", 1, "doc-1")

    assert result == (1, "첫째")


def test_fetch_indexed_chunk_returns_none_without_points() -> None:
    class _StubClient:
        def scroll(self, **kwargs: object):
            return ([], None)

    result = run_k1.fetch_indexed_chunk(_StubClient(), "documents_eval", 1, "doc-1")

    assert result is None
