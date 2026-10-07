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
