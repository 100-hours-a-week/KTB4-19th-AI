import sys
from pathlib import Path

import pytest

import zipsai.settings as settings_module
from zipsai.settings import Settings

QUERY_DIR = Path(__file__).resolve().parents[3] / "evals" / "query"
sys.path.insert(0, str(QUERY_DIR))

import run_t2


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

    assert run_t2._judge("질문", "답변", "근거") == (None, False)
