import importlib.util
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from zipsai.contracts.converse import ComplaintDraft
from zipsai.settings import Settings

_SCRIPT = Path(__file__).parents[2] / "scripts" / "test_complaint_extraction_live.py"
_SPEC = importlib.util.spec_from_file_location("test_complaint_extraction_live", _SCRIPT)
live = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(live)


def test_live_cases_cover_varied_complaints_and_have_unique_ids():
    ids = [case[0] for case in live.CASES]

    assert len(ids) >= 20
    assert len(ids) == len(set(ids))
    assert any(case[2] for case in live.CASES)
    assert all(case[1] for case in live.CASES)


def test_live_harness_calls_each_case_and_prints_extracted_fields(monkeypatch, capsys):
    seen = []
    providers = []
    locator_events = []

    class Locator:
        async def refresh(self):
            locator_events.append("refresh")

        async def aclose(self):
            locator_events.append("close")

    def interpret(request):
        seen.append(request)
        return SimpleNamespace(
            fields=ComplaintDraft(
                issue_type="plumbing",
                location="욕실 천장",
                symptom="물이 떨어짐",
                occurred_at=datetime.fromisoformat("2026-11-04T09:30:00+09:00"),
            )
        )

    def settings(provider):
        providers.append(provider)
        locator_events.append("settings")
        return Settings("key", "https://qwen.example/v1", "Qwen/test", 30)

    monkeypatch.delenv("POD_BASE_URL", raising=False)
    monkeypatch.setattr(live, "get_locator", lambda: Locator())
    monkeypatch.setattr(live, "get_settings", settings)
    monkeypatch.setattr(live, "_interpret_turn", interpret)

    live.main()

    output = capsys.readouterr().out
    assert providers == ["pod"]
    assert locator_events == ["refresh", "settings", "close"]
    assert len(seen) == len(live.CASES)
    assert len({request.turn_id for request in seen}) == len(seen)
    for case_id, text, history in live.CASES:
        request = next(request for request in seen if request.turn_id == f"turn-{case_id}")
        assert request.message.text == text
        assert len(request.conversation_history) == len(history)
        assert f"{case_id}: {text}" in output
    for field in ("issue_type", "location", "symptom", "occurred_at"):
        assert f"{field}:" in output
    assert "욕실 천장" in output
    assert "2026-11-04" in output
    assert f"{len(live.CASES)} cases inspected" in output
    assert "PASS" not in output
    assert "expected" not in output


def test_live_harness_requires_qwen(monkeypatch):
    monkeypatch.setenv("POD_BASE_URL", "https://qwen.example/v1")
    monkeypatch.setattr(live, "get_locator", lambda: None)
    monkeypatch.setattr(
        live,
        "get_settings",
        lambda provider: Settings("key", "https://other.example/v1", "Other/test", 30),
    )
    monkeypatch.setattr(
        live, "_interpret_turn", lambda _request: pytest.fail("model must not be called")
    )

    with pytest.raises(SystemExit, match="Qwen is not configured"):
        live.main()
