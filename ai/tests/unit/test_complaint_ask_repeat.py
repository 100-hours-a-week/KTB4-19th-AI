import logging

import pytest

from zipsai.complaint import node
from zipsai.complaint.node import (
    _ComplaintProgress,
    _ComplaintTurn,
    _TurnEvidence,
    _TurnInterpretation,
)
from zipsai.contracts.converse import (
    ComplaintDraft,
    ComplaintSwitch,
    ConverseRequest,
    RouteResult,
)
from zipsai.observability import JsonFormatter

_LOCATION = "어디에서 생긴 문제인가요?"


def _request() -> ConverseRequest:
    history: list[dict[str, object]] = []
    for index in range(2):
        history.append({"message_id": f"u{index}", "role": "user", "text": "보일러"})
        history.append(
            {"message_id": f"a{index}", "role": "assistant", "text": _LOCATION}
        )
    return ConverseRequest.model_validate(
        {
            "building_id": 1,
            "room_no": "301",
            "resident_id": "linda",
            "conversation_id": "conv-001",
            "turn_id": "turn-001",
            "trace_id": "trace-001",
            "current_route": "complaint",
            "current_complaint_state": "collecting",
            "message": {"message_id": "msg-001", "text": "보일러", "images": []},
            "conversation_history": history,
            "complaint_draft": {"symptom": "물이 샘"},
        }
    )


def test_third_location_ask_logs_repeat_without_resident_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    draft = ComplaintDraft(symptom="물이 샘")
    turn = _ComplaintTurn(
        result=RouteResult(complaint_draft=draft, missing_fields=["location"]),
        reply=_LOCATION,
        complaint_state=None,
        asked="location",
        reply_source="fixed",
    )
    progress = _ComplaintProgress(
        interpretation=_TurnInterpretation(switch=ComplaintSwitch.SAME),
        draft=draft,
        missing_fields=["location"],
    )
    evidence = _TurnEvidence(
        interpretation=progress.interpretation,
        image_analysis=None,
    )

    with caplog.at_level(logging.INFO):
        node._log_complaint_turn(_request(), evidence, progress, turn, 0.0)

    records = [record for record in caplog.records if record.name == node.__name__]
    summary = next(record for record in records if record.getMessage() == "complaint_turn")
    warning = next(
        record for record in records if record.getMessage() == "complaint_ask_repeated"
    )

    assert summary.ask_repeat_count == 2
    assert summary.asked_field == "location"
    assert warning.ask_repeat_count == 2
    assert "보일러" not in JsonFormatter().format(summary)
    assert "보일러" not in JsonFormatter().format(warning)
    assert "물이 샘" not in JsonFormatter().format(summary)
