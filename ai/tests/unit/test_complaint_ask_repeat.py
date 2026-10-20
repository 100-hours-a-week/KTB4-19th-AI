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
_LLM_LOCATION = "어느 공간에서 물이 새는지 알려주실 수 있을까요?"
_COMPLETED = "민원 접수가 완료되었습니다. 담당자가 확인하겠습니다."


def _request(assistant_texts: list[str] | None = None) -> ConverseRequest:
    texts = assistant_texts if assistant_texts is not None else [_LOCATION, _LOCATION]
    history: list[dict[str, object]] = []
    for index, text in enumerate(texts):
        history.append({"message_id": f"u{index}", "role": "user", "text": "보일러"})
        history.append({"message_id": f"a{index}", "role": "assistant", "text": text})
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


def _log(request: ConverseRequest, caplog: pytest.LogCaptureFixture, *, asked: str):
    draft = ComplaintDraft(symptom="물이 샘")
    turn = _ComplaintTurn(
        result=RouteResult(complaint_draft=draft, missing_fields=[asked]),
        reply="질문",
        complaint_state=None,
        asked=asked,
        reply_source="llm",
    )
    progress = _ComplaintProgress(
        interpretation=_TurnInterpretation(switch=ComplaintSwitch.SAME),
        draft=draft,
        missing_fields=[asked],
    )
    evidence = _TurnEvidence(
        interpretation=progress.interpretation,
        image_analysis=None,
    )
    with caplog.at_level(logging.INFO):
        node._log_complaint_turn(request, evidence, progress, turn, 0.0)
    return [record for record in caplog.records if record.name == node.__name__]


def test_llm_location_question_counts_as_the_same_field(
    caplog: pytest.LogCaptureFixture,
) -> None:
    records = _log(_request([_LLM_LOCATION, _LLM_LOCATION]), caplog, asked="location")
    summary = next(record for record in records if record.getMessage() == "complaint_turn")
    warning = next(
        record for record in records if record.getMessage() == "complaint_ask_repeated"
    )

    assert summary.ask_repeat_count == 2
    assert warning.ask_repeat_count == 2
    assert summary.reply_source == "llm"
    formatted = JsonFormatter().format(summary)
    assert _LLM_LOCATION not in formatted
    assert "보일러" not in formatted


def test_completion_notice_does_not_count_as_a_location_ask(
    caplog: pytest.LogCaptureFixture,
) -> None:
    records = _log(
        _request([_LLM_LOCATION, _LLM_LOCATION, _COMPLETED]),
        caplog,
        asked="location",
    )
    summary = next(record for record in records if record.getMessage() == "complaint_turn")

    assert summary.ask_repeat_count == 0
    assert not any(record.getMessage() == "complaint_ask_repeated" for record in records)
    assert _COMPLETED not in JsonFormatter().format(summary)


def test_symptom_pattern_stops_at_a_different_assistant_turn() -> None:
    history = _request(
        [
            "어떤 불편이 있는지 알려주세요",
            _COMPLETED,
            "증상이 어떻게 되나요?",
        ]
    ).conversation_history

    assert node._previous_same_field_asks(history, "symptom") == 1
