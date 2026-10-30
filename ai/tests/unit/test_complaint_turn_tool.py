
import pytest

import zipsai.complaint.tool as tool_module
from zipsai.contracts.converse import (
    ComplaintDraft,
    ComplaintState,
    ConverseRequest,
    ImageAnalysis,
    ImageObservation,
    RouteResult,
)
from zipsai.errors import (
    ComplaintExtractionError,
    LlmRateLimitedError,
    LlmTimeoutError,
    LlmUnavailableError,
    LlmUpstreamError,
)
from zipsai.orchestration.tool_contracts import (
    ComplaintTurnArgs,
    ComplaintTurnResult,
    ToolFailure,
)


def _request() -> ConverseRequest:
    return ConverseRequest.model_validate(
        {
            "building_id": 101,
            "room_no": "101",
            "resident_id": "resident-1",
            "conversation_id": "conversation-1",
            "turn_id": "turn-1",
            "trace_id": "trace-1",
            "current_route": "complaint",
            "current_complaint_state": "collecting",
            "message": {"message_id": "message-1", "text": "세탁기 아래에서 물이 새요"},
            "conversation_history": [],
            "complaint_draft": {"issue_type": "leak"},
        }
    )


def test_process_complaint_turn_reuses_one_existing_turn_with_server_context(monkeypatch):
    calls = []
    image_analysis = ImageAnalysis(
        images=[ImageObservation(attachmentId=123, summary="세탁기 아래 물이 고임")]
    )
    expected = {
        "reply": "어디에서 생긴 문제인가요?",
        "complaint_state": ComplaintState.COLLECTING,
        "result": RouteResult(
            complaint_draft=ComplaintDraft(issue_type="leak", symptom="물이 샘"),
            missing_fields=["location"],
        ),
    }

    def handle(request, *, image_analysis=None, contextualized_message=None):
        calls.append((request, image_analysis, contextualized_message))
        return expected

    monkeypatch.setattr(tool_module, "handle_complaint", handle)
    request = _request()

    result = tool_module.process_complaint_turn(
        ComplaintTurnArgs(),
        request,
        image_analysis=image_analysis,
        contextualized_message="세탁기 아래로 물이 새는 민원입니다.",
    )

    assert result == ComplaintTurnResult(**expected)
    assert calls == [
        (request, image_analysis, "세탁기 아래로 물이 새는 민원입니다.")
    ]


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (LlmRateLimitedError("limited"), "rate_limited"),
        (LlmTimeoutError("timed out"), "timeout"),
        (LlmUnavailableError("unavailable"), "unavailable"),
        (LlmUpstreamError("upstream"), "upstream_error"),
        (ComplaintExtractionError("invalid"), "upstream_error"),
    ],
)
def test_process_complaint_turn_returns_structured_failures(monkeypatch, error, code):
    monkeypatch.setattr(
        tool_module,
        "handle_complaint",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )

    result = tool_module.process_complaint_turn(ComplaintTurnArgs(), _request())

    assert result == ToolFailure(code=code, message="민원 내용을 처리할 수 없습니다.")
