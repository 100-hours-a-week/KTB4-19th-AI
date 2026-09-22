import json

from pydantic import ValidationError

from zipsai.complaint.prompts import COMPLAINT_PROMPT
from zipsai.contracts.converse import (
    ComplaintDraft,
    ComplaintState,
    ConverseRequest,
    RouteResult,
)
from zipsai.errors import ComplaintExtractionError
from zipsai.integrations.llm import generate_text


def extract_complaint_fields(request: ConverseRequest) -> ComplaintDraft:
    system_message, user_message = COMPLAINT_PROMPT.format_messages(
        conversation_history=request.conversation_history,
        complaint_draft=request.complaint_draft,
        message_text=request.message.text,
    )

    raw = generate_text(system_message.content, user_message.content)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ComplaintExtractionError("LLM returned invalid JSON") from error

    try:
        return ComplaintDraft(**data)
    except ValidationError as error:
        raise ComplaintExtractionError(
            "LLM returned an invalid complaint draft"
        ) from error


def handle_complaint(request: ConverseRequest) -> dict[str, object]:
    draft = request.complaint_draft
    missing_fields = [
        field
        for field in ("location", "symptom")
        if draft is None or not getattr(draft, field)
    ]
    reply = "민원 접수를 위해 발생 위치와 불편 증상을 알려주세요."
    if not missing_fields:
        reply = "민원 정보를 확인했습니다. 접수할 내용을 확인해 주세요."

    return {
        "complaint_state": ComplaintState.COLLECTING,
        "reply": reply,
        "result": RouteResult(missing_fields=missing_fields),
    }
