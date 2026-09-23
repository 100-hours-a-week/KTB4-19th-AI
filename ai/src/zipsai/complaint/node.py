import json

from pydantic import ValidationError

from zipsai.complaint.prompts import COMPLAINT_PROMPT, VLM_ANALYSIS_PROMPT
from zipsai.contracts.converse import (
    ComplaintDraft,
    ComplaintState,
    ConverseRequest,
    RouteResult,
)
from zipsai.errors import (
    ComplaintExtractionError,
    ImageAnalysisError,
    LlmRateLimitedError,
    LlmTimeoutError,
    LlmUnavailableError,
    LlmUpstreamError,
)
from zipsai.integrations.llm import generate_text, strip_json_code_fence
from zipsai.integrations.vlm import analyze_images


def extract_complaint_fields(request: ConverseRequest) -> ComplaintDraft:
    system_message, user_message = COMPLAINT_PROMPT.format_messages(
        conversation_history=request.conversation_history,
        complaint_draft=request.complaint_draft,
        message_text=request.message.text,
    )

    raw = generate_text(system_message.content, user_message.content)
    try:
        data = json.loads(strip_json_code_fence(raw))
    except json.JSONDecodeError as error:
        raise ComplaintExtractionError("LLM returned invalid JSON") from error

    try:
        return ComplaintDraft(**data)
    except ValidationError as error:
        raise ComplaintExtractionError(
            "LLM returned an invalid complaint draft"
        ) from error


def _merge_complaint_draft(
    current: ComplaintDraft | None, extracted: ComplaintDraft
) -> ComplaintDraft:
    updates = {
        field: getattr(extracted, field)
        for field in ("issue_type", "location", "symptom")
        if getattr(extracted, field) is not None
    }
    return (current or ComplaintDraft()).model_copy(update=updates)


def _append_image_urls(draft: ComplaintDraft, image_urls: list[str]) -> ComplaintDraft:
    return draft.model_copy(
        update={"image_urls": list(dict.fromkeys([*draft.image_urls, *image_urls]))}
    )


def handle_complaint(request: ConverseRequest) -> dict[str, object]:
    draft = _append_image_urls(
        _merge_complaint_draft(
            request.complaint_draft, extract_complaint_fields(request)
        ),
        request.message.image_urls,
    )
    try:
        image_analysis = (
            analyze_images(request.message.image_urls, VLM_ANALYSIS_PROMPT)
            if request.message.image_urls
            else None
        )
    except (
        ImageAnalysisError,
        LlmRateLimitedError,
        LlmTimeoutError,
        LlmUnavailableError,
        LlmUpstreamError,
    ):
        image_analysis = None
    missing_fields = [
        field
        for field in ("location", "symptom")
        if draft is None or not getattr(draft, field)
    ]
    reply = "민원 접수를 위해 발생 위치와 불편 증상을 알려주세요."
    if not missing_fields:
        reply = "민원 정보를 확인했습니다. 접수할 내용을 확인해 주세요."
    elif image_analysis is not None:
        reply = "사진은 확인했습니다. 정확한 접수를 위해 위치와 증상을 간단히 말씀해 주시겠어요?"

    return {
        "complaint_state": ComplaintState.COLLECTING,
        "reply": reply,
        "result": RouteResult(
            complaint_draft=draft,
            missing_fields=missing_fields,
            image_analysis=image_analysis,
        ),
    }
