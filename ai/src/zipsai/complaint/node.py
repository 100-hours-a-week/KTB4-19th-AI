import json
from datetime import datetime
from zoneinfo import ZoneInfo

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

_KST = ZoneInfo("Asia/Seoul")
# 일시적인 JSON/스키마 오류만 한 번 더 시도한다. rate limit·timeout 등 LLM 레벨 오류는
# generate_text가 별도 예외로 던지므로 여기서 재시도하지 않고 API 계층까지 그대로 올려보낸다.
_EXTRACTION_ATTEMPTS = 2


def extract_complaint_fields(request: ConverseRequest) -> ComplaintDraft:
    draft, _reply, _missing = _extract_complaint_fields_and_reply(request)
    return draft


def _extract_complaint_fields_and_reply(
    request: ConverseRequest,
) -> tuple[ComplaintDraft, str, set[str]]:
    system_message, user_message = COMPLAINT_PROMPT.format_messages(
        today=datetime.now(_KST).date().isoformat(),
        conversation_history=request.conversation_history,
        complaint_draft=request.complaint_draft,
        message_text=request.message.text,
    )

    last_error: Exception | None = None
    for _attempt in range(_EXTRACTION_ATTEMPTS):
        raw = generate_text(system_message.content, user_message.content)
        try:
            data = json.loads(strip_json_code_fence(raw))
        except json.JSONDecodeError as error:
            last_error = error
            continue

        reply = data.pop("reply", "")
        if not isinstance(reply, str):
            reply = ""

        llm_missing = data.pop("missing", [])
        if not isinstance(llm_missing, list):
            llm_missing = []
        llm_missing = {
            field for field in llm_missing if field in ("location", "symptom")
        }

        try:
            draft = ComplaintDraft(**data)
        except ValidationError as error:
            last_error = error
            continue

        return draft, reply.strip(), llm_missing

    raise ComplaintExtractionError(
        "LLM returned an invalid complaint draft after retry"
    ) from last_error


def _merge_complaint_draft(
    current: ComplaintDraft | None, extracted: ComplaintDraft
) -> ComplaintDraft:
    updates = {
        field: getattr(extracted, field)
        for field in ("issue_type", "location", "symptom", "occurred_at")
        if getattr(extracted, field) is not None
    }
    return (current or ComplaintDraft()).model_copy(update=updates)


def _append_image_urls(draft: ComplaintDraft, image_urls: list[str]) -> ComplaintDraft:
    return draft.model_copy(
        update={"image_urls": list(dict.fromkeys([*draft.image_urls, *image_urls]))}
    )


# LLM의 reply를 못 받았거나, LLM이 생각하는 missing이 실제 missing_fields와 다를 때 쓰는 fallback.
_MISSING_FIELDS_REPLY = {
    ("location", "symptom"): "민원 접수를 위해 발생 위치와 불편 증상을 알려주세요.",
    ("location",): "정확한 발생 위치를 알려주세요.",
    ("symptom",): "어떤 불편 증상인지 알려주세요.",
}
_PHOTO_ANALYZED_PREFIX = "사진은 확인했습니다. "
_PHOTO_FAILED_PREFIX = "사진을 받았지만 분석에 실패했어요. "


def handle_complaint(request: ConverseRequest) -> dict[str, object]:
    extracted, llm_reply, llm_missing = _extract_complaint_fields_and_reply(request)
    draft = _append_image_urls(
        _merge_complaint_draft(request.complaint_draft, extracted),
        request.message.image_urls,
    )
    photo_sent = bool(request.message.image_urls)
    try:
        image_analysis = (
            analyze_images(request.message.image_urls, VLM_ANALYSIS_PROMPT)
            if photo_sent
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

    if missing_fields:
        complaint_state = ComplaintState.COLLECTING
        if llm_reply and llm_missing == set(missing_fields):
            follow_up = llm_reply
        else:
            follow_up = _MISSING_FIELDS_REPLY[tuple(missing_fields)]
        if not photo_sent:
            reply = follow_up
        elif image_analysis is not None:
            reply = _PHOTO_ANALYZED_PREFIX + follow_up
        else:
            reply = _PHOTO_FAILED_PREFIX + follow_up
    else:
        if draft.issue_type is None:
            draft = draft.model_copy(update={"issue_type": "other"})
        complaint_state = ComplaintState.READY_TO_CONFIRM
        reply = "민원 정보를 확인했습니다. 접수할 내용을 확인해 주세요."

    return {
        "complaint_state": complaint_state,
        "reply": reply,
        "result": RouteResult(
            complaint_draft=draft,
            missing_fields=missing_fields,
            image_analysis=image_analysis,
        ),
    }
