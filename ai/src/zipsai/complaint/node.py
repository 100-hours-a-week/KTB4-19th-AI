import json
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from zipsai.complaint.prompts import COMPLAINT_PROMPT, VLM_ANALYSIS_PROMPT
from zipsai.contracts.converse import (
    ComplaintDraft,
    ComplaintState,
    ConverseRequest,
    ImageAnalysis,
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
from zipsai.history import format_history
from zipsai.integrations.llm import generate_text, strip_json_code_fence
from zipsai.integrations.vlm import analyze_images

logger = logging.getLogger(__name__)

_KST = ZoneInfo("Asia/Seoul")
_EXTRACTION_ATTEMPTS = 2

_REQUIRED_FIELDS = ("location", "symptom")
_UNKNOWN_LOCATION = "모름"


def extract_complaint_fields(request: ConverseRequest) -> ComplaintDraft:
    draft, _reply, _missing = _extract_complaint_fields_and_reply(request)
    return draft


def _extract_complaint_fields_and_reply(
    request: ConverseRequest,
) -> tuple[ComplaintDraft, str, set[str]]:
    system_message, user_message = COMPLAINT_PROMPT.format_messages(
        today=datetime.now(_KST).date().isoformat(),
        conversation_history=format_history(request.conversation_history),
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
        llm_missing = {field for field in llm_missing if field in _REQUIRED_FIELDS}

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
    if updates.get("location") == _UNKNOWN_LOCATION and current and current.location:
        del updates["location"]
    return (current or ComplaintDraft()).model_copy(update=updates)


def _append_image_urls(draft: ComplaintDraft, image_urls: list[str]) -> ComplaintDraft:
    return draft.model_copy(
        update={"image_urls": list(dict.fromkeys([*draft.image_urls, *image_urls]))}
    )


_MISSING_FIELD_REPLY = {
    "location": "어디에서 생긴 문제인가요?",
    "symptom": "어떤 불편 증상인지 알려주세요.",
}
_PHOTO_ANALYZED_PREFIX = "사진은 확인했습니다. "
_PHOTO_FAILED_PREFIX = "사진을 받았지만 분석에 실패했어요. "


def _photo_prefix(image_analysis: ImageAnalysis | None) -> str:
    """사진에서 읽은 내용을 노출한다. 입주민이 확인해주면 그 발화로 symptom이 채워지므로,
    초안에 직접 넣을 때 필요한 신뢰도 임계값이 필요 없다."""
    if image_analysis is None:
        return _PHOTO_FAILED_PREFIX
    summary = next(
        (observed.summary for observed in image_analysis.images if observed.summary),
        None,
    )
    if not summary:
        return _PHOTO_ANALYZED_PREFIX
    return f"사진은 확인했습니다 — {summary.rstrip('. ')}. "


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
    missing_fields = _missing_fields(draft)

    if missing_fields:
        complaint_state = ComplaintState.COLLECTING
        asked_now = missing_fields[0]
        if llm_reply and llm_missing == {asked_now}:
            follow_up = llm_reply
            reply_source = "llm"
        else:
            follow_up = _MISSING_FIELD_REPLY[asked_now]
            reply_source = "fixed"
        reply = (
            follow_up if not photo_sent else _photo_prefix(image_analysis) + follow_up
        )
    else:
        if draft.issue_type is None:
            draft = draft.model_copy(update={"issue_type": "other"})
        complaint_state = None
        reply = "민원 정보를 확인했습니다. 접수할 내용을 확인해 주세요."
        reply_source = "complete"


    logger.info(
        "complaint_turn building_id=%s conversation_id=%s trace_id=%s "
        "asked=%s missing=%s draft_changed=%s location_unknown=%s "
        "reply_source=%s text_len=%s photo=%s",
        request.building_id,
        request.conversation_id,
        request.trace_id,
        asked_now if missing_fields else None,
        missing_fields,
        draft != request.complaint_draft,
        draft.location == _UNKNOWN_LOCATION,
        reply_source,
        len((request.message.text or "").strip()),
        "analyzed" if image_analysis else ("failed" if photo_sent else "none"),
    )

    return {
        "complaint_state": complaint_state,
        "reply": reply,
        "result": RouteResult(
            complaint_draft=draft,
            missing_fields=missing_fields,
            image_analysis=image_analysis,
        ),
    }
