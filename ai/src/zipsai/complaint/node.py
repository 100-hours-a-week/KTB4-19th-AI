import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from time import perf_counter
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


@contextmanager
def _stage(request: ConverseRequest, name: str) -> Iterator[None]:
    started_at = perf_counter()
    try:
        yield
    except Exception as error:
        duration_ms = int((perf_counter() - started_at) * 1000)
        detail = (
            str(error)
            if isinstance(
                error,
                (
                    ComplaintExtractionError,
                    ImageAnalysisError,
                    LlmRateLimitedError,
                    LlmTimeoutError,
                    LlmUnavailableError,
                    LlmUpstreamError,
                ),
            )
            else None
        )
        logger.error(
            "complaint_stage trace_id=%s stage=%s status=failed duration_ms=%d "
            "error=%s cause=%s detail=%s",
            request.trace_id,
            name,
            duration_ms,
            type(error).__name__,
            type(error.__cause__).__name__ if error.__cause__ else None,
            detail,
            extra={
                "trace_id": request.trace_id,
                "building_id": request.building_id,
                "stage": name,
                "duration_ms": duration_ms,
                "outcome": "fail",
                "error_type": type(error).__name__,
                "error": detail,
            },
        )
        raise
    else:
        duration_ms = int((perf_counter() - started_at) * 1000)
        logger.info(
            "complaint_stage trace_id=%s stage=%s status=ok duration_ms=%d",
            request.trace_id,
            name,
            duration_ms,
            extra={
                "trace_id": request.trace_id,
                "building_id": request.building_id,
                "stage": name,
                "duration_ms": duration_ms,
                "outcome": "ok",
            },
        )


_KST = ZoneInfo("Asia/Seoul")
_EXTRACTION_ATTEMPTS = 2

_REQUIRED_FIELDS = ("location", "symptom")
_UNKNOWN_LOCATION = "모름"


def extract_complaint_fields(request: ConverseRequest) -> ComplaintDraft:
    with _stage(request, "text_extraction"):
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
    for attempt in range(1, _EXTRACTION_ATTEMPTS + 1):
        raw = generate_text(system_message.content, user_message.content)
        try:
            data = json.loads(strip_json_code_fence(raw))
        except json.JSONDecodeError as error:
            last_error = error
            logger.warning(
                "complaint_retry trace_id=%s stage=text_extraction attempt=%d error=%s",
                request.trace_id,
                attempt,
                type(error).__name__,
                extra={
                    "trace_id": request.trace_id,
                    "building_id": request.building_id,
                    "stage": "text_extraction",
                    "attempt": attempt,
                    "error_type": type(error).__name__,
                },
            )
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
            logger.warning(
                "complaint_retry trace_id=%s stage=text_extraction attempt=%d error=%s",
                request.trace_id,
                attempt,
                type(error).__name__,
                extra={
                    "trace_id": request.trace_id,
                    "building_id": request.building_id,
                    "stage": "text_extraction",
                    "attempt": attempt,
                    "error_type": type(error).__name__,
                },
            )
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


def _missing_fields(draft: ComplaintDraft) -> list[str]:
    return [field for field in _REQUIRED_FIELDS if not getattr(draft, field)]


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
    started_at = perf_counter()
    with _stage(request, "text_extraction"):
        extracted, llm_reply, llm_missing = _extract_complaint_fields_and_reply(request)
    with _stage(request, "draft_merge"):
        draft = _append_image_urls(
            _merge_complaint_draft(request.complaint_draft, extracted),
            request.message.image_urls,
        )
    photo_sent = bool(request.message.image_urls)
    try:
        if photo_sent:
            with _stage(request, "image_analysis"):
                image_analysis = analyze_images(
                    request.message.image_urls, VLM_ANALYSIS_PROMPT
                )
        else:
            image_analysis = None
            logger.info(
                "complaint_stage trace_id=%s stage=image_analysis "
                "status=skipped duration_ms=0",
                request.trace_id,
                extra={
                    "trace_id": request.trace_id,
                    "building_id": request.building_id,
                    "stage": "image_analysis",
                    "duration_ms": 0,
                    "outcome": "skipped",
                },
            )
    except (
        ImageAnalysisError,
        LlmRateLimitedError,
        LlmTimeoutError,
        LlmUnavailableError,
        LlmUpstreamError,
    ):
        image_analysis = None
    with _stage(request, "required_fields"):
        missing_fields = _missing_fields(draft)

    with _stage(request, "reply"):
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
                follow_up
                if not photo_sent
                else _photo_prefix(image_analysis) + follow_up
            )
        else:
            if draft.issue_type is None:
                draft = draft.model_copy(update={"issue_type": "other"})
            complaint_state = None
            reply = "민원 정보를 확인했습니다. 접수할 내용을 확인해 주세요."
            reply_source = "complete"

    total_ms = int((perf_counter() - started_at) * 1000)
    photo = "analyzed" if image_analysis else ("failed" if photo_sent else "none")
    logger.info(
        "complaint_turn building_id=%s conversation_id=%s trace_id=%s "
        "total_ms=%d asked=%s missing=%s draft_changed=%s location_unknown=%s "
        "reply_source=%s text_len=%s photo=%s",
        request.building_id,
        request.conversation_id,
        request.trace_id,
        total_ms,
        asked_now if missing_fields else None,
        missing_fields,
        draft != request.complaint_draft,
        draft.location == _UNKNOWN_LOCATION,
        reply_source,
        len((request.message.text or "").strip()),
        photo,
        extra={
            "trace_id": request.trace_id,
            "building_id": request.building_id,
            "conversation_id": request.conversation_id,
            "total_ms": total_ms,
            "outcome": "ok",
            "photo": photo,
        },
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
