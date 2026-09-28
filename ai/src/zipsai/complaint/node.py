import json
import logging
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
from zipsai.observability import elapsed_ms, skipped, stage

logger = logging.getLogger(__name__)

_KST = ZoneInfo("Asia/Seoul")
_EXTRACTION_ATTEMPTS = 2

_REQUIRED_FIELDS = ("location", "symptom")
_UNKNOWN_LOCATION = "모름"


def extract_complaint_fields(request: ConverseRequest) -> ComplaintDraft:
    # 단계 로그는 handle_complaint 한 곳에서만 낸다. 여기서도 내면 같은 stage 이름이
    # 두 번 집계된다.
    draft, _reply, _missing = _extract_complaint_fields_and_reply(request)
    return draft


def _log_retry(attempt: int, error: Exception) -> None:
    """재시도로 넘어간 회차를 남긴다. 다음 시도가 성공하면 1차 실패가 어디에도 안 남는다."""
    logger.warning(
        "extraction_retry",
        extra={
            "stage": "text_extraction",
            "attempt": attempt,
            "error_type": type(error).__name__,
        },
    )


def _extract_complaint_fields_and_reply(
    request: ConverseRequest,
) -> tuple[ComplaintDraft, str, set[str]]:
    system_message, user_message = COMPLAINT_PROMPT.format_messages(
        today=datetime.now(_KST).date().isoformat(),
        conversation_history=format_history(request.conversation_history),
        complaint_draft=request.complaint_draft,
        # 사진만 온 턴은 text가 None이다. 그대로 넘기면 "현재 발화: None"이 렌더돼
        # 모델이 None을 입주민 발화로 읽는다.
        message_text=request.message.text or "",
    )

    last_error: Exception | None = None
    for attempt in range(1, _EXTRACTION_ATTEMPTS + 1):
        raw = generate_text(system_message.content, user_message.content)
        try:
            data = json.loads(strip_json_code_fence(raw))
        except json.JSONDecodeError as error:
            last_error = error
            _log_retry(attempt, error)
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
            _log_retry(attempt, error)
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
    # "모름"은 빈 칸을 채우는 값이지 이미 확인된 값을 대체하는 값이 아니다. 증상을 물은
    # 턴에 "모르겠어요"가 오면 추출이 그걸 위치에 대한 모름으로 보고 "모름"을 넣기도 한다
    # (실제로 관측됨). 그대로 두면 확인된 위치가 지워져 관리자가 쓸 수 없는 값이 된다.
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
    with stage("text_extraction", logger):
        extracted, llm_reply, llm_missing = _extract_complaint_fields_and_reply(request)
    draft = _append_image_urls(
        _merge_complaint_draft(request.complaint_draft, extracted),
        request.message.image_urls,
    )
    photo_sent = bool(request.message.image_urls)
    image_analysis = None
    if photo_sent:
        try:
            with stage("image_analysis", logger):
                image_analysis = analyze_images(
                    request.message.image_urls, VLM_ANALYSIS_PROMPT
                )
        except (
            ImageAnalysisError,
            LlmRateLimitedError,
            LlmTimeoutError,
            LlmUnavailableError,
            LlmUpstreamError,
        ):
            # 사진 분석이 실패해도 민원 수집은 이어간다. 답변에 실패를 알리고 텍스트로 받는다.
            image_analysis = None
    else:
        # 사진이 없어 안 돈 것과 VLM이 매달려 아직 안 찍힌 것을 구분하려면 줄이 있어야 한다.
        skipped("image_analysis", logger)

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
        "complaint_turn",
        extra={
            "conversation_id": request.conversation_id,
            "total_ms": elapsed_ms(started_at),
            "asked": asked_now if missing_fields else None,
            "missing": missing_fields,
            "draft_changed": draft != request.complaint_draft,
            "location_unknown": draft.location == _UNKNOWN_LOCATION,
            "reply_source": reply_source,
            "text_len": len((request.message.text or "").strip()),
            "photo": "analyzed"
            if image_analysis
            else ("failed" if photo_sent else "none"),
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
