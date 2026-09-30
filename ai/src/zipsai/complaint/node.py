import json
import logging
from dataclasses import dataclass, field, replace
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

# 수집 중인 민원과 이번 턴이 같은 건인지에 대한 판정. 자세한 규칙은 추출 프롬프트에 있다.
_SWITCH_SAME = "same"
_SWITCH_ASK = "ask"
_SWITCH_ACCEPT = "accept"
_SWITCH_VALUES = (_SWITCH_SAME, _SWITCH_ASK, _SWITCH_ACCEPT)


@dataclass(frozen=True)
class _Extraction:
    draft: ComplaintDraft = field(default_factory=ComplaintDraft)
    reply: str = ""
    missing: frozenset[str] = frozenset()
    switch: str = _SWITCH_SAME


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


def _extract_complaint(request: ConverseRequest) -> _Extraction:
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
        missing = data.pop("missing", [])
        switch = data.pop("complaint_switch", _SWITCH_SAME)

        try:
            draft = ComplaintDraft(**data)
        except ValidationError as error:
            last_error = error
            _log_retry(attempt, error)
            continue

        # 잘못된 값은 예외로 올리지 않고 기본값으로 강등한다. 한 필드가 어긋났다고
        # 나머지가 멀쩡한 추출을 버리면 수집이 한 턴 헛돈다.
        return _Extraction(
            draft=draft,
            reply=reply.strip() if isinstance(reply, str) else "",
            missing=frozenset(missing) & frozenset(_REQUIRED_FIELDS)
            if isinstance(missing, list)
            else frozenset(),
            switch=switch if switch in _SWITCH_VALUES else _SWITCH_SAME,
        )

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


def _resolve_switch(
    switch: str, current: ComplaintDraft | None, extracted: ComplaintDraft
) -> str:
    if switch == _SWITCH_ASK and not (
        current and current.symptom and extracted.symptom
    ):
        return _SWITCH_SAME
    # 수락인데 증상을 못 읽었다. 그대로 비우면 빈 초안으로 처음부터 다시 묻게 된다.
    if switch == _SWITCH_ACCEPT and not extracted.symptom:
        return _SWITCH_SAME
    return switch


def _switch_question(current_symptom: str, new_symptom: str) -> str:
    return f"{current_symptom} 건은 접어두고 '{new_symptom}'으로 전환할까요?"


_MISSING_FIELD_REPLY = {
    "location": "어디에서 생긴 문제인가요?",
    "symptom": "어떤 불편 증상인지 알려주세요.",
}
_PHOTO_ANALYZED_PREFIX = "사진은 확인했습니다. "
_PHOTO_FAILED_PREFIX = "사진을 받았지만 분석에 실패했어요. "


@dataclass
class _ComplaintTurn:
    result: RouteResult
    reply: str
    complaint_state: ComplaintState | None
    asked: str | None
    reply_source: str


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


def _analyze_photos(image_urls: list[str]) -> ImageAnalysis | None:
    """사진 분석 한 단계. 돌았든 안 돌았든 실패했든 줄을 하나 남긴다.

    사진이 없어 안 돈 것과 VLM이 매달려 아직 안 찍힌 것을 로그로 구분해야 한다.
    분석이 실패해도 예외를 올리지 않는다. 민원 수집은 텍스트만으로도 이어가고,
    실패는 답변 문구로 입주민에게 알린다.
    """
    if not image_urls:
        skipped("image_analysis", logger, skip_reason="no_image")
        return None
    try:
        with stage("image_analysis", logger):
            return analyze_images(image_urls, VLM_ANALYSIS_PROMPT)
    except (
        ImageAnalysisError,
        LlmRateLimitedError,
        LlmTimeoutError,
        LlmUnavailableError,
        LlmUpstreamError,
    ):
        return None


def _photo_label(
    image_urls: list[str], image_analysis: ImageAnalysis | None, switch: str
) -> str:
    if not image_urls:
        return "none"
    if switch == _SWITCH_ASK:
        return "pending"
    return "analyzed" if image_analysis else "failed"


def _follow_up(
    asked: str, llm_reply: str, llm_missing: frozenset[str]
) -> tuple[str, str]:
    """부족한 항목 하나를 묻는 문구와 그 출처.

    LLM이 쓴 질문은 그것이 우리가 물으려는 항목과 정확히 같을 때만 쓴다. 어긋나면
    두 항목을 묻거나 이미 받은 값을 다시 묻는 문구가 나간다.
    """
    if llm_reply and llm_missing == {asked}:
        return llm_reply, "llm"
    return _MISSING_FIELD_REPLY[asked], "fixed"


def _confirm_switch(
    current: ComplaintDraft, extracted: ComplaintDraft
) -> _ComplaintTurn:
    skipped("image_analysis", logger, skip_reason="switch_pending")
    # 백엔드는 missing_fields가 비면 카드를 만들고 대화를 끝낸다. 옛 초안이 이미
    # 완성돼 있어도 그 값을 그대로 쓸 수 없다 — 전환을 묻는 중에 카드가 뜬다.
    # 판정 대상은 새 민원이고, 수락 전에는 그 민원의 어떤 항목도 확정되지 않았다.
    return _ComplaintTurn(
        result=RouteResult(
            complaint_draft=current, missing_fields=list(_REQUIRED_FIELDS)
        ),
        reply=_switch_question(current.symptom, extracted.symptom),
        complaint_state=ComplaintState.COLLECTING,
        asked=None,
        reply_source="switch_ask",
    )


def _collect_complaint(
    request: ConverseRequest, extraction: _Extraction
) -> _ComplaintTurn:
    # 수락이면 옛 초안을 버린다. base가 None이면 텍스트 필드와 image_urls가 함께
    # 비므로, 이전 민원의 사진이 새 카드에 딸려가지 않는다.
    base = None if extraction.switch == _SWITCH_ACCEPT else request.complaint_draft
    image_urls = request.message.image_urls
    draft = _append_image_urls(
        _merge_complaint_draft(base, extraction.draft), image_urls
    )
    image_analysis = _analyze_photos(image_urls)
    missing_fields = _missing_fields(draft)

    if missing_fields:
        asked = missing_fields[0]
        question, reply_source = _follow_up(asked, extraction.reply, extraction.missing)
        reply = _photo_prefix(image_analysis) + question if image_urls else question
        complaint_state = ComplaintState.COLLECTING
    else:
        if draft.issue_type is None:
            draft = draft.model_copy(update={"issue_type": "other"})
        asked = None
        reply = "민원 정보를 확인했습니다. 접수할 내용을 확인해 주세요."
        reply_source = "complete"
        complaint_state = None

    return _ComplaintTurn(
        result=RouteResult(
            complaint_draft=draft,
            missing_fields=missing_fields,
            image_analysis=image_analysis,
        ),
        reply=reply,
        complaint_state=complaint_state,
        asked=asked,
        reply_source=reply_source,
    )


def handle_complaint(request: ConverseRequest) -> dict[str, object]:
    started_at = perf_counter()
    with stage("text_extraction", logger):
        extraction = _extract_complaint(request)

    switch = _resolve_switch(
        extraction.switch, request.complaint_draft, extraction.draft
    )
    extraction = replace(extraction, switch=switch)

    if switch == _SWITCH_ASK:
        turn = _confirm_switch(request.complaint_draft, extraction.draft)
    else:
        turn = _collect_complaint(request, extraction)

    draft = turn.result.complaint_draft
    logger.info(
        "complaint_turn",
        extra={
            "conversation_id": request.conversation_id,
            "total_ms": elapsed_ms(started_at),
            "switch": switch,
            "asked": turn.asked,
            "missing": turn.result.missing_fields,
            "draft_changed": draft != request.complaint_draft,
            "location_unknown": draft.location == _UNKNOWN_LOCATION,
            "reply_source": turn.reply_source,
            "text_len": len((request.message.text or "").strip()),
            "photo": _photo_label(
                request.message.image_urls, turn.result.image_analysis, switch
            ),
        },
    )
    return {
        "complaint_state": turn.complaint_state,
        "reply": turn.reply,
        "result": turn.result,
    }
