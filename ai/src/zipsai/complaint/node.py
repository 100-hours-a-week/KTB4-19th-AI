import json
import logging
from dataclasses import dataclass, field, replace
from datetime import datetime
from time import perf_counter
from zoneinfo import ZoneInfo

from pydantic import BaseModel

from zipsai.complaint.prompts import (
    COMPLAINT_PROMPT,
    TURN_FINALIZATION_PROMPT,
    VLM_ANALYSIS_PROMPT,
)
from zipsai.contracts.converse import (
    ComplaintDraft,
    ComplaintState,
    ComplaintSwitch,
    ConverseRequest,
    ImageAnalysis,
    IssueType,
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
from zipsai.integrations.llm import generate_structured
from zipsai.integrations.vlm import analyze_images
from zipsai.observability import elapsed_ms, skipped, stage

logger = logging.getLogger(__name__)

_KST = ZoneInfo("Asia/Seoul")
_INTERPRETATION_ATTEMPTS = 2

_REQUIRED_FIELDS = ("location", "symptom")
_UNKNOWN_LOCATION = "모름"


class _InterpretationOutput(BaseModel):
    complaint_switch: ComplaintSwitch
    issue_type: IssueType | None
    location: str | None
    symptom: str | None
    occurred_at: datetime | None


class _TurnFinalization(BaseModel):
    symptom: str | None
    reply: str


@dataclass(frozen=True)
class _TurnInterpretation:
    draft: ComplaintDraft = field(default_factory=ComplaintDraft)
    switch: ComplaintSwitch = ComplaintSwitch.SAME


def _log_interpretation_retry(attempt: int, error: Exception) -> None:
    """재시도로 넘어간 회차를 남긴다. 다음 시도가 성공하면 1차 실패가 어디에도 안 남는다."""
    logger.warning(
        "interpretation_retry",
        extra={
            "stage": "text_interpretation",
            "attempt": attempt,
            "error_type": type(error).__name__,
        },
    )


def _interpret_turn(
    request: ConverseRequest, usage_sink: dict[str, object] | None = None
) -> _TurnInterpretation:
    """현재 발화에서 새로 말한 민원 사실과 전환 판정만 받는다."""
    system_message, user_message = COMPLAINT_PROMPT.format_messages(
        today=datetime.now(_KST).date().isoformat(),
        conversation_history=format_history(request.conversation_history),
        complaint_draft=request.complaint_draft,
        # 사진만 온 턴은 text가 None이다. 그대로 넘기면 "현재 발화: None"이 렌더돼
        # 모델이 None을 입주민 발화로 읽는다.
        message_text=request.message.text or "",
    )

    last_error: Exception | None = None
    for attempt in range(1, _INTERPRETATION_ATTEMPTS + 1):
        parsed = generate_structured(
            system_prompt=str(system_message.content),
            user_prompt=str(user_message.content),
            response_format=_InterpretationOutput,
            usage_sink=usage_sink,
        )
        if parsed is not None:
            return _TurnInterpretation(
                draft=ComplaintDraft(
                    issue_type=parsed.issue_type,
                    location=parsed.location,
                    symptom=parsed.symptom,
                    occurred_at=parsed.occurred_at,
                ),
                switch=parsed.complaint_switch,
            )
        last_error = ComplaintExtractionError("LLM did not return an interpretation")
        _log_interpretation_retry(attempt, last_error)

    raise ComplaintExtractionError(
        "LLM did not return a valid complaint interpretation after retry"
    ) from last_error


def _merge_structured_fields(
    current: ComplaintDraft | None, interpreted: ComplaintDraft
) -> ComplaintDraft:
    """의미 결합이 필요 없는 구조 필드만 기존 초안에 얹는다."""
    updates = {
        field: getattr(interpreted, field)
        for field in ("issue_type", "location", "occurred_at")
        if getattr(interpreted, field) is not None
    }

    if updates.get("location") == _UNKNOWN_LOCATION and current and current.location:
        del updates["location"]
    return (current or ComplaintDraft()).model_copy(update=updates)


def _append_image_urls(draft: ComplaintDraft, image_urls: list[str]) -> ComplaintDraft:
    """초안의 사진 목록에 이번 턴 사진을 중복 없이 잇는다."""
    return draft.model_copy(
        update={"image_urls": list(dict.fromkeys([*draft.image_urls, *image_urls]))}
    )


def _missing_fields(draft: ComplaintDraft) -> list[str]:
    """접수에 필요한데 아직 비어 있는 항목."""
    return [field for field in _REQUIRED_FIELDS if not getattr(draft, field)]


def _settle_switch(
    interpretation: _TurnInterpretation, current: ComplaintDraft | None
) -> _TurnInterpretation:
    """전환 질문을 만들 재료가 없는 ask를 같은 민원으로 떨어뜨린다.

    accept는 떨어뜨리지 않는다. 입주민이 이미 바꾸겠다고 답한 턴이라, same으로
    돌리면 방금 접어두기로 한 민원이 완성 상태로 카드까지 나간다.
    """
    if interpretation.switch != ComplaintSwitch.ASK:
        return interpretation
    if not (current and current.symptom):
        # 전환할 대상 자체가 없다. 평범한 수집 턴이므로 추출값은 그대로 쓴다.
        return replace(interpretation, switch=ComplaintSwitch.SAME)
    if interpretation.draft.symptom:
        return interpretation
    # "다른 민원"이라면서 증상이 없다. 모순된 판정이라 이 턴의 값을 믿지 않는다.
    # 병합하면 다른 민원의 issue_type이 지금 초안에 얹혀 짜깁기 카드가 된다.
    return _TurnInterpretation(switch=ComplaintSwitch.SAME)


def _switch_question(current_symptom: str, new_symptom: str) -> str:
    """전환 확인 문구. 수락 턴의 추출이 낫표 안에서 새 증상을 되읽는다.

    입주민 발화에 낫표가 섞일 일은 없다. 따옴표를 쓰면 "에러 'E1'이 떠요" 같은
    증상에서 경계가 겹쳐 회수가 조용히 깨진다.
    """
    return f"{current_symptom} 건은 접어두고 「{new_symptom}」으로 전환할까요?"


_MISSING_FIELD_REPLY = {
    "location": "어디에서 생긴 문제인가요?",
    "symptom": "어떤 불편 증상인지 알려주세요.",
}
_PHOTO_ANALYZED_PREFIX = "사진은 확인했습니다. "
_PHOTO_FAILED_PREFIX = "사진을 받았지만 분석에 실패했어요. "


@dataclass(frozen=True)
class _ComplaintTurn:
    result: RouteResult
    reply: str
    complaint_state: ComplaintState | None
    asked: str | None
    reply_source: str


@dataclass(frozen=True)
class _TurnEvidence:
    interpretation: _TurnInterpretation
    image_analysis: ImageAnalysis | None
    image_urls: list[str] = field(default_factory=list)
    message_text: str = ""


@dataclass(frozen=True)
class _ComplaintProgress:
    draft: ComplaintDraft
    missing_fields: list[str]
    previous_symptom: str | None = None
    current_symptom: str | None = None


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


def _run_image_analysis(image_urls: list[str]) -> ImageAnalysis | None:
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
    image_urls: list[str], image_analysis: ImageAnalysis | None, switch: ComplaintSwitch
) -> str:
    """이번 턴 사진이 어떻게 처리됐는지 나타내는 로그 값."""
    if not image_urls:
        return "none"
    if switch == ComplaintSwitch.ASK:
        # 전환을 묻는 턴은 분석도 누적도 하지 않는다. 이 URL은 여기서 사라진다.
        return "dropped"
    return "analyzed" if image_analysis else "failed"


def _follow_up(asked: str, llm_reply: str) -> tuple[str, str]:
    """부족한 항목 하나를 묻는 문구와 그 출처.

    어떤 항목이 부족한지는 이미 _missing_fields(draft)로 직접 계산했으므로, LLM
    문구는 표현만 다듬는 역할이다. 있으면 그대로 신뢰하고, 없으면 고정 문구로 묻는다.
    """
    if llm_reply:
        return llm_reply, "llm"
    return _MISSING_FIELD_REPLY[asked], "fixed"


def _build_switch_confirmation_turn(
    current: ComplaintDraft, extracted: ComplaintDraft
) -> _ComplaintTurn:
    """새 민원으로 바꿀지 입주민에게 묻는 턴."""
    # 돌려주는 초안이 옛 민원이므로 빈 칸도 옛 민원 기준으로 적는다.
    return _ComplaintTurn(
        result=RouteResult(
            complaint_draft=current, missing_fields=_missing_fields(current)
        ),
        reply=_switch_question(current.symptom, extracted.symptom),
        complaint_state=ComplaintState.COLLECTING,
        asked=None,
        reply_source="switch_ask",
    )


def _ask_for_missing(
    draft: ComplaintDraft,
    missing_fields: list[str],
    image_analysis: ImageAnalysis | None,
    image_urls: list[str],
    llm_reply: str,
) -> _ComplaintTurn:
    """부족한 항목 유도하는 턴."""
    asked = missing_fields[0]
    question, reply_source = _follow_up(asked, llm_reply)
    return _ComplaintTurn(
        result=RouteResult(
            complaint_draft=draft,
            missing_fields=missing_fields,
            image_analysis=image_analysis,
        ),
        reply=_photo_prefix(image_analysis) + question if image_urls else question,
        complaint_state=ComplaintState.COLLECTING,
        asked=asked,
        reply_source=reply_source,
    )


def _ready_for_card(
    draft: ComplaintDraft, image_analysis: ImageAnalysis | None
) -> _ComplaintTurn:
    """백엔드가 확인 카드를 만드는 유일한 경로. missing_fields가 비는 곳도 여기뿐이다."""
    # issue_type은 필수 항목이 아니지만, 분류 없는 카드는 관리자가 담당을 나눌 수 없다.
    if draft.issue_type is None:
        draft = draft.model_copy(update={"issue_type": "other"})
    return _ComplaintTurn(
        result=RouteResult(
            complaint_draft=draft, missing_fields=[], image_analysis=image_analysis
        ),
        reply="민원 정보를 확인했습니다. 접수할 내용을 확인해 주세요.",
        complaint_state=None,
        asked=None,
        reply_source="complete",
    )


def _gather_turn_evidence(request: ConverseRequest) -> _TurnEvidence:
    with stage("text_interpretation", logger) as step:
        interpretation = _interpret_turn(request, usage_sink=step)
    interpretation = _settle_switch(interpretation, request.complaint_draft)
    image_urls = request.message.image_urls
    if interpretation.switch == ComplaintSwitch.ASK:
        skipped("image_analysis", logger, skip_reason="switch_pending")
        image_analysis = None
    else:
        image_analysis = _run_image_analysis(image_urls)
    return _TurnEvidence(
        interpretation,
        image_analysis,
        image_urls,
        request.message.text or "",
    )


def _advance_complaint(
    request: ConverseRequest, evidence: _TurnEvidence
) -> _ComplaintProgress:
    interpretation = evidence.interpretation
    if interpretation.switch == ComplaintSwitch.ASK:
        draft = request.complaint_draft
        assert draft is not None
        previous_symptom = draft.symptom
    else:
        previous = (
            None
            if interpretation.switch == ComplaintSwitch.ACCEPT
            else request.complaint_draft
        )
        draft = _append_image_urls(
            _merge_structured_fields(previous, interpretation.draft),
            evidence.image_urls,
        )
        previous_symptom = previous.symptom if previous else None

    current_symptom = interpretation.draft.symptom
    if not previous_symptom or previous_symptom == current_symptom:
        draft = draft.model_copy(
            update={"symptom": current_symptom or previous_symptom}
        )
    return _ComplaintProgress(
        draft=draft,
        missing_fields=_missing_fields(draft),
        previous_symptom=previous_symptom,
        current_symptom=current_symptom,
    )


def _generate_turn_finalization(
    progress: _ComplaintProgress,
    evidence: _TurnEvidence,
    next_action: str,
) -> _TurnFinalization | None:
    messages = TURN_FINALIZATION_PROMPT.format_messages(
        previous_symptom=json.dumps(progress.previous_symptom, ensure_ascii=False),
        current_symptom=json.dumps(progress.current_symptom, ensure_ascii=False),
        current_message=json.dumps(evidence.message_text, ensure_ascii=False),
        next_action=next_action,
    )
    try:
        with stage("turn_finalization", logger) as step:
            return generate_structured(
                system_prompt=str(messages[0].content),
                user_prompt=str(messages[1].content),
                response_format=_TurnFinalization,
                usage_sink=step,
            )
    except (
        LlmRateLimitedError,
        LlmTimeoutError,
        LlmUnavailableError,
        LlmUpstreamError,
    ):
        return None


def _finalize_turn(
    progress: _ComplaintProgress, evidence: _TurnEvidence
) -> _ComplaintTurn:
    interpretation = evidence.interpretation
    if interpretation.switch == ComplaintSwitch.ASK:
        skipped("turn_finalization", logger, skip_reason="switch_confirmation")
        return _build_switch_confirmation_turn(progress.draft, interpretation.draft)

    symptom_merge_needed = bool(
        progress.previous_symptom
        and progress.current_symptom
        and progress.previous_symptom != progress.current_symptom
    )
    asked = progress.missing_fields[0] if progress.missing_fields else None
    if not symptom_merge_needed and asked is None:
        skipped("turn_finalization", logger, skip_reason="fixed_completion")
        return _ready_for_card(progress.draft, evidence.image_analysis)

    next_action = f"ask_{asked}" if asked else "complete"
    finalization = _generate_turn_finalization(progress, evidence, next_action)
    fallback_symptom = progress.current_symptom or progress.previous_symptom
    finalized_symptom = (
        finalization.symptom.strip()
        if symptom_merge_needed and finalization and finalization.symptom
        else ""
    )
    symptom = finalized_symptom or fallback_symptom
    draft = progress.draft.model_copy(update={"symptom": symptom})

    if progress.missing_fields:
        return _ask_for_missing(
            draft,
            progress.missing_fields,
            evidence.image_analysis,
            evidence.image_urls,
            finalization.reply.strip() if finalization else "",
        )
    return _ready_for_card(draft, evidence.image_analysis)


def _log_complaint_turn(
    request: ConverseRequest,
    evidence: _TurnEvidence,
    turn: _ComplaintTurn,
    started_at: float,
) -> None:
    draft = turn.result.complaint_draft
    switch = evidence.interpretation.switch
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
                evidence.image_urls, turn.result.image_analysis, switch
            ),
        },
    )


def handle_complaint(request: ConverseRequest) -> dict[str, object]:
    started_at = perf_counter()
    evidence = _gather_turn_evidence(request)
    progress = _advance_complaint(request, evidence)
    turn = _finalize_turn(progress, evidence)
    _log_complaint_turn(request, evidence, turn, started_at)
    return {
        "complaint_state": turn.complaint_state,
        "reply": turn.reply,
        "result": turn.result,
    }
