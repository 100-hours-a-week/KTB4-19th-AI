import json
import logging
import math
from dataclasses import dataclass, field, replace
from datetime import datetime
from time import perf_counter
from zoneinfo import ZoneInfo

from pydantic import BaseModel

from zipsai.complaint.prompts import (
    COMPLAINT_PROMPT,
    TURN_FINALIZATION_PROMPT,
)
from zipsai.contracts.converse import (
    ComplaintDraft,
    CompletedComplaintDraft,
    ComplaintState,
    ComplaintSwitch,
    ConverseRequest,
    ImageAnalysis,
    ImageAttachment,
    ImageObservation,
    IssueType,
    RouteResult,
)
from zipsai.errors import (
    ComplaintExtractionError,
    EmbeddingError,
    LlmRateLimitedError,
    LlmTimeoutError,
    LlmUnavailableError,
    LlmUpstreamError,
)
from zipsai.history import format_history
from zipsai.integrations.llm import generate_structured
from zipsai.knowledge.retrieve import query_encoder
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
    fields: ComplaintDraft = field(default_factory=ComplaintDraft)
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
    request: ConverseRequest,
    image_analysis: ImageAnalysis | None = None,
    usage_sink: dict[str, object] | None = None,
) -> _TurnInterpretation:
    """현재 발화에서 새로 말한 민원 사실과 전환 판정만 받는다."""
    system_message, user_message = COMPLAINT_PROMPT.format_messages(
        today=datetime.now(_KST).date().isoformat(),
        conversation_history=format_history(request.conversation_history),
        complaint_draft=request.complaint_draft,
        # 사진만 온 턴은 text가 None이다. 그대로 넘기면 "현재 발화: None"이 렌더돼
        # 모델이 None을 입주민 발화로 읽는다.
        message_text=request.message.text or "",
        image_evidence=_format_image_evidence(image_analysis),
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
                fields=ComplaintDraft(
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


def _append_images(
    draft: ComplaintDraft, images: list[ImageAttachment | ImageObservation]
) -> ComplaintDraft:
    """초안에는 이미지 분석을 복사하지 않고 연결된 ID만 중복 없이 누적한다."""
    attachment_ids = dict.fromkeys(
        [*draft.attachment_ids, *(image.attachment_id for image in images)]
    )
    return draft.model_copy(update={"attachment_ids": list(attachment_ids)})


def _image_symptom(image_analysis: ImageAnalysis | None) -> str | None:
    if image_analysis is None:
        return None
    summaries = [
        observation.summary.strip()
        for observation in image_analysis.images
        if observation.summary and observation.summary.strip()
    ]
    return " ".join(summaries) or None


def _format_image_evidence(image_analysis: ImageAnalysis | None) -> str:
    if not image_analysis:
        return "없음"
    return json.dumps(
        [
            {
                "attachmentId": image.attachment_id,
                "summary": image.summary,
                "ocrText": image.ocr_text,
            }
            for image in image_analysis.images
        ],
        ensure_ascii=False,
    )


def _missing_fields(draft: ComplaintDraft) -> list[str]:
    """접수에 필요한데 아직 비어 있는 항목."""
    return [field for field in _REQUIRED_FIELDS if not getattr(draft, field)]


def _can_switch_away_from(current: ComplaintDraft | None) -> bool:
    return bool(current and current.symptom)


def _settle_switch(
    interpretation: _TurnInterpretation, current: ComplaintDraft | None
) -> _TurnInterpretation:
    """전환 질문을 만들 재료가 없는 ask를 같은 민원으로 떨어뜨린다.

    accept는 떨어뜨리지 않는다. 입주민이 이미 바꾸겠다고 답한 턴이라, same으로
    돌리면 방금 접어두기로 한 민원이 완성 상태로 카드까지 나간다.
    """
    if interpretation.switch != ComplaintSwitch.ASK:
        return interpretation
    if not _can_switch_away_from(current):
        # 전환할 대상 자체가 없다. 평범한 수집 턴이므로 추출값은 그대로 쓴다.
        return replace(interpretation, switch=ComplaintSwitch.SAME)
    if interpretation.fields.symptom:
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
    images: list[ImageAttachment] = field(default_factory=list)
    message_text: str = ""


@dataclass(frozen=True)
class _ComplaintProgress:
    interpretation: _TurnInterpretation
    draft: ComplaintDraft
    missing_fields: list[str]
    previous_symptom: str | None = None
    current_symptom: str | None = None
    image_symptom: str | None = None
    image_context: str | None = None


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


def _photo_label(
    images: list[ImageAttachment], image_analysis: ImageAnalysis | None, switch: ComplaintSwitch
) -> str:
    """이번 턴 사진이 어떻게 처리됐는지 나타내는 로그 값."""
    if not images:
        return "none"
    if switch == ComplaintSwitch.ASK:
        # 새 민원 수락 전까지 연결만 보류하고, 분석은 history에 보존한다.
        return "pending_switch"
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
    current: ComplaintDraft,
    extracted: ComplaintDraft,
    image_analysis: ImageAnalysis | None,
) -> _ComplaintTurn:
    """새 민원으로 바꿀지 입주민에게 묻는 턴."""
    # 돌려주는 초안이 옛 민원이므로 빈 칸도 옛 민원 기준으로 적는다.
    return _ComplaintTurn(
        result=RouteResult(
            complaint_draft=current,
            missing_fields=_missing_fields(current),
            image_analysis=image_analysis,
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
    images: list[ImageAttachment],
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
        reply=_photo_prefix(image_analysis) + question if images else question,
        complaint_state=ComplaintState.COLLECTING,
        asked=asked,
        reply_source=reply_source,
    )


def _ready_for_card(
    request: ConverseRequest,
    draft: ComplaintDraft,
    image_analysis: ImageAnalysis | None,
) -> _ComplaintTurn:
    """백엔드가 확인 카드를 만드는 유일한 경로. missing_fields가 비는 곳도 여기뿐이다."""
    # issue_type은 필수 항목이 아니지만, 분류 없는 카드는 관리자가 담당을 나눌 수 없다.
    if draft.issue_type is None:
        draft = draft.model_copy(update={"issue_type": "other"})
    representative_id = _representative_attachment_id(
        request, draft, image_analysis
    )
    response_draft = CompletedComplaintDraft(
        **draft.model_dump(), representative_attachment_id=representative_id
    )
    return _ComplaintTurn(
        result=RouteResult(
            complaint_draft=response_draft,
            missing_fields=[],
            image_analysis=image_analysis,
        ),
        reply="민원 정보를 확인했습니다. 접수할 내용을 확인해 주세요.",
        complaint_state=None,
        asked=None,
        reply_source="complete",
    )


def _representative_attachment_id(
    request: ConverseRequest,
    draft: ComplaintDraft,
    image_analysis: ImageAnalysis | None,
) -> int | None:
    attachment_ids = list(dict.fromkeys(draft.attachment_ids))
    if not attachment_ids:
        return None
    if len(attachment_ids) == 1:
        return attachment_ids[0]

    summaries = {
        image.attachment_id: image.summary.strip()
        for turn in request.conversation_history
        if turn.role == "user"
        for image in turn.images or []
        if image.summary and image.summary.strip()
    }
    if image_analysis:
        summaries.update(
            {
                image.attachment_id: image.summary.strip()
                for image in image_analysis.images
                if image.summary and image.summary.strip()
            }
        )

    candidate_summaries = [
        summaries.get(attachment_id) for attachment_id in attachment_ids
    ]
    # 사진이 있으면 대표 사진도 항상 있어야 한다. 유사도를 못 구하는 경우엔
    # 첫 번째 사진으로 떨어진다
    if any(summary is None for summary in candidate_summaries):
        return attachment_ids[0]

    complaint_text = " ".join(
        value.strip()
        for value in (draft.issue_type, draft.location, draft.symptom)
        if value and value.strip()
    )
    texts = [complaint_text, *(summary for summary in candidate_summaries if summary)]
    try:
        dense, _ = query_encoder().encode(texts, request.trace_id)
    except EmbeddingError as error:
        logger.warning(
            "representative_image_embedding_failed",
            extra={"trace_id": request.trace_id, "error_type": type(error).__name__},
        )
        return attachment_ids[0]

    if len(dense) != len(texts):
        return attachment_ids[0]
    complaint_vector = dense[0]
    scored: list[tuple[float, int]] = []
    for attachment_id, vector in zip(attachment_ids, dense[1:], strict=True):
        score = _cosine_similarity(complaint_vector, vector)
        if score is None:
            return attachment_ids[0]
        scored.append((score, attachment_id))
    return max(scored, key=lambda item: item[0])[1] if scored else attachment_ids[0]


def _cosine_similarity(left: list[float], right: list[float]) -> float | None:
    if not left or len(left) != len(right):
        return None
    if any(not math.isfinite(value) for value in left) or any(
        not math.isfinite(value) for value in right
    ):
        return None
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return None
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def _interpret_with_stage(
    request: ConverseRequest, image_analysis: ImageAnalysis | None = None
) -> _TurnInterpretation:
    with stage("text_interpretation", logger) as step:
        return _interpret_turn(
            request, image_analysis=image_analysis, usage_sink=step
        )


def _gather_turn_evidence(
    request: ConverseRequest, image_analysis: ImageAnalysis | None
) -> _TurnEvidence:
    text = (request.message.text or "").strip()
    images = request.message.images
    analysis = image_analysis if image_analysis and image_analysis.images else None

    if text or analysis:
        interpretation = (
            _interpret_with_stage(request, analysis)
            if analysis
            else _interpret_with_stage(request)
        )
    else:
        skipped("text_interpretation", logger, skip_reason="no_text_or_image_evidence")
        interpretation = _TurnInterpretation()

    return _TurnEvidence(
        interpretation,
        analysis,
        images,
        text,
    )


def _pending_switch_images(request: ConverseRequest) -> list[ImageObservation]:
    """직전 assistant의 전환 확인을 만든 user 턴 사진만 다음 민원에 넘긴다."""
    history = request.conversation_history
    if not history or history[-1].role != "assistant":
        return []
    previous_user = next(
        (turn for turn in reversed(history[:-1]) if turn.role == "user"), None
    )
    return previous_user.images or [] if previous_user else []


def _advance_complaint(
    request: ConverseRequest, evidence: _TurnEvidence
) -> _ComplaintProgress:
    interpretation = _settle_switch(evidence.interpretation, request.complaint_draft)
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
        current_images = evidence.images
        pending_images = (
            _pending_switch_images(request)
            if interpretation.switch == ComplaintSwitch.ACCEPT
            else []
        )
        draft = _append_images(
            _merge_structured_fields(previous, interpretation.fields),
            [*pending_images, *current_images],
        )
        previous_symptom = previous.symptom if previous else None

    image_symptom = _image_symptom(evidence.image_analysis)
    current_symptom = interpretation.fields.symptom or image_symptom
    if not previous_symptom or previous_symptom == current_symptom:
        draft = draft.model_copy(
            update={"symptom": current_symptom or previous_symptom}
        )
    return _ComplaintProgress(
        interpretation=interpretation,
        draft=draft,
        missing_fields=_missing_fields(draft),
        previous_symptom=previous_symptom,
        current_symptom=current_symptom,
        image_symptom=image_symptom,
        image_context=_format_image_evidence(evidence.image_analysis),
    )


def _generate_turn_finalization(
    progress: _ComplaintProgress,
    evidence: _TurnEvidence,
    next_action: str,
) -> _TurnFinalization | None:
    messages = TURN_FINALIZATION_PROMPT.format_messages(
        previous_symptom=json.dumps(progress.previous_symptom, ensure_ascii=False),
        current_symptom=json.dumps(progress.current_symptom, ensure_ascii=False),
        image_context=json.dumps(progress.image_context, ensure_ascii=False),
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
    request: ConverseRequest,
    progress: _ComplaintProgress,
    evidence: _TurnEvidence,
) -> _ComplaintTurn:
    interpretation = progress.interpretation
    if interpretation.switch == ComplaintSwitch.ASK:
        skipped("turn_finalization", logger, skip_reason="switch_confirmation")
        return _build_switch_confirmation_turn(
            progress.draft, interpretation.fields, evidence.image_analysis
        )

    symptom_merge_needed = bool(
        (
            progress.previous_symptom
            and progress.current_symptom
            and progress.previous_symptom != progress.current_symptom
        )
        or (
            progress.image_symptom
            and progress.interpretation.fields.symptom
            and progress.image_symptom != progress.interpretation.fields.symptom
        )
    )
    asked = progress.missing_fields[0] if progress.missing_fields else None
    if not symptom_merge_needed:
        skipped(
            "turn_finalization",
            logger,
            skip_reason="fixed_completion" if asked is None else "fixed_missing_field",
        )
        if asked:
            return _ask_for_missing(
                progress.draft,
                progress.missing_fields,
                evidence.image_analysis,
                evidence.images,
                "",
            )
        return _ready_for_card(request, progress.draft, evidence.image_analysis)

    next_action = f"ask_{asked}" if asked else "complete"
    finalization = _generate_turn_finalization(progress, evidence, next_action)
    fallback_symptom = progress.current_symptom or progress.previous_symptom
    finalized_symptom = (
        finalization.symptom.strip() if finalization and finalization.symptom else ""
    )
    symptom = finalized_symptom or fallback_symptom
    draft = progress.draft.model_copy(update={"symptom": symptom})

    if progress.missing_fields:
        return _ask_for_missing(
            draft,
            progress.missing_fields,
            evidence.image_analysis,
            evidence.images,
            finalization.reply.strip() if finalization else "",
        )
    return _ready_for_card(request, draft, evidence.image_analysis)


def _log_complaint_turn(
    request: ConverseRequest,
    evidence: _TurnEvidence,
    progress: _ComplaintProgress,
    turn: _ComplaintTurn,
    started_at: float,
) -> None:
    draft = turn.result.complaint_draft
    switch = progress.interpretation.switch
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
                evidence.images, turn.result.image_analysis, switch
            ),
        },
    )


def handle_complaint(
    request: ConverseRequest, *, image_analysis: ImageAnalysis | None = None
) -> dict[str, object]:
    started_at = perf_counter()
    evidence = _gather_turn_evidence(request, image_analysis)
    progress = _advance_complaint(request, evidence)
    turn = _finalize_turn(request, progress, evidence)
    _log_complaint_turn(request, evidence, progress, turn, started_at)
    return {
        "complaint_state": turn.complaint_state,
        "reply": turn.reply,
        "result": turn.result,
    }
