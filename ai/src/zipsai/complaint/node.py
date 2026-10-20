import asyncio
import json
import logging
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime
from time import perf_counter
from zoneinfo import ZoneInfo

from pydantic import BaseModel

from zipsai.complaint.prompts import (
    COMPLAINT_PROMPT,
    TURN_FINALIZATION_PROMPT,
    TURN_REPLY_PROMPT,
    TURN_SYMPTOM_PROMPT,
)
from zipsai.complaint.urgency_llm import UrgencyResult, assess_urgency
from zipsai.contracts.converse import (
    ComplaintDraft,
    ComplaintState,
    ComplaintSwitch,
    CompletedComplaintDraft,
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
from zipsai.integrations.llm import generate_structured, generate_text_stream
from zipsai.knowledge.retrieve import query_encoder
from zipsai.metrics import observe_ask_repeat
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


class _TurnSymptomOutput(BaseModel):
    symptom: str | None


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
    contextualized_message: str | None = None,
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
        contextualized_message=contextualized_message or "없음",
        image_evidence=_format_image_evidence(image_analysis),
    )

    last_error: Exception | None = None
    for attempt in range(1, _INTERPRETATION_ATTEMPTS + 1):
        parsed = generate_structured(
            system_prompt=str(system_message.content),
            user_prompt=str(user_message.content),
            response_format=_InterpretationOutput,
            usage_sink=usage_sink,
            purpose="complaint_interpretation",
            attempt=attempt,
            retry_reason=None if attempt == 1 else "empty_output",
        )
        if parsed is not None:
            if usage_sink is not None:
                issue = parsed.issue_type
                usage_sink["complaint_switch"] = getattr(
                    parsed.complaint_switch, "value", parsed.complaint_switch
                )
                usage_sink["issue_type"] = (
                    getattr(issue, "value", issue) if issue else None
                )
                usage_sink["filled_fields"] = [
                    name
                    for name in ("issue_type", "location", "symptom", "occurred_at")
                    if getattr(parsed, name, None)
                ]
                usage_sink["interpretation_attempts"] = attempt
                usage_sink["parse_failed"] = False
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
# 고정 문구는 LLM이 비었을 때만 나간다. 모델이 다듬은 질문도 같은 항목으로 센다.
_ASK_FIELD_PATTERNS = {
    "location": re.compile(r"어디|위치|장소|어느 (곳|방|공간)"),
    "symptom": re.compile(r"증상|어떤 (문제|불편)|어떻게 (되|안)"),
}


@dataclass(frozen=True)
class _ComplaintTurn:
    result: RouteResult
    reply: str
    complaint_state: ComplaintState | None
    asked: str | None
    reply_source: str
    urgency_level: str | None = None


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


@dataclass(frozen=True)
class ComplaintAnalysis:
    """민원 reply를 만들기 전에 완료되어야 하는 구조화 분석 결과."""

    request: ConverseRequest
    evidence: _TurnEvidence
    progress: _ComplaintProgress
    symptom_merge_needed: bool
    asked: str | None


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
        reply=question,
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
    urgency = _assess_urgency_with_stage(draft, image_analysis)
    response_draft = CompletedComplaintDraft(
        **draft.model_dump(),
        representative_attachment_id=representative_id,
        urgency_score=urgency.score,
        is_urgent=urgency.is_urgent,
        urgency_version=urgency.urgency_version,
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
        urgency_level=getattr(urgency, "level", None),
    )


def _representative_attachment_id(
    request: ConverseRequest,
    draft: ComplaintDraft,
    image_analysis: ImageAnalysis | None,
) -> int | None:
    attachment_ids = list(dict.fromkeys(draft.attachment_ids))
    if not attachment_ids:
        # 사진 없는 민원이 다수라 여기서부터 로그를 남기면 신호 없는 줄만 늘어난다.
        return None
    if len(attachment_ids) == 1:
        skipped("representative_image", logger, skip_reason="single_attachment")
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
    # 첫 번째 사진으로 떨어진다 — 전부 null보다는 기계적 선택이라도 낫다.
    if any(summary is None for summary in candidate_summaries):
        skipped("representative_image", logger, skip_reason="missing_summary")
        return attachment_ids[0]

    complaint_text = " ".join(
        value.strip()
        for value in (draft.issue_type, draft.location, draft.symptom)
        if value and value.strip()
    )
    texts = [complaint_text, *(summary for summary in candidate_summaries if summary)]
    try:
        with stage("representative_image", logger) as step:
            dense, _ = query_encoder().encode(texts, request.trace_id)
            if len(dense) != len(texts):
                raise EmbeddingError("embedding count did not match requested texts")
            complaint_vector = dense[0]
            scored: list[tuple[float, int]] = []
            for attachment_id, vector in zip(attachment_ids, dense[1:], strict=True):
                score = _cosine_similarity(complaint_vector, vector)
                if score is None:
                    raise EmbeddingError("cosine similarity undefined for embedding")
                scored.append((score, attachment_id))
            representative_id = max(scored, key=lambda item: item[0])[1]
            step["representative_attachment_id"] = representative_id
            return representative_id
    except EmbeddingError as error:
        logger.warning(
            "representative_image_embedding_failed",
            extra={"trace_id": request.trace_id, "error_type": type(error).__name__},
        )
        return attachment_ids[0]


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
    request: ConverseRequest,
    image_analysis: ImageAnalysis | None = None,
    contextualized_message: str | None = None,
) -> _TurnInterpretation:
    with stage("text_interpretation", logger) as step:
        return _interpret_turn(
            request,
            image_analysis=image_analysis,
            contextualized_message=contextualized_message,
            usage_sink=step,
        )


def _assess_urgency_with_stage(
    draft: ComplaintDraft, image_analysis: ImageAnalysis | None
) -> UrgencyResult:
    """Phase 0 — 내부 로그 전용. assess_urgency가 실패를 null로 흡수하므로 여기선
    성공/실패를 가리지 않고 항상 결과를 기록한다."""
    with stage("urgency_assessment", logger) as step:
        result = assess_urgency(draft, image_analysis)
        step["urgency_score"] = result.score
        step["is_urgent"] = result.is_urgent
        step["urgency_version"] = result.urgency_version
        step["urgency_dimensions"] = result.dimensions
        return result


def _gather_turn_evidence(
    request: ConverseRequest,
    image_analysis: ImageAnalysis | None,
    contextualized_message: str | None = None,
) -> _TurnEvidence:
    text = (request.message.text or "").strip()
    images = request.message.images
    analysis = image_analysis if image_analysis and image_analysis.images else None

    if text or analysis:
        interpretation = (
            _interpret_with_stage(request, analysis, contextualized_message)
            if analysis
            else _interpret_with_stage(
                request, contextualized_message=contextualized_message
            )
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


def _analyze_complaint(
    request: ConverseRequest,
    image_analysis: ImageAnalysis | None = None,
    contextualized_message: str | None = None,
) -> ComplaintAnalysis:
    """동기·스트리밍 경로가 공유하는 민원 분석 단계."""
    evidence = _gather_turn_evidence(
        request, image_analysis, contextualized_message
    )
    progress = _advance_complaint(request, evidence)
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
    return ComplaintAnalysis(
        request=request,
        evidence=evidence,
        progress=progress,
        symptom_merge_needed=symptom_merge_needed,
        asked=progress.missing_fields[0] if progress.missing_fields else None,
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
                purpose="turn_finalization",
            )
    except (
        LlmRateLimitedError,
        LlmTimeoutError,
        LlmUnavailableError,
        LlmUpstreamError,
    ):
        return None


def _generate_turn_symptom(
    progress: _ComplaintProgress,
    evidence: _TurnEvidence,
) -> str | None:
    messages = TURN_SYMPTOM_PROMPT.format_messages(
        previous_symptom=json.dumps(progress.previous_symptom, ensure_ascii=False),
        current_symptom=json.dumps(progress.current_symptom, ensure_ascii=False),
        image_context=json.dumps(progress.image_context, ensure_ascii=False),
        current_message=json.dumps(evidence.message_text, ensure_ascii=False),
    )
    with stage("turn_symptom_finalize", logger) as step:
        result = generate_structured(
            system_prompt=str(messages[0].content),
            user_prompt=str(messages[1].content),
            response_format=_TurnSymptomOutput,
            usage_sink=step,
            purpose="symptom_finalize",
        )
        return result.symptom.strip() if result and result.symptom else None


def _prepare_streamed_reply(
    analysis: ComplaintAnalysis,
) -> tuple[ComplaintDraft, str] | None:
    """LLM reply streaming이 필요한 경우의 최종 증상과 다음 행동을 준비한다."""
    if not analysis.symptom_merge_needed or not analysis.asked:
        return None

    progress = analysis.progress
    symptom = _generate_turn_symptom(progress, analysis.evidence)
    symptom = symptom or progress.current_symptom or progress.previous_symptom
    draft = progress.draft.model_copy(update={"symptom": symptom})
    return draft, f"ask_{analysis.asked}"


async def _stream_generated_reply(
    analysis: ComplaintAnalysis,
    emit: Callable[[dict[str, str]], None],
) -> _ComplaintTurn | None:
    prepared = await asyncio.to_thread(_prepare_streamed_reply, analysis)
    if prepared is None:
        return None

    draft, next_action = prepared
    messages = TURN_REPLY_PROMPT.format_messages(
        draft=json.dumps(draft.model_dump(mode="json"), ensure_ascii=False),
        previous_symptom=json.dumps(
            analysis.progress.previous_symptom, ensure_ascii=False
        ),
        current_symptom=json.dumps(draft.symptom, ensure_ascii=False),
        next_action=next_action,
    )
    parts: list[str] = []
    with stage("turn_reply_generate", logger) as step:
        async for text in generate_text_stream(
            system_prompt=str(messages[0].content),
            user_prompt=str(messages[1].content),
            usage_sink=step,
            purpose="turn_reply",
        ):
            parts.append(text)
            emit({"type": "delta", "field": "reply", "text": text})

    reply = "".join(parts).strip()
    if not reply:
        return None
    return _ask_for_missing(
        draft,
        analysis.progress.missing_fields,
        analysis.evidence.image_analysis,
        reply,
    )


def _finalize_turn(
    analysis: ComplaintAnalysis,
) -> _ComplaintTurn:
    request = analysis.request
    progress = analysis.progress
    evidence = analysis.evidence
    interpretation = progress.interpretation
    if interpretation.switch == ComplaintSwitch.ASK:
        skipped("turn_finalization", logger, skip_reason="switch_confirmation")
        return _build_switch_confirmation_turn(
            progress.draft, interpretation.fields, evidence.image_analysis
        )

    symptom_merge_needed = analysis.symptom_merge_needed
    asked = analysis.asked
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
    repeat = _previous_same_field_asks(request.conversation_history, turn.asked)
    observations = (
        turn.result.image_analysis.images if turn.result.image_analysis else []
    )
    logger.info(
        "complaint_turn",
        extra={
            "conversation_id": request.conversation_id,
            "total_ms": elapsed_ms(started_at),
            "switch": switch,
            "asked": turn.asked,
            "asked_field": turn.asked,
            "ask_repeat_count": repeat,
            "missing": turn.result.missing_fields,
            "missing_fields": turn.result.missing_fields,
            "draft_filled": _draft_filled(draft),
            "draft_changed": draft != request.complaint_draft,
            "location_unknown": draft.location == _UNKNOWN_LOCATION,
            "reply_source": turn.reply_source,
            "text_len": len((request.message.text or "").strip()),
            "urgency_level": turn.urgency_level,
            "image_count": len(evidence.images),
            "images_analyzed": len(observations),
            "summary_chars": sum(len(image.summary or "") for image in observations),
            "ocr_chars": sum(len(image.ocr_text or "") for image in observations),
            "photo": _photo_label(
                evidence.images, turn.result.image_analysis, switch
            ),
        },
    )
    if turn.asked and repeat >= 2:
        logger.warning(
            "complaint_ask_repeated",
            extra={"asked_field": turn.asked, "ask_repeat_count": repeat},
        )
        observe_ask_repeat(turn.asked)


def _same_field_ask(text: str, field: str) -> bool:
    question = _MISSING_FIELD_REPLY.get(field, "")
    if question and question in text:
        return True
    pattern = _ASK_FIELD_PATTERNS.get(field)
    return bool(pattern and pattern.search(text))


def _previous_same_field_asks(history: list, field: str | None) -> int:
    """바로 앞에서 같은 항목을 물은 횟수. 이번 질문이 세 번째면 2다."""
    if not field or field not in _ASK_FIELD_PATTERNS:
        return 0
    count = 0
    for turn in reversed(history):
        if turn.role == "user":
            continue
        if _same_field_ask(turn.text or "", field):
            count += 1
            continue
        break
    return count


def _draft_filled(draft: ComplaintDraft) -> list[str]:
    names: list[str] = []
    for name in ("issue_type", "location", "symptom", "occurred_at"):
        value = getattr(draft, name, None)
        if not value:
            continue
        if name == "location" and value == _UNKNOWN_LOCATION:
            continue
        names.append(name)
    return names


def handle_complaint(
    request: ConverseRequest,
    *,
    image_analysis: ImageAnalysis | None = None,
    contextualized_message: str | None = None,
) -> dict[str, object]:
    started_at = perf_counter()
    analysis = _analyze_complaint(request, image_analysis, contextualized_message)
    turn = _finalize_turn(analysis)
    _log_complaint_turn(
        request, analysis.evidence, analysis.progress, turn, started_at
    )
    return {
        "complaint_state": turn.complaint_state,
        "reply": turn.reply,
        "result": turn.result,
    }


async def handle_complaint_stream(
    request: ConverseRequest,
    emit: Callable[[dict[str, str]], None],
    *,
    image_analysis: ImageAnalysis | None = None,
    contextualized_message: str | None = None,
) -> dict[str, object]:
    emit({"type": "status", "status": "민원 내용을 분석하고 있습니다."})
    started_at = perf_counter()
    analysis = await asyncio.to_thread(
        _analyze_complaint, request, image_analysis, contextualized_message
    )
    streamed_turn = await _stream_generated_reply(analysis, emit)
    turn = streamed_turn or await asyncio.to_thread(_finalize_turn, analysis)
    _log_complaint_turn(
        request, analysis.evidence, analysis.progress, turn, started_at
    )
    result = {
        "complaint_state": turn.complaint_state,
        "reply": turn.reply,
        "result": turn.result,
    }
    reply = result.get("reply")
    if streamed_turn is None and isinstance(reply, str) and reply:
        emit({"type": "delta", "field": "reply", "text": reply})
    return result
