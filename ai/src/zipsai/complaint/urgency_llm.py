from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

DimensionName = Literal["severity", "imminence", "scope"]
DimensionStatus = Literal["supported", "explicitly_absent", "unknown"]
EvidenceSource = Literal["symptom", "image"]

_MAX_VALUE: dict[DimensionName, int] = {
    "severity": 4,
    "imminence": 4,
    "scope": 2,
}
_MIN_SUPPORTED_VALUE: dict[DimensionName, int] = {
    "severity": 1,
    "imminence": 1,
    "scope": 0,
}


class DimensionJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: DimensionStatus
    value: int | None = None
    source: EvidenceSource | None = None
    attachment_id: int | None = None
    quote: str | None = None


def _validate_dimension(name: DimensionName, dimension: DimensionJudgment) -> None:
    max_value = _MAX_VALUE[name]
    if dimension.status == "unknown":
        if (
            dimension.value is not None
            or dimension.source is not None
            or dimension.attachment_id is not None
            or dimension.quote is not None
        ):
            raise ValueError(
                f"{name}: unknown status must carry no value, source, or quote"
            )
        return
    if dimension.status == "explicitly_absent" and dimension.value != 0:
        raise ValueError(f"{name}: explicitly_absent status must carry value 0")
    if dimension.status == "supported" and (
        dimension.value is None
        or not (_MIN_SUPPORTED_VALUE[name] <= dimension.value <= max_value)
    ):
        raise ValueError(
            f"{name}: supported status must carry a value between "
            f"{_MIN_SUPPORTED_VALUE[name]} and {max_value}"
        )
    if not dimension.quote:
        raise ValueError(f"{name}: {dimension.status} status must carry a quote")
    if dimension.source is None:
        raise ValueError(f"{name}: {dimension.status} status must carry a source")
    if dimension.source == "image" and dimension.attachment_id is None:
        raise ValueError(f"{name}: source=image must carry an attachment_id")


class UrgencyEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    severity: DimensionJudgment
    imminence: DimensionJudgment
    scope: DimensionJudgment

    @model_validator(mode="after")
    def validate_dimensions(self) -> "UrgencyEvidence":
        for name in ("severity", "imminence", "scope"):
            try:
                _validate_dimension(name, getattr(self, name))
            except ValueError:
                # 한 축의 잘못된 판단이 다른 축의 유효한 근거를 버리지 않게 한다.
                setattr(self, name, DimensionJudgment(status="unknown"))
        return self


from dataclasses import dataclass, field


@dataclass(frozen=True)
class ScoringConfig:
    severity_weight: int
    imminence_weight: int
    scope_weight: int
    urgent_threshold: int
    version: str


URGENCY_VERSION = "urgency-llm-v1-phase0"

DEFAULT_SCORING_CONFIG = ScoringConfig(
    severity_weight=50,
    imminence_weight=30,
    scope_weight=20,
    urgent_threshold=60,
    version=URGENCY_VERSION,
)


@dataclass(frozen=True)
class UrgencyResult:
    score: int | None
    is_urgent: bool | None
    urgency_version: str
    dimensions: dict[str, dict[str, object]] = field(default_factory=dict)


def _numeric_value(dimension: DimensionJudgment, *, unknown_value: int) -> int:
    if dimension.status == "unknown":
        return unknown_value
    assert dimension.value is not None
    return dimension.value


def _dimension_summary(evidence: UrgencyEvidence) -> dict[str, dict[str, object]]:
    return {
        name: {"status": dimension.status, "value": dimension.value}
        for name, dimension in (
            ("severity", evidence.severity),
            ("imminence", evidence.imminence),
            ("scope", evidence.scope),
        )
    }


def score_urgency(evidence: UrgencyEvidence, config: ScoringConfig) -> UrgencyResult:
    """합산 공식이 임박성 낮은 S=4를 과소평가하는 걸 막으려고 severity==4면 점수와
    무관하게 is_urgent를 강제한다 — 2026-10-17 재설계 문서의 override 규칙."""
    dimensions = (evidence.severity, evidence.imminence, evidence.scope)
    if all(dimension.status == "unknown" for dimension in dimensions):
        return UrgencyResult(
            score=None,
            is_urgent=None,
            urgency_version=config.version,
            dimensions=_dimension_summary(evidence),
        )

    severity_value = _numeric_value(evidence.severity, unknown_value=1)
    imminence_value = _numeric_value(evidence.imminence, unknown_value=1)
    scope_value = _numeric_value(evidence.scope, unknown_value=0)

    score = round(
        config.severity_weight * severity_value / 4
        + config.imminence_weight * imminence_value / 4
        + config.scope_weight * scope_value / 2
    )
    is_urgent = score >= config.urgent_threshold or (
        evidence.severity.status == "supported" and evidence.severity.value == 4
    )
    return UrgencyResult(
        score=score,
        is_urgent=is_urgent,
        urgency_version=config.version,
        dimensions=_dimension_summary(evidence),
    )


import json
import logging

from pydantic import ValidationError

from zipsai.complaint.prompts import URGENCY_ASSESSMENT_PROMPT
from zipsai.contracts.converse import ComplaintDraft, ImageAnalysis
from zipsai.errors import (
    LlmRateLimitedError,
    LlmTimeoutError,
    LlmUnavailableError,
    LlmUpstreamError,
)
from zipsai.integrations.llm import generate_structured

logger = logging.getLogger(__name__)

_RECOVERABLE_LLM_ERRORS = (
    LlmRateLimitedError,
    LlmTimeoutError,
    LlmUnavailableError,
    LlmUpstreamError,
)

_UNKNOWN_DIMENSION = DimensionJudgment(status="unknown")


def _unscored_result() -> UrgencyResult:
    return UrgencyResult(score=None, is_urgent=None, urgency_version=URGENCY_VERSION)


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


def _image_summary(
    image_analysis: ImageAnalysis | None, attachment_id: int | None
) -> str | None:
    if image_analysis is None or attachment_id is None:
        return None
    for image in image_analysis.images:
        if image.attachment_id == attachment_id:
            return image.summary
    return None


def _verify_dimension(
    dimension: DimensionJudgment, symptom: str, image_analysis: ImageAnalysis | None
) -> DimensionJudgment:
    if dimension.status == "unknown":
        return dimension
    if not dimension.quote:
        return _UNKNOWN_DIMENSION
    if dimension.source == "symptom":
        source_text = symptom
    elif dimension.source == "image":
        source_text = _image_summary(image_analysis, dimension.attachment_id)
    else:
        source_text = None
    if not source_text or dimension.quote not in source_text:
        return _UNKNOWN_DIMENSION
    return dimension


def _verify_evidence(
    evidence: UrgencyEvidence, symptom: str, image_analysis: ImageAnalysis | None
) -> UrgencyEvidence:
    return UrgencyEvidence(
        severity=_verify_dimension(evidence.severity, symptom, image_analysis),
        imminence=_verify_dimension(evidence.imminence, symptom, image_analysis),
        scope=_verify_dimension(evidence.scope, symptom, image_analysis),
    )


def assess_urgency(
    draft: ComplaintDraft, image_analysis: ImageAnalysis | None
) -> UrgencyResult:
    """완료 카드 시점에만 호출한다. 실패해도 예외를 던지지 않고 null 결과로 흡수한다 —
    긴급도는 참고 정보일 뿐 접수 흐름을 막으면 안 된다(2026-10-17 설계 "실패·안전 동작")."""
    symptom = draft.symptom or ""
    system_message, user_message = URGENCY_ASSESSMENT_PROMPT.format_messages(
        symptom=symptom,
        location=draft.location or "미상",
        occurred_at=draft.occurred_at.isoformat() if draft.occurred_at else "미상",
        image_evidence=_format_image_evidence(image_analysis),
    )
    try:
        evidence = generate_structured(
            system_prompt=str(system_message.content),
            user_prompt=str(user_message.content),
            response_format=UrgencyEvidence,
            purpose="urgency",
        )
    except (*_RECOVERABLE_LLM_ERRORS, ValidationError) as error:
        logger.exception(
            "urgency_assessment_failed",
            extra={
                "error_type": type(error).__name__,
                "urgency_version": URGENCY_VERSION,
            },
        )
        return _unscored_result()

    if evidence is None:
        logger.warning(
            "urgency_assessment_empty", extra={"urgency_version": URGENCY_VERSION}
        )
        return _unscored_result()

    verified = _verify_evidence(evidence, symptom, image_analysis)
    return score_urgency(verified, DEFAULT_SCORING_CONFIG)
