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
        dimension.value is None or not (1 <= dimension.value <= max_value)
    ):
        raise ValueError(
            f"{name}: supported status must carry a value between 1 and {max_value}"
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
        _validate_dimension("severity", self.severity)
        _validate_dimension("imminence", self.imminence)
        _validate_dimension("scope", self.scope)
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
