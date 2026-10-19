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
