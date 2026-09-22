from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Route(str, Enum):
    COMPLAINT = "complaint"
    KNOWLEDGE = "knowledge"
    CLARIFY = "clarify"


class ComplaintState(str, Enum):
    COLLECTING = "collecting"
    GUIDING = "guiding"
    CLARIFYING = "clarifying"


def _require_complaint_state_only_for_complaint_route(
    route: Route | None, complaint_state: ComplaintState | None, field_name: str
) -> None:
    if route is not Route.COMPLAINT and complaint_state is not None:
        raise ValueError(f"{field_name} requires route=complaint")


IssueType = Literal[
    "water_supply",
    "drain",
    "leak",
    "heating",
    "electricity",
    "mold",
    "pest",
    "facility",
    "noise",
    "other",
]
MissingField = Literal["location", "symptom"]
CitationSource = Literal["building_document", "qa_history", "web"]


class ComplaintDraft(BaseModel):
    issue_type: IssueType | None = None
    location: str | None = None
    symptom: str | None = None
    occurred_at: datetime | None = None
    image_urls: list[str] = Field(default_factory=list)


class IncomingMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message_id: str
    text: str | None
    image_urls: list[str]


class HistoryTurn(BaseModel):
    message_id: str
    role: Literal["user", "assistant"]
    text: str | None
    image_urls: list[str]


class ConverseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    building_id: int
    room_no: str
    resident_id: str
    conversation_id: str
    trace_id: str
    current_route: Route | None
    current_complaint_state: ComplaintState | None
    message: IncomingMessage
    conversation_history: list[HistoryTurn]
    complaint_draft: ComplaintDraft | None

    @model_validator(mode="after")
    def validate_complaint_state(self) -> "ConverseRequest":
        _require_complaint_state_only_for_complaint_route(
            self.current_route,
            self.current_complaint_state,
            "current_complaint_state",
        )
        return self


class Meta(BaseModel):
    model: str
    timing_ms: int


class QaCardDraft(BaseModel):
    question: str


class Citation(BaseModel):
    source_type: CitationSource
    source_id: str | None
    title: str
    snippet: str | None
    url: str | None


class RouteResult(BaseModel):
    complaint_draft: ComplaintDraft | None = None
    qa_card_draft: QaCardDraft | None = None
    missing_fields: list[MissingField] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    has_sufficient_evidence: bool | None = None
    image_analysis: dict[str, Any] | None = None


class ConverseData(BaseModel):
    route: Route
    complaint_intent: Literal["guide", "register", "unknown"] | None = None
    next_complaint_state: ComplaintState | None = None
    reply: str
    result: RouteResult
    meta: Meta

    @model_validator(mode="after")
    def validate_next_complaint_state(self) -> "ConverseData":
        _require_complaint_state_only_for_complaint_route(
            self.route, self.next_complaint_state, "next_complaint_state"
        )
        return self


class ConverseResponse(BaseModel):
    code: Literal["ai_response_success"]
    trace_id: str
    data: ConverseData
