from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Route(str, Enum):
    COMPLAINT = "complaint"
    KNOWLEDGE = "knowledge"
    CLARIFY = "clarify"


class ConversationState(str, Enum):
    COLLECTING = "collecting"
    ACTION_SELECTION = "action_selection"
    GUIDING = "guiding"
    READY_TO_CONFIRM = "ready_to_confirm"
    CLARIFYING = "clarifying"


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
    "qa_card",
    "other",
]


class ResidentContext(BaseModel):
    unit_id: int
    resident_id: str


class ComplaintDraft(BaseModel):
    issue_type: IssueType | None = None
    location: str | None = None
    symptom: str | None = None
    occurred_at: datetime | None = None
    image_urls: list[str] = Field(default_factory=list)


class IncomingMessage(BaseModel):
    message_id: str
    text: str | None = None
    image_urls: list[str] = Field(default_factory=list)
    user_action: Literal["REQUEST_GUIDE", "REQUEST_REGISTER"] | None = None


class HistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ConverseRequest(BaseModel):
    building_id: int
    resident_context: ResidentContext
    conversation_id: str
    trace_id: str
    current_route: Route | None = None
    conversation_state: ConversationState | None = None
    message: IncomingMessage
    conversation_history: list[HistoryTurn] = Field(default_factory=list)
    complaint_draft: ComplaintDraft | None = None


class Meta(BaseModel):
    model: str
    timing_ms: int


class RouteResult(BaseModel):
    draft_patch: ComplaintDraft | None = None
    missing_fields: list[str] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    has_sufficient_evidence: bool | None = None
    image_analysis: dict[str, Any] | None = None


class ConverseData(BaseModel):
    route: Route
    requested_action: Literal["guide", "register", "unknown"] | None = None
    next_state: ConversationState | None = None
    reply: str
    result: RouteResult
    trace_id: str
    meta: Meta


class ApiResponse[T](BaseModel):
    message: str
    data: T


ConverseResponse = ApiResponse[ConverseData]
