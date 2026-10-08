from datetime import datetime
from enum import Enum
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Route(str, Enum):
    COMPLAINT = "complaint"
    KNOWLEDGE = "knowledge"
    CLARIFY = "clarify"


class ComplaintState(str, Enum):
    COLLECTING = "collecting"
    GUIDING = "guiding"
    CLARIFYING = "clarifying"


class ComplaintSwitch(str, Enum):
    """수집 중인 민원과 이번 턴이 같은 건인지에 대한 판정. 자세한 규칙은 추출 프롬프트에 있다."""

    SAME = "same"
    ASK = "ask"
    ACCEPT = "accept"


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


class ImageObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    attachment_id: int = Field(alias="attachmentId")
    summary: str | None = None
    ocr_text: str | None = Field(default=None, alias="ocrText")


class ImageAttachment(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    attachment_id: int = Field(alias="attachmentId")
    url: str


class ComplaintDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    issue_type: IssueType | None = None
    location: str | None = None
    symptom: str | None = None
    occurred_at: datetime | None = None
    attachment_ids: list[int] = Field(default_factory=list, alias="attachmentIds")


class ImageAnalysis(BaseModel):
    images: list[ImageObservation]


class IncomingMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message_id: str
    text: str | None
    images: list[ImageAttachment] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_images(self) -> "IncomingMessage":
        for image in self.images:
            extension = urlparse(image.url).path.rsplit(".", 1)[-1].lower()
            if extension not in {"jpg", "jpeg", "png", "webp"}:
                raise ValueError(f"Unsupported image extension: {image.url}")
        return self


class HistoryTurn(BaseModel):
    message_id: str
    role: Literal["user", "assistant"]
    text: str | None
    images: list[ImageObservation] | None = None

    @model_validator(mode="after")
    def validate_images_role(self) -> "HistoryTurn":
        if self.role == "assistant" and "images" in self.model_fields_set:
            raise ValueError("Assistant history turns must not include images")
        return self


class ConverseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    building_id: int
    room_no: str
    resident_id: str
    conversation_id: str
    turn_id: str
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
    image_analysis: ImageAnalysis | None = None


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
    turn_id: str
    trace_id: str
    data: ConverseData
