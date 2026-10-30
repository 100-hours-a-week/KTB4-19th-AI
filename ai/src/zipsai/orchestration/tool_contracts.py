"""Agent가 고를 수 있는 도구의 입력·결과 계약. 실행 함수는 후속 태스크에서 연결한다."""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from zipsai.contracts.converse import Citation, ComplaintState, RouteResult


class BuildingDocsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    query: str = Field(min_length=1, description="건물 문서에서 확인할 질문")


class ComplaintTurnArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1)
    citation: Citation


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["ok", "no_results"]
    evidence: list[SearchEvidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_evidence(self) -> "SearchResult":
        if self.status == "ok" and not self.evidence:
            raise ValueError("ok requires evidence")
        if self.status == "no_results" and self.evidence:
            raise ValueError("no_results must not include evidence")
        return self


class BuildingDocsResult(SearchResult):
    @model_validator(mode="after")
    def validate_sources(self) -> "BuildingDocsResult":
        if any(
            item.citation.source_type not in {"building_document", "qa_history"}
            for item in self.evidence
        ):
            raise ValueError("building search requires building_document or qa_history")
        return self


class ToolFailure(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["error"] = "error"
    code: Literal[
        "invalid_arguments",
        "permission_denied",
        "timeout",
        "unavailable",
        "rate_limited",
        "upstream_error",
    ]
    message: str = Field(min_length=1)


class ComplaintTurnResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reply: str
    complaint_state: ComplaintState | None
    result: RouteResult


@dataclass(frozen=True)
class ToolContract:
    name: str
    description: str
    args_model: type[BaseModel]
    result_model: type[BaseModel]
    terminal: bool = False

    def model_spec(self) -> dict[str, object]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_model.model_json_schema(),
            },
        }


TOOL_CONTRACTS = (
    ToolContract(
        name="search_building_docs",
        description=(
            "현재 입주민 건물의 규정·시설·운영 정보가 필요한 경우 문서와 과거 공개 답변에서 "
            "근거를 검색한다. 건물 범위는 서버가 정하며, 결과가 없으면 건물 규칙을 추측하지 않는다."
        ),
        args_model=BuildingDocsArgs,
        result_model=BuildingDocsResult,
    ),
    ToolContract(
        name="process_complaint_turn",
        description=(
            "시설 문제 신고를 작성·이어가거나 기존 민원 초안을 확인할 때 현재 대화 한 턴을 "
            "처리한다. 요청·대화 이력·민원 상태·이미지 관찰은 서버가 전달한다. "
            "완성된 답변과 초안을 반환하며 실제 접수는 수행하지 않는다."
        ),
        args_model=ComplaintTurnArgs,
        result_model=ComplaintTurnResult,
        terminal=True,
    ),
)
