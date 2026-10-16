from typing import Annotated, Any, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    Field,
    StrictInt,
    StrictStr,
    StringConstraints,
    model_validator,
)

from zipsai.contracts.converse import IssueType

MAX_ITEMS = 100

# 공백뿐인 값은 내용이 없는 것과 같다. 앞뒤 공백을 떼고 길이를 본다.
NonBlank = Annotated[
    str, StringConstraints(strict=True, strip_whitespace=True, min_length=1)
]


class CommentItem(BaseModel):
    complaint_detail_id: StrictInt
    building_id: StrictInt
    complaint_type: Literal["COMPLAINT", "QA"]
    symptom: NonBlank
    comment: NonBlank
    # 오프셋이 없으면 어느 쪽이 더 최신인지 비교가 틀어진다. 오프셋 없는 값은 거절한다.
    updated_at: AwareDatetime
    # 통지에 요청 원문을 그대로 돌려주려고 둔다. 파싱한 값을 다시 직렬화하면
    # 밀리초 자릿수가 달라져 BE가 문자열로 건을 찾지 못한다.
    updated_at_raw: str = Field(default="", exclude=True)
    # 키는 반드시 와야 하고 값만 null일 수 있다. 위치를 모르는 QA는 null로 온다.
    location: StrictStr | None
    issue_type: IssueType | None
    # 검색에 쓰지 않는다. BE가 보내도 받기만 한다.
    occurred_time: StrictStr | None = None

    @model_validator(mode="before")
    @classmethod
    def keep_raw_updated_at(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("updated_at"), str):
            return {**data, "updated_at_raw": data["updated_at"]}
        return data


class CommentRequest(BaseModel):
    trace_id: NonBlank
    items: Annotated[list[CommentItem], Field(min_length=1, max_length=MAX_ITEMS)]


class CommentDeleteItem(BaseModel):
    building_id: StrictInt
    complaint_detail_id: StrictInt


class CommentDeleteRequest(BaseModel):
    trace_id: NonBlank
    items: Annotated[list[CommentDeleteItem], Field(min_length=1, max_length=MAX_ITEMS)]
