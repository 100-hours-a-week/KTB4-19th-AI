from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, StrictInt, StrictStr, model_validator

DOCUMENT_FIELDS = ("doc_id", "title", "file_key")


class JobStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class IndexingJobRequest(BaseModel):
    """색인과 정리를 한 요청으로 받는다.

    문서 세 필드는 묶어서 있거나 없고, 목록을 얹은 요청만 삭제가 돈다.
    """

    building_id: StrictInt
    # 빈 값이면 같은 building의 서로 다른 문서가 같은 교체 필터를 쓴다.
    doc_id: Annotated[StrictStr, Field(min_length=1)] | None = None
    # 검색 결과의 출처 표기에 쓴다. 입주민에게 doc_id를 보여줄 수는 없다.
    title: StrictStr | None = None
    file_key: StrictStr | None = None
    # 빈 목록을 허용하면 백엔드 실수 한 번에 건물 문서가 전멸한다.
    valid_doc_ids: Annotated[list[StrictStr], Field(min_length=1)] | None = None

    @model_validator(mode="after")
    def check_shape(self) -> "IndexingJobRequest":
        present = [name for name in DOCUMENT_FIELDS if getattr(self, name) is not None]
        if present and len(present) != len(DOCUMENT_FIELDS):
            missing = [name for name in DOCUMENT_FIELDS if name not in present]
            raise ValueError(
                f"send {', '.join(DOCUMENT_FIELDS)} together, missing {missing}"
            )
        if not present and self.valid_doc_ids is None:
            raise ValueError("send a document, valid_doc_ids, or both")
        return self

    @property
    def has_document(self) -> bool:
        return self.doc_id is not None
