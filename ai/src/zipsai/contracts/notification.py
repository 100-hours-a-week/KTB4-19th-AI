from enum import Enum

from pydantic import BaseModel


class ResultStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class ItemResult(BaseModel):
    """처리 대상 한 건의 결과. 대상 키는 종류에 따라 doc_id 또는 comment id다."""

    status: ResultStatus
    reason_code: str | None = None
    doc_id: str | None = None
    complaint_detail_id: int | None = None
    updated_at: str | None = None
