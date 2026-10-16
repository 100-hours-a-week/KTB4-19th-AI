from enum import Enum

from pydantic import BaseModel


class NotificationKind(str, Enum):
    DOCUMENT_INDEX = "document_index"
    DOCUMENT_CLEANUP = "document_cleanup"
    COMMENT_UPSERT = "comment_upsert"
    COMMENT_DELETE = "comment_delete"


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


class Notification(BaseModel):
    kind: NotificationKind
    trace_id: str
    # 댓글은 대상 키가 results에 있어 건물을 싣지 않는다.
    building_id: int | None = None
    results: list[ItemResult]

    def to_message(self) -> dict:
        # reason_code는 성공일 때도 null로 나가야 하지만, 쓰지 않는 대상 키는 빠져야 한다.
        body = self.model_dump(mode="json", exclude_none=True)
        for raw, result in zip(body["results"], self.results, strict=True):
            raw["reason_code"] = result.reason_code
        return body
