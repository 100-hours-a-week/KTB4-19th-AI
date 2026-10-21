from typing import Literal

from pydantic import BaseModel

from zipsai.contracts.notification import ResultStatus


class DocumentResult(BaseModel):
    """문서 색인 또는 정리 한 건. doc_id는 색인에만 실리고 정리는 빠진다."""

    kind: Literal["document_index", "document_cleanup"]
    status: ResultStatus
    reason_code: str | None = None
    retryable: bool = False
    doc_id: str | None = None

    def to_dict(self) -> dict:
        body = self.model_dump(mode="json", exclude_none=True)
        body["reason_code"] = self.reason_code
        if self.kind != "document_index":
            body.pop("doc_id", None)
        return body


class CommentResult(BaseModel):
    """답변 반영·삭제 한 건. 없는 대상 키도 null로 남긴다."""

    status: ResultStatus
    reason_code: str | None = None
    retryable: bool = False
    complaint_detail_id: int | None = None
    updated_at: str | None = None

    def to_dict(self) -> dict:
        body = self.model_dump(mode="json", exclude_none=True)
        body["reason_code"] = self.reason_code
        body["complaint_detail_id"] = self.complaint_detail_id
        body["updated_at"] = self.updated_at
        return body


class SyncBody(BaseModel):
    trace_id: str
    results: list[DocumentResult] | list[CommentResult]

    def to_body(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "results": [item.to_dict() for item in self.results],
        }
