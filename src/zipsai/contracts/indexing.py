from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, StrictInt, StrictStr


class SourceType(str, Enum):
    RULE = "rule"
    NOTICE = "notice"
    FACILITY = "facility"


class JobStatus(str, Enum):
    ACCEPTED = "accepted"
    RUNNING = "running"
    NEEDS_REVIEW = "needs_review"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class IndexingJobRequest(BaseModel):
    building_id: StrictInt
    # 빈 값이면 같은 building의 서로 다른 문서가 같은 교체 필터를 쓴다.
    doc_id: StrictStr = Field(min_length=1)
    source_type: SourceType
    title: StrictStr
    published_at: datetime
    file_key: StrictStr
    trace_id: StrictStr


class IndexingJobResponse(BaseModel):
    job_id: str
    status: JobStatus
