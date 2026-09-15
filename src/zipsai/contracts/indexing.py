from datetime import datetime
from enum import Enum

from pydantic import BaseModel, StrictInt, StrictStr


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
    doc_id: StrictStr
    source_type: SourceType
    title: StrictStr
    published_at: datetime
    file_key: StrictStr
    trace_id: StrictStr


class IndexingJobResponse(BaseModel):
    job_id: str
    status: JobStatus
