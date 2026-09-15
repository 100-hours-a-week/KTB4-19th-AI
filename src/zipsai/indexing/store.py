from uuid import uuid4

from zipsai.contracts.indexing import JobStatus
from zipsai.errors import JobNotFoundError


class InMemoryJobStore:
    def __init__(self) -> None:
        self._statuses: dict[str, JobStatus] = {}

    def create(self) -> str:
        job_id = str(uuid4())
        self._statuses[job_id] = JobStatus.ACCEPTED
        return job_id

    def get_status(self, job_id: str) -> JobStatus:
        try:
            return self._statuses[job_id]
        except KeyError as exc:
            raise JobNotFoundError(job_id) from exc
