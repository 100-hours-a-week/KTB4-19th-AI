import threading

from zipsai.contracts.indexing import IndexingJobRequest


class Rejected(Exception):
    def __init__(self, status_code: int, error_code: str, retry_after: int) -> None:
        self.status_code = status_code
        self.error_code = error_code
        self.retry_after = retry_after


_lock = threading.Lock()
_active: set[str] = set()


def document_key(payload: IndexingJobRequest) -> str:
    if payload.doc_id is not None:
        return f"doc:{payload.doc_id}"
    return f"building:{payload.building_id}"


def acquire(key: str | None) -> None:
    if key is None:
        return
    with _lock:
        if key in _active:
            raise Rejected(429, "IN_PROGRESS", 30)
        _active.add(key)


def release(key: str | None) -> None:
    if key is None:
        return
    with _lock:
        _active.discard(key)


def reset() -> None:
    with _lock:
        _active.clear()
