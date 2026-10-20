import threading

from zipsai.contracts.indexing import IndexingJobRequest
from zipsai.metrics import INDEXING_IN_FLIGHT
from zipsai.settings import INDEXING_MAX_CONCURRENCY


class Rejected(Exception):
    def __init__(self, status_code: int, error_code: str, retry_after: int) -> None:
        self.status_code = status_code
        self.error_code = error_code
        self.retry_after = retry_after


_lock = threading.Lock()
_active: set[str] = set()
_slots: threading.BoundedSemaphore | None = None
_in_flight = 0


def document_key(payload: IndexingJobRequest) -> str:
    if payload.doc_id is not None:
        return f"doc:{payload.doc_id}"
    return f"building:{payload.building_id}"


def in_flight() -> int:
    with _lock:
        return _in_flight


def _semaphore() -> threading.BoundedSemaphore:
    global _slots
    if _slots is None:
        _slots = threading.BoundedSemaphore(INDEXING_MAX_CONCURRENCY)
    return _slots


def acquire(key: str | None) -> None:
    global _in_flight
    with _lock:
        if key is not None and key in _active:
            raise Rejected(429, "IN_PROGRESS", 30)
        if not _semaphore().acquire(blocking=False):
            raise Rejected(503, "BUSY", 10)
        if key is not None:
            _active.add(key)
        _in_flight += 1
        INDEXING_IN_FLIGHT.inc()


def release(key: str | None) -> None:
    global _in_flight
    with _lock:
        if key is not None:
            _active.discard(key)
        _semaphore().release()
        _in_flight -= 1
        INDEXING_IN_FLIGHT.dec()


def reset() -> None:
    global _slots, _in_flight
    with _lock:
        _active.clear()
        _slots = None
        _in_flight = 0
        INDEXING_IN_FLIGHT.set(0)
