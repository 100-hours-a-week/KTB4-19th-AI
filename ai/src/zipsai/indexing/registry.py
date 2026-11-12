"""문서 색인 작업 기록. 프로세스 메모리에만 둔다.

작업 스레드는 하나다. 대기열에 있는 작업은 시작 전에 빼면 파이프라인을 타지 않는다.
"""

from __future__ import annotations

import logging
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal

from zipsai.contracts.indexing import IndexingJobRequest
from zipsai.contracts.notification import ResultStatus
from zipsai.contracts.sync_response import SyncBody
from zipsai.errors import JobCancelled
from zipsai.metrics import record_indexing_result

logger = logging.getLogger("zipsai.api.indexing")

Action = Literal["replay", "reject", "wait"]
State = Literal["queued", "running", "done", "failed", "cancelled"]


@dataclass
class Decision:
    action: Action
    record: Job | None = None
    body: dict | None = None
    error_code: str | None = None
    status_code: int = 200
    retry_after: int = 30


class _Hooks:
    def __init__(self, record: Job, lock: threading.Condition) -> None:
        self._record = record
        self._lock = lock

    def should_cancel(self) -> bool:
        return self._record.cancel.is_set()

    def on_progress(self, done: int, total: int) -> None:
        with self._lock:
            self._record.done = done
            self._record.total = total


_hooks: ContextVar[_Hooks | None] = ContextVar("indexing_job_hooks", default=None)


def current_hooks() -> _Hooks | None:
    return _hooks.get()


@dataclass
class Job:
    payload: IndexingJobRequest
    job_id: str
    state: State
    cancel: threading.Event
    finished: threading.Event
    done: int = 0
    total: int = 0
    body: dict | None = None
    retryable: bool = False
    cancel_reason: str | None = None
    finished_at: float | None = None
    started: float = 0
    queue_len_at_accept: int = 0

    @property
    def key(self) -> tuple[str, str]:
        return (self.payload.doc_id or "", self.payload.file_key or "")

    def progress(self) -> dict[str, int]:
        return {"done": self.done, "total": self.total}


class Registry:
    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._jobs: dict[tuple[str, str], Job] = {}
        self._active_doc: dict[str, Job] = {}
        self._queue: list[Job] = []
        self._clock = time.monotonic
        self._stop = False
        self._thread = threading.Thread(
            target=self._loop, name="indexing-job", daemon=True
        )
        self._thread.start()

    def set_clock(self, clock) -> None:
        with self._cv:
            self._clock = clock

    def queue_len(self) -> int:
        with self._cv:
            return len(self._queue)

    def close(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify()
        self._thread.join(timeout=2)

    def cancel_absent(self, building_id: int, valid_doc_ids: set[str]) -> None:
        with self._cv:
            for record in list(self._active_doc.values()):
                if record.payload.building_id != building_id:
                    continue
                if record.payload.doc_id in valid_doc_ids:
                    continue
                self._cancel_locked(record, "deleted")

    def admit(self, payload: IndexingJobRequest, job_id: str) -> Decision:
        with self._cv:
            self._expire_locked()
            assert payload.doc_id is not None
            assert payload.file_key is not None
            key = (payload.doc_id, payload.file_key)
            existing = self._jobs.get(key)
            if existing is not None and existing.state == "cancelled":
                self._forget_locked(existing)
                existing = None
            if existing is not None and existing.state in ("queued", "running"):
                code = "IN_PROGRESS" if existing.state == "running" else "QUEUED"
                return Decision(
                    "reject",
                    existing,
                    error_code=code,
                    status_code=429,
                    retry_after=_retry_after(),
                )
            if existing is not None and existing.state == "done":
                return Decision("replay", existing, existing.body)
            if existing is not None and existing.state == "failed":
                if not existing.retryable:
                    return Decision("replay", existing, existing.body)
                self._forget_locked(existing)

            active = self._active_doc.get(payload.doc_id)
            if active is not None and active.payload.file_key != payload.file_key:
                self._cancel_locked(active, "replaced")
                _log_replaced(payload, job_id)

            if len(self._queue) >= _queue_limit():
                return Decision(
                    "reject",
                    error_code="QUEUE_FULL",
                    status_code=503,
                    retry_after=10,
                )

            record = Job(
                payload=payload,
                job_id=job_id,
                state="queued",
                cancel=threading.Event(),
                finished=threading.Event(),
            )
            self._jobs[key] = record
            self._active_doc[payload.doc_id] = record
            self._queue.append(record)
            record.queue_len_at_accept = len(self._queue)
            # 접수 로그를 남긴 다음에 깨운다. 먼저 깨우면 거절 로그와 섞인다.
            return Decision("wait", record)

    def notify(self) -> None:
        with self._cv:
            self._cv.notify()

    def wait(self, record: Job, timeout: float) -> Job:
        record.finished.wait(timeout)
        return record

    def _expire_locked(self) -> None:
        now = self._clock()
        ttl = _ttl()
        for record in list(self._jobs.values()):
            if record.state not in ("done", "failed"):
                continue
            if record.finished_at is None:
                continue
            if now - record.finished_at >= ttl:
                self._forget_locked(record)

    def _forget_locked(self, record: Job) -> None:
        current = self._jobs.get(record.key)
        if current is record:
            self._jobs.pop(record.key, None)
        active = self._active_doc.get(record.payload.doc_id or "")
        if active is record:
            self._active_doc.pop(record.payload.doc_id or "", None)

    def _cancel_locked(self, record: Job, reason: str) -> None:
        record.cancel_reason = reason
        record.cancel.set()
        if record.state != "queued":
            return
        self._queue = [item for item in self._queue if item is not record]
        record.state = "cancelled"
        record.finished_at = self._clock()
        active = self._active_doc.get(record.payload.doc_id or "")
        if active is record:
            self._active_doc.pop(record.payload.doc_id or "", None)
        _log_cancelled(record)
        record.finished.set()

    def _loop(self) -> None:
        while True:
            with self._cv:
                while not self._stop and not self._queue:
                    self._cv.wait()
                if self._stop and not self._queue:
                    return
                record = self._take_locked()
            if record is None:
                continue
            self._execute(record)

    def _take_locked(self) -> Job | None:
        while self._queue:
            record = self._queue.pop(0)
            if record.state == "queued":
                record.state = "running"
                return record
        return None

    def _execute(self, record: Job) -> None:
        record.started = time.perf_counter()
        hooks = _Hooks(record, self._cv)
        token = _hooks.set(hooks)
        try:
            from zipsai.api.indexing import _run_job

            try:
                outcome = _run_job(record.payload, record.job_id)
            except JobCancelled:
                self._mark_cancelled(record)
                return
            except Exception as error:  # noqa: BLE001 — 작업 스레드가 죽으면 다음 요청이 429에 남는다
                self._mark_unknown_failure(record, error)
                return
            if record.cancel.is_set():
                self._mark_cancelled(record)
                return
            self._mark_finished(record, outcome)
        finally:
            _hooks.reset(token)

    def _mark_cancelled(self, record: Job) -> None:
        with self._cv:
            record.state = "cancelled"
            record.finished_at = self._clock()
            active = self._active_doc.get(record.payload.doc_id or "")
            if active is record:
                self._active_doc.pop(record.payload.doc_id or "", None)
        _log_cancelled(record)
        record.finished.set()

    def _mark_unknown_failure(self, record: Job, error: Exception) -> None:
        from zipsai.api.indexing import _document_result, _outcome

        results = [
            _document_result(
                "document_index",
                ResultStatus.FAILED,
                "INDEX_FAILED",
                record.payload.doc_id,
            )
        ]
        outcome = _outcome(results, {"pages": None, "chunks": None, "points": None})
        body = SyncBody(trace_id=record.payload.trace_id, results=results).to_body()
        logger.info(
            "job_done",
            extra={
                "job_id": record.job_id,
                "trace_id": record.payload.trace_id,
                "doc_id": record.payload.doc_id,
                "building_id": record.payload.building_id,
                "outcome": "fail",
                "total_ms": 0,
                "failed_stage": None,
                "error_type": type(error).__name__,
            },
        )
        for item in outcome.results:
            record_indexing_result(item.kind, item.status.value, item.reason_code)
        self._store_locked(record, body, retryable=False, state="failed")

    def _mark_finished(self, record: Job, outcome) -> None:
        from zipsai.api.indexing import _log_index_result

        failed = any(item.status is ResultStatus.FAILED for item in outcome.results)
        retryable = any(
            item.retryable
            for item in outcome.results
            if item.status is ResultStatus.FAILED
        )
        body = SyncBody(
            trace_id=record.payload.trace_id, results=outcome.results
        ).to_body()
        # 기다리는 요청은 finished가 켜지면 바로 돌아간다. 로그와 지표를 먼저 남긴다.
        _log_index_result(record.payload, record.job_id, outcome, record.started)
        for item in outcome.results:
            record_indexing_result(item.kind, item.status.value, item.reason_code)
        self._store_locked(
            record, body, retryable=retryable, state="failed" if failed else "done"
        )

    def _store_locked(
        self, record: Job, body: dict, *, retryable: bool, state: State
    ) -> None:
        with self._cv:
            record.body = body
            record.retryable = retryable
            record.state = state
            record.finished_at = self._clock()
            active = self._active_doc.get(record.payload.doc_id or "")
            if active is record:
                self._active_doc.pop(record.payload.doc_id or "", None)
            record.finished.set()


def _queue_limit() -> int:
    # 테스트가 설정 값을 바꾼 뒤에도 그 값을 보게 매번 읽는다.
    from zipsai import settings

    return int(settings.INDEXING_QUEUE_LIMIT)


def _ttl() -> float:
    from zipsai import settings

    return float(settings.INDEXING_RECORD_TTL_SECONDS)


def _retry_after() -> int:
    from zipsai import settings

    return int(settings.INDEXING_RETRY_AFTER_SECONDS)


def _file_key_parts(file_key: str | None) -> tuple[str | None, str | None]:
    if not file_key:
        return None, None
    path = file_key.split("?", 1)[0]
    name = path.rsplit("/", 1)[-1]
    if "." not in name:
        return path, None
    suffix = name.rsplit(".", 1)[-1]
    return path, suffix or None


def _log_cancelled(record: Job) -> None:
    path, ext = _file_key_parts(record.payload.file_key)
    logger.info(
        "job_cancelled",
        extra={
            "job_id": record.job_id,
            "trace_id": record.payload.trace_id,
            "doc_id": record.payload.doc_id,
            "building_id": record.payload.building_id,
            "file_key_path": path,
            "file_key_ext": ext,
            "reason": record.cancel_reason,
        },
    )


def _log_replaced(payload: IndexingJobRequest, job_id: str) -> None:
    path, ext = _file_key_parts(payload.file_key)
    logger.info(
        "job_replaced",
        extra={
            "job_id": job_id,
            "trace_id": payload.trace_id,
            "doc_id": payload.doc_id,
            "building_id": payload.building_id,
            "file_key_path": path,
            "file_key_ext": ext,
        },
    )


_registry: Registry | None = None
_registry_guard = threading.Lock()


def get_registry() -> Registry:
    global _registry
    with _registry_guard:
        if _registry is None:
            _registry = Registry()
        return _registry


def reset_registry() -> None:
    global _registry
    with _registry_guard:
        if _registry is not None:
            _registry.close()
            _registry = None
