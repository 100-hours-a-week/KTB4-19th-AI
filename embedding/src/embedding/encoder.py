import heapq
import itertools
import logging
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import CancelledError, Future
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)

MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "BAAI/bge-m3")
# 질의가 기다리는 최대 시간은 처리 중인 묶음 하나가 끝나는 시간이다(≈ 묶음 크기 × 가장
# 긴 텍스트). 키우면 처리량은 늘지만 질의 대기 한도(3.5초)를 넘길 수 있어 1에서 시작한다.
MAX_INDEX_BATCH = int(os.getenv("EMBEDDING_MAX_INDEX_BATCH", "1"))

Priority = Literal["query", "index"]
_RANK: dict[Priority, int] = {"query": 0, "index": 1}


class ModelNotLoadedError(RuntimeError):
    pass


class ModelBusyError(RuntimeError):
    pass


def _normalize(
    output: dict[str, Any],
) -> tuple[list[list[float]], list[dict[str, float]]]:
    dense = [[float(value) for value in vector] for vector in output["dense_vecs"]]
    sparse = [
        {str(token): float(weight) for token, weight in weights.items()}
        for weights in output["lexical_weights"]
    ]
    return dense, sparse


@dataclass
class _Item:
    text: str
    priority: int
    future: Future = field(default_factory=Future)
    # 요청의 첫 텍스트에만 건다. 첫 텍스트가 시작되면 나머지는 기다림 없이 이어서 돈다.
    deadline: float | None = None
    enqueued_at: float = 0.0


@dataclass(frozen=True)
class _TextResult:
    dense: list[float]
    sparse: dict[str, float]
    queue_wait_s: float
    infer_s: float


@dataclass(frozen=True)
class EncodeResult:
    dense: list[list[float]]
    sparse: list[dict[str, float]]
    queue_wait_ms: float
    infer_ms: float
    queue_len: int


class BgeM3Encoder:
    def __init__(
        self,
        model_id: str = MODEL_ID,
        *,
        max_index_batch: int = MAX_INDEX_BATCH,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_index_batch < 1:
            raise ValueError("max_index_batch must be at least 1")
        self.model_id = model_id
        self._model = None
        self._max_index_batch = max_index_batch
        self._clock = clock
        self._cond = threading.Condition()
        self._heap: list[tuple[int, int, _Item]] = []
        self._seq = itertools.count()
        self._thread: threading.Thread | None = None
        self._stopping = False

    @property
    def ready(self) -> bool:
        return (
            self._model is not None
            and self._thread is not None
            and self._thread.is_alive()
        )

    def pending(self) -> int:
        with self._cond:
            return len(self._heap)

    def load(self) -> None:
        # CPU 전용 컨테이너라 fp16을 쓰지 않는다.
        from FlagEmbedding import BGEM3FlagModel

        self._model = BGEM3FlagModel(self.model_id, use_fp16=False)

    def start(self) -> None:
        with self._cond:
            if self._model is None or (self._thread and self._thread.is_alive()):
                return
            self._stopping = False
            self._thread = threading.Thread(
                target=self._run, name="embed-worker", daemon=True
            )
            self._thread.start()

    def stop(self) -> None:
        with self._cond:
            self._stopping = True
            queued = [entry[2] for entry in self._heap]
            self._heap.clear()
            thread = self._thread
            self._cond.notify_all()
        # 대기하던 요청이 영원히 걸려 있지 않게 깨운다. 진행 중인 묶음은 끝나고 멈춘다.
        for item in queued:
            item.future.cancel()
        if thread:
            thread.join()
        self._thread = None

    def submit(
        self,
        texts: list[str],
        priority: Priority = "index",
        max_wait: float | None = None,
    ) -> list[Future]:
        deadline = None if max_wait is None else self._clock() + max_wait
        return self._enqueue(texts, priority, deadline)

    def encode(
        self,
        texts: list[str],
        max_wait: float | None = None,
        priority: Priority = "index",
    ) -> EncodeResult:
        if not self.ready:
            raise ModelNotLoadedError(f"Model '{self.model_id}' is not loaded")

        # FastAPI가 동기 핸들러를 스레드풀에서 돌리므로 요청이 겹치면 추론도 겹친다.
        # CPU에서는 서로 코어를 뺏어 느려지기만 하니 추론은 워커 하나만 한다.
        # 요청을 텍스트 단위로 쪼개 우선순위 큐에 넣어, 32개짜리 색인 요청 하나가
        # 끝날 때까지 질의가 기다리는 일이 없게 한다.
        queue_len = self.pending()
        deadline = None if max_wait is None else self._clock() + max_wait
        futures = self._enqueue(texts, priority, deadline)
        self._await_start(futures, deadline)
        return self._collect(futures, queue_len)

    def _enqueue(
        self, texts: list[str], priority: Priority, deadline: float | None
    ) -> list[Future]:
        now = self._clock()
        items = [
            _Item(
                text=text,
                priority=_RANK[priority],
                deadline=deadline if index == 0 else None,
                enqueued_at=now,
            )
            for index, text in enumerate(texts)
        ]
        with self._cond:
            for item in items:
                heapq.heappush(self._heap, (item.priority, next(self._seq), item))
            self._cond.notify()
        return [item.future for item in items]

    def _await_start(self, futures: list[Future], deadline: float | None) -> None:
        """첫 텍스트가 시작되기 전까지만 마감을 적용한다. 시작된 뒤에는 건드리지 않는다."""
        if not futures or deadline is None:
            return
        first = futures[0]
        try:
            # wait()는 cancel()로는 안 깨어난다. stop()이 대기자를 바로 깨우려면 exception()이어야 한다.
            first.exception(timeout=max(0.0, deadline - self._clock()))
        except TimeoutError:
            # 취소가 성공하면 아직 대기 중이었다는 뜻이다. 이미 돌고 있으면 False라 계속 기다린다.
            if first.cancel():
                self._cancel_all(futures)
                raise ModelBusyError("Embedding queue wait exceeded the deadline")
        except CancelledError:
            pass  # _collect가 busy로 바꾼다

    def _collect(self, futures: list[Future], queue_len: int = 0) -> EncodeResult:
        results: list[_TextResult] = []
        for future in futures:
            try:
                results.append(future.result())
            except CancelledError as error:
                self._cancel_all(futures)
                raise ModelBusyError("Embedding request was cancelled") from error
            except Exception:
                self._cancel_all(futures)
                raise
        return EncodeResult(
            dense=[r.dense for r in results],
            sparse=[r.sparse for r in results],
            queue_wait_ms=results[0].queue_wait_s * 1000 if results else 0.0,
            infer_ms=sum(r.infer_s for r in results) * 1000,
            queue_len=queue_len,
        )

    @staticmethod
    def _cancel_all(futures: list[Future]) -> None:
        for future in futures:
            future.cancel()

    def _run(self) -> None:
        while (batch := self._next_batch()) is not None:
            self._process(batch)

    def _next_batch(self) -> list[_Item] | None:
        with self._cond:
            while True:
                if self._stopping:
                    return None
                # 취소된 항목은 "대기 중인 질의가 있는가" 판단에서 빠져야 한다.
                while self._heap and self._heap[0][2].future.cancelled():
                    heapq.heappop(self._heap)
                if self._heap:
                    break
                self._cond.wait()
            # 질의가 맨 앞이면 하나씩, 맨 앞이 색인이면 대기 중인 질의가 없다는 뜻이라 묶는다.
            limit = 1 if self._heap[0][0] == _RANK["query"] else self._max_index_batch
            batch: list[_Item] = []
            while self._heap and len(batch) < limit:
                item = heapq.heappop(self._heap)[2]
                if not item.future.cancelled():
                    batch.append(item)
            return batch

    def _process(self, batch: list[_Item]) -> None:
        live: list[_Item] = []
        for item in batch:
            if not item.future.set_running_or_notify_cancel():
                continue
            if item.deadline is not None and self._clock() >= item.deadline:
                item.future.set_exception(
                    ModelBusyError("Embedding queue wait exceeded the deadline")
                )
                continue
            live.append(item)
        if not live:
            return
        started = self._clock()
        try:
            output = self._model.encode(
                [item.text for item in live],
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )
            dense, sparse = _normalize(output)
        except Exception as error:
            # 워커가 죽으면 이후 모든 요청이 멈춘다. 예외는 요청에 돌려주고 계속 돈다.
            logger.exception("embedding inference failed")
            for item in live:
                item.future.set_exception(error)
            return
        infer_s = (self._clock() - started) / len(live)
        for index, item in enumerate(live):
            item.future.set_result(
                _TextResult(
                    dense=dense[index],
                    sparse=sparse[index],
                    queue_wait_s=started - item.enqueued_at,
                    infer_s=infer_s,
                )
            )


# 라우터와 기동 코드가 같은 인스턴스를 본다.
encoder = BgeM3Encoder()
