import os
import threading
from typing import Any

MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "BAAI/bge-m3")


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


class BgeM3Encoder:
    def __init__(self, model_id: str = MODEL_ID) -> None:
        self.model_id = model_id
        self._model = None
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        # CPU 전용 컨테이너라 fp16을 쓰지 않는다.
        from FlagEmbedding import BGEM3FlagModel

        self._model = BGEM3FlagModel(self.model_id, use_fp16=False)

    def encode(
        self, texts: list[str], max_wait: float | None = None
    ) -> tuple[list[list[float]], list[dict[str, float]]]:
        if self._model is None:
            raise ModelNotLoadedError(f"Model '{self.model_id}' is not loaded")

        # FastAPI가 동기 핸들러를 스레드풀에서 돌리므로 요청이 겹치면 추론도 겹친다.
        # CPU에서는 서로 코어를 뺏어 느려지기만 하니 한 번에 한 배치만 돌린다.
        # 처리량이 모자라면 요청을 모아 한 번에 추론하는 배치 큐로 바꾼다.
        # max_wait=None이면 색인 배치처럼 끝날 때까지 기다린다. 질의는 화면 앞에서
        # 기다리는 사람이 있어 짧은 한도를 주고, 못 잡으면 바로 포기시킨다.
        acquired = self._lock.acquire(timeout=max_wait if max_wait is not None else -1)
        if not acquired:
            raise ModelBusyError("Embedding model is busy with another batch")
        try:
            output = self._model.encode(
                texts,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )
        finally:
            self._lock.release()
        return _normalize(output)


# 라우터와 기동 코드가 같은 인스턴스를 본다.
encoder = BgeM3Encoder()
