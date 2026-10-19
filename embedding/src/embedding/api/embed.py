import logging
import os
import time
from functools import lru_cache

from fastapi import APIRouter, HTTPException
from langfuse import Langfuse
from pydantic import BaseModel

from embedding.encoder import MODEL_ID, ModelBusyError, Priority, encoder
from embedding.metrics import record_embed

logger = logging.getLogger(__name__)
router = APIRouter()

# CPU 추론이라 한 요청이 쥐는 메모리와 시간을 여기서 묶는다.
# 호출 측(ai-api)이 이 크기로 잘라 보내며, 값을 올리면 양쪽을 같이 올려야 한다.
MAX_BATCH_SIZE = 32
# 청크는 400자 기준이라 정상 요청은 여기 닿지 않는다. 오용을 막는 상한이다.
MAX_TEXT_CHARS = 8000


class EmbedRequest(BaseModel):
    texts: list[str]
    trace_id: str
    max_wait_seconds: float | None = None
    priority: Priority = "index"


class EmbedResponse(BaseModel):
    dense: list[list[float]]
    sparse: list[dict[str, float]]


@lru_cache(maxsize=1)
def _tracing_client() -> Langfuse | None:
    if os.getenv("LANGFUSE_TRACING_ENABLED", "true").lower() == "false" or not (
        os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")
    ):
        return None
    return Langfuse()


def _size(texts: list[str]) -> dict[str, int]:
    return {
        "texts": len(texts),
        "total_chars": sum(map(len, texts)),
        "max_chars": max((len(text) for text in texts), default=0),
    }


def _log_embed(
    event: str,
    payload: EmbedRequest,
    *,
    outcome: str,
    started: float,
    status_code: int,
    error_code: str | None = None,
    queue_wait_ms: float | None = None,
    infer_ms: float | None = None,
    queue_len: int | None = None,
) -> None:
    logger.info(
        event,
        extra={
            "trace_id": payload.trace_id,
            **_size(payload.texts),
            "priority": payload.priority,
            "model": MODEL_ID,
            "device": "cpu",
            "load_avg": [round(value, 2) for value in os.getloadavg()],
            "outcome": outcome,
            "status_code": status_code,
            "error_code": error_code,
            "duration_ms": int((time.perf_counter() - started) * 1000),
            "queue_wait_ms": queue_wait_ms,
            "encode_ms": infer_ms,
            "infer_ms": infer_ms,
            "queue_len": queue_len,
        },
    )


@router.post("/embed", response_model=EmbedResponse)
def embed(payload: EmbedRequest) -> EmbedResponse:
    client = _tracing_client()
    observation = (
        client.start_observation(
            as_type="embedding",
            name="embed-texts",
            model=MODEL_ID,
            input={
                "batch_size": len(payload.texts),
                "total_chars": sum(map(len, payload.texts)),
            },
            metadata={"trace_id": payload.trace_id},
        )
        if client
        else None
    )
    started = time.perf_counter()
    _log_embed("embed_request", payload, outcome="started", started=started, status_code=0)
    try:
        if not encoder.ready:
            raise HTTPException(status_code=503, detail="Model is still loading")
        if len(payload.texts) > MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=422,
                detail=f"texts must hold at most {MAX_BATCH_SIZE} items",
            )
        if any(len(text) > MAX_TEXT_CHARS for text in payload.texts):
            raise HTTPException(
                status_code=422,
                detail=f"each text must be at most {MAX_TEXT_CHARS} characters",
            )
        if payload.texts:
            result = encoder.encode(
                payload.texts,
                max_wait=payload.max_wait_seconds,
                priority=payload.priority,
            )
            dense, sparse = result.dense, result.sparse
            queue_wait_ms = result.queue_wait_ms
            infer_ms = result.infer_ms
            queue_len = result.queue_len
        else:
            dense, sparse = [], []
            queue_wait_ms = None
            infer_ms = None
            queue_len = None
    except HTTPException as error:
        if observation:
            observation.update(output={"status_code": error.status_code}, level="ERROR")
        outcome = "rejected" if error.status_code == 422 else "unavailable"
        _log_embed(
            "embed_failed",
            payload,
            outcome=outcome,
            started=started,
            status_code=error.status_code,
            error_code=outcome,
        )
        record_embed(
            outcome=outcome,
            texts=len(payload.texts),
            duration_seconds=(time.perf_counter() - started),
            queue_wait_seconds=None,
        )
        raise
    except ModelBusyError as error:
        if observation:
            observation.update(output={"status_code": 503}, level="ERROR")
        _log_embed(
            "embed_failed",
            payload,
            outcome="busy",
            started=started,
            status_code=503,
            error_code="busy",
        )
        record_embed(
            outcome="busy",
            texts=len(payload.texts),
            duration_seconds=(time.perf_counter() - started),
            queue_wait_seconds=None,
        )
        raise HTTPException(status_code=503, detail="Embedding model is busy") from error
    except Exception:
        if observation:
            observation.update(output={"status_code": 500}, level="ERROR")
        _log_embed(
            "embed_failed",
            payload,
            outcome="error",
            started=started,
            status_code=500,
            error_code="error",
        )
        record_embed(
            outcome="error",
            texts=len(payload.texts),
            duration_seconds=(time.perf_counter() - started),
            queue_wait_seconds=None,
        )
        raise
    else:
        if observation:
            observation.update(output={"status_code": 200, "vectors": len(dense)})
        _log_embed(
            "embed_done",
            payload,
            outcome="ok",
            started=started,
            status_code=200,
            queue_wait_ms=queue_wait_ms,
            infer_ms=infer_ms,
            queue_len=queue_len,
        )
        record_embed(
            outcome="ok",
            texts=len(payload.texts),
            duration_seconds=(time.perf_counter() - started),
            queue_wait_seconds=None if queue_wait_ms is None else queue_wait_ms / 1000,
        )
        return EmbedResponse(dense=dense, sparse=sparse)
    finally:
        if observation:
            observation.end()
