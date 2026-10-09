"""Langfuse tracing with request content removed before export."""

import os
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache

from langfuse import Langfuse, propagate_attributes
from langfuse.types import MaskOtelSpansParams, MaskOtelSpansResult, OtelSpanPatch

from zipsai import settings  # noqa: F401 - Load .env before reading Langfuse keys.
from zipsai.contracts.converse import ConverseRequest
from zipsai.contracts.indexing import IndexingJobRequest

_in_trace: ContextVar[bool] = ContextVar("langfuse_in_trace", default=False)


def _mask_otel_spans(*, params: MaskOtelSpansParams) -> MaskOtelSpansResult:
    patches = {}
    for identifier, span in params.spans.items():
        model_call = span.attributes.get("langfuse.observation.type") in {
            "generation",
            "embedding",
        }
        private = tuple(
            key
            for key in span.attributes
            if key.startswith("exception.")
            or key in {"otel.status_description", "langfuse.observation.status_message"}
            or (
                model_call
                and (
                    key
                    in {
                        "langfuse.observation.input",
                        "langfuse.observation.output",
                        "langfuse.trace.input",
                        "langfuse.trace.output",
                    }
                    or key.startswith(("gen_ai.prompt", "gen_ai.completion"))
                )
            )
        )
        if private:
            patches[identifier] = OtelSpanPatch(delete_attributes=private)
    return MaskOtelSpansResult(span_patches=patches)


@lru_cache(maxsize=1)
def get_tracing_client() -> Langfuse | None:
    if os.getenv("LANGFUSE_TRACING_ENABLED", "true").lower() == "false" or not (
        os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")
    ):
        return None
    return Langfuse(mask_otel_spans=_mask_otel_spans)


@contextmanager
def trace_turn(request: ConverseRequest):
    client = get_tracing_client()
    if client is None:
        yield None
        return
    with (
        client.start_as_current_observation(
            name="converse-turn",
            input={
                "has_text": bool(request.message.text),
                "image_count": len(request.message.images),
                "history_turns": len(request.conversation_history),
            },
            metadata={"turn_id": request.turn_id, "trace_id": request.trace_id},
        ) as span,
        propagate_attributes(session_id=request.conversation_id, tags=["converse"]),
    ):
        token = _in_trace.set(True)
        try:
            yield span
        finally:
            _in_trace.reset(token)


@contextmanager
def trace_job(request: IndexingJobRequest, job_id: str):
    client = get_tracing_client()
    if client is None:
        yield None
        return
    with (
        client.start_as_current_observation(
            name="indexing-job",
            input={
                "has_document": request.has_document,
                "reconcile_count": len(request.valid_doc_ids or []),
            },
            metadata={"job_id": job_id, "trace_id": request.trace_id},
        ) as span,
        propagate_attributes(tags=["indexing"]),
    ):
        token = _in_trace.set(True)
        try:
            yield span
        finally:
            _in_trace.reset(token)


@contextmanager
def trace_step(name: str):
    client = get_tracing_client() if _in_trace.get() else None
    if client is None:
        yield None
        return
    with client.start_as_current_observation(name=name.replace("_", "-")) as span:
        yield span
