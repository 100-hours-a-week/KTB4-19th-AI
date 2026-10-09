import logging
from types import SimpleNamespace

import httpx
import pytest
from langfuse import Langfuse
from langfuse.openai import OpenAI
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from zipsai import tracing
from zipsai.api import indexing
from zipsai.contracts.indexing import IndexingJobRequest, JobStatus
from zipsai.observability import stage


def test_converse_trace_keeps_usage_and_excludes_private_content(monkeypatch):
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test-tracing")
    exporter = InMemorySpanExporter()
    client = Langfuse(
        public_key="pk-test-tracing",
        secret_key="sk-test-tracing",
        base_url="http://localhost:1",
        span_exporter=exporter,
        mask_otel_spans=tracing._mask_otel_spans,
    )
    monkeypatch.setattr(tracing, "get_tracing_client", lambda: client)

    def respond(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "PRIVATE_OUTPUT"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                },
            },
        )

    request = SimpleNamespace(
        message=SimpleNamespace(text="PRIVATE_INPUT", images=[]),
        conversation_history=[],
        conversation_id="conversation-1",
        turn_id="turn-1",
        trace_id="trace-converse-1",
    )
    openai_client = OpenAI(
        api_key="test",
        base_url="http://localhost:1",
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    )
    with tracing.trace_turn(request), tracing.trace_step("intent"):
        openai_client.chat.completions.create(
            model="test-model",
            messages=[{"role": "user", "content": "PRIVATE_INPUT"}],
            name="generate-text",
        )
    client.flush()

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == ["generate-text", "intent", "converse-turn"]
    generation = spans[0]
    assert generation.attributes["langfuse.observation.model.name"] == "test-model"
    assert (
        '"prompt_tokens": 3'
        in generation.attributes["langfuse.observation.usage_details"]
    )
    assert all("PRIVATE_" not in str(span.attributes) for span in spans)
    assert spans[2].attributes["session.id"] == "conversation-1"


@pytest.mark.parametrize(
    ("index_status", "expected_names", "outcome"),
    [
        (JobStatus.SUCCEEDED, ["setup", "download", "reconcile", "indexing-job"], "ok"),
        (JobStatus.FAILED, ["setup", "download", "indexing-job"], "fail"),
    ],
)
def test_background_indexing_job_traces_stages_without_document_data(
    monkeypatch, index_status, expected_names, outcome
):
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "true")
    public_key = f"pk-test-indexing-{index_status.value}"
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", public_key)
    exporter = InMemorySpanExporter()
    client = Langfuse(
        public_key=public_key,
        secret_key="sk-test-indexing",
        base_url="http://localhost:1",
        span_exporter=exporter,
        mask_otel_spans=tracing._mask_otel_spans,
    )
    monkeypatch.setattr(tracing, "get_tracing_client", lambda: client)
    monkeypatch.setattr(indexing, "_dependencies", lambda: (object(), object()))

    def fake_index(*_args, **_kwargs):
        with stage("download", logging.getLogger(__name__)):
            pass
        return index_status

    monkeypatch.setattr(indexing, "run_indexing_job", fake_index)
    monkeypatch.setattr(indexing, "delete_missing_documents", lambda *_: None)
    payload = IndexingJobRequest(
        building_id=101,
        trace_id="trace-indexing-1",
        doc_id="doc-1",
        title="PRIVATE_TITLE",
        file_key="PRIVATE_FILE_KEY",
        valid_doc_ids=["doc-1"],
    )
    indexing._run_job(payload, "job-1")
    client.flush()

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == expected_names
    assert all("PRIVATE_" not in str(span.attributes) for span in spans)
    assert (
        f'"outcome": "{outcome}"' in spans[-1].attributes["langfuse.observation.output"]
    )
