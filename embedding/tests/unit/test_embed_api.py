import pytest
from fastapi.testclient import TestClient
from langfuse import Langfuse
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from embedding.api import embed as embed_api
from embedding.api import health as health_api
from embedding.api.embed import MAX_BATCH_SIZE, MAX_TEXT_CHARS
from embedding.encoder import EncodeResult, ModelBusyError
from embedding.main import app


class FakeEncoder:
    def __init__(self, *, ready: bool = True, busy: bool = False) -> None:
        self.ready = ready
        self.busy = busy
        self.calls: list[list[str]] = []
        self.max_waits: list[float | None] = []
        self.priorities: list[str] = []

    def encode(
        self,
        texts: list[str],
        max_wait: float | None = None,
        priority: str = "index",
    ) -> EncodeResult:
        self.calls.append(texts)
        self.max_waits.append(max_wait)
        self.priorities.append(priority)
        if self.busy:
            raise ModelBusyError("busy")
        return EncodeResult(
            dense=[[0.1, 0.2] for _ in texts],
            sparse=[{"7": 0.5} for _ in texts],
            queue_wait_ms=12.4,
            infer_ms=34.6,
            queue_len=3,
        )


@pytest.fixture
def client() -> TestClient:
    # lifespan을 실행하지 않는다. 실행하면 실제 BGE-M3를 내려받는다.
    return TestClient(app)


@pytest.fixture
def encoder(monkeypatch: pytest.MonkeyPatch) -> FakeEncoder:
    fake = FakeEncoder()
    monkeypatch.setattr(embed_api, "encoder", fake)
    monkeypatch.setattr(health_api, "encoder", fake)
    return fake


def test_health_reports_ok_once_the_model_is_loaded(
    client: TestClient, encoder: FakeEncoder
) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_reports_unavailable_while_the_model_loads(
    client: TestClient, encoder: FakeEncoder
) -> None:
    encoder.ready = False

    assert client.get("/health").status_code == 503


def test_embed_returns_dense_and_sparse_for_every_text(
    client: TestClient, encoder: FakeEncoder
) -> None:
    response = client.post(
        "/embed", json={"texts": ["첫 청크", "둘째 청크"], "trace_id": "trace-001"}
    )

    assert response.status_code == 200
    assert response.json() == {
        "dense": [[0.1, 0.2], [0.1, 0.2]],
        "sparse": [{"7": 0.5}, {"7": 0.5}],
    }
    assert encoder.calls == [["첫 청크", "둘째 청크"]]


def test_embed_trace_records_counts_without_text_or_vectors(
    client: TestClient, encoder: FakeEncoder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "true")
    exporter = InMemorySpanExporter()
    langfuse = Langfuse(
        public_key="pk-test-embed",
        secret_key="sk-test-embed",
        base_url="http://localhost:1",
        span_exporter=exporter,
    )
    monkeypatch.setattr(embed_api, "_tracing_client", lambda: langfuse)

    response = client.post(
        "/embed", json={"texts": ["PRIVATE_TEXT"], "trace_id": "trace-001"}
    )
    langfuse.flush()

    assert response.status_code == 200
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "embed-texts"
    assert spans[0].attributes["langfuse.observation.type"] == "embedding"
    assert "PRIVATE_TEXT" not in str(spans[0].attributes)
    assert "0.1" not in str(spans[0].attributes)


def test_embed_returns_empty_result_without_calling_the_model(
    client: TestClient, encoder: FakeEncoder
) -> None:
    response = client.post("/embed", json={"texts": [], "trace_id": "trace-001"})

    assert response.status_code == 200
    assert response.json() == {"dense": [], "sparse": []}
    assert encoder.calls == []


def test_embed_reports_unavailable_while_the_model_loads(
    client: TestClient, encoder: FakeEncoder
) -> None:
    encoder.ready = False

    response = client.post("/embed", json={"texts": ["청크"], "trace_id": "trace-001"})

    assert response.status_code == 503
    assert encoder.calls == []


def test_embed_rejects_a_batch_over_the_limit(
    client: TestClient, encoder: FakeEncoder
) -> None:
    response = client.post(
        "/embed",
        json={"texts": ["청크"] * (MAX_BATCH_SIZE + 1), "trace_id": "trace-001"},
    )

    assert response.status_code == 422
    assert encoder.calls == []


def test_embed_passes_max_wait_seconds_to_the_encoder(
    client: TestClient, encoder: FakeEncoder
) -> None:
    response = client.post(
        "/embed",
        json={"texts": ["청크"], "trace_id": "trace-001", "max_wait_seconds": 3.5},
    )

    assert response.status_code == 200
    assert encoder.max_waits == [3.5]


def test_embed_without_priority_is_treated_as_index(
    client: TestClient, encoder: FakeEncoder
) -> None:
    response = client.post(
        "/embed",
        json={"texts": ["청크"], "trace_id": "trace-001", "max_wait_seconds": 3.5},
    )

    assert response.status_code == 200
    assert encoder.priorities == ["index"]
    assert encoder.max_waits == [3.5]


@pytest.mark.parametrize("priority", ["query", "index"])
def test_embed_passes_priority_to_the_encoder(
    client: TestClient, encoder: FakeEncoder, priority: str
) -> None:
    response = client.post(
        "/embed",
        json={"texts": ["청크"], "trace_id": "trace-001", "priority": priority},
    )

    assert response.status_code == 200
    assert encoder.priorities == [priority]


def test_embed_rejects_an_unknown_priority(
    client: TestClient, encoder: FakeEncoder
) -> None:
    response = client.post(
        "/embed",
        json={"texts": ["청크"], "trace_id": "trace-001", "priority": "urgent"},
    )

    assert response.status_code == 422
    assert encoder.calls == []


def test_embed_done_log_carries_queue_and_inference_timings(
    client: TestClient, encoder: FakeEncoder, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("INFO", logger=embed_api.logger.name):
        client.post(
            "/embed",
            json={"texts": ["청크"], "trace_id": "trace-001", "priority": "query"},
        )

    done = [record for record in caplog.records if record.getMessage() == "embed_done"]
    assert len(done) == 1
    assert done[0].trace_id == "trace-001"
    assert done[0].texts == 1
    assert done[0].priority == "query"
    assert done[0].queue_wait_ms == 12.4
    assert done[0].infer_ms == 34.6
    assert done[0].queue_len == 3
    assert "청크" not in done[0].getMessage()


def test_embed_reports_busy_when_the_lock_is_not_acquired_in_time(
    client: TestClient, encoder: FakeEncoder
) -> None:
    encoder.busy = True

    response = client.post(
        "/embed",
        json={"texts": ["청크"], "trace_id": "trace-001", "max_wait_seconds": 3.5},
    )

    assert response.status_code == 503


def test_embed_rejects_a_text_over_the_character_limit(
    client: TestClient, encoder: FakeEncoder
) -> None:
    response = client.post(
        "/embed", json={"texts": ["가" * (MAX_TEXT_CHARS + 1)], "trace_id": "trace-001"}
    )

    assert response.status_code == 422
    assert encoder.calls == []
