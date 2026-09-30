import pytest
from fastapi.testclient import TestClient
from langfuse import Langfuse
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from embedding.api import embed as embed_api
from embedding.api import health as health_api
from embedding.api.embed import MAX_BATCH_SIZE, MAX_TEXT_CHARS
from embedding.main import app


class FakeEncoder:
    def __init__(self, *, ready: bool = True) -> None:
        self.ready = ready
        self.calls: list[list[str]] = []

    def encode(
        self, texts: list[str]
    ) -> tuple[list[list[float]], list[dict[str, float]]]:
        self.calls.append(texts)
        return [[0.1, 0.2] for _ in texts], [{"7": 0.5} for _ in texts]


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


def test_embed_rejects_a_text_over_the_character_limit(
    client: TestClient, encoder: FakeEncoder
) -> None:
    response = client.post(
        "/embed", json={"texts": ["가" * (MAX_TEXT_CHARS + 1)], "trace_id": "trace-001"}
    )

    assert response.status_code == 422
    assert encoder.calls == []
