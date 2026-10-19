import os

from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from embedding.main import app
from embedding.metrics import record_embed

_NAMES = (
    "zipsai_build_info",
    "zipsai_http_requests_total",
    "zipsai_http_request_duration_seconds",
    "zipsai_requests_in_flight",
    "zipsai_embed_requests_total",
    "zipsai_embed_duration_seconds",
    "zipsai_embed_queue_wait_seconds",
    "zipsai_embed_queue_length",
    "zipsai_embed_texts_total",
)


def test_metrics_lists_the_embedding_series_without_auth() -> None:
    response = TestClient(app).get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    for name in _NAMES:
        assert name in response.text
    version = os.getenv("ZIPSAI_VERSION") or "unknown"
    assert (
        f'zipsai_build_info{{service="embedding",version="{version}"}} 1.0'
        in response.text
    )
    assert "zipsai_embed_lock_wait_seconds" not in response.text


def test_record_embed_sets_queue_length_and_wait() -> None:
    before = REGISTRY.get_sample_value("zipsai_embed_queue_wait_seconds_count") or 0.0

    record_embed(
        outcome="ok",
        texts=1,
        duration_seconds=0.1,
        queue_wait_seconds=0.2,
        queue_len=4,
    )

    assert REGISTRY.get_sample_value("zipsai_embed_queue_length") == 4
    assert (
        REGISTRY.get_sample_value("zipsai_embed_queue_wait_seconds_count") or 0.0
    ) == before + 1


def _request_samples(body: str) -> list[str]:
    prefix = ("embedding_requests_total{", "zipsai_http_requests_total{")
    return [line for line in body.splitlines() if line.startswith(prefix)]


def test_health_does_not_record_an_embedding_request() -> None:
    client = TestClient(app)
    before = _request_samples(client.get("/metrics").text)
    client.get("/health")
    after = _request_samples(client.get("/metrics").text)

    assert after == before
