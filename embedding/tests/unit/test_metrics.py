from fastapi.testclient import TestClient

from embedding.main import app

_NAMES = (
    "zipsai_http_requests_total",
    "zipsai_http_request_duration_seconds",
    "zipsai_requests_in_flight",
    "zipsai_embed_requests_total",
    "zipsai_embed_duration_seconds",
    "zipsai_embed_lock_wait_seconds",
    "zipsai_embed_texts_total",
)


def test_metrics_lists_the_embedding_series_without_auth() -> None:
    response = TestClient(app).get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    for name in _NAMES:
        assert name in response.text


def _request_samples(body: str) -> list[str]:
    prefix = ("embedding_requests_total{", "zipsai_http_requests_total{")
    return [line for line in body.splitlines() if line.startswith(prefix)]


def test_health_does_not_record_an_embedding_request() -> None:
    client = TestClient(app)
    before = _request_samples(client.get("/metrics").text)
    client.get("/health")
    after = _request_samples(client.get("/metrics").text)

    assert before == []
    assert after == []
