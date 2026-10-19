import os

from fastapi.testclient import TestClient

from zipsai.main import app

_NAMES = (
    "zipsai_build_info",
    "zipsai_http_requests_total",
    "zipsai_http_request_duration_seconds",
    "zipsai_requests_in_flight",
    "zipsai_converse_duration_seconds",
    "zipsai_stage_duration_seconds",
    "zipsai_stage_failures_total",
    "zipsai_llm_tokens_total",
    "zipsai_llm_cost_usd_total",
    "zipsai_llm_call_duration_seconds",
    "zipsai_llm_ttft_seconds",
    "zipsai_llm_structured_output_failures_total",
    "zipsai_gate_decisions_total",
    "zipsai_fallbacks_total",
    "zipsai_complaint_ask_repeats_total",
    "zipsai_index_jobs_total",
    "zipsai_parse_tables_total",
)


def _requests_total(body: str) -> float:
    total = 0.0
    prefix = ("zipsai_http_requests_total{", "zipsai_http_requests_total ")
    for line in body.splitlines():
        if line.startswith(prefix):
            total += float(line.rsplit(" ", 1)[-1])
    return total


def test_metrics_lists_the_ai_series_without_auth() -> None:
    response = TestClient(app).get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    for name in _NAMES:
        assert name in response.text
    version = os.getenv("ZIPSAI_VERSION") or "unknown"
    assert (
        f'zipsai_build_info{{service="ai-api",version="{version}"}} 1.0'
        in response.text
    )


def test_health_does_not_increment_the_request_counter() -> None:
    client = TestClient(app)
    before = _requests_total(client.get("/metrics").text)
    client.get("/health")
    after = _requests_total(client.get("/metrics").text)

    assert after == before
