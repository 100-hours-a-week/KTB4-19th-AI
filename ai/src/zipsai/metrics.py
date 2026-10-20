"""Prometheus 지표. 워커는 1개라 multiprocess 모드는 쓰지 않는다."""

import os

from prometheus_client import Counter, Gauge, Histogram, generate_latest

SERVICE = "ai-api"

BUILD_INFO = Gauge(
    "zipsai_build_info",
    "Running build. The sample value is always 1.",
    ["service", "version"],
)
BUILD_INFO.labels(
    service=SERVICE, version=os.getenv("ZIPSAI_VERSION") or "unknown"
).set(1)

# 운영 목표(초). 경계 안쪽 비율로 목표 이내 여부를 바로 센다.
_REQUEST_BUCKETS = (0.5, 1, 2, 4, 6, 12, 15, 18, 20, 25, 30, 60, 120)
_STAGE_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6, 10, 12, 18, 30)
_TTFT_BUCKETS = (0.25, 0.5, 1, 1.5, 2, 2.5, 3, 5, 10)

HTTP_REQUESTS = Counter(
    "zipsai_http_requests_total",
    "HTTP requests",
    ["service", "method", "path", "status"],
)
HTTP_DURATION = Histogram(
    "zipsai_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["service", "method", "path"],
    buckets=_REQUEST_BUCKETS,
)
IN_FLIGHT = Gauge(
    "zipsai_requests_in_flight",
    "HTTP requests being handled",
    ["service"],
)
CONVERSE_DURATION = Histogram(
    "zipsai_converse_duration_seconds",
    "Converse request duration in seconds",
    ["route", "outcome"],
    buckets=_REQUEST_BUCKETS,
)
STAGE_DURATION = Histogram(
    "zipsai_stage_duration_seconds",
    "Stage duration in seconds",
    ["stage", "outcome"],
    buckets=_STAGE_BUCKETS,
)
STAGE_FAILURES = Counter(
    "zipsai_stage_failures_total",
    "Stage failures",
    ["stage", "error_code"],
)
LLM_TOKENS = Counter(
    "zipsai_llm_tokens_total",
    "LLM or VLM tokens",
    ["purpose", "model", "kind"],
)
LLM_COST = Counter(
    "zipsai_llm_cost_usd_total",
    "LLM or VLM cost in USD",
    ["purpose", "model"],
)
LLM_DURATION = Histogram(
    "zipsai_llm_call_duration_seconds",
    "LLM or VLM call duration in seconds",
    ["purpose", "model", "outcome"],
    buckets=_STAGE_BUCKETS,
)
LLM_TTFT = Histogram(
    "zipsai_llm_ttft_seconds",
    "Time to first token in seconds",
    ["purpose", "model"],
    buckets=_TTFT_BUCKETS,
)
LLM_STRUCTURED_FAILURES = Counter(
    "zipsai_llm_structured_output_failures_total",
    "Structured output that failed validation",
    ["purpose", "model"],
)
GATE = Counter("zipsai_gate_decisions_total", "Retrieval gate results", ["passed"])
FALLBACKS = Counter("zipsai_fallbacks_total", "Answer fallbacks", ["reason"])
ASK_REPEATS = Counter(
    "zipsai_complaint_ask_repeats_total",
    "Complaint questions repeated for the same field",
    ["field"],
)
INDEX_JOBS = Counter(
    "zipsai_index_jobs_total",
    "Indexing jobs",
    ["outcome", "failed_stage"],
)
TABLES = Counter("zipsai_parse_tables_total", "Document tables", ["result"])


def metrics_payload() -> bytes:
    return generate_latest()


def start_http() -> None:
    IN_FLIGHT.labels(service=SERVICE).inc()


def finish_http(method: str, path: str, status: int, duration_seconds: float) -> None:
    IN_FLIGHT.labels(service=SERVICE).dec()
    HTTP_REQUESTS.labels(
        service=SERVICE, method=method, path=path, status=str(status)
    ).inc()
    HTTP_DURATION.labels(service=SERVICE, method=method, path=path).observe(
        duration_seconds
    )


def observe_stage(
    stage: str,
    outcome: str,
    duration_seconds: float,
    error_code: str | None = None,
) -> None:
    STAGE_DURATION.labels(stage=stage, outcome=outcome).observe(duration_seconds)
    if outcome == "fail":
        STAGE_FAILURES.labels(stage=stage, error_code=error_code or "unknown").inc()


def observe_converse(route: str, outcome: str, duration_seconds: float) -> None:
    CONVERSE_DURATION.labels(route=route or "unknown", outcome=outcome).observe(
        duration_seconds
    )


def observe_llm(
    *,
    purpose: str,
    model: str,
    outcome: str,
    duration_seconds: float,
    ttft_seconds: float | None,
    tokens: dict[str, int | None],
    cost_usd: float | None,
    structured_output_ok: bool | None,
) -> None:
    purpose = purpose or "unknown"
    model = model or "unknown"
    LLM_DURATION.labels(purpose=purpose, model=model, outcome=outcome).observe(
        duration_seconds
    )
    if ttft_seconds is not None:
        LLM_TTFT.labels(purpose=purpose, model=model).observe(ttft_seconds)
    for kind, amount in tokens.items():
        if amount:
            LLM_TOKENS.labels(purpose=purpose, model=model, kind=kind).inc(amount)
    if cost_usd:
        LLM_COST.labels(purpose=purpose, model=model).inc(cost_usd)
    if structured_output_ok is False:
        LLM_STRUCTURED_FAILURES.labels(purpose=purpose, model=model).inc()


def observe_gate(passed: bool) -> None:
    GATE.labels(passed="true" if passed else "false").inc()


def observe_fallback(reason: str) -> None:
    FALLBACKS.labels(reason=reason or "unknown").inc()


def observe_ask_repeat(field: str) -> None:
    ASK_REPEATS.labels(field=field or "unknown").inc()


def observe_index_job(outcome: str, failed_stage: str | None) -> None:
    INDEX_JOBS.labels(outcome=outcome, failed_stage=failed_stage or "none").inc()


def observe_tables(*, parsed: int, failed: int | None) -> None:
    if parsed:
        TABLES.labels(result="parsed").inc(parsed)
    if failed:
        TABLES.labels(result="failed").inc(failed)
