"""Prometheus 지표. 워커는 1개라 multiprocess 모드는 쓰지 않는다."""

from prometheus_client import Counter, Gauge, Histogram, generate_latest

SERVICE = "embedding"

# AI와 같은 요청·단계 경계. 임베딩 추론은 단계 쪽에 가깝다.
_REQUEST_BUCKETS = (0.5, 1, 2, 4, 6, 12, 15, 18, 20, 25, 30, 60, 120)
_STAGE_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6, 10, 12, 18, 30)

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
EMBED_REQUESTS = Counter(
    "zipsai_embed_requests_total",
    "Embedding requests",
    ["outcome"],
)
EMBED_DURATION = Histogram(
    "zipsai_embed_duration_seconds",
    "Embedding request duration in seconds",
    ["outcome"],
    buckets=_STAGE_BUCKETS,
)
# #126 이후 락이 아니라 우선순위 큐다. 대기 시간은 이 지표에 넣는다.
EMBED_LOCK_WAIT = Histogram(
    "zipsai_embed_lock_wait_seconds",
    "Time waiting for the embedding worker",
    buckets=_STAGE_BUCKETS,
)
EMBED_TEXTS = Counter("zipsai_embed_texts_total", "Texts submitted for embedding")


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


def record_embed(
    *,
    outcome: str,
    texts: int,
    duration_seconds: float,
    queue_wait_seconds: float | None,
) -> None:
    EMBED_REQUESTS.labels(outcome=outcome).inc()
    EMBED_DURATION.labels(outcome=outcome).observe(duration_seconds)
    if texts:
        EMBED_TEXTS.inc(texts)
    if queue_wait_seconds is not None:
        EMBED_LOCK_WAIT.observe(queue_wait_seconds)
