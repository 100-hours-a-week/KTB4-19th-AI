import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST

from zipsai.access import AccessLogMiddleware
from zipsai.api.comment import router as comment_router
from zipsai.api.converse import router as converse_router
from zipsai.api.error_responses import (
    request_validation_error_handler,
    unhandled_exception_handler,
)
from zipsai.api.health import router as health_router
from zipsai.api.indexing import router as indexing_router
from zipsai.integrations.runpod import get_locator
from zipsai.metrics import finish_http, metrics_payload, start_http
from zipsai.observability import configure_logging
from zipsai.settings import (
    ACTIVE_MODEL_PROVIDER,
    API_PREFIX,
    COMMENT_API_PREFIX,
    CORS_ORIGINS,
    missing_required_settings,
)
from zipsai.tracing import get_tracing_client

configure_logging()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    missing = missing_required_settings()
    if missing:
        raise RuntimeError(f"Missing required settings: {', '.join(missing)}")
    locator = get_locator() if ACTIVE_MODEL_PROVIDER == "pod" else None
    task = None
    try:
        if locator is not None:
            await locator.refresh()
            task = asyncio.create_task(locator.run())
        yield
    finally:
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        if locator is not None:
            await locator.aclose()
            get_locator.cache_clear()
        client = get_tracing_client()
        if client is not None:
            client.flush()


app = FastAPI(title="zipsai", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(CORS_ORIGINS),
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)
app.add_middleware(AccessLogMiddleware, record_http=finish_http, start_http=start_http)


@app.get("/metrics")
def metrics() -> Response:
    return Response(metrics_payload(), media_type=CONTENT_TYPE_LATEST)


app.include_router(converse_router)
app.include_router(health_router)
app.include_router(indexing_router, prefix=API_PREFIX)
app.include_router(comment_router, prefix=COMMENT_API_PREFIX)
app.add_exception_handler(RequestValidationError, request_validation_error_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)
