import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from zipsai.api.converse import router as converse_router
from zipsai.api.error_responses import (
    request_validation_error_handler,
    unhandled_exception_handler,
)
from zipsai.api.health import router as health_router
from zipsai.api.indexing import router as indexing_router
from zipsai.settings import API_PREFIX, missing_required_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    missing = missing_required_settings()
    if missing:
        raise RuntimeError(f"Missing required settings: {', '.join(missing)}")
    yield


app = FastAPI(title="zipsai", lifespan=lifespan)
app.include_router(converse_router)
app.include_router(health_router)
app.include_router(indexing_router, prefix=API_PREFIX)
app.add_exception_handler(RequestValidationError,
request_validation_error_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)
