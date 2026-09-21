from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from zipsai.api.converse import router as converse_router
from zipsai.api.error_responses import (
    request_validation_error_handler,
    unhandled_exception_handler,
)
from zipsai.api.health import router as health_router

app = FastAPI(title="zipsai")
app.include_router(converse_router)
app.include_router(health_router)
app.add_exception_handler(RequestValidationError, request_validation_error_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)
