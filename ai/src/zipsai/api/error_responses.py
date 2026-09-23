from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from zipsai.errors import (
    ComplaintExtractionError,
    IntentClassificationError,
    LlmRateLimitedError,
    LlmTimeoutError,
    LlmUnavailableError,
    LlmUpstreamError,
)


class ErrorDetail(BaseModel):
    code: str
    detail: str
    retryable: bool
    retry_after_seconds: int | None = None


class ErrorResponse(BaseModel):
    message: str
    error: ErrorDetail
    trace_id: str | None


# exception type -> (status_code, code, detail)
_AGENT_ERROR_MAPPING: dict[type[Exception], tuple[int, str, str]] = {
    LlmRateLimitedError: (429, "MODEL_RATE_LIMITED", "AI model rate limit exceeded"),
    LlmTimeoutError: (504, "MODEL_TIMEOUT", "AI model did not respond in time"),
    LlmUpstreamError: (
        502,
        "MODEL_UPSTREAM_ERROR",
        "AI model provider returned an upstream error",
    ),
    LlmUnavailableError: (503, "DEPENDENCY_NOT_READY", "AI model is unavailable"),
    IntentClassificationError: (
        502,
        "MODEL_UPSTREAM_ERROR",
        "AI model returned an unsupported route",
    ),
    ComplaintExtractionError: (
        502,
        "MODEL_UPSTREAM_ERROR",
        "AI model returned an invalid complaint draft",
    ),
}


def agent_error_response(error: Exception, trace_id: str) -> JSONResponse:
    status_code, code, detail = _AGENT_ERROR_MAPPING[type(error)]
    return error_response(
        status_code=status_code,
        code=code,
        detail=detail,
        trace_id=trace_id,
        retryable=True,
        retry_after_seconds=getattr(error, "retry_after_seconds", None),
    )


def error_response(
    *,
    status_code: int,
    code: str,
    detail: str,
    trace_id: str | None,
    retryable: bool,
    retry_after_seconds: int | None = None,
) -> JSONResponse:
    body = ErrorResponse(
        message="ai_response_error",
        error=ErrorDetail(
            code=code,
            detail=detail,
            retryable=retryable,
            retry_after_seconds=retry_after_seconds,
        ),
        trace_id=trace_id,
    )
    return JSONResponse(
        status_code=status_code, content=body.model_dump(exclude_none=True)
    )


async def request_validation_error_handler(
    _request: Request,
    error: RequestValidationError,
) -> JSONResponse:
    bad_request = any(
        item["type"] in {"missing", "json_invalid"} for item in error.errors()
    )
    body = error.body
    trace_id = body.get("trace_id") if isinstance(body, dict) else None
    return error_response(
        status_code=400 if bad_request else 422,
        code="MISSING_REQUIRED_FIELD" if bad_request else "VALIDATION_ERROR",
        detail="Request validation failed",
        trace_id=trace_id if isinstance(trace_id, str) else None,
        retryable=False,
    )


async def unhandled_exception_handler(
    request: Request,
    _error: Exception,
) -> JSONResponse:
    trace_id = getattr(request.state, "trace_id", None)
    return error_response(
        status_code=500,
        code="INTERNAL_SERVER_ERROR",
        detail="Unexpected server error",
        trace_id=trace_id if isinstance(trace_id, str) else None,
        retryable=False,
    )
