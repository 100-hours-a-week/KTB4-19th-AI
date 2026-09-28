import logging

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from zipsai.errors import (
    ComplaintExtractionError,
    EmbeddingError,
    IntentClassificationError,
    LlmRateLimitedError,
    LlmTimeoutError,
    LlmUnavailableError,
    LlmUpstreamError,
    VectorStoreError,
)

logger = logging.getLogger(__name__)


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
    EmbeddingError: (
        503,
        "DEPENDENCY_NOT_READY",
        "Embedding service is unavailable",
    ),
    VectorStoreError: (
        503,
        "DEPENDENCY_NOT_READY",
        "Vector store is unavailable",
    ),
}


def agent_error_response(error: Exception, trace_id: str) -> JSONResponse:
    status_code, code, detail = _AGENT_ERROR_MAPPING[type(error)]
    # 예외를 JSON으로 바꾸고 로그를 남기지 않으면 무엇이 터졌는지가 영영 사라진다.
    logger.warning(
        "agent_error",
        extra={
            "trace_id": trace_id,
            "status": status_code,
            "code": code,
            "error_type": type(error).__name__,
            "error": str(error),
        },
    )
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
    # 어느 필드가 어긋났는지 남기지 않으면 백엔드와 계약을 맞출 근거가 없다.
    logger.warning(
        "validation_error",
        extra={
            "trace_id": trace_id if isinstance(trace_id, str) else None,
            "status": 400 if bad_request else 422,
            "fields": [
                ".".join(str(part) for part in item["loc"]) for item in error.errors()
            ],
            "error_types": [item["type"] for item in error.errors()],
        },
    )
    return error_response(
        status_code=400 if bad_request else 422,
        code="MISSING_REQUIRED_FIELD" if bad_request else "VALIDATION_ERROR",
        detail="Request validation failed",
        trace_id=trace_id if isinstance(trace_id, str) else None,
        retryable=False,
    )


async def unhandled_exception_handler(
    request: Request,
    error: Exception,
) -> JSONResponse:
    trace_id = getattr(request.state, "trace_id", None)
    # 500은 우리가 예상하지 못한 경로다. 스택트레이스가 유일한 단서다.
    logger.exception(
        "unhandled_error",
        exc_info=error,
        extra={
            "trace_id": trace_id if isinstance(trace_id, str) else None,
            "status": 500,
            "error_type": type(error).__name__,
            "path": request.url.path,
        },
    )
    return error_response(
        status_code=500,
        code="INTERNAL_SERVER_ERROR",
        detail="Unexpected server error",
        trace_id=trace_id if isinstance(trace_id, str) else None,
        retryable=False,
    )
