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
    error_code,
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
    turn_id: str | None


# exception type -> (status_code, detail). 오류 코드는 errors.ERROR_CODES가 단일 출처다.
_AGENT_ERROR_MAPPING: dict[type[Exception], tuple[int, str]] = {
    LlmRateLimitedError: (429, "AI model rate limit exceeded"),
    LlmTimeoutError: (504, "AI model did not respond in time"),
    LlmUpstreamError: (502, "AI model provider returned an upstream error"),
    LlmUnavailableError: (503, "AI model is unavailable"),
    IntentClassificationError: (502, "AI model returned an unsupported route"),
    ComplaintExtractionError: (502, "AI model returned an invalid complaint draft"),
    EmbeddingError: (503, "Embedding service is unavailable"),
    VectorStoreError: (503, "Vector store is unavailable"),
}


def agent_error_response(error: Exception, turn_id: str) -> JSONResponse:
    status_code, detail = _AGENT_ERROR_MAPPING[type(error)]
    code = error_code(error)
    # 예외를 JSON으로 바꾸고 로그를 남기지 않으면 무엇이 터졌는지가 영영 사라진다.
    logger.warning(
        "agent_error",
        extra={
            "turn_id": turn_id,
            "status_code": status_code,
            "error_code": code,
            "error_type": type(error).__name__,
            "error": str(error),
        },
    )
    return error_response(
        status_code=status_code,
        code=code,
        detail=detail,
        turn_id=turn_id,
        retryable=True,
        retry_after_seconds=getattr(error, "retry_after_seconds", None),
    )


def error_response(
    *,
    status_code: int,
    code: str,
    detail: str,
    turn_id: str | None,
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
        turn_id=turn_id,
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
    turn_id = body.get("turn_id") if isinstance(body, dict) else None
    status_code = 400 if bad_request else 422
    code = "MISSING_REQUIRED_FIELD" if bad_request else "VALIDATION_ERROR"
    # 어느 필드가 어긋났는지 남기지 않으면 백엔드와 계약을 맞출 근거가 없다.
    logger.warning(
        "validation_error",
        extra={
            "turn_id": turn_id if isinstance(turn_id, str) else None,
            "status_code": status_code,
            "error_code": code,
            "fields": [
                ".".join(str(part) for part in item["loc"]) for item in error.errors()
            ],
            "error_types": [item["type"] for item in error.errors()],
        },
    )
    return error_response(
        status_code=status_code,
        code=code,
        detail="Request validation failed",
        turn_id=turn_id if isinstance(turn_id, str) else None,
        retryable=False,
    )


async def unhandled_exception_handler(
    request: Request,
    error: Exception,
) -> JSONResponse:
    turn_id = getattr(request.state, "turn_id", None)
    # 500은 우리가 예상하지 못한 경로다. 스택트레이스가 유일한 단서다.
    logger.exception(
        "unhandled_error",
        exc_info=error,
        extra={
            "turn_id": turn_id if isinstance(turn_id, str) else None,
            "status_code": 500,
            "error_code": "INTERNAL_SERVER_ERROR",
            "error_type": type(error).__name__,
            "route": request.url.path,
        },
    )
    return error_response(
        status_code=500,
        code="INTERNAL_SERVER_ERROR",
        detail="Unexpected server error",
        turn_id=turn_id if isinstance(turn_id, str) else None,
        retryable=False,
    )
