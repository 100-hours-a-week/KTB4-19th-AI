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
from zipsai.settings import COMMENT_API_PREFIX

logger = logging.getLogger(__name__)


class ErrorDetail(BaseModel):
    code: str
    detail: str
    retryable: bool
    retry_after_seconds: int | None = None
    invalid_items: list[dict[str, object]] | None = None


class ErrorResponse(BaseModel):
    message: str
    error: ErrorDetail
    turn_id: str | None
    trace_id: str | None


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


def agent_error_response(error: Exception, turn_id: str, trace_id: str) -> JSONResponse:
    status_code, detail = _AGENT_ERROR_MAPPING[type(error)]
    code = error_code(error)
    # 예외를 JSON으로 바꾸고 로그를 남기지 않으면 무엇이 터졌는지가 영영 사라진다.
    logger.warning(
        "agent_error",
        extra={
            "turn_id": turn_id,
            "trace_id": trace_id,
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
        trace_id=trace_id,
        retryable=True,
        retry_after_seconds=getattr(error, "retry_after_seconds", None),
    )


def error_response(
    *,
    status_code: int,
    code: str,
    detail: str,
    turn_id: str | None,
    trace_id: str | None,
    retryable: bool,
    retry_after_seconds: int | None = None,
    invalid_items: list[dict[str, object]] | None = None,
) -> JSONResponse:
    body = ErrorResponse(
        message="ai_response_error",
        error=ErrorDetail(
            code=code,
            detail=detail,
            retryable=retryable,
            retry_after_seconds=retry_after_seconds,
            invalid_items=invalid_items,
        ),
        turn_id=turn_id,
        trace_id=trace_id,
    )
    return JSONResponse(
        status_code=status_code, content=body.model_dump(exclude_none=True)
    )


async def request_validation_error_handler(
    request: Request,
    error: RequestValidationError,
) -> JSONResponse:
    is_comment = request.url.path.startswith(COMMENT_API_PREFIX)
    # 댓글 API는 필드 누락도 422가 된게 한다. 400은 JSON을 읽을 수 없을 때만 쓴다.
    bad_codes = {"json_invalid"} if is_comment else {"missing", "json_invalid"}
    bad_request = any(item["type"] in bad_codes for item in error.errors())
    body = error.body
    turn_id = body.get("turn_id") if isinstance(body, dict) else None
    trace_id = body.get("trace_id") if isinstance(body, dict) else None
    status_code = 400 if bad_request else 422
    code = "MISSING_REQUIRED_FIELD" if bad_request else "VALIDATION_ERROR"
    # 어느 필드가 어긋났는지 남기지 않으면 백엔드와 계약을 맞출 근거가 없다.
    logger.warning(
        "validation_error",
        extra={
            "turn_id": turn_id if isinstance(turn_id, str) else None,
            "trace_id": trace_id if isinstance(trace_id, str) else None,
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
        trace_id=trace_id if isinstance(trace_id, str) else None,
        retryable=False,
        invalid_items=_invalid_items(error) if is_comment else None,
    )


def _invalid_items(error: RequestValidationError) -> list[dict[str, object]] | None:
    """items[i] 아래에서 난 오류를 항목별로 모은다. BE가 문제 건만 빼고 다시 보내게 한다."""
    body = error.body
    raw_items = body.get("items") if isinstance(body, dict) else None
    found: list[dict[str, object]] = []
    for item in error.errors():
        loc = item["loc"]
        if len(loc) < 3 or loc[1] != "items" or not isinstance(loc[2], int):
            continue
        index = loc[2]
        detail_id = None
        if isinstance(raw_items, list) and index < len(raw_items):
            candidate = (
                raw_items[index].get("complaint_detail_id")
                if isinstance(raw_items[index], dict)
                else None
            )
            if isinstance(candidate, int) and not isinstance(candidate, bool):
                detail_id = candidate
        found.append(
            {
                "index": index,
                "complaint_detail_id": detail_id,
                "field": ".".join(str(part) for part in loc[3:]),
                "reason": item["msg"],
            }
        )
    # 특정 항목에 속하지 않는 오류만이면 필드를 생략한다.
    return found or None


async def unhandled_exception_handler(
    request: Request,
    error: Exception,
) -> JSONResponse:
    turn_id = getattr(request.state, "turn_id", None)
    trace_id = getattr(request.state, "trace_id", None)
    # 500은 우리가 예상하지 못한 경로다. 스택트레이스가 유일한 단서다.
    logger.exception(
        "unhandled_error",
        exc_info=error,
        extra={
            "turn_id": turn_id if isinstance(turn_id, str) else None,
            "trace_id": trace_id if isinstance(trace_id, str) else None,
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
        trace_id=trace_id if isinstance(trace_id, str) else None,
        retryable=False,
    )
