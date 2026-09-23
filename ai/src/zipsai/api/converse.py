from time import perf_counter

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from zipsai.api.error_responses import agent_error_response, error_response
from zipsai.contracts.converse import (
    ConverseData,
    ConverseRequest,
    ConverseResponse,
    Meta,
    RouteResult,
)
from zipsai.errors import (
    EmbeddingError,
    IntentClassificationError,
    LlmRateLimitedError,
    LlmTimeoutError,
    LlmUnavailableError,
    LlmUpstreamError,
    VectorStoreError,
)
from zipsai.orchestration.graph import build_graph
from zipsai.settings import get_settings

router = APIRouter(prefix="/api/v3/ai", tags=["converse"])


@router.post("/converse", response_model=ConverseResponse)
def converse(
    request: ConverseRequest, http_request: Request
) -> ConverseResponse | JSONResponse:
    http_request.state.trace_id = request.trace_id
    if not _has_message_content(request):
        return error_response(
            status_code=400,
            code="MISSING_REQUIRED_FIELD",
            detail="A message requires text or image_urls",
            trace_id=request.trace_id,
            retryable=False,
        )

    started_at = perf_counter()
    try:
        settings = get_settings()
        result = build_graph().invoke(
            {
                "request": request,
                "route": None,
                "complaint_state": request.current_complaint_state,
                "reply": None,
                "result": RouteResult(),
            }
        )
    except (
        LlmRateLimitedError,
        LlmTimeoutError,
        LlmUpstreamError,
        LlmUnavailableError,
        IntentClassificationError,
        EmbeddingError,
        VectorStoreError,
    ) as error:
        return agent_error_response(error, request.trace_id)

    reply = result["reply"]
    if reply is None:
        return error_response(
            status_code=500,
            code="INTERNAL_SERVER_ERROR",
            detail="Agent route returned no reply",
            trace_id=request.trace_id,
            retryable=False,
        )

    return ConverseResponse(
        code="ai_response_success",
        trace_id=request.trace_id,
        data=ConverseData(
            route=result["route"],
            next_complaint_state=result["complaint_state"],
            reply=reply,
            result=result["result"],
            meta=Meta(
                model=settings.llm_model,
                timing_ms=int((perf_counter() - started_at) * 1000),
            ),
        ),
    )


def _has_message_content(request: ConverseRequest) -> bool:
    message = request.message
    return bool(message.text and message.text.strip()) or bool(message.image_urls)
