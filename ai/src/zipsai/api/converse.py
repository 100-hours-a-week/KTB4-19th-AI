import logging
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
    ComplaintExtractionError,
    EmbeddingError,
    IntentClassificationError,
    LlmRateLimitedError,
    LlmTimeoutError,
    LlmUnavailableError,
    LlmUpstreamError,
    VectorStoreError,
)
from zipsai.observability import bind, collect_timings, track_in_flight
from zipsai.orchestration.graph import build_graph
from zipsai.settings import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v3/ai", tags=["converse"])

AGENT_ERRORS = (
    LlmRateLimitedError,
    LlmTimeoutError,
    LlmUpstreamError,
    LlmUnavailableError,
    IntentClassificationError,
    ComplaintExtractionError,
    EmbeddingError,
    VectorStoreError,
)


@router.post("/converse", response_model=ConverseResponse)
def converse(
    request: ConverseRequest, http_request: Request
) -> ConverseResponse | JSONResponse:
    http_request.state.trace_id = request.trace_id
    with (
        bind(trace_id=request.trace_id, building_id=request.building_id),
        track_in_flight() as in_flight,
        collect_timings() as timings,
    ):
        return _handle(request, in_flight=in_flight, timings=timings)


def _handle(
    request: ConverseRequest, *, in_flight: int, timings: dict[str, int]
) -> ConverseResponse | JSONResponse:
    started_at = perf_counter()
    message = request.message
    received = {
        "has_text": bool(message.text and message.text.strip()),
        "has_image": bool(message.image_urls),
        "history_turns": len(request.conversation_history),
        "in_flight": in_flight,
    }

    if not (received["has_text"] or received["has_image"]):
        _request_done(started_at, timings, received, outcome="fail", status=400)
        return error_response(
            status_code=400,
            code="MISSING_REQUIRED_FIELD",
            detail="A message requires text or image_urls",
            trace_id=request.trace_id,
            retryable=False,
        )

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
    except AGENT_ERRORS as error:
        _request_done(
            started_at,
            timings,
            received,
            outcome="fail",
            error_type=type(error).__name__,
        )
        return agent_error_response(error, request.trace_id)

    reply = result["reply"]
    if reply is None:
        _request_done(started_at, timings, received, outcome="fail", status=500)
        return error_response(
            status_code=500,
            code="INTERNAL_SERVER_ERROR",
            detail="Agent route returned no reply",
            trace_id=request.trace_id,
            retryable=False,
        )

    route_result = result["result"]
    _request_done(
        started_at,
        timings,
        received,
        outcome="ok",
        status=200,
        route=result["route"].value if result["route"] else None,
        has_evidence=route_result.has_sufficient_evidence,
        citations=len(route_result.citations or []),
        reply_chars=len(reply),
    )
    return ConverseResponse(
        code="ai_response_success",
        trace_id=request.trace_id,
        data=ConverseData(
            route=result["route"],
            next_complaint_state=result["complaint_state"],
            reply=reply,
            result=route_result,
            meta=Meta(
                model=settings.llm_model,
                timing_ms=int((perf_counter() - started_at) * 1000),
            ),
        ),
    )


def _request_done(
    started_at: float,
    timings: dict[str, int],
    received: dict[str, object],
    **fields: object,
) -> None:
    """요청 하나에 한 줄. 단계별 소요시간을 여기에 모아 병목을 한눈에 본다."""
    logger.info(
        "request_done",
        extra={
            "total_ms": int((perf_counter() - started_at) * 1000),
            **received,
            **{f"{name}_ms": elapsed for name, elapsed in timings.items()},
            **fields,
        },
    )
