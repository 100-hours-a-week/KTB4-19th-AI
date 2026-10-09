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
    error_code,
)
from zipsai.observability import bind, collect_timings, is_cold, track_in_flight
from zipsai.orchestration.graph import build_graph
from zipsai.settings import get_settings
from zipsai.tracing import trace_turn

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
    http_request.state.turn_id = request.turn_id
    http_request.state.trace_id = request.trace_id
    with (
        bind(
            turn_id=request.turn_id,
            trace_id=request.trace_id,
            building_id=request.building_id,
            # 경로를 리터럴로 적으면 라우터 prefix와 두 곳이 된다.
            route=http_request.url.path,
            # 의도 분류가 실패하면 그 단계 로그에 경로가 비어 조회가 끊긴다.
            # 백엔드가 준 경로로 시작해 두고, 분류가 끝나면 그 결과가 덮는다.
            intent_route=(
                request.current_route.value if request.current_route else None
            ),
        ),
        track_in_flight() as in_flight,
        collect_timings() as timings,
        trace_turn(request) as trace,
    ):
        response = _handle(request, in_flight=in_flight, timings=timings)
        if trace is not None:
            trace.update(
                output={
                    "status_code": response.status_code
                    if isinstance(response, JSONResponse)
                    else 200,
                    "route": response.data.route.value
                    if isinstance(response, ConverseResponse)
                    else None,
                }
            )
        return response


def _handle(
    request: ConverseRequest, *, in_flight: int, timings: dict[str, int]
) -> ConverseResponse | JSONResponse:
    started_at = perf_counter()
    message = request.message
    received = {
        "has_text": bool(message.text and message.text.strip()),
        "has_image": bool(message.images),
        "history_turns": len(request.conversation_history),
        "in_flight": in_flight,
        # 인코더·Qdrant 클라이언트를 만드는 첫 질의는 수 초가 더 걸린다.
        "cold": is_cold("converse"),
    }
    # 끝난 요청만 남기면 처리 중 컨테이너가 죽은 요청은 흔적이 없다.
    logger.info("request_started")

    if not (received["has_text"] or received["has_image"]):
        _request_done(
            started_at,
            timings,
            received,
            outcome="fail",
            status_code=400,
            error_code="MISSING_REQUIRED_FIELD",
        )
        return error_response(
            status_code=400,
            code="MISSING_REQUIRED_FIELD",
            detail="A message requires text or images",
            turn_id=request.turn_id,
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
                "image_analysis": None,
            }
        )
    except AGENT_ERRORS as error:
        # 상태 코드의 출처는 error_responses의 매핑 하나다. 여기서 다시 정하지 않고
        # 만들어진 응답에서 읽는다.
        response = agent_error_response(error, request.turn_id, request.trace_id)
        _request_done(
            started_at,
            timings,
            received,
            outcome="fail",
            status_code=response.status_code,
            error_code=error_code(error),
            error_type=type(error).__name__,
        )
        return response

    reply = result["reply"]
    if reply is None:
        _request_done(
            started_at,
            timings,
            received,
            outcome="fail",
            status_code=500,
            error_code="INTERNAL_SERVER_ERROR",
        )
        return error_response(
            status_code=500,
            code="INTERNAL_SERVER_ERROR",
            detail="Agent route returned no reply",
            turn_id=request.turn_id,
            trace_id=request.trace_id,
            retryable=False,
        )

    route_result = result["result"]
    _request_done(
        started_at,
        timings,
        received,
        outcome="ok",
        status_code=200,
        error_code=None,
        intent_route=result["route"].value if result["route"] else None,
        has_evidence=route_result.has_sufficient_evidence,
        citations=len(route_result.citations or []),
        reply_chars=len(reply),
    )
    return ConverseResponse(
        code="ai_response_success",
        turn_id=request.turn_id,
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
            **_qdrant_ms(timings),
            **fields,
        },
    )


def _qdrant_ms(timings: dict[str, int]) -> dict[str, int]:
    """게이트와 하이브리드를 합친 Qdrant 총 소요시간.

    검색을 타지 않은 민원 경로에서는 필드를 넣지 않는다. 0을 넣으면 "Qdrant가
    0ms"로 읽혀 병목 판단이 뒤집힌다.
    """
    parts = [timings[name] for name in ("gate", "hybrid") if name in timings]
    return {"qdrant_ms": sum(parts)} if parts else {}
