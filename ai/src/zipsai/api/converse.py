import asyncio
import json
import logging
from time import perf_counter

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

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
from zipsai.metrics import observe_converse
from zipsai.observability import (
    bind,
    collect_timings,
    is_cold,
    llm_token_total,
    track_in_flight,
)
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


def _sse(data: dict[str, object]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


_CONVERSE_OUTCOMES = {"ok", "fail", "cancelled"}


def _converse_outcome(value: object) -> str:
    text = str(value) if value else "fail"
    return text if text in _CONVERSE_OUTCOMES else "fail"


def _stream_request_done(
    started_at: float,
    timings: dict[str, int],
    in_flight: int,
    *,
    outcome: str,
    intent_route: object,
    model: str | None = None,
    error_code: str | None = None,
    error_type: str | None = None,
    reply_chars: int | None = None,
) -> None:
    total_ms = int((perf_counter() - started_at) * 1000)
    route_name = getattr(intent_route, "value", intent_route)
    logger.info(
        "stream_request_done",
        extra={
            "in_flight": in_flight,
            "total_ms": total_ms,
            "timings": timings,
            "streaming": True,
            "http_status": 200,
            "stage_ms": dict(timings),
            "tokens_total": llm_token_total(),
            "cost_usd_total": None,
            "model": model,
            "intent_route": route_name,
            "outcome": outcome,
            "error_code": error_code,
            "error_type": error_type,
            "reply_chars": reply_chars,
        },
    )
    observe_converse(
        str(route_name) if route_name else "unknown",
        outcome,
        total_ms / 1000,
    )


async def _stream_converse(request: ConverseRequest):
    started_at = perf_counter()
    with (
        bind(
            turn_id=request.turn_id,
            trace_id=request.trace_id,
            building_id=request.building_id,
            route="/api/v3/ai/converse/stream",
            intent_route=(
                request.current_route.value if request.current_route else None
            ),
        ),
        track_in_flight() as in_flight,
        collect_timings() as timings,
        trace_turn(request) as trace,
    ):
        state = {
            "request": request,
            "route": None,
            "complaint_state": request.current_complaint_state,
            "reply": None,
            "result": RouteResult(),
        }
        streamed_reply = False
        settings = None
        logged = False
        try:
            logger.info("stream_request_started", extra={"in_flight": in_flight})
            yield _sse(
                {
                    "type": "status",
                    "status": "응답을 준비하고 있습니다.",
                }
            )
            settings = get_settings()
            async for mode, event in build_graph(streaming=True).astream(
                state,
                stream_mode=["custom", "values"],
            ):
                if mode == "custom":
                    if isinstance(event, str):
                        streamed_reply = True
                        yield _sse(
                            {"type": "delta", "field": "reply", "text": event}
                        )
                    elif isinstance(event, dict):
                        if event.get("type") == "delta":
                            streamed_reply = True
                        yield _sse(event)
                elif mode == "values":
                    state = event
        except AGENT_ERRORS as error:
            logger.exception("stream_request_failed")
            if trace is not None:
                trace.update(
                    output={"outcome": "fail", "error_type": type(error).__name__}
                )
            _stream_request_done(
                started_at,
                timings,
                in_flight,
                outcome="fail",
                intent_route=state.get("route") or request.current_route,
                model=None if settings is None else settings.llm_model,
                error_code=error_code(error),
                error_type=type(error).__name__,
            )
            logged = True
            yield _sse(
                {
                    "type": "error",
                    "message": "응답 생성에 실패했습니다.",
                }
            )
            yield _sse({"type": "done"})
            return
        except (asyncio.CancelledError, GeneratorExit) as error:
            if not logged:
                _stream_request_done(
                    started_at,
                    timings,
                    in_flight,
                    outcome="cancelled",
                    intent_route=state.get("route") or request.current_route,
                    model=None if settings is None else settings.llm_model,
                    error_type=type(error).__name__,
                )
            raise
        except Exception as error:
            # 스트림이 시작된 뒤에는 HTTP 상태 코드를 바꿀 수 없으므로,
            # 예상하지 못한 예외도 SSE 오류 이벤트로 알리고 정상 종료한다.
            logger.exception("stream_request_failed_unexpected")
            if trace is not None:
                trace.update(
                    output={"outcome": "fail", "error_type": type(error).__name__}
                )
            _stream_request_done(
                started_at,
                timings,
                in_flight,
                outcome="fail",
                intent_route=state.get("route") or request.current_route,
                model=None if settings is None else settings.llm_model,
                error_code=error_code(error),
                error_type=type(error).__name__,
            )
            logged = True
            yield _sse(
                {
                    "type": "error",
                    "message": "응답 생성에 실패했습니다.",
                }
            )
            yield _sse({"type": "done"})
            return

        if not streamed_reply and state.get("reply"):
            yield _sse(
                {"type": "delta", "field": "reply", "text": state["reply"]}
            )
        if not state.get("reply") or not state.get("route"):
            _stream_request_done(
                started_at,
                timings,
                in_flight,
                outcome="fail",
                intent_route=state.get("route") or request.current_route,
                model=None if settings is None else settings.llm_model,
                error_code="INTERNAL_SERVER_ERROR",
            )
            logged = True
            yield _sse(
                {
                    "type": "error",
                    "message": "응답 생성에 실패했습니다.",
                }
            )
            yield _sse({"type": "done"})
            return
        complete = _build_success_response(
            request,
            state,
            settings=settings,
            started_at=started_at,
        )
        route = state.get("route")
        _stream_request_done(
            started_at,
            timings,
            in_flight,
            outcome="ok",
            intent_route=route,
            model=settings.llm_model,
            reply_chars=len(state.get("reply") or ""),
        )
        logged = True
        if trace is not None:
            trace.update(
                output={
                    "outcome": "ok",
                    "route": state.get("route").value
                    if state.get("route")
                    else None,
                }
            )
        yield _sse(
            {
                "type": "complete",
                "response": complete.model_dump(mode="json"),
            }
        )
        yield _sse({"type": "done"})


@router.post("/converse/stream")
def converse_stream(
    request: ConverseRequest, http_request: Request
) -> StreamingResponse:
    http_request.state.turn_id = request.turn_id
    http_request.state.trace_id = request.trace_id

    return StreamingResponse(
        _stream_converse(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


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
        model=settings.llm_model,
    )
    return _build_success_response(
        request,
        result,
        settings=settings,
        started_at=started_at,
    )


def _build_success_response(
    request: ConverseRequest,
    result: dict[str, object],
    *,
    settings,
    started_at: float,
) -> ConverseResponse:
    reply = result["reply"]
    route_result = result.get("result") or RouteResult()
    if isinstance(route_result, dict):
        route_result = RouteResult.model_validate(route_result)
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
    total_ms = int((perf_counter() - started_at) * 1000)
    logger.info(
        "request_done",
        extra={
            "total_ms": total_ms,
            **received,
            **{f"{name}_ms": elapsed for name, elapsed in timings.items()},
            **_qdrant_ms(timings),
            **fields,
            "http_status": fields.get("status_code"),
            "streaming": False,
            "stage_ms": dict(timings),
            "tokens_total": llm_token_total(),
            "cost_usd_total": None,
        },
    )
    route = fields.get("intent_route")
    observe_converse(
        str(route) if route else "unknown",
        _converse_outcome(fields.get("outcome")),
        total_ms / 1000,
    )


def _qdrant_ms(timings: dict[str, int]) -> dict[str, int]:
    """게이트와 하이브리드를 합친 Qdrant 총 소요시간.

    검색을 타지 않은 민원 경로에서는 필드를 넣지 않는다. 0을 넣으면 "Qdrant가
    0ms"로 읽혀 병목 판단이 뒤집힌다.
    """
    parts = [timings[name] for name in ("gate", "hybrid") if name in timings]
    return {"qdrant_ms": sum(parts)} if parts else {}
