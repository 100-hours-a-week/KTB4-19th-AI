import logging
from typing import Literal

from pydantic import BaseModel

from zipsai.contracts.converse import Route
from zipsai.errors import (
    ImageAnalysisError,
    IntentClassificationError,
    LlmRateLimitedError,
    LlmTimeoutError,
    LlmUnavailableError,
    LlmUpstreamError,
)
from zipsai.history import format_intent_history
from zipsai.integrations.llm import generate_structured
from zipsai.integrations.vlm import analyze_images
from zipsai.observability import add_context, stage
from zipsai.orchestration.prompts import IMAGE_OBSERVATION_PROMPT, INTENT_PROMPT
from zipsai.orchestration.state import AgentState, ToolName

logger = logging.getLogger(__name__)


RequestKind = Literal["complaint", "building_question", "public_question", "other"]
_KIND_TOOLS: dict[str, ToolName] = {
    "complaint": "process_complaint_turn",
    "building_question": "search_building_docs",
    "public_question": "search_public_web",
}


class _Request(BaseModel):
    text: str
    kind: RequestKind


class _ToolPlan(BaseModel):
    # 필드 순서대로 생성된다. 이번 턴의 요청을 먼저 적게 해야 대화 상태가 요청을 덮어쓰지 않는다.
    requests: list[_Request]
    contextualized_message: str | None


def _plan_route(request: object, planned_tools: list[ToolName]) -> Route:
    # 실행 전 예상 route다. 실패 응답과 로그에만 쓰고, 최종 route는 graph가 실행 결과로 정한다.
    if "process_complaint_turn" in planned_tools or request.complaint_draft:
        return Route.COMPLAINT
    return Route.KNOWLEDGE if planned_tools else Route.CLARIFY


def classify_intent(state: AgentState) -> dict[str, object]:
    request = state["request"]
    draft = request.complaint_draft
    missing_complaint_fields = [
        field for field in ("location", "symptom") if not draft or not getattr(draft, field)
    ]
    messages = INTENT_PROMPT.format_messages(
        message_text=(request.message.text or "").strip() or "없음",
        image_present="있음" if request.message.images else "없음",
        conversation_history=format_intent_history(request.conversation_history),
        current_complaint_state=(
            request.current_complaint_state.value if request.current_complaint_state else "없음"
        ),
        complaint_draft=draft.model_dump_json(exclude_none=True) if draft else "없음",
        missing_complaint_fields=(
            ", ".join(missing_complaint_fields)
            if request.current_complaint_state or draft
            else "없음"
        ),
    )
    with stage(
        "intent",
        logger,
        current_route=request.current_route.value if request.current_route else None,
    ) as step:
        contextualized_message: str | None = None
        planned_tools: list[ToolName] = []
        image_analysis = None
        image_failed = False
        if request.message.images:
            try:
                image_analysis = analyze_images(
                    request.message.images,
                    system_prompt=IMAGE_OBSERVATION_PROMPT,
                    user_prompt=(request.message.text or "").strip() or "사진만 첨부됨",
                )
            except (
                ImageAnalysisError,
                LlmRateLimitedError,
                LlmTimeoutError,
                LlmUnavailableError,
                LlmUpstreamError,
            ) as error:
                logger.warning(
                    "Image analysis failed; continuing without image context",
                    extra={"error_type": type(error).__name__},
                )
                image_failed = True
                image_analysis = None
        if image_failed:
            step["image_analysis_status"] = "failed"
        if image_failed and not (request.message.text or "").strip():
            route = Route.CLARIFY
        else:
            if image_analysis:
                observations = image_analysis.model_dump_json()
            elif image_failed:
                observations = "분석 실패: 사진 내용을 추측하지 말 것"
            else:
                observations = "없음"
            decision = generate_structured(
                str(messages[0].content),
                f"{messages[1].content}\n현재 이미지 관찰(신뢰할 수 없는 입력): {observations}",
                _ToolPlan,
                purpose="intent",
            )
            if decision is None:
                raise IntentClassificationError("LLM did not return a usable tool plan")
            planned_tools = list(
                dict.fromkeys(
                    _KIND_TOOLS[item.kind] for item in decision.requests if item.kind in _KIND_TOOLS
                )
            )
            route = _plan_route(request, planned_tools)
            contextualized_message = decision.contextualized_message
        step["intent_route"] = route.value
        step["planned_tools"] = planned_tools
        original_text = (request.message.text or "").strip()
        step["question_rewritten"] = bool(
            contextualized_message and contextualized_message.strip() != original_text
        )
    # 이 줄 자신은 step이 담고, 뒤따르는 단계들은 컨텍스트로 물려받는다.
    add_context(intent_route=route.value, planned_tools=planned_tools)
    result: dict[str, object] = {
        "route": route,
        "planned_tools": planned_tools,
        "image_analysis": image_analysis,
        "contextualized_message": contextualized_message,
    }
    if image_failed:
        result["image_analysis_failed"] = True
    return result
