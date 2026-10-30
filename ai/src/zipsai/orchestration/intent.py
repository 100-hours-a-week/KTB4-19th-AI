import logging

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
from zipsai.orchestration.state import AgentState

logger = logging.getLogger(__name__)


class _RouteDecision(BaseModel):
    route: Route
    contextualized_message: str | None


def classify_intent(state: AgentState) -> dict[str, object]:
    request = state["request"]
    messages = INTENT_PROMPT.format_messages(
        message_text=(request.message.text or "").strip() or "없음",
        image_present="있음" if request.message.images else "없음",
        conversation_history=format_intent_history(request.conversation_history),
        current_route=request.current_route.value if request.current_route else "없음",
    )
    with stage(
        "intent",
        logger,
        current_route=request.current_route.value if request.current_route else None,
    ) as step:
        contextualized_message: str | None = None
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
                _RouteDecision,
                purpose="intent",
            )
            if decision is None:
                raise IntentClassificationError("LLM did not return a usable intent route")
            route = decision.route
            contextualized_message = decision.contextualized_message
        step["intent_route"] = route.value
        original_text = (request.message.text or "").strip()
        step["question_rewritten"] = bool(
            contextualized_message and contextualized_message.strip() != original_text
        )
    # 이 줄 자신은 step이 담고, 뒤따르는 단계들은 컨텍스트로 물려받는다.
    add_context(intent_route=route.value)
    result: dict[str, object] = {
        "route": route,
        "image_analysis": image_analysis,
        "contextualized_message": contextualized_message,
    }
    if image_failed:
        result["image_analysis_failed"] = True
    return result
