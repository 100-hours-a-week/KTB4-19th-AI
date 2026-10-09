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
from zipsai.history import format_history
from zipsai.integrations.llm import generate_structured
from zipsai.integrations.vlm import classify_and_analyze
from zipsai.observability import add_context, stage
from zipsai.orchestration.prompts import INTENT_PROMPT
from zipsai.orchestration.state import AgentState

logger = logging.getLogger(__name__)


class _RouteOnly(BaseModel):
    route: Route


def classify_intent(state: AgentState) -> dict[str, object]:
    request = state["request"]
    messages = INTENT_PROMPT.format_messages(
        message_text=(request.message.text or "").strip() or "없음",
        image_present="있음" if request.message.images else "없음",
        conversation_history=format_history(request.conversation_history),
        current_route=request.current_route.value if request.current_route else "없음",
    )
    with stage(
        "intent",
        logger,
        current_route=request.current_route.value if request.current_route else None,
    ) as step:
        try:
            route, image_analysis = classify_and_analyze(
                request.message.images,
                system_prompt=str(messages[0].content),
                user_prompt=str(messages[1].content),
            )
        except (
            ImageAnalysisError,
            LlmRateLimitedError,
            LlmTimeoutError,
            LlmUnavailableError,
            LlmUpstreamError,
        ) as error:
            if not request.message.images:
                if isinstance(error, ImageAnalysisError):
                    raise IntentClassificationError(
                        "VLM did not return a usable intent route"
                    ) from error
                raise

            logger.warning(
                "Image analysis failed; continuing without image context",
                extra={"error_type": type(error).__name__},
            )
            image_analysis = None
            if (request.message.text or "").strip():
                fallback = generate_structured(
                    "Classify the resident's intent as complaint, knowledge, or clarify. "
                    "Use the current text and conversation history. The newly attached image "
                    "could not be analyzed, so do not infer its contents. Use current_route only "
                    "as context for a brief reply, not as the sole reason to choose a route. "
                    "If intent is unclear, choose clarify.",
                    "Current text: "
                    f"{request.message.text}\n"
                    f"Conversation history:\n{format_history(request.conversation_history)}\n"
                    "Current route: "
                    f"{request.current_route.value if request.current_route else 'none'}",
                    _RouteOnly,
                )
                route = fallback.route if fallback else Route.CLARIFY
            else:
                route = Route.CLARIFY
            step["image_analysis_status"] = "failed"
        step["intent_route"] = route.value
    # 이 줄 자신은 step이 담고, 뒤따르는 단계들은 컨텍스트로 물려받는다.
    add_context(intent_route=route.value)
    result: dict[str, object] = {
        "route": route,
        "image_analysis": image_analysis if request.message.images else None,
    }
    if request.message.images and image_analysis is None:
        result["image_analysis_failed"] = True
    return result
