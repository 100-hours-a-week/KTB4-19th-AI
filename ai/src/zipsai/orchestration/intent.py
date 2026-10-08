import logging

from zipsai.errors import ImageAnalysisError, IntentClassificationError
from zipsai.history import format_history
from zipsai.integrations.vlm import classify_and_analyze
from zipsai.observability import add_context, stage
from zipsai.orchestration.prompts import INTENT_PROMPT
from zipsai.orchestration.state import AgentState

logger = logging.getLogger(__name__)

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
        except ImageAnalysisError as error:
            raise IntentClassificationError(
                "VLM did not return a usable intent route"
            ) from error
        step["intent_route"] = route.value
    # 이 줄 자신은 step이 담고, 뒤따르는 단계들은 컨텍스트로 물려받는다.
    add_context(intent_route=route.value)
    return {
        "route": route,
        "image_analysis": image_analysis if request.message.images else None,
    }
