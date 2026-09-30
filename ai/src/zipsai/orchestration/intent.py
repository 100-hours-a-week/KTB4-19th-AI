import logging

from pydantic import BaseModel

from zipsai.contracts.converse import Route
from zipsai.errors import IntentClassificationError
from zipsai.history import format_history
from zipsai.integrations.llm import generate_structured
from zipsai.observability import add_context, stage
from zipsai.orchestration.prompts import INTENT_PROMPT
from zipsai.orchestration.state import AgentState

logger = logging.getLogger(__name__)


class _RouteResponse(BaseModel):
    route: Route


def classify_intent(state: AgentState) -> dict[str, Route]:
    request = state["request"]
    messages = INTENT_PROMPT.format_messages(
        message_text=(request.message.text or "").strip() or "없음",
        image_present="있음" if request.message.image_urls else "없음",
        conversation_history=format_history(request.conversation_history),
        current_route=request.current_route.value if request.current_route else "없음",
    )
    with stage(
        "intent",
        logger,
        current_route=request.current_route.value if request.current_route else None,
    ) as step:
        parsed = generate_structured(
            system_prompt=str(messages[0].content),
            user_prompt=str(messages[1].content),
            response_format=_RouteResponse,
        )
        if parsed is None:
            raise IntentClassificationError("LLM did not return a usable intent route")
        route = parsed.route
        step["intent_route"] = route.value
    # 이 줄 자신은 step이 담고, 뒤따르는 단계들은 컨텍스트로 물려받는다.
    add_context(intent_route=route.value)
    return {"route": route}
