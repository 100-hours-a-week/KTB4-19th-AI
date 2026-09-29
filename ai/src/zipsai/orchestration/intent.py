import logging

from zipsai.contracts.converse import Route
from zipsai.errors import IntentClassificationError
from zipsai.history import format_history
from zipsai.integrations.llm import generate_text
from zipsai.observability import add_context, stage
from zipsai.orchestration.prompts import INTENT_PROMPT
from zipsai.orchestration.state import AgentState

logger = logging.getLogger(__name__)


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
        raw_response = generate_text(
            system_prompt=str(messages[0].content),
            user_prompt=str(messages[1].content),
        )
        route = parse_route(raw_response)
        step["intent_route"] = route.value
    # 이 줄 자신은 step이 담고, 뒤따르는 단계들은 컨텍스트로 물려받는다.
    add_context(intent_route=route.value)
    return {"route": route}


def parse_route(raw_response: str) -> Route:
    try:
        return Route(raw_response.strip())
    except ValueError as error:
        raise IntentClassificationError("Unsupported intent route") from error
