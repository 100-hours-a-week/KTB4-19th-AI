from zipsai.contracts.converse import Route
from zipsai.errors import IntentClassificationError
from zipsai.integrations.llm import generate_text
from zipsai.orchestration.prompts import INTENT_PROMPT
from zipsai.orchestration.state import AgentState


def classify_intent(state: AgentState) -> dict[str, Route]:
    request = state["request"]
    messages = INTENT_PROMPT.format_messages(
        message_text=request.message.text or "없음",
        image_present="있음" if request.message.image_urls else "없음",
        user_action=request.message.user_action or "없음",
        conversation_history=_format_history(state),
        current_route=request.current_route.value if request.current_route else "없음",
    )
    raw_response = generate_text(
        system_prompt=str(messages[0].content),
        user_prompt=str(messages[1].content),
    )
    return {"route": parse_route(raw_response)}


def parse_route(raw_response: str) -> Route:
    try:
        return Route(raw_response.strip())
    except ValueError as error:
        raise IntentClassificationError("Unsupported intent route") from error


def _format_history(state: AgentState) -> str:
    history = state["request"].conversation_history
    if not history:
        return "없음"
    return "\n".join(f"{turn.role}: {turn.content}" for turn in history)
