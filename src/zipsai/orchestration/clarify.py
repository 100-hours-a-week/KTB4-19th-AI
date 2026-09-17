from zipsai.contracts.converse import Route
from zipsai.integrations.llm import generate_text
from zipsai.orchestration.prompts import CLARIFY_PROMPT
from zipsai.orchestration.state import AgentState

CLARIFY_REPLY = "어떤 것을 도와드릴까요? 민원/시설 문제인가요, 건물 정보 질문인가요?"


def handle_clarify(state: AgentState) -> dict[str, str]:
    request = state["request"]
    if request.current_route is not Route.CLARIFY:
        return {"reply": CLARIFY_REPLY}

    messages = CLARIFY_PROMPT.format_messages(
        message_text=request.message.text or "없음",
        conversation_history=_format_history(state),
    )
    reply = generate_text(
        system_prompt=str(messages[0].content),
        user_prompt=str(messages[1].content),
    )
    return {"reply": reply}


def _format_history(state: AgentState) -> str:
    history = state["request"].conversation_history
    if not history:
        return "없음"
    return "\n".join(f"{turn.role}: {turn.content}" for turn in history)
