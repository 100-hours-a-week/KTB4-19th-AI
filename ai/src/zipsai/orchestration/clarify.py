from zipsai.contracts.converse import Route, RouteResult
from zipsai.history import format_history
from zipsai.integrations.llm import generate_text
from zipsai.orchestration.prompts import CLARIFY_PROMPT
from zipsai.orchestration.state import AgentState

CLARIFY_REPLY = "어떤 것을 도와드릴까요? 민원/시설 문제인가요, 건물 정보 질문인가요?"


def handle_clarify(state: AgentState) -> dict[str, str | None]:
    request = state["request"]
    if request.current_route is not Route.CLARIFY:
        reply = CLARIFY_REPLY
    else:
        messages = CLARIFY_PROMPT.format_messages(
            message_text=(request.message.text or "").strip() or "없음",
            conversation_history=format_history(request.conversation_history),
        )
        reply = generate_text(
            system_prompt=str(messages[0].content),
            user_prompt=str(messages[1].content),
        )

    return {
        "reply": reply,
        "complaint_state": None,
        "result": RouteResult(complaint_draft=request.complaint_draft),
    }
