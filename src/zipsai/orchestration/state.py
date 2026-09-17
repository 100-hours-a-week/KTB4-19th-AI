from typing import TypedDict

from zipsai.contracts.converse import (
    ConversationState,
    ConverseData,
    ConverseRequest,
    Route,
)


class AgentState(TypedDict):
    request: ConverseRequest
    route: Route | None
    conversation_state: ConversationState | None
    response: ConverseData | None
    reply: str | None
