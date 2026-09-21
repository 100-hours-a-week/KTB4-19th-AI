from typing import TypedDict

from zipsai.contracts.converse import (
    ComplaintState,
    ConverseRequest,
    Route,
    RouteResult,
)


class AgentState(TypedDict):
    request: ConverseRequest
    route: Route | None
    complaint_state: ComplaintState | None
    reply: str | None
    result: RouteResult
