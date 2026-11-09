from typing import Literal, NotRequired, TypedDict

from zipsai.contracts.converse import (
    ComplaintState,
    ConverseRequest,
    ImageAnalysis,
    Route,
    RouteResult,
)

ToolName = Literal[
    "search_building_docs", "process_complaint_turn", "search_public_web"
]


class AgentState(TypedDict):
    request: ConverseRequest
    route: Route | None
    complaint_state: ComplaintState | None
    reply: str | None
    result: RouteResult
    image_analysis: ImageAnalysis | None
    image_analysis_failed: NotRequired[bool]
    contextualized_message: NotRequired[str | None]
    planned_tools: NotRequired[list[ToolName]]
    attempted_tools: NotRequired[tuple[ToolName, ...]]
    completed_tools: NotRequired[tuple[ToolName, ...]]
    agent_messages: NotRequired[list[object]]
    pending_tool_calls: NotRequired[tuple[object, ...]]
    tool_call_count: NotRequired[int]
    protocol_retry_count: NotRequired[int]
    protocol_retry_pending: NotRequired[bool]
    last_tool_result: NotRequired[object]
    search_result: NotRequired[object]
    search_query: NotRequired[str]
    web_search_query: NotRequired[str]
    web_search_result: NotRequired[object]
    complaint_result: NotRequired[object]
