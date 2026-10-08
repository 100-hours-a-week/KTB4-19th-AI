import logging
from functools import lru_cache

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from zipsai.complaint.node import handle_complaint
from zipsai.contracts.converse import Route
from zipsai.knowledge.node import handle_knowledge
from zipsai.orchestration.clarify import handle_clarify
from zipsai.orchestration.intent import classify_intent
from zipsai.orchestration.state import AgentState

logger = logging.getLogger(__name__)


def select_next_node(state: AgentState) -> str:
    return (state["route"] or Route.CLARIFY).value


def _run_complaint(state: AgentState) -> dict[str, object]:
    result = handle_complaint(
        state["request"], image_analysis=state.get("image_analysis")
    ) or {}
    return {"route": Route.COMPLAINT, **result}


def _run_knowledge(state: AgentState) -> dict[str, object]:
    return handle_knowledge(state["request"])


def _run_classify_intent(state: AgentState) -> dict[str, object]:
    return classify_intent(state)


def _run_clarify(state: AgentState) -> dict[str, object]:
    return handle_clarify(state)


@lru_cache(maxsize=1)
def build_graph() -> CompiledStateGraph:
    builder = StateGraph(AgentState)
    builder.add_node("classify_intent", _run_classify_intent)
    builder.add_node("complaint", _run_complaint)
    builder.add_node("knowledge", _run_knowledge)
    builder.add_node("clarify", _run_clarify)
    builder.add_edge(START, "classify_intent")
    builder.add_conditional_edges(
        "classify_intent",
        select_next_node,
        {"complaint": "complaint", "knowledge": "knowledge", "clarify": "clarify"},
    )
    builder.add_edge("complaint", END)
    builder.add_edge("knowledge", END)
    builder.add_edge("clarify", END)
    return builder.compile()
