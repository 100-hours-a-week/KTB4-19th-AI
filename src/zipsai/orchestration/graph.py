from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from zipsai.complaint.node import handle_complaint
from zipsai.contracts.converse import Route
from zipsai.knowledge.node import handle_knowledge
from zipsai.orchestration.clarify import handle_clarify
from zipsai.orchestration.intent import classify_intent
from zipsai.orchestration.state import AgentState


def select_next_node(state: AgentState) -> str:
    return (state["route"] or Route.CLARIFY).value


def _run_complaint(state: AgentState) -> dict[str, object]:
    return handle_complaint(state["request"])


def _run_knowledge(state: AgentState) -> dict[str, object]:
    return handle_knowledge(state["request"])


def build_graph() -> CompiledStateGraph:
    builder = StateGraph(AgentState)
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("complaint", _run_complaint)
    builder.add_node("knowledge", _run_knowledge)
    builder.add_node("clarify", handle_clarify)
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
