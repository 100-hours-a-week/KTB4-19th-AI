from zipsai.contracts.converse import Route
from zipsai.orchestration.state import AgentState


def classify_intent(state: AgentState) -> dict[str, Route]:
    """Keep an injected route until the LLM classifier is added."""
    return {"route": state["route"] or Route.CLARIFY}
