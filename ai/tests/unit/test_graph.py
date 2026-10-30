from zipsai.orchestration.graph import build_graph


def test_graph_has_prepare_agent_and_tool_nodes():
    graph = build_graph()

    assert {"prepare", "agent", "tool"} <= set(graph.get_graph().nodes)
