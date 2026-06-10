from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agent.nodes.agent_node import agent_node
from agent.nodes.response_node import response_node
from agent.nodes.routing import route_after_agent
from agent.nodes.tool_context_node import tool_context_node
from agent.state import AgentState
from agent.tools import get_tools

_compiled_graph = None


def create_agent_graph() -> StateGraph:
    """Build the uncompiled LangGraph StateGraph for the multi-tool agent."""
    tools = get_tools()
    tool_executor_node = ToolNode(tools)

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tool_executor", tool_executor_node)
    graph.add_node("tool_context", tool_context_node)
    graph.add_node("response", response_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tool_executor": "tool_executor",
            "response": "response",
        },
    )
    graph.add_edge("tool_executor", "tool_context")
    graph.add_edge("tool_context", "agent")
    graph.add_edge("response", END)

    return graph


agent_graph = create_agent_graph()


def build_graph():
    """Return a cached compiled agent graph."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = agent_graph.compile()
    return _compiled_graph


def run_agent(
    query: str,
    *,
    messages: list[Any] | None = None,
) -> dict[str, Any]:
    """Run the compiled agent for a single user query."""
    agent = build_graph()
    initial_messages = list(messages or [])
    initial_messages.append(HumanMessage(content=query))

    return agent.invoke(
        {
            "messages": initial_messages,
            "tool_calls": [],
            "context": "",
            "final_answer": "",
        }
    )
