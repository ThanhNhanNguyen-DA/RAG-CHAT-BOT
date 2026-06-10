from typing import Literal

from langchain_core.messages import AIMessage

from agent.state import AgentState

RouteDecision = Literal["tool_executor", "response"]


def route_after_agent(state: AgentState) -> RouteDecision:
    """Route to tool execution when the model requested tools, otherwise respond."""
    messages = state["messages"]
    if not messages:
        return "response"

    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_executor"
    return "response"
