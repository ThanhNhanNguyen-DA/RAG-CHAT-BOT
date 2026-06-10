from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from agent.state import AgentState


def _format_tool_log(ai_message: AIMessage) -> list[dict[str, Any]]:
    logs: list[dict[str, Any]] = []
    for call in ai_message.tool_calls or []:
        logs.append(
            {
                "id": call.get("id"),
                "name": call.get("name"),
                "args": call.get("args", {}),
            }
        )
    return logs


def tool_context_node(state: AgentState) -> dict[str, Any]:
    """Aggregate ToolMessage outputs into context and append tool call logs."""
    messages = state["messages"]
    tool_results: list[str] = []
    new_tool_calls: list[dict[str, Any]] = list(state.get("tool_calls", []))

    last_ai: AIMessage | None = None
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.tool_calls:
            last_ai = message
            break

    if last_ai is not None:
        new_tool_calls.extend(_format_tool_log(last_ai))

    for message in messages:
        if isinstance(message, ToolMessage):
            tool_results.append(
                f"[{message.name}] {message.content}"
            )

    existing_context = state.get("context", "")
    joined_results = "\n\n".join(tool_results)
    if existing_context and joined_results:
        context = f"{existing_context}\n\n---\n\n{joined_results}"
    else:
        context = existing_context or joined_results

    return {
        "context": context,
        "tool_calls": new_tool_calls,
    }
