from typing import Any

from langchain_core.messages import AIMessage

from agent.state import AgentState


def _extract_text(message: AIMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts).strip()
    return str(content).strip()


def response_node(state: AgentState) -> dict[str, str]:
    """Set final_answer from the latest assistant message or accumulated context."""
    messages = state["messages"]
    final_answer = ""

    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            final_answer = _extract_text(message)
            if final_answer:
                break

    if not final_answer and state.get("context"):
        final_answer = state["context"]

    if not final_answer:
        final_answer = "I could not generate a response. Please try again."

    return {"final_answer": final_answer}
