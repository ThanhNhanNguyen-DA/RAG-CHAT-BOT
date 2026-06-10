from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph

from agent.graph import agent_graph
from agent.tools.rag_tool import _get_supabase_client
from config import GEMINI_MODEL


def session_messages_to_langchain(messages: list[dict[str, str]]) -> list[BaseMessage]:
    """Convert Streamlit session message dicts to LangChain message objects."""
    langchain_messages: list[BaseMessage] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        if role == "user":
            langchain_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            langchain_messages.append(AIMessage(content=content))
    return langchain_messages


def check_vector_store_status() -> str:
    """Return connected/disconnected based on a lightweight Supabase probe."""
    try:
        client = _get_supabase_client()
        client.table("document_chunks").select("id").limit(1).execute()
        return "connected"
    except Exception:
        return "disconnected"


def handle_chat_turn(
    user_input: str,
    conversation_history: list[dict[str, str]],
    *,
    compiled_agent: CompiledStateGraph | None = None,
) -> dict[str, Any]:
    """
    Invoke the LangGraph agent for one user turn.

    Uses conversation history from st.session_state to preserve multi-turn context.
    """
    agent = compiled_agent or agent_graph.compile()
    messages = session_messages_to_langchain(conversation_history)
    messages.append(HumanMessage(content=user_input))

    return agent.invoke(
        {
            "messages": messages,
            "tool_calls": [],
            "context": "",
            "final_answer": "",
        }
    )


def get_available_tool_names() -> list[str]:
    """Return registered tool names for sidebar display."""
    from agent.tools import get_tools

    return [tool.name for tool in get_tools()]


def get_agent_model_name() -> str:
    """Return the configured Gemini model name."""
    return GEMINI_MODEL
