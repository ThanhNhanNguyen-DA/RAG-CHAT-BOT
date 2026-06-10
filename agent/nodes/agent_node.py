from typing import Any

from langchain_core.messages import AIMessage, SystemMessage

from agent.llm import get_llm_with_tools
from agent.state import AgentState
from agent.tools import get_tools

AGENT_SYSTEM_PROMPT = """You are SA Agent, a Solution Architect assistant for CMC Cloud.

You have access to these tools:
- rag_retriever_tool: search internal documentation (Supabase pgvector)
- web_search_tool: search the public web via Tavily for external/current information
- calculator_tool: evaluate arithmetic expressions

Rules:
1. Use rag_retriever_tool for internal CMC Cloud product, service, or policy questions.
2. Use web_search_tool when internal docs are insufficient or the question needs public/current data.
3. Use calculator_tool only for numeric calculations.
4. You may call multiple tools across turns before giving a final answer.
5. Respond in the same language as the user (Vietnamese or English).
6. Cite sources when using retrieved content.
"""


def _ensure_system_message(messages: list[Any]) -> list[Any]:
    if messages and isinstance(messages[0], SystemMessage):
        return messages
    return [SystemMessage(content=AGENT_SYSTEM_PROMPT), *messages]


def agent_node(state: AgentState) -> dict[str, Any]:
    """Invoke Gemini with bound tools; append the model response to messages."""
    tools = get_tools()
    llm = get_llm_with_tools(tools)
    messages = _ensure_system_message(state["messages"])
    response = llm.invoke(messages)
    if not isinstance(response, AIMessage):
        response = AIMessage(content=str(response))
    return {"messages": [response]}
