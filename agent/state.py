from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """LangGraph state schema for the multi-tool agent."""

    messages: Annotated[list[BaseMessage], add_messages]
    tool_calls: list[dict]
    context: str
    final_answer: str
