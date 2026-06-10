from langchain_core.tools import BaseTool

from agent.tools.calculator_tool import create_calculator_tool
from agent.tools.rag_tool import create_rag_retriever_tool
from agent.tools.search_tool import create_web_search_tool


def get_tools() -> list[BaseTool]:
    """Return all agent tools in registration order."""
    return [
        create_rag_retriever_tool(),
        create_web_search_tool(),
        create_calculator_tool(),
    ]
