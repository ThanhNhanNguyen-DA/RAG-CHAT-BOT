import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.graph import agent_graph
from agent.tools.rag_tool import rag_retriever_tool
from agent.tools.search_tool import web_search_tool


def test_rag_tool_returns_chunks() -> None:
    """Mock Supabase retrieval and assert chunks include content and source."""
    mock_chunks = [
        {
            "content": "CMC Cloud platform overview",
            "source": "page_1",
            "document_id": "doc-123",
            "chunk_index": 0,
            "similarity": 0.88,
            "metadata": {"page": 1},
        }
    ]

    with patch("agent.tools.rag_tool._retrieve_chunks", return_value=mock_chunks):
        result = json.loads(rag_retriever_tool("What is CMC Cloud?"))

    assert "chunks" in result
    assert isinstance(result["chunks"], list)
    assert len(result["chunks"]) > 0
    for chunk in result["chunks"]:
        assert "content" in chunk
        assert "source" in chunk


def test_web_search_tool_callable() -> None:
    """Mock Tavily and assert the tool returns a non-empty string."""
    mock_results = [
        {
            "title": "Cloud computing news",
            "url": "https://example.com/article",
            "content": "Latest updates on cloud platforms.",
        }
    ]
    mock_search = MagicMock()
    mock_search.invoke.return_value = mock_results

    with patch("agent.tools.search_tool.TAVILY_API_KEY", "test-tavily-key"):
        with patch(
            "langchain_community.tools.tavily_search.TavilySearchResults",
            return_value=mock_search,
        ):
            result = web_search_tool("latest cloud news")

    assert isinstance(result, str)
    assert result.strip() != ""
    payload = json.loads(result)
    assert payload["count"] == 1


def test_agent_graph_compiles() -> None:
    """Assert the LangGraph StateGraph compiles without error."""
    compiled = agent_graph.compile()
    assert compiled is not None
