import json
import logging
from typing import Any

from langchain_core.tools import StructuredTool
from langchain_community.tools.tavily_search import TavilySearchResults
from config import TAVILY_API_KEY, TAVILY_MAX_RESULTS

logger = logging.getLogger(__name__)


def _summarize_tavily_results(results: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Normalize Tavily hits into concise records for the LLM."""
    summarized: list[dict[str, str]] = []
    for item in results:
        summarized.append(
            {
                "title": str(item.get("title", "")),
                "url": str(item.get("url", "")),
                "content": str(item.get("content", item.get("snippet", ""))),
            }
        )
    return summarized


def web_search_tool(query: str) -> str:
    """
    Search the public web via Tavily for up-to-date external information.

    Use this tool when the question requires current events, public documentation
    outside internal docs, or facts not likely stored in the vector database.
    Returns a JSON list of summarized results with title, URL, and content snippet.
    """
    if not TAVILY_API_KEY:
        return json.dumps(
            {
                "results": [],
                "error": "TAVILY_API_KEY is not configured. Web search is unavailable.",
            },
            ensure_ascii=False,
        )

    try:
        search = TavilySearchResults(
            max_results=TAVILY_MAX_RESULTS,
            tavily_api_key=TAVILY_API_KEY,
        )
        raw_results = search.invoke({"query": query})
        if not isinstance(raw_results, list):
            raw_results = [raw_results]
        summarized = _summarize_tavily_results(raw_results)
        return json.dumps(
            {"results": summarized, "count": len(summarized)},
            ensure_ascii=False,
        )
    except Exception as exc:
        logger.error("Web search failed: %s", exc)
        return json.dumps(
            {"results": [], "error": f"Web search failed: {exc}"},
            ensure_ascii=False,
        )


def create_web_search_tool() -> StructuredTool:
    """Factory for the Tavily web search LangChain tool."""
    return StructuredTool.from_function(
        func=web_search_tool,
        name="web_search_tool",
        description=(
            "Search the public web using Tavily. "
            "Use for current events or information not in internal documents."
        ),
    )
