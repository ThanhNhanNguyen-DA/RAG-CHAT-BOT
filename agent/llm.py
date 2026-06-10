import itertools
from typing import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI

from config import (
    GEMINI_API_KEYS,
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_MODEL,
    GEMINI_TEMPERATURE,
)

_key_cycle = itertools.cycle(GEMINI_API_KEYS)


def get_llm() -> ChatGoogleGenerativeAI:
    """Return a Gemini chat model instance with rotated API keys."""
    api_key = next(_key_cycle)
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        api_key=api_key,
        temperature=GEMINI_TEMPERATURE,
        max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
    )


def get_llm_with_tools(tools: Sequence[BaseTool]) -> BaseChatModel:
    """Bind tools to the Gemini model for agent tool selection."""
    return get_llm().bind_tools(tools)
