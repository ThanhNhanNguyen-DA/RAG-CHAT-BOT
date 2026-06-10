import json
import logging
from typing import Any

from langchain_core.tools import StructuredTool
from supabase import Client, create_client

from config import (
    HF_EMBEDDING_MODEL,
    RAG_TOP_K,
    SUPABASE_KEY,
    SUPABASE_URL,
    VECTOR_MATCH_THRESHOLD,
)
from src.embedding import get_embedding_model

logger = logging.getLogger(__name__)

_supabase_client: Client | None = None


def _get_supabase_client() -> Client:
    global _supabase_client
    if _supabase_client is None:
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase_client


def _embed_query(query: str) -> list[float]:
    model = get_embedding_model()
    text = query
    if "e5" in HF_EMBEDDING_MODEL.lower():
        text = f"query: {query}"
    return model.encode(text).tolist()


def _retrieve_chunks(query: str) -> list[dict[str, Any]]:
    """Embed the query and search Supabase pgvector for similar document chunks."""
    try:
        query_embedding = _embed_query(query)
        response = (
            _get_supabase_client()
            .rpc(
                "match_document_chunks",
                {
                    "query_embedding": query_embedding,
                    "match_threshold": VECTOR_MATCH_THRESHOLD,
                    "match_count": RAG_TOP_K,
                },
            )
            .execute()
        )
        rows = response.data or []
        chunks: list[dict[str, Any]] = []
        for row in rows:
            metadata = row.get("metadata") or {}
            source = metadata.get("source") or metadata.get("page") or row.get("document_id", "unknown")
            chunks.append(
                {
                    "content": row.get("content", ""),
                    "source": str(source),
                    "document_id": row.get("document_id"),
                    "chunk_index": row.get("chunk_index"),
                    "similarity": row.get("similarity"),
                    "metadata": metadata,
                }
            )
        logger.info("RAG tool retrieved %d chunks for query.", len(chunks))
        return chunks
    except Exception as exc:
        logger.error("RAG retrieval failed: %s", exc)
        return []


def rag_retriever_tool(query: str) -> str:
    """
    Search internal CMC Cloud documentation stored in Supabase pgvector.

    Use this tool when the user asks about internal products, services, policies,
    technical specifications, or any topic that may be covered by uploaded documents.
    Returns top-k text chunks with source metadata and similarity scores.
    """
    chunks = _retrieve_chunks(query)
    if not chunks:
        return json.dumps(
            {"chunks": [], "message": "No relevant internal documents found."},
            ensure_ascii=False,
        )
    return json.dumps({"chunks": chunks, "count": len(chunks)}, ensure_ascii=False)


def create_rag_retriever_tool() -> StructuredTool:
    """Factory for the RAG retriever LangChain tool."""
    return StructuredTool.from_function(
        func=rag_retriever_tool,
        name="rag_retriever_tool",
        description=(
            "Search internal company documentation via Supabase pgvector. "
            "Use for CMC Cloud services, products, policies, and technical docs."
        ),
    )
