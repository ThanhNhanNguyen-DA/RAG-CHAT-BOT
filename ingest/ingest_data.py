import os
import sys
import uuid
import argparse
import logging
from pathlib import Path
from typing import Optional

# LangChain loaders & splitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Hugging Face embedding
from sentence_transformers import SentenceTransformer

# Supabase client
from supabase import create_client, Client

# Local config

from config import (
    HUGGINGFACE_API_KEY,
    HF_EMBEDDING_MODEL,
    HF_DEVICE,
    SUPABASE_URL,
    SUPABASE_KEY,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_DIM,
)

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest")

# ── Supabase client ────────────────────────────────────────────────────────
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── Hugging Face model ─────────────────────────────────────────────────────
_embedding_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        log.info(f"Loading Hugging Face model: {HF_EMBEDDING_MODEL} on {HF_DEVICE}")
        _embedding_model = SentenceTransformer(
            HF_EMBEDDING_MODEL,
            device=HF_DEVICE,
            token=HUGGINGFACE_API_KEY
        )
    return _embedding_model


# ── Text splitter ──────────────────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n#{1-6} ",
                "```\n",
                "\n\\*\\*\\*+\n",
                "\n---+\n",
                "\n---+\n",
                "\n\n",
                "\n",
                " ",
                ""
                ],
    length_function=len,
)

# ═══════════════════════════════════════════════════════════════════════════
# Loaders
# ═══════════════════════════════════════════════════════════════════════════

def load_pdf(path: Path) -> list[str]:
    """Extract pages from a PDF and return as list of page-text strings."""
    loader = PyPDFLoader(str(path))
    pages = loader.load()
    return [p.page_content for p in pages if p.page_content.strip()]


def load_docx(path: Path) -> list[str]:
    """Extract text from a DOCX file, returned as a single-element list."""
    loader = Docx2txtLoader(str(path))
    docs = loader.load()
    return [d.page_content for d in docs if d.page_content.strip()]


def load_file(path: Path) -> tuple[list[str], str]:
    """
    Auto-detect file type and return (text_pages, file_type).
    Raises ValueError for unsupported extensions.
    """
    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(path), "pdf"
    elif ext in (".docx", ".doc"):
        return load_docx(path), "docx"
    else:
        raise ValueError(f"Unsupported file type: {ext}. Only PDF and DOCX are supported.")


# ═══════════════════════════════════════════════════════════════════════════
# Chunking
# ═══════════════════════════════════════════════════════════════════════════

def chunk_texts(pages: list[str]) -> list[dict]:
    """
    Split pages into overlapping chunks using RecursiveCharacterTextSplitter.
    Returns list of dicts: {chunk_index, content, token_count, metadata}
    """
    all_chunks = []
    global_index = 0

    for page_num, page_text in enumerate(pages, start=1):
        splits = splitter.split_text(page_text)
        for split in splits:
            text = split.strip()
            if not text:
                continue
            all_chunks.append({
                "chunk_index": global_index,
                "content": text,
                "token_count": len(text.split()),
                "metadata": {
                    "page": page_num,
                    "char_count": len(text),
                },
            })
            global_index += 1

    log.info(f"  → {len(all_chunks)} chunks created from {len(pages)} page(s)")
    return all_chunks


# ═══════════════════════════════════════════════════════════════════════════
# Embedding
# ═══════════════════════════════════════════════════════════════════════════

def embed_chunks(chunks: list[dict], batch_size: int = 32) -> list[dict]:
    """
    Generate Hugging Face embeddings for each chunk.
    Returns chunks enriched with 'embedding' key.
    """
    model = get_embedding_model()
    texts = [c["content"] for c in chunks]
    
    # E5 models require a prefix, other models like mpnet do not.
    if "e5" in HF_EMBEDDING_MODEL.lower():
        texts = [f"passage: {text}" for text in texts]

    log.info(f"  Embedding {len(texts)} chunks with batch_size={batch_size} ...")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    actual_dim = embeddings.shape[1]
    if actual_dim != EMBEDDING_DIM:
        raise ValueError(
            f"❌ Embedding dimension mismatch: model returned {actual_dim}, "
            f"but EMBEDDING_DIM={EMBEDDING_DIM}"
        )

    for chunk, vec in zip(chunks, embeddings):
        chunk["embedding"] = vec.tolist()

    return chunks


# ═══════════════════════════════════════════════════════════════════════════
# Supabase upsert
# ═══════════════════════════════════════════════════════════════════════════

def upsert_document(file_path: Path, file_type: str, member_id: Optional[str] = None) -> str:
    """
    Insert a record into the `documents` table.
    Returns the new document UUID.
    """
    doc_id = str(uuid.uuid4())
    payload = {
        "id": doc_id,
        "name": file_path.name,
        "file_type": file_type,
        "file_size": file_path.stat().st_size,
        "uploaded_by": member_id,
    }
    supabase.table("documents").insert(payload).execute()
    log.info(f"  Document record created → {doc_id}")
    return doc_id


def upsert_chunks(document_id: str, chunks: list[dict], batch_size: int = 100) -> None:
    """
    Bulk-insert document_chunks rows into Supabase.
    Splits into batches to stay within Supabase request limits.
    """
    rows = [
        {
            "document_id": document_id,
            "chunk_index": c["chunk_index"],
            "content": c["content"],
            "embedding": c["embedding"],
            "token_count": c["token_count"],
            "metadata": c["metadata"],
        }
        for c in chunks
    ]

    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        supabase.table("document_chunks").insert(batch).execute()
        log.info(f"  Inserted chunks {i + 1}–{i + len(batch)}")

    log.info(f"  ✓ {len(rows)} chunks stored in Supabase")


# ═══════════════════════════════════════════════════════════════════════════
# Main ingestion flow
# ═══════════════════════════════════════════════════════════════════════════

def ingest_file(file_path: Path, member_id: Optional[str] = None) -> None:
    """Full pipeline: load → chunk → embed → store."""
    log.info(f"━━━ Ingesting: {file_path.name} ━━━")

    pages, file_type = load_file(file_path)
    log.info(f"  Loaded {len(pages)} page(s) [{file_type.upper()}]")

    chunks = chunk_texts(pages)
    if not chunks:
        log.warning("  No text content found — skipping file.")
        return

    chunks = embed_chunks(chunks)

    doc_id = upsert_document(file_path, file_type, member_id)
    upsert_chunks(doc_id, chunks)

    log.info(f"  ✅ Done: {file_path.name} ({len(chunks)} chunks, doc_id={doc_id})\n")


def ingest_directory(dir_path: Path, member_id: Optional[str] = None) -> None:
    """Recursively ingest all PDF and DOCX files in a directory."""
    supported = {".pdf", ".docx", ".doc"}
    files = [f for f in dir_path.rglob("*") if f.suffix.lower() in supported]
    log.info(f"Found {len(files)} file(s) in {dir_path}\n")

    for f in files:
        try:
            ingest_file(f, member_id)
        except Exception as e:
            log.error(f"  ✗ Failed {f.name}: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Ingest PDF/DOCX files into Supabase vector store")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=Path, help="Single file to ingest (.pdf or .docx)")
    group.add_argument("--dir", type=Path, help="Directory to batch-ingest")
    parser.add_argument(
        "--member-id",
        type=str,
        default=None,
        help="Supabase auth user UUID of the uploader (optional)"
    )
    args = parser.parse_args()

    if args.file:
        if not args.file.exists():
            log.error(f"File not found: {args.file}")
            sys.exit(1)
        ingest_file(args.file, args.member_id)

    elif args.dir:
        if not args.dir.is_dir():
            log.error(f"Directory not found: {args.dir}")
            sys.exit(1)
        ingest_directory(args.dir, args.member_id)


if __name__ == "__main__":
    main()