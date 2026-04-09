from sentence_transformers import SentenceTransformer
from config import (
    HUGGINGFACE_API_KEY,
    HF_EMBEDDING_MODEL,
    HF_DEVICE,
    EMBEDDING_DIM,
)
import logging
from typing import Optional

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("embedding")
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