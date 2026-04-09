import os
from dotenv import load_dotenv

load_dotenv()

# ==============================
# HUGGING FACE EMBEDDING
# ==============================
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

HF_EMBEDDING_MODEL = os.getenv(
    "HF_EMBEDDING_MODEL",
    "sentence-transformers/all-mpnet-base-v2"
)

# multilingual-e5-base và all-mpnet-base-v2 đều có 768 dimensions
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 768))

# Optional: chọn device thủ công: cpu / cuda / mps
HF_DEVICE = os.getenv("HF_DEVICE", "cpu")

# ==============================
# SUPABASE
# ==============================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL:
    raise ValueError("❌ SUPABASE_URL is missing")

if not SUPABASE_KEY:
    raise ValueError("❌ SUPABASE_KEY is missing")

# ==============================
# GEMINI (Google Generative AI)
# ==============================
GEMINI_API_KEYS = [
    k.strip() for k in os.getenv("GEMINI_API_KEYS", "").split(",") if k.strip()
]

GEMINI_MODEL = "models/gemini-3-flash-preview"

# Reasonable defaults for internal Q&A
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.2"))
GEMINI_TOP_P = float(os.getenv("GEMINI_TOP_P", "0.95"))
GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "4096"))


# ==============================
# RAG RETRIEVAL (match_document_chunks)
# ==============================
# SQL default is 0.75; lower values return more chunks (cosine similarity).
VECTOR_MATCH_THRESHOLD = float(os.getenv("VECTOR_MATCH_THRESHOLD", "0.5"))

# ==============================
# INGESTION
# ==============================
TOP_K_DEFAULT = int(os.getenv("TOP_K_DEFAULT", 5))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))