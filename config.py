import os
from dotenv import load_dotenv

load_dotenv(override=True)

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
VECTOR_MATCH_THRESHOLD = float(os.getenv("VECTOR_MATCH_THRESHOLD", 0.5))
if not SUPABASE_URL:
    raise ValueError("❌ SUPABASE_URL is missing")

if not SUPABASE_KEY:
    raise ValueError("❌ SUPABASE_KEY is missing")

# ==============================
# GEMINI (Google Generative AI)
# ==============================
_raw_gemini_keys = (os.getenv("GEMINI_API_KEYS") or "").strip()
if not _raw_gemini_keys:
    raise ValueError("❌ GEMINI_API_KEYS is missing")

GEMINI_API_KEYS = [k.strip() for k in _raw_gemini_keys.split(",") if k.strip()]
if not GEMINI_API_KEYS:
    raise ValueError("❌ GEMINI_API_KEYS is empty after parsing")

# Some .env files may accidentally contain leading/trailing spaces.
# Preview IDs like gemini-2.5-flash-preview-05-20 are retired; use gemini-3-flash-preview or gemini-2.5-flash.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview").strip()

# Reasonable defaults for internal Q&A
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.2"))
GEMINI_TOP_P = float(os.getenv("GEMINI_TOP_P", "0.95"))
GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "4096"))

# ==============================
# CHAT PERSISTENCE (Supabase)
# ==============================
# Must be a UUID that exists in auth.users (FK on chat_sessions.user_id).
# Leave unset to skip persisting chat_sessions / chat_messages.
CHAT_USER_ID = (os.getenv("CHAT_USER_ID") or os.getenv("SUPABASE_CHAT_USER_ID") or "").strip() or None

# ==============================
# AGENT / TOOLS
# ==============================
RAG_TOP_K = int(os.getenv("RAG_TOP_K", os.getenv("TOP_K_DEFAULT", "5")))
TAVILY_API_KEY = (os.getenv("TAVILY_API_KEY") or "").strip() or None
TAVILY_MAX_RESULTS = int(os.getenv("TAVILY_MAX_RESULTS", "5"))

# ==============================
# INGESTION
# ==============================
TOP_K_DEFAULT = int(os.getenv("TOP_K_DEFAULT", 5))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))
