# RAG API Chatbot

An internal virtual assistant (Solution Architect style) built on **RAG** (Retrieval-Augmented Generation): retrieve context from vectorized documents in Supabase, assemble a prompt, then generate answers with **Google Gemini API**. Chat UI is **Streamlit**.

## Features

- **End-to-end RAG**: embed question → similar chunks (pgvector) → prompt → LLM.
- **Multi-format ingest**: PDF and DOCX (`.doc` is handled via the DOCX loader; stored `file_type` is `docx`).
- **Vector search**: Supabase (PostgreSQL + **pgvector**), sample schema includes an **HNSW** index.
- **Gemini + API key rotation**: multiple keys in `GEMINI_API_KEYS`, cycled per call to ease rate limits.
- **Streamlit UI**: chat history, Markdown answers with a word-by-word “typing” effect (client-side).
- **Sample schema with RLS**: `documents` / `document_chunks` and related tables include policies (fits Supabase Auth). The Python client currently uses a **service role key**, which typically **bypasses RLS**; if you switch to a user or anon key, align policies and JWT accordingly.

## Tech stack


| Layer     | Technology                                                                                       |
| --------- | ------------------------------------------------------------------------------------------------ |
| Language  | Python 3                                                                                         |
| UI        | Streamlit                                                                                        |
| Embedding | `sentence-transformers` (default `sentence-transformers/all-mpnet-base-v2`, **768**-dim vectors) |
| LLM       | `langchain-google-genai` → `ChatGoogleGenerativeAI` (Gemini)                                     |
| Vector DB | Supabase (`supabase` Python client)                                                              |
| Documents | `langchain-community` (PyPDF, Docx2txt), `langchain-text-splitters`                              |


## Repository layout

```text
.
├── app/
│   ├── main.py                 # Entry: runs `streamlit run` for the UI
│   ├── core/
│   │   └── rag_pipeline.py    # RAG: embed → retrieve → prompt → generate
│   └── ui/
│       └── streamlit_app.py   # Chat UI
├── ingest/
│   ├── ingest_data.py         # CLI: Load PDF/DOCX → ETL → chunk → embed → Supabase
│   └── vector_store_schema.sql # DDL, RPC `match_document_chunks`, RLS, HNSW index
├── src/
│   ├── embedding.py           # SentenceTransformer (same pattern for query + ingest)
│   ├── retriever.py           # Supabase RPC `match_document_chunks`
│   ├── prompts.py             # Prompt template (LangChain)
│   └── generation.py          # Gemini + normalize/format Markdown
├── config.py                  # Environment variables (required: Supabase; Gemini keys)
├── requirements.txt
└── README.md
```

Run scripts from the **repository root** so `config` and `src` imports resolve correctly.

## Setup

1. **Prerequisites**: Python 3, a Supabase project, Google AI (Gemini) API keys, and a Hugging Face token (recommended if your model or HF policy requires auth).
2. **Clone and install**:
  ```bash
   cd chatbot
   pip install -r requirements.txt
  ```
   The codebase also depends on (install if anything is missing after the step above):
3. **Supabase**: create a project, open the SQL Editor, and execute the full contents of `ingest/vector_store_schema.sql` (enable `vector`, create tables, `match_document_chunks` RPC, RLS, indexes).
4. `**.env`** at the repo root (loaded by `python-dotenv` in `config.py`):
  ```env
   # Required
   SUPABASE_URL=https://xxxx.supabase.co
   SUPABASE_KEY=your_service_role_or_secret_key

   # Gemini: comma-separated keys (not a JSON array)
   GEMINI_API_KEYS=key1,key2,key3

   # Hugging Face (depends on model / HF policy)
   HUGGINGFACE_API_KEY=your_hf_token

   # Optional — must match vector(768) in SQL and your actual model output
   HF_EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
   HF_DEVICE=cpu
   EMBEDDING_DIM=768

   # Cosine similarity threshold for the RPC (code default: 0.5)
   VECTOR_MATCH_THRESHOLD=0.5

   # Chunking (config defaults: 800 / 150)
   CHUNK_SIZE=1200
   CHUNK_OVERLAP=200

   # Other Gemini knobs (temperature, top_p, max tokens) — see `config.py`;
   # `generation.get_llm()` currently sets fixed temperature/max_output_tokens on the LLM.
  ```
   The default Gemini model name is set in `config.py` (`GEMINI_MODEL`, e.g. `models/gemini-3-flash-preview`). Change it in `config.py` or extend the code to read from the environment.

## Usage

**1. Ingest documents into the vector store** (from the repo root):

```bash
python ingest/ingest_data.py --dir /path/to/your/documents
```

Single file:

```bash
python ingest/ingest_data.py --file /path/to/document.pdf
```

Optional: tie the uploader to `auth.users` (UUID):

```bash
python ingest/ingest_data.py --file ./doc.pdf --member-id <uuid>
```

**2. Run the chatbot**:

```bash
python app/main.py
```

Or:

```bash
streamlit run app/ui/streamlit_app.py
```

By default Streamlit serves at `http://localhost:8501`.

## RAG flow (summary)

1. User submits a question in `app/ui/streamlit_app.py`.
2. `app/core/rag_pipeline.py` → `src/embedding.py` encodes the question as a 768-dimensional vector.
3. `src/retriever.py` calls Supabase RPC `match_document_chunks` with `match_threshold` from `VECTOR_MATCH_THRESHOLD` and `match_count` (the pipeline currently uses **top_k = 5**).
4. `src/prompts.py` merges context and question.
5. `src/generation.py` calls Gemini (`get_llm()` rotates keys), normalizes and formats Markdown.
6. The UI renders the result (`st.write_stream` simulates streaming word-by-word).
<img width="1916" height="955" alt="image" src="https://github.com/user-attachments/assets/9fefa9db-ae7c-412f-830c-90d28baeacd4" />


---
