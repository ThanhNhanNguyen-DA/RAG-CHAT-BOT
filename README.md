☁️ CMC Cloud RAG Chatbot

The CMC Cloud RAG Chatbot is an internal virtual assistant (acting as a Solution Architect) built on the RAG (Retrieval-Augmented Generation) architecture. The application helps look up, synthesize, and accurately answer technical questions about CMC Cloud services based on the provided documentation.

✨ Key Features

Complete RAG Architecture: Seamlessly combines semantic search capabilities with the powerful text generation of LLMs.

Multi-format Document Processing: Supports automatic extraction, chunking, and vectorization of data from both PDF and DOCX files.

High-Speed Vector Search: Utilizes Supabase (PostgreSQL + pgvector) with HNSW (Hierarchical Navigable Small World) indexing for maximum retrieval performance.

LLM Optimization: Leverages Google Gemini paired with an automatic API Key rotation mechanism to prevent Rate Limit bottlenecks.

User-Friendly Interface: Provides an intuitive, smooth chat application built with Streamlit, featuring a typing effect (streaming text).

Role-Based Security: Supports Row Level Security (RLS) on the database to ensure the strict safety of internal document information.

🛠 Tech Stack

Language: Python 3.9+

User Interface (UI): Streamlit

LLM / AI Model:

Answer Generation: Google Gemini (langchain-google-genai)

Embedding: Hugging Face Sentence Transformers (text-embedding-004 or equivalent with a 768D vector dimension)

Vector Database: Supabase (PostgreSQL + pgvector)

RAG Framework: LangChain (Text Splitters, Document Loaders, Prompts)

📂 Directory Structure

.
├── app/
│   ├── core/
│   │   └── rag_pipeline.py       # Main RAG flow (Vectorization -> Retrieval -> Prompt -> LLM)
│   └── ui/
│       └── streamlit_app.py      # Streamlit user interface
├── src/
│   ├── embedding.py              # Load and process Hugging Face model
│   ├── generation.py             # Call Gemini LLM & handle output/Rate Limits
│   ├── ingest_data.py            # CLI script to process, chunk, and push docs to Supabase
│   ├── prompts.py                # System prompt configuration for the Solution Architect role
│   └── retriever.py              # Connect to Supabase & execute Semantic Search
├── vector_store_schema.sql       # SQL code to initialize tables, functions, and RLS on Supabase
├── main.py                       # Application entry point
├── config.py                     # (Requires configuration) Loads environment variables
├── .env                          # File containing API Keys
└── README.md


⚙️ Installation Guide

1. System Requirements

Python 3.9 or higher.

A Supabase account.

Accounts/Tokens for Hugging Face and the Google Gemini API.

2. Install Dependencies

Clone this repository to your local machine and install the required libraries:

git clone <repository_url>
cd cmc-cloud-rag-chatbot
pip install -r requirements.txt



(Ensure the following libraries are installed: streamlit, langchain, langchain-google-genai, langchain-community, sentence-transformers, supabase, pypdf, docx2txt)

3. Database Setup (Supabase)

Create a new project on Supabase.

Open the SQL Editor, copy the entire contents of the vector_store_schema.sql file, and Run it to construct the table schemas, enable the pgvector extension, and create the search functions (RPC).

4. Environment Variables

Create a .env file in the root directory of the project (which will be read by the config.py file) and include the following configurations:

# Supabase Configuration
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_service_role_key

# Gemini Configuration
GEMINI_API_KEYS=["key_1", "key_2", "key_3"] # Supports rotation to bypass Rate Limits
GEMINI_MODEL=gemini-1.5-flash # Or whichever model you are currently using

# Hugging Face Embedding Configuration
HUGGINGFACE_API_KEY=your_hf_token
HF_EMBEDDING_MODEL=your_hf_model_id (e.g., keepitreal/vietnamese-sbert)
HF_DEVICE=cpu # Use 'cuda' if a GPU is available
EMBEDDING_DIM=768
VECTOR_MATCH_THRESHOLD=0.75

# Chunking Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200



🚀 Usage Guide

Step 1: Data Ingestion (Vector Database)

You must process and push your technical documents (PDF, DOCX) into the Supabase database before you can chat with the assistant.

Run the following CLI command to ingest an entire directory:

python src/ingest_data.py --dir /path/to/your/documents



Or to ingest a single file:

python src/ingest_data.py --file /path/to/document.pdf



Step 2: Start the Chatbot UI

Once the data is ready, run the following command to launch the Streamlit interface:

python main.py



The system will automatically initialize Streamlit and open a new browser tab at: http://localhost:8501.

🧠 System Flow (RAG Pipeline)

The user asks a question via the streamlit_app.py interface.

The question is passed into rag_pipeline.py.

embedding.py uses Hugging Face to convert the query into a Vector (768D).

retriever.py calls the match_document_chunks RPC function on Supabase to retrieve the top 5-15 text segments (chunks) with the highest semantic similarity.

The retrieved Context is combined with the Query and the System Prompt in prompts.py.

generation.py sends the fully constructed Prompt to Google Gemini. The get_llm() function automatically handles API Key rotation.

The Markdown-formatted answer is returned and streamed (displayed word-by-word) onto the Streamlit interface.

Developed for the CMC Cloud ecosystem. ☁️