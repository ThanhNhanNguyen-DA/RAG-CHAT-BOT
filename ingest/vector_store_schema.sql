-- ============================================================
-- RAG Vector Store Schema — Supabase + pgvector (768D Gemini)
-- ============================================================

-- 1. Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================
-- 2. Documents table — tracks source files
-- ============================================================
CREATE TABLE IF NOT EXISTS documents (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT NOT NULL,                     -- original filename
    file_type   TEXT NOT NULL CHECK (file_type IN ('pdf', 'docx')),
    file_size   BIGINT,                            -- bytes
    source_url  TEXT,                              -- optional: storage path
    uploaded_by UUID REFERENCES auth.users(id),   -- member who uploaded
    created_at  TIMESTAMPTZ DEFAULT now(),
    metadata    JSONB DEFAULT '{}'::jsonb          -- extra fields (tags, dept, etc.)
);

-- ============================================================
-- 3. Document chunks — stores text + vector embeddings (768D)
-- ============================================================
CREATE TABLE IF NOT EXISTS document_chunks (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id  UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index  INT  NOT NULL,                    -- order within the document
    content      TEXT NOT NULL,                    -- raw chunk text
    embedding    vector(768),                      -- Gemini text-embedding-004
    token_count  INT,                              -- approx token count
    metadata     JSONB DEFAULT '{}'::jsonb,        -- page number, section, etc.
    created_at   TIMESTAMPTZ DEFAULT now()
);

-- ============================================================
-- 4. Members table (private access control)
-- ============================================================
CREATE TABLE IF NOT EXISTS members (
    id         UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    full_name  TEXT,
    email      TEXT UNIQUE NOT NULL,
    role       TEXT NOT NULL DEFAULT 'member' CHECK (role IN ('admin', 'member')),
    is_active  BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- ============================================================
-- 5. Chat sessions — for future RAG Agent
-- ============================================================
CREATE TABLE IF NOT EXISTS chat_sessions (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id    UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    title      TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- ============================================================
-- 6. Chat messages — conversation history per session
-- ============================================================
CREATE TABLE IF NOT EXISTS chat_messages (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content     TEXT NOT NULL,
    sources     JSONB,                             -- chunk IDs used for answer
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- ============================================================
-- 7. Indexes
-- ============================================================

-- HNSW index for fast ANN vector search (recommended for pgvector >= 0.5)
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON document_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- IVFFlat fallback (use if HNSW not available)
-- CREATE INDEX IF NOT EXISTS idx_chunks_embedding_ivf
--     ON document_chunks
--     USING ivfflat (embedding vector_cosine_ops)
--     WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id  ON document_chunks (document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_metadata     ON document_chunks USING gin (metadata);
CREATE INDEX IF NOT EXISTS idx_documents_uploaded  ON documents (uploaded_by);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_user  ON chat_sessions (user_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_sess  ON chat_messages (session_id);

-- ============================================================
-- 8. Similarity search function (called from Python / LangChain)
-- ============================================================
CREATE OR REPLACE FUNCTION match_document_chunks(
    query_embedding  vector(768),
    match_threshold  FLOAT    DEFAULT 0.75,
    match_count      INT      DEFAULT 5,
    filter_doc_ids   UUID[]   DEFAULT NULL   -- optional: scope to specific docs
)
RETURNS TABLE (
    id           UUID,
    document_id  UUID,
    chunk_index  INT,
    content      TEXT,
    metadata     JSONB,
    similarity   FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        dc.id,
        dc.document_id,
        dc.chunk_index,
        dc.content,
        dc.metadata,
        1 - (dc.embedding <=> query_embedding) AS similarity
    FROM document_chunks dc
    WHERE
        (filter_doc_ids IS NULL OR dc.document_id = ANY(filter_doc_ids))
        AND 1 - (dc.embedding <=> query_embedding) >= match_threshold
    ORDER BY dc.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- ============================================================
-- 9. Row Level Security (RLS) — private member-only access
-- ============================================================

ALTER TABLE documents       ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE members         ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_sessions   ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_messages   ENABLE ROW LEVEL SECURITY;

-- Only active members can read documents
CREATE POLICY "members_read_documents" ON documents
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM members m
            WHERE m.id = auth.uid() AND m.is_active = TRUE
        )
    );

-- Only admins can insert/delete documents
CREATE POLICY "admins_write_documents" ON documents
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM members m
            WHERE m.id = auth.uid() AND m.role = 'admin' AND m.is_active = TRUE
        )
    );

-- Members can read chunks
CREATE POLICY "members_read_chunks" ON document_chunks
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM members m
            WHERE m.id = auth.uid() AND m.is_active = TRUE
        )
    );

-- Members manage their own chat sessions
CREATE POLICY "own_sessions" ON chat_sessions
    FOR ALL USING (user_id = auth.uid());

-- Members manage their own messages
CREATE POLICY "own_messages" ON chat_messages
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM chat_sessions s
            WHERE s.id = session_id AND s.user_id = auth.uid()
        )
    );

-- Members view their own profile; admins view all
CREATE POLICY "members_read_members" ON members
    FOR SELECT USING (
        id = auth.uid()
        OR EXISTS (
            SELECT 1 FROM members m
            WHERE m.id = auth.uid() AND m.role = 'admin'
        )
    );
