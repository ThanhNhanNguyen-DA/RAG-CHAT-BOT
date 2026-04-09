from supabase import create_client
from config import SUPABASE_URL, SUPABASE_KEY, VECTOR_MATCH_THRESHOLD
import logging

# Cấu hình log để dễ debug
log = logging.getLogger("retriever")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def retrieve_documents(query_embedding, top_k=15):
    try:
        # Gọi hàm RPC với đầy đủ 3 tham số tiêu chuẩn của pgvector
        # Tăng top_k mặc định lên 15 để tránh lỗi Under-retrieval đối với các câu hỏi liệt kê dài
        res = supabase.rpc(
            "match_document_chunks",
            {
                "query_embedding": query_embedding,
                "match_threshold": VECTOR_MATCH_THRESHOLD,  # Thêm ngưỡng tương đồng
                "match_count": top_k
            }
        ).execute()

        docs = res.data or []
        log.info(f"Supabase trả về {len(docs)} đoạn văn bản (chunks).")
        
        # Trích xuất nội dung văn bản
        return [d["content"] for d in docs if "content" in d]
        
    except Exception as e:
        log.error(f"Lỗi khi gọi RPC Supabase: {e}")
        return []