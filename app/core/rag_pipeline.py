import sys
import os
import logging

# Thêm thư mục gốc (chứa các file .py của bạn) vào sys.path để import
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Đã xóa bỏ hoàn toàn dòng import GeminiRouter
from src.embedding import get_embedding_model
from src.retriever import retrieve_documents
from src.prompts import build_prompt
from src.generation import generate_answer

# Cấu hình logging cơ bản cho pipeline
logger = logging.getLogger("rag_pipeline")

def ask_question(query: str) -> str:
    """
    Luồng RAG tối giản và siêu tốc: 
    Query -> Embedding -> Retrieve (Top 15) -> Master Prompt -> LLM -> Answer
    """
    try:
        # Bước 1: Khởi tạo model embedding và chuyển câu hỏi thành vector
        logger.info(f"Đang vector hóa câu hỏi: '{query}'")
        embed_model = get_embedding_model()
        query_embedding = embed_model.encode(query).tolist()

        # Bước 2: Lưới rộng tìm kiếm ngữ cảnh (Lấy hẳn 15 chunks để có bức tranh toàn cảnh)
        logger.info("Đang truy xuất Top 15 tài liệu từ cơ sở dữ liệu...")
        docs = retrieve_documents(query_embedding, top_k=15)
        
        if not docs:
            context = "Không tìm thấy tài liệu phù hợp trong cơ sở dữ liệu."
            logger.warning("Không có chunk nào được trả về từ Supabase.")
        else:
            # Gộp các chunk lại thành một đoạn ngữ cảnh duy nhất
            context = "\n\n---\n\n".join(docs)
            logger.info(f"Đã ghép {len(docs)} chunks thành ngữ cảnh chuẩn bị đưa vào Prompt.")

        # Bước 3: Đưa ngữ cảnh và câu hỏi vào Master Prompt
        # (Ở bước này, Master Prompt sẽ tự động chỉ thị cho Gemini cách xử lý)
        logger.info("Đang xây dựng Master Prompt...")
        final_prompt = build_prompt(context=context, question=query)

        # Bước 4: Gọi Gemini để sinh câu trả lời
        logger.info("Đang gọi Gemini xử lý và tổng hợp...")
        answer = generate_answer(final_prompt)

        return answer

    except Exception as e:
        logger.error(f"Lỗi trong quá trình RAG: {e}")
        return f"Xin lỗi, đã xảy ra lỗi trong hệ thống: {str(e)}"