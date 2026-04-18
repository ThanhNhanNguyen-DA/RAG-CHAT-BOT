import os
import logging
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Cấu hình logging để dễ debug trên WSL
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class GeminiRouter:
    def __init__(self, api_key: str = None):
        """Khởi tạo Router sử dụng Gemini 1.5 Flash siêu nhẹ và miễn phí."""
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("❌ Không tìm thấy GOOGLE_API_KEY. Vui lòng cấu hình biến môi trường.")

        # Sử dụng nhiệt độ = 0 để buộc Gemini trả lời logic, không sáng tạo lung tung
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            max_tokens=10,  # Ép token nhỏ nhất có thể để phản hồi chớp nhoáng
            google_api_key=self.api_key
        )

        self.prompt = PromptTemplate.from_template(
            """Bạn là một hệ thống phân loại câu hỏi cho tài liệu kỹ thuật điện toán đám mây.
            Hãy phân tích câu hỏi sau và phân loại nó thành 1 trong 2 loại:
            - 'GENERAL': Nếu câu hỏi mang tính khái quát, hỏi về danh mục dịch vụ, tổng quan hệ thống.
            - 'SPECIFIC': Nếu câu hỏi hỏi chi tiết về thông số, cách cấu hình, lỗi kỹ thuật của 1 dịch vụ.
            
            Chỉ trả về duy nhất 1 từ 'GENERAL' hoặc 'SPECIFIC', tuyệt đối không giải thích thêm.
            Câu hỏi: {question}"""
        )
        
        self.chain = self.prompt | self.llm

    def classify_intent(self, question: str) -> str:
        """Phân tích ngữ nghĩa câu hỏi để trả về luồng truy xuất tương ứng."""
        try:
            response = self.chain.invoke({"question": question})
            intent = response.content.strip().upper()
            
            # Lọc kết quả an toàn (tránh trường hợp LLM sinh thêm dấu chấm/phẩy)
            if "GENERAL" in intent:
                return "GENERAL"
            return "SPECIFIC"
            
        except Exception as e:
            logger.error(f"⚠️ Lỗi kết nối Gemini API: {e}. Tự động Fallback về SPECIFIC.")
            return "SPECIFIC" # Mặc định an toàn nhất là tìm kiếm thông thường

    def smart_retrieve(self, question: str, vectorstore):
        """
        Hàm trung tâm: Nhận câu hỏi, phân loại và gọi VectorDB bằng thuật toán tối ưu nhất.
        (Cần truyền đối tượng vectorstore - Supabase từ file retriever.py của bạn vào đây)
        """
        intent = self.classify_intent(question)
        
        if intent == "GENERAL":
            logger.info(f"⚡ [ROUTER] Phân loại: {intent} -> Kích hoạt thuật toán đa dạng hóa (MMR)")
            retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 5, 'fetch_k': 20})
        else:
            logger.info(f"⚡ [ROUTER] Phân loại: {intent} -> Kích hoạt thuật toán chính xác (Similarity)")
            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 5})
            
        docs = retriever.invoke(question)
        return docs, intent