import re
import itertools
from langchain_google_genai import ChatGoogleGenerativeAI
from config import GEMINI_API_KEYS, GEMINI_MODEL

# Khởi tạo chuỗi xoay vòng cho API Keys
if not GEMINI_API_KEYS:
    raise ValueError("❌ GEMINI_API_KEYS is empty in .env")

_key_cycle = itertools.cycle(GEMINI_API_KEYS)

def get_llm() -> ChatGoogleGenerativeAI:
    """
    Lấy instance của LLM với key xoay vòng để tránh nghẽn Rate Limit
    """
    api_key = next(_key_cycle)
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        api_key=api_key,
        temperature=0.3,
        max_output_tokens=2048
    )

def normalize_llm_output(response) -> str:
    """
    Chuẩn hoá output từ LangChain 4.x (ChatGoogleGenerativeAI)
    -> trả về string sạch
    """
    if response is None:
        return ""

    content = response.content

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        return "\n".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip()

    return str(content).strip()

def format_answer_markdown(text: str) -> str:
    """
    Làm đẹp output để render Markdown
    """
    if not text:
        return ""

    text = text.replace("\r\n", "\n").strip()

    # Convert "1. Service:" -> Markdown heading
    text = re.sub(
        r"\n?(\d+)\.\s*([A-Za-zÀ-ỹ0-9\s\(\)&\-]+):",
        r"\n\n### \1. **\2**\n",
        text
    )

    # Bullet phụ
    text = re.sub(r"\n-\s*", "\n• ", text)

    return text.strip()

def generate_answer(prompt: str) -> str:
    """
    Gọi LLM + normalize + format output
    """
    if not prompt or not prompt.strip():
        return "Không tìm thấy tài liệu phù hợp."

    try:
        # Sử dụng hàm get_llm() để tự động lấy API key tiếp theo trong danh sách
        llm = get_llm()
        response = llm.invoke(prompt)

        text = normalize_llm_output(response)
        text = format_answer_markdown(text)

        return text

    except Exception as e:
        return f"Lỗi hệ thống: {str(e)}"