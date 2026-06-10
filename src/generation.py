import re
from agent.llm import get_llm
from src.prompts import build_prompt

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