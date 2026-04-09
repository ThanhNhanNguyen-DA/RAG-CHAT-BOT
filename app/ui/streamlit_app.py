import streamlit as st
import time
import sys
import os

# Đảm bảo có thể import app.core và các file ở thư mục gốc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from app.core.rag_pipeline import ask_question

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="CMC Cloud RAG Chatbot",
    page_icon="☁️",
    layout="centered"
)

st.title("☁️ CMC Cloud Assistant")
st.caption("Chatbot nội bộ hỗ trợ giải đáp thông tin (RAG Architecture)")

# Khởi tạo bộ nhớ tạm (session state) để lưu lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Xin chào! Tôi là Kỹ sư Giải pháp của CMC Cloud. Tôi có thể giúp gì cho bạn hôm nay?"}
    ]

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Hàm mô phỏng hiệu ứng truyền phát (streaming) do hàm gốc trả về full string
def stream_text(text: str):
    """Chia nhỏ text và yield từng phần để tạo hiệu ứng gõ chữ"""
    # Tách theo từng từ (cách nhau bởi khoảng trắng)
    words = text.split(" ")
    for word in words:
        yield word + " "
        time.sleep(0.02) # Tốc độ gõ chữ

# Nhận input từ người dùng
if prompt := st.chat_input("Nhập câu hỏi của bạn về dịch vụ CMC Cloud..."):
    # 1. Hiển thị câu hỏi của người dùng
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Xử lý và hiển thị câu trả lời từ Bot
    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm tài liệu và phân tích..."):
            # Gọi RAG pipeline
            final_answer = ask_question(prompt)
        
        # In ra với hiệu ứng streaming
        st.write_stream(stream_text(final_answer))
        
    # 3. Lưu lại vào lịch sử
    st.session_state.messages.append({"role": "assistant", "content": final_answer})