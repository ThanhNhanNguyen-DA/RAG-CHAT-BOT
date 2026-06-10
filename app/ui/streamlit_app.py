import streamlit as st
import time
import sys
import os
import uuid

# Đảm bảo có thể import app.core và các file ở thư mục gốc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from app.ui.chat_handler import (
    check_vector_store_status,
    get_agent_model_name,
    get_available_tool_names,
    handle_chat_turn,
)

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="SA Agent Chatbot",
    page_icon="☁️",
    layout="centered"
)

# Login functionality
if "acknowledged" not in st.session_state:
    st.session_state.acknowledged = False

# Check if user is logged in
try:
    user_logged_in = st.user.is_logged_in
except AttributeError:
    user_logged_in = False

with st.sidebar:
    logo_col, _ = st.columns([30, 1])

    with logo_col:
        st.title("SA Agent Chatbot")
        st.title("Xác nhận")

        acknowledged = st.checkbox(
            "Tôi xác nhận đã đọc cảnh báo và đồng ý với các điều khoản sử dụng",
            key="acknowledged",
            label_visibility="visible",
        )
        if not user_logged_in:
            submit = st.button(
                "Vui lòng đăng nhập trước khi sử dụng",
                disabled=not (acknowledged),
            )
            if submit:
                st.login("google")

        if user_logged_in:
            if "conversation_id" not in st.session_state:
                st.session_state.conversation_id = str(uuid.uuid4())
            st.session_state.user_authenticated = True

    st.divider()
    st.subheader("Agent Status")
    st.markdown("**Tools available**")
    for tool_name in get_available_tool_names():
        st.markdown(f"- `{tool_name}`")
    st.markdown(f"**Model:** `{get_agent_model_name()}`")
    vector_status = check_vector_store_status()
    status_icon = "🟢" if vector_status == "connected" else "🔴"
    st.markdown(f"**Vector store:** {status_icon} {vector_status}")

if not user_logged_in:
    st.markdown(
        "<h1 style='text-align: center;'>Welcome to SA Agent Chatbot</h1>",
        unsafe_allow_html=True,
    )
else:
    st.title("☁️ SA Agent")
    st.caption("Chatbot nội bộ hỗ trợ giải đáp thông tin")

    # Khởi tạo bộ nhớ tạm (session state) để lưu lịch sử chat
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Xin chào! Tôi là SA Agent CMC Cloud. Tôi có thể giúp gì cho bạn hôm nay?"}
        ]

    # Hiển thị lịch sử chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    @st.cache_resource
    def get_compiled_agent():
        from agent.graph import agent_graph
        return agent_graph.compile()

    def _stream_text(text: str):
        """Yield text word by word for typing effect."""
        words = text.split(" ")
        for word in words:
            yield word + " "
            time.sleep(0.02)

    def _display_tool_calls(tool_calls: list[dict]) -> None:
        """Display executed tool calls in an expander."""
        if tool_calls:
            with st.expander("Công cụ đã sử dụng"):
                for call in tool_calls:
                    st.markdown(f"**{call.get('name')}**")
                    st.code(call.get("args", {}), language="json")

    def _process_user_query(prompt: str) -> tuple[str, list[dict]]:
        """Run the LangGraph agent for one chat turn."""
        prior_history = st.session_state.messages[:-1]
        agent_result = handle_chat_turn(
            prompt,
            prior_history,
            compiled_agent=get_compiled_agent(),
        )
        return (
            agent_result.get("final_answer", ""),
            agent_result.get("tool_calls", []),
        )

    # Nhận input từ người dùng
    if prompt := st.chat_input("Nhập câu hỏi của bạn về dịch vụ CMC Cloud..."):
        # 1. Hiển thị câu hỏi của người dùng
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Xử lý và hiển thị câu trả lời từ Bot
        with st.chat_message("assistant"):
            with st.spinner("Đang phân tích và chọn công cụ phù hợp..."):
                final_answer, tool_calls = _process_user_query(prompt)

            _display_tool_calls(tool_calls)
            st.write_stream(_stream_text(final_answer))

        # 3. Lưu lại vào lịch sử
        st.session_state.messages.append({"role": "assistant", "content": final_answer})