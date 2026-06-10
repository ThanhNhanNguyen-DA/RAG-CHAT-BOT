import uuid
import streamlit as st

def login():
    # Customer login form as sidebar and enable session tabs only when the username is entered
    # user chưa xác nhận điều khoản thì không được vào trang chủ
    if "acknowledged" not in st.session_state:
        st.session_state.acknowledged = False

    # tạo sidebar bên trái
    with st.sidebar:
        # tạo cột logo và cột rỗng tỉ lệ 30:1
        logo_col, _ = st.columns([30, 1])

        with logo_col:
            st.title("SA SRE Agent Gen App")
            st.title("Xác nhận")

            # tạo checkbox để user xác nhận điều khoản
            acknowledged = st.checkbox(
                "Tôi xác nhận đã đọc cảnh báo và đồng ý với các điều khoản sử dụng",
                key="acknowledged",
                label_visibility="visible",
            )
            # nếu user chưa đăng nhập thì hiển thị button đăng nhập
            # nút này bị vô hiệu hóa khi user chưa xác nhận điều khoản
            if not st.user.is_logged_in:
                submit = st.button(
                    "Vui lòng đăng nhập trước khi sử dụng",
                    disabled=not (acknowledged),
                )
                # nếu user click vào nút đăng nhập thì đăng nhập với provider microsoft
                if submit:
                    st.login("google")

            if st.user.is_logged_in:
                st.session_state.conversation_id = str(uuid.uuid4())
                st.session_state.user_authenticated = True
                st.session_state.ask_user = True
                st.session_state.isEnabledPrompt = True


                st.rerun()

    # Description and Disclaimer
    # Main page content
    st.markdown(
        "<h1 style='text-align: center;'>Welcome to LLM Evaluator</h1>",
        unsafe_allow_html=True,
    )

    # Description section
    st.header("Description")
    st.write(
        """
    SA Agent Chatbot là ứng dụng hỗ trợ giải đáp thông tin về dịch vụ CMC Cloud.
    Với SA Agent Chatbot, bạn có thể:
    - Tìm kiếm thông tin về dịch vụ CMC Cloud.
    - Tìm kiếm thông tin về các sản phẩm và dịch vụ của CMC Cloud.
    - Tìm kiếm thông tin về các giải pháp của CMC Cloud.
    - Tìm kiếm thông tin về các chính sách của CMC Cloud.
    - Tìm kiếm thông tin về các quy định của CMC Cloud.
    - Tìm kiếm thông tin về các quy trình của CMC Cloud.
    """
    )
    # Add some space between sections
    st.markdown("---")

    # Phần cảnh báo (Disclaimer)
    st.header("Cảnh báo & Điều khoản sử dụng")
    st.write(
        """
    Bằng việc sử dụng SA Agent Chatbot, bạn xác nhận đã đọc, hiểu và đồng ý với các điều khoản.
    SA Agent Chatbot không phải là tư vấn chuyên môn, pháp lý, tài chính hoặc y tế.
    Chỉ là dự án cá nhân để hỗ trợ giải đáp thông tin về dịch vụ CMC Cloud.
        """
    )