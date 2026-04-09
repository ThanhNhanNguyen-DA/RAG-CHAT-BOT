import os
import sys
import subprocess

def main():
    """
    Điểm vào chính của ứng dụng. 
    Script này sẽ tự động gọi Streamlit để khởi chạy giao diện người dùng.
    """
    print("🚀 Đang khởi động CMC Cloud RAG Chatbot...")
    
    # Lấy đường dẫn tuyệt đối đến tệp streamlit_app.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ui_path = os.path.join(current_dir, "ui", "streamlit_app.py")
    
    # Chạy lệnh: streamlit run app/ui/streamlit_app.py
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", ui_path])
    except KeyboardInterrupt:
        print("\n👋 Đã tắt Chatbot.")
    except Exception as e:
        print(f"❌ Lỗi khi khởi động UI: {e}")

if __name__ == "__main__":
    main()