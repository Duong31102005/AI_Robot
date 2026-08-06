import sys
from utils.logger import get_logger

logger = get_logger("Main")

def print_menu():
    print("\n===============================================")
    print("      ROBOT AI SERVER - MENU BẮT ĐẦU           ")
    print("===============================================")
    print("1. Chạy Vision (Phát hiện & Theo dõi người Face-to-Face)")
    print("2. Chạy STT + LLM (Trò chuyện & Lệnh giọng nói Kim Qui)")
    print("6. 💬 CHẠY CHẾ ĐỘ NHẬP VĂN BẢN WEB CHAT + VISION (Không mở Mic, Nhập Text -> Loa Xe)")
    print("5. 🚀 CHẠY HỢP NHẤT TOÀN BỘ (Vision + STT + LLM Đa Luồng)")
    print("3. Chạy ROS 2 Command Node (Giả lập Topic /robot/command)")
    print("4. Gửi lệnh văn bản thử nghiệm")
    print("0. Thoát")
    print("===============================================")


def run_text_chat():
    from scripts.main_vision import main as run_vision
    logger.info("💬 [PURE WEB TEXT CHAT MODE] Khởi chạy Hệ thống Vision + Web Text Chat (Microphone OFF)...")
    logger.info("👉 Hãy nhập câu hỏi/câu lệnh trực tiếp ở giao diện Web Chat, Kim Qui sẽ trả lời ra Loa Xe Robot!")
    run_vision()


def run_combined():
    import threading
    from scripts.main_vision import main as run_vision
    from scripts.main_stt import main as run_stt

    logger.info("🚀 Đang khởi chạy Hệ thống Hợp nhất Vision + STT Đa luồng...")
    t_vision = threading.Thread(target=run_vision, daemon=True)
    t_stt = threading.Thread(target=run_stt, daemon=True)

    t_vision.start()
    t_stt.start()

    t_vision.join()
    t_stt.join()


def main():
    # Hỗ trợ truyền tham số dòng lệnh trực tiếp (Ví dụ: python main.py 5 hoặc python main.py full)
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower().strip()
        if arg in ("5", "full", "--full", "all"):
            run_combined()
            return
        elif arg in ("6", "text", "--text", "chat"):
            run_text_chat()
            return
        elif arg in ("2", "stt", "--stt"):
            from scripts.main_stt import main as run_stt
            run_stt()
            return
        elif arg in ("1", "vision", "--vision"):
            from scripts.main_vision import main as run_vision
            run_vision()
            return

    while True:
        print_menu()
        choice = input("Nhập lựa chọn của bạn (0-6): ").strip()

        if choice in ("6", "text"):
            run_text_chat()
            break
        elif choice == "1":
            from scripts.main_vision import main as run_vision
            run_vision()
            break
        elif choice == "2":
            from scripts.main_stt import main as run_stt
            run_stt()
            break
        elif choice == "5":
            run_combined()
            break
        elif choice == "3":
            from communication.ros_command_node import run_ros_node
            run_ros_node()
        elif choice == "4":
            from scripts.send_command import main as run_send
            run_send()
        elif choice == "0":
            logger.info("Đã thoát Robot AI Server.")
            sys.exit(0)
        else:
            print("Lựa chọn không hợp lệ, vui lòng thử lại.")

if __name__ == "__main__":
    main()
