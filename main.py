import sys
from utils.logger import get_logger

logger = get_logger("Main")

def print_menu():
    print("\n===============================================")
    print("      ROBOT AI SERVER - MENU BẮT ĐẦU           ")
    print("===============================================")
    print("1. Chạy Vision (Phát hiện & Theo dõi người Face-to-Face)")
    print("2. Chạy STT (Nhận dạng giọng nói tiếng Việt bằng Whisper)")
    print("3. Chạy ROS 2 Command Node (Lắng nghe topic /robot/command)")
    print("4. Gửi lệnh văn bản thử nghiệm")
    print("0. Thoát")
    print("===============================================")

def main():
    while True:
        print_menu()
        choice = input("Nhập lựa chọn của bạn (0-4): ").strip()

        if choice == "1":
            from scripts.main_vision import main as run_vision
            run_vision()
        elif choice == "2":
            from scripts.main_stt import main as run_stt
            run_stt()
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
