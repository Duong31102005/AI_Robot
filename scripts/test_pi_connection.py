import os
import sys
import time

# Đảm bảo import các module từ thư mục gốc dự án
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from communication.pi_client import PiClient
from config.settings import PI_COMMAND_URL
from utils.logger import get_logger

logger = get_logger("TestPiConn")

def main():
    logger.info(f"=== TEST KẾT NỐI MẠNG HTTP TỚI RASPBERRY PI ({PI_COMMAND_URL}) ===")

    client = PiClient()

    # 1. Test kết nối ban đầu
    connected = client.test_connection()
    if connected:
        logger.info("[PI TEST] Kết nối mạng HTTP hoạt động TỐT!")
    else:
        logger.warning("[PI TEST] Không kết nối được Raspberry Pi. Hãy kiểm tra IP/Port và đảm bảo http_bridge_node đang chạy trên Pi.")

    # 2. Test gửi chuỗi các lệnh điều hướng
    test_commands = ["giu_nguyen", "quay_trai", "quay_phai", "tiens_len", "lui_lai", "giu_nguyen"]

    logger.info("Đang gửi chuỗi lệnh kiểm tra đến Pi...")
    for cmd in test_commands:
        client.send_command(cmd)
        time.sleep(1.0)

    logger.info("=== ĐÃ HOÀN THÀNH TEST KẾT NỐI PI ===")

if __name__ == "__main__":
    main()
