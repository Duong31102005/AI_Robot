import os
import sys

# Đảm bảo import các module từ thư mục gốc dự án
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio.whisper_stt import WhisperSTT
from communication.pi_client import PiClient
from utils.logger import get_logger

logger = get_logger("MainSTT")

def main():
    logger.info("--- KHỞI CHẠY HỆ THỐNG NHẬN DẠNG GIỌNG NÓI STT ---")

    stt = WhisperSTT()
    pi_client = PiClient()

    try:
        text = stt.record_and_transcribe()
        if text:
            logger.info(f"Đã nhận câu thoại: '{text}'")
            success = pi_client.send_command(text)
            if success:
                logger.info("Đã gửi lệnh thành công tới Raspberry Pi.")
            else:
                logger.warning("Gửi lệnh không thành công.")
        else:
            logger.warning("Không nhận diện được giọng nói.")
    except Exception as e:
        logger.error(f"Xảy ra lỗi trong quá trình thu âm/nhận dạng: {e}")

if __name__ == "__main__":
    main()
