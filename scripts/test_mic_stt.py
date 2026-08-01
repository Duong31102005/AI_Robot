"""
Script Test Micro Thời Gian Thực (Using Factory Class WhisperSTT)
================================================================
Tự động chuyển đổi giữa PhoWhisperSTT và MoonshineSTT theo cấu hình STT_ENGINE.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio.whisper_stt import WhisperSTT
from config.settings import STT_ENGINE
from utils.logger import get_logger

logger = get_logger("TestMicSTT")


def main():
    print("\n=======================================================")
    print(f"  🎙️ KIỂM TRA STT PRODUCTION ENGINE ({STT_ENGINE.upper()}) ")
    print("=======================================================")
    print("Mô tả: Bạn hãy nói vào Micro máy tính.")
    print(f"Hệ thống đang sử dụng Backend STT: {STT_ENGINE.upper()}")
    print("Nhấn Ctrl+C để dừng chương trình.\n")

    stt = WhisperSTT()

    logger.info(">>> ĐÃ BẬT MICROPHONE! Hãy bắt đầu nói...")

    try:
        while True:
            text, vad_ms, stt_ms = stt.listen_and_transcribe()
            if text:
                print(f"\n🗣️ BẠN VỪA NÓI: \"{text}\" [VAD: {vad_ms:.0f}ms | STT: {stt_ms:.1f}ms]\n")
    except KeyboardInterrupt:
        print("\n\nĐã dừng kiểm tra Microphone.")


if __name__ == "__main__":
    main()
