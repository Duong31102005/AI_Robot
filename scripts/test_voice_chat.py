"""
Standalone Script Test Giọng Nói & Trả Lời AI Qua Loa (Voice STT -> LLM -> TTS Speaker)
========================================================================================
Chỉ tập trung kiểm tra:
1. Nhận dạng giọng nói qua Microphone (STT)
2. Hỏi đáp Trí tuệ nhân tạo Ollama LLM (qwen2.5:3b)
3. Cất giọng trả lời trực tiếp ra Loa PC và Loa Bluetooth Raspberry Pi (TTS)
"""

import os
import sys
import re
import time
import threading

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio.whisper_stt import WhisperSTT
from audio.tts_engine import TTSEngine
from llm.ollama_llm import OllamaLLM
from communication.pi_client import PiClient
from utils.logger import get_logger

logger = get_logger("TestVoiceChat")

WAKE_KEYWORDS = [
    "rùa", "rùa ơi", "ơi rùa", "con rùa", "robot rùa", "bạn rùa", "chú rùa",
    "kim qui", "kim quy", "kim quý", "kim qui ơi", "ơi kim qui",
    "minh quý", "nguyễn minh quý", "kỳ quý", "kín quý", "chim quý",
    "phương nam", "phương nam ơi", "ơi phương nam", "robot phương nam"
]


def parse_wake_word(text: str) -> tuple[bool, str]:
    text_lower = text.lower().strip()
    for wake in WAKE_KEYWORDS:
        if wake in text_lower:
            cleaned = re.sub(re.escape(wake), '', text_lower, flags=re.IGNORECASE).strip()
            cleaned = re.sub(r'^(ơi|hãy|làm ơn|bạn|giúp)\s*', '', cleaned).strip()
            return True, cleaned
    return False, text


def main():
    print("\n=======================================================")
    print(" 🎙️ TEST CHUYÊN SÂU: GIỌNG NÓI -> AI TRẢ LỜI QUA LOA ")
    print("=======================================================")
    print("Mô tả: Bạn hãy nói vào Micro (Ví dụ: 'Kim Qui ơi, bạn là ai?').")
    print("AI sẽ đọc hiểu và cất giọng trả lời ra Loa PC + Loa Pi.")
    print("Nhấn Ctrl+C để dừng.\n")

    stt = WhisperSTT()
    llm = OllamaLLM()
    tts = TTSEngine()
    pi_client = PiClient()

    if llm.is_available():
        logger.info("Ollama LLM: CONNECTED & ONLINE")
    else:
        logger.warning("Ollama LLM: DISCONNECTED (Hãy chắc chắn bạn đã bật Ollama).")

    logger.info(">>> ĐÃ BẬT MICROPHONE! Hãy thử gọi 'Kim Qui ơi, bạn tên là gì?'...")

    try:
        while True:
            start_pipeline_t = time.perf_counter()
            raw_text, vad_ms, stt_ms = stt.listen_and_transcribe()

            if not raw_text:
                continue

            raw_text = raw_text.strip()
            print(f"\n🗣️ BẠN NÓI: \"{raw_text}\" [VAD: {vad_ms:.0f}ms | STT: {stt_ms:.0f}ms]")

            has_wake, prompt = parse_wake_word(raw_text)

            # BẮT BUỘC có từ khóa kích hoạt 'Rùa' hoặc 'Phương Nam' mới trả lời
            if not has_wake:
                logger.info(f"[STT] Bỏ qua giọng nói (Robot bảo vệ, chưa gọi 'Rùa' / 'Phương Nam'): '{raw_text}'")
                continue

            question = prompt

            if not question or question in ["ơi", "à", "ơi bạn", "bạn ơi"]:
                greeting = "Dạ, Rùa nghe đây!"
                print(f"🤖 ROBOT TRẢ LỜI: \"{greeting}\"\n")
                tts.speak(greeting, sync=False)
                pi_client.send_tts(greeting)
                continue

            logger.info(f"Hỏi AI: '{question}'...")
            llm_start_t = time.perf_counter()
            reply = llm.generate_response(question)
            llm_ms = (time.perf_counter() - llm_start_t) * 1000.0

            if reply:
                total_pipeline_ms = (time.perf_counter() - start_pipeline_t) * 1000.0
                print(f"🤖 AI TRẢ LỜI: \"{reply}\" [LLM: {llm_ms:.0f}ms | Tổng: {total_pipeline_ms:.0f}ms]\n")

                # Phát âm thanh ra Loa PC và Loa Bluetooth Pi
                tts.speak(reply, sync=False)
                threading.Thread(target=pi_client.send_tts, args=(reply,), daemon=True).start()

    except KeyboardInterrupt:
        print("\n\nĐã dừng test Voice Chat.")


if __name__ == "__main__":
    main()
