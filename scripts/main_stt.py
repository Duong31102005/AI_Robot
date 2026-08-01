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
from utils.command_normalizer import CommandNormalizer
from config.settings import ENABLE_LLM_CHAT, ENABLE_TTS_SPEAKER
from utils.logger import get_logger

logger = get_logger("MainSTT")

WAKE_KEYWORDS = [
    "rùa", "rùa ơi", "ơi rùa", "con rùa", "robot rùa", "bạn rùa", "chú rùa",
    "kim qui", "kim quy", "kim quý", "kim qui ơi", "ơi kim qui",
    "minh quý", "nguyễn minh quý", "kỳ quý", "kín quý", "chim quý",
    "phương nam", "phương nam ơi", "ơi phương nam", "robot phương nam"
]


def parse_wake_word(text: str) -> tuple[bool, str]:
    """
    Kiểm tra Từ khóa kích hoạt 'Kim Qui'.
    Nếu có từ khóa 'Kim Qui', trả về (True, câu_hỏi_đã_lọc).
    Nếu không có từ khóa, trả về (False, text).
    """
    text_lower = text.lower().strip()

    # Lệnh dừng khẩn cấp được ưu tiên không cần wake word
    if text_lower in ["thoát", "kết thúc", "dừng chương trình", "dừng", "dừng lại"]:
        return True, text

    for wake in WAKE_KEYWORDS:
        if wake in text_lower:
            cleaned = re.sub(re.escape(wake), '', text_lower, flags=re.IGNORECASE).strip()
            cleaned = re.sub(r'^(ơi|hãy|làm ơn|bạn|giúp)\s*', '', cleaned).strip()
            return True, cleaned

    return False, text


def main():
    logger.info("==============================================")
    logger.info("    ROBOT KIM QUI (PHOWHISPER / MOONSHINE) - ONLINE   ")
    logger.info("==============================================")

    stt = WhisperSTT()
    pi_client = PiClient()
    normalizer = CommandNormalizer()

    llm = OllamaLLM() if ENABLE_LLM_CHAT else None
    tts = TTSEngine() if ENABLE_TTS_SPEAKER else None

    # Kiểm tra kết nối Pi
    if not pi_client.test_connection():
        logger.error("Không kết nối được Raspberry Pi.")
        return

    logger.info("Raspberry Pi: CONNECTED")

    # Kiểm tra kết nối Ollama LLM
    if llm and llm.is_available():
        logger.info(f"Ollama LLM ({llm.model}): CONNECTED & ONLINE")
    else:
        logger.warning("Ollama LLM Server: DISCONNECTED (Chạy 'ollama run qwen2.5:3b' để bật tính năng trò chuyện).")

    logger.info("STT lắng nghe liên tục.")
    logger.info("Tên kích hoạt Robot: 'KIM QUI' (ví dụ: 'Kim Qui ơi', 'Kim Qui đi thẳng').")
    logger.info("Nói 'thoát' để kết thúc.")

    is_robot_moving = False

    try:
        while True:
            # Continuous VAD Listening + Moonshine Transcribe
            start_pipeline_t = time.perf_counter()
            raw_text, vad_ms, stt_ms = stt.listen_and_transcribe()

            if not raw_text:
                continue

            raw_text = raw_text.strip()
            raw_lower = raw_text.lower()

            # Thoát chương trình khẩn cấp
            if raw_lower in ["thoát", "kết thúc", "dừng chương trình"]:
                pi_client.send_command("dừng")
                is_robot_moving = False
                logger.info("Đã gửi lệnh dừng robot.")
                break

            # 🔍 ĐIỀU KIỆN KÍCH HOẠT: Kiểm tra từ khóa "Kim Qui"
            has_wake, prompt = parse_wake_word(raw_text)

            if not has_wake:
                logger.info(f"[STT] Bỏ qua giọng nói (Robot đang bảo vệ, không gọi 'Kim Qui'): '{raw_text}'")
                continue

            # Nếu chỉ gọi "Kim Qui ơi" mà không có câu lệnh
            if not prompt or prompt in ["ơi", "à", "ơi bạn"]:
                greeting = "Dạ, Kim Qui nghe đây!"
                total_pipeline_ms = (time.perf_counter() - start_pipeline_t) * 1000.0
                logger.info(f"[PERF] VAD: {vad_ms:.0f} ms | Moonshine: {stt_ms:.0f} ms | LLM: 0 ms | TTS: 10 ms | TOTAL: {total_pipeline_ms:.0f} ms")
                logger.info(f"[STT] WAKE WORD DETECTED -> Respondent: '{greeting}'")
                if tts:
                    tts.speak(greeting, sync=False)
                threading.Thread(target=pi_client.send_tts, args=(greeting,), daemon=True).start()
                threading.Thread(target=pi_client.send_conversation, args=("Kim Qui ơi", greeting), daemon=True).start()
                continue

            # 1. Kiểm tra Lệnh di chuyển Robot (Chỉ kích hoạt khi gọi Kim Qui)
            command = normalizer.normalize(prompt)

            if command:
                if command == "dừng":
                    is_robot_moving = False
                    total_pipeline_ms = (time.perf_counter() - start_pipeline_t) * 1000.0
                    logger.info(f"[PERF] VAD: {vad_ms:.0f} ms | Moonshine: {stt_ms:.0f} ms | LLM: 0 ms | TTS: 10 ms | TOTAL: {total_pipeline_ms:.0f} ms")
                    logger.info("[STT] WAKE COMMAND: Dừng Robot!")
                    pi_client.send_command("dừng")
                    response_str = "Kim Qui đã dừng lại"
                    if tts:
                        tts.speak(response_str, sync=False)
                    threading.Thread(target=pi_client.send_tts, args=(response_str,), daemon=True).start()
                    threading.Thread(target=pi_client.send_conversation, args=(prompt, response_str), daemon=True).start()
                else:
                    is_robot_moving = True
                    total_pipeline_ms = (time.perf_counter() - start_pipeline_t) * 1000.0
                    logger.info(f"[PERF] VAD: {vad_ms:.0f} ms | Moonshine: {stt_ms:.0f} ms | LLM: 0 ms | TTS: 10 ms | TOTAL: {total_pipeline_ms:.0f} ms")
                    logger.info(f"[STT] WAKE COMMAND: '{raw_text}' -> '{command}' (Robot bắt đầu di chuyển)")
                    success = pi_client.send_command(command)
                    response_str = f"Kim Qui đã nhận lệnh {command}"
                    if tts:
                        tts.speak(response_str, sync=False)
                    if success:
                        threading.Thread(target=pi_client.send_tts, args=(response_str,), daemon=True).start()
                        threading.Thread(target=pi_client.send_conversation, args=(prompt, response_str), daemon=True).start()
                    elif not success:
                        logger.warning("Gửi lệnh thất bại.")
            else:
                # 2. Xử lý Trò chuyện / Hỏi đáp LLM trong luồng ngầm không làm đơ hệ thống
                logger.info(f"[STT] WAKE CHAT QUESTION: '{prompt}'")
                def _async_chat_process(q_prompt, pipeline_t, v_ms, s_ms):
                    if llm and llm.is_available():
                        llm_start_t = time.perf_counter()
                        reply = llm.generate_response(q_prompt)
                        llm_ms = (time.perf_counter() - llm_start_t) * 1000.0
                        if reply:
                            total_pipeline_ms = (time.perf_counter() - pipeline_t) * 1000.0
                            logger.info(f"[PERF] VAD: {v_ms:.0f} ms | STT: {s_ms:.0f} ms | LLM: {llm_ms:.0f} ms | TOTAL: {total_pipeline_ms:.0f} ms")
                            logger.info(f"🗣️ AI TRẢ LỜI: '{reply}'")
                            if tts:
                                tts.speak(reply, sync=False)
                            pi_client.send_tts(reply)
                            pi_client.send_conversation(q_prompt, reply)
                    else:
                        logger.warning(f"Bỏ qua câu trò chuyện (LLM không khả dụng): '{q_prompt}'")

                threading.Thread(target=_async_chat_process, args=(prompt, start_pipeline_t, vad_ms, stt_ms), daemon=True).start()

    except KeyboardInterrupt:
        logger.info("CTRL+C - dừng robot.")
        pi_client.send_command("dừng")

    except Exception as e:
        logger.error(f"Lỗi STT: {e}")
        pi_client.send_command("dừng")


if __name__ == "__main__":
    main()