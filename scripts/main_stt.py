import os
import sys
import re
import time
import threading

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio.streaming_stt import WhisperStreamingSTT
from api.stt_websocket import broadcast_stt_event
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
    logger.info("  ROBOT KIM QUI (STREAMING STT YOUTUBE-LIKE)  ")
    logger.info("==============================================")

    stt = WhisperStreamingSTT()
    pi_client = PiClient()
    normalizer = CommandNormalizer()

    llm = OllamaLLM() if ENABLE_LLM_CHAT else None
    tts = TTSEngine() if ENABLE_TTS_SPEAKER else None

    # Callback hiển thị chữ tạm thời (Partial Subtitle) kiểu YouTube thời gian thực
    def handle_partial_subtitle(partial_text: str):
        print(f"\r💬 [LIVE SUBTITLE] {partial_text}...", end="", flush=True)
        broadcast_stt_event("partial", partial_text)
        threading.Thread(target=pi_client.send_partial_stt, args=(partial_text,), daemon=True).start()

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

    logger.info("==========================================================================")
    logger.info("🎙️ [CHẾ ĐỘ SẴN SÀNG NGHE AM THANH] ROBOT KIM QUI ĐANG LẮNG NGHE TỪ MICRO CAMERA UGREEN!")
    logger.info("👉 Hãy nói rõ ràng vào Microphone của Camera UGREEN: 'Kim Qui ơi!' hoặc 'Kim Qui đi thẳng'")
    logger.info("==========================================================================")

    is_robot_moving = False
    active_until_time = 0.0  # Thời điểm hết hạn phiên hội thoại liên tục (20 giây)

    try:
        while True:
            # 🎧 STATE: LISTENING
            logger.info("🎧 [ROBOT STATE: LISTENING] Robot đang lắng nghe âm thanh từ Micro UGREEN...")
            
            # Streaming VAD Listening + Realtime Subtitle
            start_pipeline_t = time.perf_counter()
            raw_text, vad_ms, stt_ms = stt.listen_and_stream(on_partial=handle_partial_subtitle)
            print()  # Đổi dòng sau khi chốt câu

            if not raw_text:
                continue

            # ⚡ STATE: PROCESSING_STT
            logger.info(f"⚡ [ROBOT STATE: PROCESSING_STT] Đã dịch giọng nói: '{raw_text}'")

            raw_text = raw_text.strip()
            raw_lower = raw_text.lower()

            # Thoát chương trình khẩn cấp
            if raw_lower in ["thoát", "kết thúc", "dừng chương trình"]:
                pi_client.send_command("dừng")
                is_robot_moving = False
                logger.info("Đã gửi lệnh dừng robot.")
                break

            # 🔍 ĐIỀU KIỆN KÍCH HOẠT: Kiểm tra từ khóa "Kim Qui" hoặc phiên hội thoại đang mở (20s)
            has_wake, prompt = parse_wake_word(raw_text)
            now_t = time.time()

            if not has_wake:
                if now_t < active_until_time:
                    # Trong phiên hội thoại 20 giây: Chấp nhận câu nói trực tiếp không cần nhắc lại "Kim Qui"
                    has_wake = True
                    prompt = raw_text
                else:
                    logger.info(f"[STT] Bỏ qua giọng nói (Robot đang bảo vệ, không gọi 'Kim Qui'): '{raw_text}'")
                    continue

            # Gia hạn phiên hội thoại 20 giây cho câu nói tiếp theo
            active_until_time = time.time() + 20.0

            # Nếu chỉ gọi "Kim Qui ơi" mà không có câu lệnh
            if not prompt or prompt in ["ơi", "à", "ơi bạn", "kim qui", "kim quy"]:
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
                        # 🧠 STATE: THINKING_LLM
                        logger.info(f"🧠 [ROBOT STATE: THINKING_LLM] Đang suy luận câu hỏi: '{q_prompt}'...")
                        llm_start_t = time.perf_counter()
                        reply = llm.generate_response(q_prompt)
                        llm_ms = (time.perf_counter() - llm_start_t) * 1000.0
                        if reply:
                            # 🔊 STATE: SPEAKING_TTS
                            total_pipeline_ms = (time.perf_counter() - pipeline_t) * 1000.0
                            logger.info(f"🔊 [ROBOT STATE: SPEAKING_TTS] AI Trả lời ({llm_ms:.0f}ms): '{reply}'")
                            logger.info(f"[PERF SUMMARY] VAD: {v_ms:.0f} ms | STT: {s_ms:.0f} ms | LLM: {llm_ms:.0f} ms | TOTAL: {total_pipeline_ms:.0f} ms")
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