import os
import sys
import re
import time
import threading

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import STT_ENGINE, ENABLE_LLM_CHAT, ENABLE_TTS_SPEAKER

if STT_ENGINE == "parakeet":
    from audio.parakeet_stt import ParakeetSTT as WhisperStreamingSTT
else:
    from audio.phowhisper_stt import PhoWhisperSTT as WhisperStreamingSTT

from api.stt_websocket import broadcast_stt_event
from audio.tts_engine import TTSEngine
from llm.ollama_llm import OllamaLLM
from communication.pi_client import PiClient
from utils.command_normalizer import CommandNormalizer
from utils.logger import get_logger

logger = get_logger("MainSTT")

WAKE_KEYWORDS = [
    "kim", "quy", "qui", "kìm", "quỳ", "quý",
    "rùa", "rùa ơi", "ơi rùa", "con rùa", "robot rùa", "bạn rùa", "chú rùa",
    "kim qui", "kim quy", "kim quý", "kim qui ơi", "ơi kim qui",
    "minh quý", "nguyễn minh quý", "kỳ quý", "kín quý", "chim quý",
    "phương nam", "phương nam ơi", "ơi phương nam", "robot phương nam",
    "rẽ", "rẽ lên", "rẽ trái", "rẽ phải", "đi", "đi thẳng", "tiến", "lùi", "dừng", "dừng lại", "xoay", "quẹo", "chào"
]


def parse_wake_word(text: str) -> tuple[bool, str]:
    """
    Kiểm tra Từ khóa kích hoạt 'Kim Qui'.
    Nếu có từ khóa 'Kim Qui', giữ NGUYÊN BẢN 100% toàn bộ câu nói của người dùng.
    """
    text_lower = text.lower().strip()

    # Lệnh dừng khẩn cấp được ưu tiên không cần wake word
    if text_lower in ["thoát", "kết thúc", "dừng chương trình", "dừng", "dừng lại"]:
        return True, text

    for wake in WAKE_KEYWORDS:
        if wake in text_lower:
            return True, text.strip()

    return False, text.strip()


def main():
    logger.info("==============================================")
    logger.info("  ROBOT KIM QUI (STREAMING STT YOUTUBE-LIKE)  ")
    logger.info("==============================================")

    stt = WhisperStreamingSTT()
    pi_client = PiClient()
    normalizer = CommandNormalizer()

    llm = OllamaLLM() if ENABLE_LLM_CHAT else None
    tts = TTSEngine() if ENABLE_TTS_SPEAKER else None

    global _global_pi_client, _global_llm
    _global_pi_client = pi_client
    _global_llm = llm

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

            # 🔍 ĐIỀU KIỆN KÍCH HOẠT THEO KIẾN TRÚC ROBOT XIAOZHI (小智 AI)
            from audio.audio_session import session_manager
            has_wake, prompt = parse_wake_word(raw_text)

            if has_wake:
                session_manager.trigger_wake_word()
            elif session_manager.is_active():
                # Trong phiên hội thoại 12 giây XiaoZhi: Chấp nhận câu nói trực tiếp không cần gọi lại "Kim Qui"
                prompt = raw_text
            else:
                logger.info(f"[STT] Bỏ qua giọng nói (Robot IDLE - chờ gọi 'Kim Qui'): '{raw_text}'")
                continue

            # Lọc bỏ câu rác, câu rỗng hoặc ký tự đặc biệt vô nghĩa
            prompt_clean = prompt.strip().strip(".!?,")
            if not prompt_clean or len(prompt_clean) < 2 or prompt_clean in ["à", "ừ", "ơ", "thì", "nhé", "nha"]:
                greeting = "Dạ, Kim Qui nghe đây!"
                total_pipeline_ms = (time.perf_counter() - start_pipeline_t) * 1000.0
                logger.info(f"[PERF] VAD: {vad_ms:.0f} ms | STT: {stt_ms:.0f} ms | LLM: 0 ms | TOTAL: {total_pipeline_ms:.0f} ms")
                logger.info(f"[STT] WAKE WORD DETECTED -> Respondent: '{greeting}'")
                if tts:
                    tts.speak(greeting, sync=False)
                threading.Thread(target=pi_client.send_tts, args=(greeting,), daemon=True).start()
                threading.Thread(target=pi_client.send_conversation, args=("Kim Qui ơi", greeting), daemon=True).start()
                continue

            # Gia hạn phiên hội thoại 12 giây XiaoZhi cho câu hỏi tiếp theo
            session_manager.refresh_session()

            # 1. Kiểm tra Lệnh di chuyển Robot (Ưu tiên xử lý lệnh di chuyển)
            command = normalizer.normalize(prompt_clean)

            if command:
                if command == "dừng":
                    is_robot_moving = False
                    total_pipeline_ms = (time.perf_counter() - start_pipeline_t) * 1000.0
                    logger.info(f"[PERF] VAD: {vad_ms:.0f} ms | STT: {stt_ms:.0f} ms | LLM: 0 ms | TOTAL: {total_pipeline_ms:.0f} ms")
                    logger.info("[STT] WAKE COMMAND: Dừng Robot!")
                    pi_client.send_command("dừng")
                    response_str = "Kim Qui đã dừng lại"
                    if tts:
                        tts.speak(response_str, sync=False)
                    threading.Thread(target=pi_client.send_tts, args=(response_str,), daemon=True).start()
                    threading.Thread(target=pi_client.send_conversation, args=(prompt_clean, response_str), daemon=True).start()
                else:
                    is_robot_moving = True
                    total_pipeline_ms = (time.perf_counter() - start_pipeline_t) * 1000.0
                    logger.info(f"[PERF] VAD: {vad_ms:.0f} ms | STT: {stt_ms:.0f} ms | LLM: 0 ms | TOTAL: {total_pipeline_ms:.0f} ms")
                    logger.info(f"[STT] WAKE COMMAND: '{raw_text}' -> '{command}' (Robot bắt đầu di chuyển)")
                    success = pi_client.send_command(command)
                    response_str = f"Kim Qui đã nhận lệnh {command}"
                    if tts:
                        tts.speak(response_str, sync=False)
                    if success:
                        threading.Thread(target=pi_client.send_tts, args=(response_str,), daemon=True).start()
                        threading.Thread(target=pi_client.send_conversation, args=(prompt_clean, response_str), daemon=True).start()
                    elif not success:
                        logger.warning("Gửi lệnh thất bại.")
            else:
                # 2. Xử lý Trò chuyện / Hỏi đáp LLM trong luồng ngầm không làm đơ hệ thống
                logger.info(f"[STT] XIAOZHI SESSION CHAT: '{prompt_clean}'")
                def _async_chat_process(q_prompt, pipeline_t, v_ms, s_ms):
                    if llm and llm.is_available():
                        # 🧠 STATE: THINKING_LLM
                        from vision.vision_intelligence import vision_intelligence
                        vision_intelligence.current_expression = "(🧠_🧠) THINKING"
                        vision_intelligence.expression_color = (255, 255, 0)
                        logger.info(f"🧠 [ROBOT STATE: THINKING_LLM] Đang suy luận câu hỏi: '{q_prompt}'...")
                        llm_start_t = time.perf_counter()

                        # 👁️ HỎI ĐÁP THỊ GIÁC AI (VISION VLM) KHI HỎI "KIM QUI NHÌN THẤY GÌ"
                        if any(kw in q_prompt.lower() for kw in ["nhìn thấy gì", "thấy gì", "phía trước có gì", "trước mặt có gì"]):
                            b64_img = vision_intelligence.encode_frame_to_base64(getattr(vision_intelligence, 'current_frame', None))
                            if b64_img and hasattr(llm, 'describe_image'):
                                reply = llm.describe_image(b64_img, q_prompt)
                            else:
                                reply = llm.generate_response(q_prompt)
                        else:
                            reply = llm.generate_response(q_prompt)

                        llm_ms = (time.perf_counter() - llm_start_t) * 1000.0
                        if reply:
                            # 🔊 STATE: SPEAKING_TTS
                            vision_intelligence.current_expression = "(🔊_🔊) SPEAKING"
                            vision_intelligence.expression_color = (0, 255, 255)
                            total_pipeline_ms = (time.perf_counter() - pipeline_t) * 1000.0
                            logger.info(f"🔊 [ROBOT STATE: SPEAKING_TTS] AI Trả lời ({llm_ms:.0f}ms): '{reply}'")
                            logger.info(f"[PERF SUMMARY] VAD: {v_ms:.0f} ms | STT: {s_ms:.0f} ms | LLM: {llm_ms:.0f} ms | TOTAL: {total_pipeline_ms:.0f} ms")
                            if tts:
                                tts.speak(reply, sync=False)
                            pi_client.send_tts(reply)
                            pi_client.send_conversation(q_prompt, reply)

                            # 🟢 GIA HẠN THÊM PHIÊN 12S XIAOZHI TÍNH TỪ THỜI ĐIỂM LOA ĐỌC XONG
                            session_manager.trigger_wake_word()
                            vision_intelligence.current_expression = "(🎧_🎧) LISTENING"
                            vision_intelligence.expression_color = (0, 255, 0)
                    else:
                        logger.warning(f"Bỏ qua câu trò chuyện (LLM không khả dụng): '{q_prompt}'")

                threading.Thread(target=_async_chat_process, args=(prompt_clean, start_pipeline_t, vad_ms, stt_ms), daemon=True).start()

    except KeyboardInterrupt:
        logger.info("CTRL+C - dừng robot.")
        pi_client.send_command("dừng")

    except Exception as e:
        logger.error(f"Lỗi STT: {e}")
        pi_client.send_command("dừng")

_global_pi_client = None
_global_llm = None

def process_text_prompt(prompt: str) -> str:
    """
    Xử lý câu hỏi / câu lệnh nhập từ Web Chat Text.
    In Log terminal ĐẦY ĐỦ Y HỆT KHI NÓI QUA MICROPHONE:
    - 🧠 [ROBOT STATE: THINKING_LLM]
    - 🔊 [ROBOT STATE: SPEAKING_TTS]
    - Gửi loa TTS ra Pi và thực thi lệnh bánh xe.
    """
    global _global_pi_client, _global_llm
    prompt_clean = prompt.strip()
    if not prompt_clean:
        return "Dạ, Kim Qui đang lắng nghe bạn đây ạ!"

    logger.info(f"💬 [WEB TEXT PROMPT RECEIVED] Nhận câu hỏi từ Web Chat Text: '{prompt_clean}'")
    logger.info(f"🧠 [ROBOT STATE: THINKING_LLM] Đang suy luận câu hỏi: '{prompt_clean}'...")

    try:
        from vision.vision_intelligence import vision_intelligence
        vision_intelligence.current_expression = "(🧠_🧠) THINKING"
        vision_intelligence.expression_color = (255, 255, 0)
    except Exception:
        pass

    # 1. Thực thi Lệnh Bánh Xe nếu là câu lệnh di chuyển
    cmd_norm = CommandNormalizer.normalize(prompt_clean)
    if cmd_norm:
        logger.info(f"🚗 [WEB COMMAND EXECUTE] Nhận diện lệnh bánh xe: '{cmd_norm}'")
        if _global_pi_client:
            _global_pi_client.send_command(cmd_norm)

    # 2. Suy luận câu trả lời qua Knowledge Base hoặc LLM
    reply = ""

    # a. Tra cứu tri thức nhanh (Instant Knowledge Base & Fuzzy Typo ASCII Match - 0ms)
    p_lower = prompt_clean.lower()
    ascii_clean = CommandNormalizer.remove_accents(p_lower)

    if any(kw in ascii_clean for kw in ["giam khao", "giam thao", "giam kho", "ban giam", "thay co", "hoi dong"]):
        reply = "Dạ, Kim Qui xin kính chào Ban Giám khảo và quý Thầy Cô! Kim Qui rất vinh dự được đồng hành và phục vụ quý Thầy Cô hôm nay ạ!"
    elif any(kw in ascii_clean for kw in ["ten gi", "ten la gi", "ten ban", "cau ten gi", "gioi thieu", "ban la ai"]):
        reply = "Dạ, em tên là Kim Qui, được sáng tạo bởi nhóm sinh viên Galacticos Khoa CNTT Trường Đại học Đại Nam ạ!"
    elif any(kw in ascii_clean for kw in ["xin chao", "chao ban", "hello", "hi kim qui", "chao"]):
        reply = "Dạ, Kim Qui xin chào bạn! Kim Qui đang sẵn sàng hỗ trợ di chuyển và trả lời câu hỏi nè!"

    # b. Nếu chưa có trong KB -> Gọi Ollama LLM
    if not reply:
        if _global_llm is None:
            try:
                _global_llm = OllamaLLM()
            except Exception:
                pass

        if _global_llm and _global_llm.is_available():
            reply = _global_llm.generate_response(prompt_clean)

    if not reply:
        reply = f"Dạ, Kim Qui đã nhận được câu hỏi: '{prompt_clean}'. Kim Qui luôn sẵn sàng hỗ trợ bạn ạ!"

    logger.info(f"🔊 [ROBOT STATE: SPEAKING_TTS] AI Trả lời: '{reply}'")
    sys.stdout.flush()

    try:
        from vision.vision_intelligence import vision_intelligence
        vision_intelligence.current_expression = "(🔊_🔊) SPEAKING"
        vision_intelligence.expression_color = (0, 255, 255)
    except Exception:
        pass

    # 3. Gửi tiếng nói TTS TRỰC TIẾP RA LOA CỦA ROBOT TRÊN RASPBERRY PI
    if _global_pi_client is None:
        try:
            _global_pi_client = PiClient()
        except Exception:
            pass

    if _global_pi_client:
        target_client = _global_pi_client
        threading.Thread(target=target_client.send_tts, args=(reply,), daemon=True).start()
        threading.Thread(target=target_client.send_conversation, args=(prompt_clean, reply), daemon=True).start()

    return reply


if __name__ == "__main__":
    main()