import os
import sys
import threading
try:
    import pyttsx3
    HAS_PYTTSX3 = True
except ImportError:
    HAS_PYTTSX3 = False

from utils.logger import get_logger

logger = get_logger("TTSEngine")


class TTSEngine:
    """
    Module Text-to-Speech (TTS) đọc câu trả lời ra Loa 100% offline.
    Mặc định sử dụng pyttsx3 tích hợp sẵn trên hệ điều hành.
    """

    def __init__(self, rate: int = 160, volume: float = 1.0):
        self.rate = rate
        self.volume = volume
        self._lock = threading.Lock()
        self._engine = None
        self._init_engine()

    def _init_engine(self):
        try:
            self._engine = pyttsx3.init()
            self._engine.setProperty('rate', self.rate)
            self._engine.setProperty('volume', self.volume)

            # Chọn giọng tiếng Việt hoặc giọng nữ mặc định nếu có
            voices = self._engine.getProperty('voices')
            for voice in voices:
                if 'vietnam' in voice.name.lower() or 'vi' in voice.id.lower():
                    self._engine.setProperty('voice', voice.id)
                    break
        except Exception as e:
            logger.warning(f"[TTS] pyttsx3 init warning: {e}")

    def speak(self, text: str, sync: bool = True):
        """Phát âm thanh đọc đoạn văn bản ra Loa."""
        if not text or not text.strip():
            return

        logger.info(f"[TTS] Speaking: '{text}'")

        def _run_speech():
            with self._lock:
                try:
                    engine = pyttsx3.init()
                    engine.setProperty('rate', self.rate)
                    engine.setProperty('volume', self.volume)
                    engine.say(text)
                    engine.runAndWait()
                    engine.stop()
                except Exception as e:
                    logger.error(f"[TTS] Speak error: {e}")

        if sync:
            _run_speech()
        else:
            thread = threading.Thread(target=_run_speech, daemon=True)
            thread.start()
