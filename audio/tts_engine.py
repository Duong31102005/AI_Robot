import os
import re
import sys
import tempfile
import urllib.parse
import urllib.request
import subprocess
import threading

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

try:
    import pyttsx3
    HAS_PYTTSX3 = True
except ImportError:
    HAS_PYTTSX3 = False

from utils.logger import get_logger

logger = get_logger("TTSEngine")


class TTSEngine:
    """
    Module Text-to-Speech (TTS) đọc câu trả lời Tiếng Việt 100% chuẩn xác.
    Ưu tiên giọng đọc Google Tiếng Việt (ngắt câu tự động), fallback pyttsx3 nếu offline.
    """

    def __init__(self, rate: int = 160, volume: float = 1.0):
        self.rate = rate
        self.volume = volume
        self._lock = threading.Lock()

    def speak(self, text: str, sync: bool = True):
        """Phát âm thanh đọc đoạn văn bản Tiếng Việt ra Loa."""
        if not text or not text.strip():
            return

        logger.info(f"[TTS] Speaking (Tiếng Việt): '{text}'")

        def _run_speech():
            with self._lock:
                # 1. Thử dùng Google TTS Tiếng Việt chuẩn (có ngắt đoạn <= 150 ký tự)
                if self._play_google_tts_vi(text):
                    return

                # 2. Fallback offline: Dùng pyttsx3 nếu không có mạng
                if HAS_PYTTSX3:
                    try:
                        engine = pyttsx3.init()
                        engine.setProperty('rate', self.rate)
                        engine.setProperty('volume', self.volume)
                        voices = engine.getProperty('voices')
                        for voice in voices:
                            v_str = (voice.name + " " + voice.id).lower()
                            if any(k in v_str for k in ['vietnam', 'vietnamese', 'vi_vn', 'vi-vn', 'hoaimy', 'an']):
                                engine.setProperty('voice', voice.id)
                                break
                        engine.say(text)
                        engine.runAndWait()
                        engine.stop()
                    except Exception as e:
                        logger.error(f"[TTS] pyttsx3 speak error: {e}")

        if sync:
            _run_speech()
        else:
            thread = threading.Thread(target=_run_speech, daemon=True)
            thread.start()

    def _play_google_tts_vi(self, text: str) -> bool:
        """Tải và phát giọng Google Tiếng Việt chuẩn với chia nhỏ văn bản tự động."""
        try:
            sentences = re.split(r'([.!?,;\n])', text)
            chunks = []
            current = ""
            for item in sentences:
                if len(current) + len(item) <= 150:
                    current += item
                else:
                    if current.strip():
                        chunks.append(current.strip())
                    current = item
            if current.strip():
                chunks.append(current.strip())

            if not chunks:
                chunks = [text[:150]]

            headers = {'User-Agent': 'Mozilla/5.0'}
            combined_mp3 = bytearray()

            for chunk in chunks:
                url = f"https://translate.google.com/translate_tts?ie=UTF-8&q={urllib.parse.quote(chunk)}&tl=vi&client=tw-ob"
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=5) as response:
                    combined_mp3.extend(response.read())

            if not combined_mp3:
                return False

            temp_dir = tempfile.gettempdir()
            mp3_file = os.path.join(temp_dir, "robot_tts.mp3")
            wav_file = os.path.join(temp_dir, "robot_tts.wav")

            with open(mp3_file, 'wb') as f:
                f.write(combined_mp3)

            # Chuyển MP3 thành WAV bằng ffmpeg để phát 100% mượt mà
            subprocess.run(["ffmpeg", "-y", "-i", mp3_file, "-ar", "22050", "-ac", "1", wav_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            if os.path.exists(wav_file):
                if HAS_WINSOUND and os.name == 'nt':
                    winsound.PlaySound(wav_file, winsound.SND_FILENAME)
                    return True
                else:
                    res = subprocess.run(["paplay", wav_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    if res.returncode == 0:
                        return True
                    res = subprocess.run(["aplay", "-D", "default", wav_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    if res.returncode == 0:
                        return True
            return False
        except Exception as e:
            logger.warning(f"[TTS] Google TTS online warning: {e}")
            return False
