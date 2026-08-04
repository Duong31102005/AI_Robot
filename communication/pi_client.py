import requests
from config.settings import PI_COMMAND_URL, DRY_RUN
from utils.logger import get_logger

logger = get_logger("PiClient")

class PiClient:
    """Client giao tiếp HTTP với Raspberry Pi của Robot."""

    def __init__(self, url: str = PI_COMMAND_URL, dry_run: bool = DRY_RUN):
        self.url = url
        self.dry_run = dry_run
        self.last_connected_status = False

    def test_connection(self, timeout: float = 3.0) -> bool:
        """Kiểm tra đường truyền HTTP tới Raspberry Pi server."""
        if self.dry_run:
            logger.info("[PI] DRY_RUN Mode enabled: Giả lập kết nối thành công.")
            self.last_connected_status = True
            return True

        logger.info(f"[PI] Checking connection to Raspberry Pi ({self.url})...")
        try:
            # Gửi thử 1 lệnh 'giu_nguyen' test
            response = requests.post(
                self.url,
                json={"text": "giu_nguyen"},
                timeout=timeout
            )
            is_ok = (response.status_code == 200)
            self.last_connected_status = is_ok
            if is_ok:
                logger.info(f"[PI] Connected successfully! (HTTP {response.status_code})")
            else:
                logger.warning(f"[PI] Connection response HTTP {response.status_code}")
            return is_ok
        except Exception as e:
            logger.error(f"[PI] Connection failed: {e}")
            self.last_connected_status = False
            return False

    check_connection = test_connection

    def send_command(self, text: str, timeout: float = 0.5) -> bool:
        """Gửi lệnh văn bản dạng JSON {"text": text} tới Raspberry Pi."""
        if not text:
            return False

        if self.dry_run:
            logger.info(f"[PI] [DRY_RUN] Command simulated: '{text}' -> {self.url}")
            return True

        import time
        if hasattr(self, '_last_fail_time'):
            if not self.last_connected_status and (time.time() - self._last_fail_time < 2.0):
                return False

        try:
            response = requests.post(
                self.url,
                json={"text": text},
                timeout=timeout
            )
            is_ok = (response.status_code == 200)
            self.last_connected_status = is_ok
            if is_ok:
                logger.info(f"[PI] Command sent successfully: '{text}' (HTTP 200)")
            else:
                logger.warning(f"[PI] Send failed HTTP {response.status_code}: {response.text}")
            return is_ok
        except Exception as e:
            self._last_fail_time = time.time()
            if self.last_connected_status:
                logger.error(f"[PI] Error sending command '{text}' to Pi: {e}")
            self.last_connected_status = False
            return False

    def send_tts(self, text: str, timeout: float = 3.0) -> bool:
        """Gửi câu văn bản TTS xuống Raspberry Pi (Cổng 8001 /tts) để đọc ra Loa Bluetooth cắm ở Pi."""
        if not text:
            return False

        # Khóa Micro thu âm trong lúc Loa đang phát giọng đọc
        try:
            from audio.audio_session import set_tts_speaking_text
            set_tts_speaking_text(text)
        except Exception:
            pass

        if self.dry_run:
            logger.info(f"[PI] [DRY_RUN] TTS simulated: '{text}'")
            return True

        tts_url = self.url.replace("/command", "/tts")
        try:
            response = requests.post(
                tts_url,
                json={"text": text},
                timeout=timeout
            )
            is_ok = (response.status_code == 200)
            if is_ok:
                logger.info(f"[PI] TTS sent to Pi Speaker successfully: '{text}' (HTTP 200)")
            else:
                logger.warning(f"[PI] TTS send failed HTTP {response.status_code}")
            return is_ok
        except Exception as e:
            logger.error(f"[PI] Error sending TTS to Pi: {e}")
            return False

    def is_connected(self) -> bool:
        """Trả về trạng thái kết nối gần nhất."""
        return self.last_connected_status

    def send_conversation(self, prompt: str, reply: str, mission_id: int = 1) -> bool:
        """Gửi nhật ký hội thoại lên HTTP bridge của Pi (port 8001)."""
        if not prompt or not reply:
            return False
        if self.dry_run:
            logger.info(f"[PI] [DRY_RUN] Conversation simulated: User='{prompt}' -> Robot='{reply}'")
            return True

        import json
        conversation_url = self.url.replace("/command", "/conversation")
        try:
            payload = {
                "prompt": prompt,
                "reply": reply,
                "mission_id": mission_id
            }
            # Đồng thời lưu lịch sử hội thoại vào Web Database (Port 8000)
            try:
                requests.post("http://localhost:8000/api/v1/robot/conversation", json={"prompt": prompt, "reply": reply}, timeout=1.0)
            except Exception:
                pass

            response = requests.post(
                conversation_url,
                json={"text": json.dumps(payload)},
                timeout=3.0
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"[PI] Error sending conversation to Pi: {e}")
            return False

    def send_detections(self, detections: list, timeout: float = 1.0) -> bool:
        """Gửi danh sách vật thể YOLO nhận dạng được lên Web Dashboard (port 8000) & HTTP bridge của Pi (port 8001)."""
        if not detections:
            return False
        if self.dry_run:
            return True

        # Gửi danh sách vật thể nhận diện trực tiếp tới Web Backend
        try:
            requests.post("http://localhost:8000/api/v1/robot/detections", json={"detections": detections}, timeout=0.5)
        except Exception:
            pass

        import json
        detection_url = self.url.replace("/command", "/detection")
        try:
            response = requests.post(
                detection_url,
                json={"text": json.dumps(detections)},
                timeout=timeout
            )
            return response.status_code == 200
        except Exception as e:
            return False

    def send_partial_stt(self, text: str, timeout: float = 0.3) -> bool:
        """Gửi phụ đề tạm thời (Partial STT) lên ROS2 topic /speech/partial_text của Pi."""
        if not text or self.dry_run:
            return True
        try:
            url = self.url.replace("/command", "/speech/partial")
            response = requests.post(url, json={"text": text}, timeout=timeout)
            return response.status_code == 200
        except Exception:
            return False

    def send_final_stt(self, text: str, timeout: float = 1.0) -> bool:
        """Gửi phụ đề chính thức (Final STT) lên ROS2 topic /speech/final_text của Pi."""
        if not text or self.dry_run:
            return True
        try:
            url = self.url.replace("/command", "/speech/final")
            response = requests.post(url, json={"text": text}, timeout=timeout)
            return response.status_code == 200
        except Exception:
            return False
