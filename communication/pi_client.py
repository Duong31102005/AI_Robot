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

    def send_command(self, text: str, timeout: float = 3.0) -> bool:
        """Gửi lệnh văn bản dạng JSON {"text": text} tới Raspberry Pi."""
        if not text:
            logger.warning("[PI] Lệnh rỗng, không gửi.")
            return False

        if self.dry_run:
            logger.info(f"[PI] [DRY_RUN] Command simulated: '{text}' -> {self.url}")
            return True

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
            logger.error(f"[PI] Error sending command '{text}' to Pi: {e}")
            self.last_connected_status = False
            return False

    def send_tts(self, text: str, timeout: float = 3.0) -> bool:
        """Gửi câu văn bản TTS xuống Raspberry Pi (Cổng 8001 /tts) để đọc ra Loa Bluetooth cắm ở Pi."""
        if not text:
            return False

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
        """Gửi danh sách vật thể YOLO nhận dạng được lên HTTP bridge của Pi (port 8001)."""
        if not detections:
            return False
        if self.dry_run:
            return True

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
