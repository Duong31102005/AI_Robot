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

    def is_connected(self) -> bool:
        """Trả về trạng thái kết nối gần nhất."""
        return self.last_connected_status
