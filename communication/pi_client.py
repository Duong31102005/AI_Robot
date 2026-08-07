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
        # Danh sách IP dự phòng tự động khi IP Pi thay đổi giữa các mạng Wi-Fi
        self.candidate_ips = ["10.68.9.203", "192.168.61.135", "127.0.0.1"]

    def test_connection(self, timeout: float = 2.0) -> bool:
        """Kiểm tra đường truyền HTTP tới Raspberry Pi server (tự động thử danh sách IP dự phòng)."""
        if self.dry_run:
            logger.info("[PI] DRY_RUN Mode enabled: Giả lập kết nối thành công.")
            self.last_connected_status = True
            return True

        # Thu thập danh sách URL thử nghiệm
        urls_to_try = [self.url]
        for ip in self.candidate_ips:
            alt_url = f"http://{ip}:8000/command"
            if alt_url not in urls_to_try:
                urls_to_try.append(alt_url)

        for test_url in urls_to_try:
            logger.info(f"[PI] Checking connection to Raspberry Pi ({test_url})...")
            try:
                response = requests.post(
                    test_url,
                    json={"text": "giu_nguyen"},
                    timeout=timeout
                )
                if response.status_code == 200:
                    self.url = test_url
                    self.last_connected_status = True
                    logger.info(f"[PI] Connected successfully to Pi at '{test_url}'! (HTTP 200)")
                    return True
            except Exception:
                pass

        logger.error(f"[PI] Connection failed to all Pi candidate IPs: {urls_to_try}")
        self.last_connected_status = False
        return False

    def get_current_mode(self, timeout: float = 0.8) -> str:
        """Truy vấn Mode hiện tại từ Raspberry Pi Backend (GET /api/robot/status)."""
        if self.dry_run:
            return "MANUAL"

        try:
            status_url = self.url.replace("/command", "/api/robot/status")
            response = requests.get(status_url, timeout=timeout)
            if response.status_code == 200:
                data = response.json()
                return data.get("mode", "MANUAL")
        except Exception:
            pass
        return "MANUAL"

    check_connection = test_connection

    def send_command(self, text: str, timeout: float = 0.5) -> bool:
        """Gửi lệnh văn bản dạng JSON {"text": text} tới Raspberry Pi."""
        if not text:
            return False

        # Chuẩn hóa câu lệnh tiếng Việt (Cho phép truyền tốc độ linh hoạt 0..255)
        cmd_lower = text.strip().lower()
        mapping = {
            "đi thẳng": "tien",
            "đi lùi": "lui",
            "rẽ trái": "trai",
            "rẽ phải": "phai",
            "xoay trái": "xoay_trai",
            "xoay phải": "xoay_phai",
            "chéo trái": "cheo_tt",
            "chéo phải": "cheo_tp",
            "lùi chéo trái": "cheo_st",
            "lùi chéo phải": "cheo_sp",
            "dừng": "dung",
            "dừng lại": "dung"
        }
        send_text = mapping.get(cmd_lower, text)

        if self.dry_run:
            logger.info(f"[PI] [DRY_RUN] Command simulated: '{send_text}' -> {self.url}")
            return True

        import time
        if hasattr(self, '_last_fail_time'):
            if not self.last_connected_status and (time.time() - self._last_fail_time < 2.0):
                return False

        try:
            response = requests.post(
                self.url,
                json={"text": send_text},
                timeout=timeout
            )
            is_ok = (response.status_code == 200)
            self.last_connected_status = is_ok
            if is_ok:
                logger.info(f"[PI] Command sent successfully: '{send_text}' (HTTP 200)")
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

        # Thử lần lượt danh sách IP dự phòng trên các cổng TTS của Pi (Port 8001 /tts, /speech/tts và Port 8000 /command với 'say')
        tts_urls = []
        for ip in self.candidate_ips:
            tts_urls.append(f"http://{ip}:8001/tts")
            tts_urls.append(f"http://{ip}:8001/speech/tts")
            tts_urls.append(f"http://{ip}:8000/command")

        for tts_url in tts_urls:
            try:
                payload = {"text": f"say {text}"} if "command" in tts_url else {"text": text}
                response = requests.post(tts_url, json=payload, timeout=timeout)
                if response.status_code == 200:
                    logger.info(f"🟢 [PI SPEAKER LA16 SUCCESS] Played audio on Pi ({tts_url}): '{text}'")
                    return True
            except Exception:
                pass

        logger.error(f"[PI] Error sending TTS to Pi across URLs: {tts_urls}")
        return False

    def is_connected(self) -> bool:
        """Trả về trạng thái kết nối gần nhất."""
        return self.last_connected_status

    def send_conversation(self, prompt: str, reply: str, mission_id: int = 1) -> bool:
        """Gửi nhật ký hội thoại đồng bộ lên NoSQL Database & HTTP bridge của Pi."""
        if not prompt or not reply:
            return False
        if self.dry_run:
            logger.info(f"[PI] [DRY_RUN] Conversation simulated: User='{prompt}' -> Robot='{reply}'")
            return True

        import json
        payload = {
            "prompt": prompt,
            "reply": reply,
            "mission_id": mission_id
        }

        # Đồng bộ lịch sử hội thoại lên tất cả các cổng NoSQL DB & Pi Bridge
        urls_to_sync = [
            "http://localhost:8000/api/v1/ai/conversation",
            "http://localhost:8000/api/v1/robot/conversation"
        ]
        for ip in self.candidate_ips:
            urls_to_sync.append(f"http://{ip}:8001/conversation")
            urls_to_sync.append(f"http://{ip}:8000/api/v1/ai/conversation")

        for sync_url in urls_to_sync:
            try:
                body = payload if "api/v1" in sync_url else {"text": json.dumps(payload)}
                res = requests.post(sync_url, json=body, timeout=1.5)
                if res.status_code == 200:
                    logger.info(f"🟢 [CONVERSATION SYNC SUCCESS] Synced: User='{prompt}' -> Robot='{reply}' ({sync_url})")
            except Exception:
                pass

        return True

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
