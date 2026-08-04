import time
import threading
from utils.logger import get_logger

logger = get_logger("AudioSession")

# --- 1. TTS MUTE GUARD (Triệt tiêu 100% tiếng vọng Loa) ---
_tts_mute_until = 0.0
_tts_lock = threading.Lock()


def set_tts_speaking_duration(seconds: float):
    """
    Kích hoạt MUTE tạm thời luồng Micro STT khi Loa đang phát giọng nói.
    Thêm 1.0 giây Cooldown sau khi loa vừa phát xong để triệt tiêu đuôi âm thanh.
    """
    global _tts_mute_until
    with _tts_lock:
        mute_until = time.time() + seconds + 1.0
        if mute_until > _tts_mute_until:
            _tts_mute_until = mute_until
            logger.info(f"🔇 [TTS MUTE GUARD] Đã khóa Micro thu âm trong {seconds:.1f}s (+1.0s cooldown)...")


def set_tts_speaking_text(text: str):
    """Tính toán thời lượng đọc dựa trên số từ tiếng Việt."""
    if not text:
        return
    words = text.split()
    # Tốc độ đọc trung bình 2.8 từ / giây + 1.0s khởi tạo
    duration = (len(words) / 2.8) + 1.0
    set_tts_speaking_duration(duration)


def is_tts_speaking() -> bool:
    """Trả về True nếu Loa đang phát giọng nói hoặc trong thời gian Cooldown."""
    with _tts_lock:
        return time.time() < _tts_mute_until


# --- 2. XIAOZHI CONVERSATION SESSION MANAGER ---
class XiaoZhiSessionManager:
    """
    Quản lý Trạng Thái Phiên Trò Chuyện theo kiến trúc Robot XiaoZhi (小智 AI):
    - IDLE: Ở trạng thái chờ, CHỈ lắng nghe Wake Word ("Kim Qui", "Rùa ơi"). Lọc bỏ 100% âm thanh nhiễu rác.
    - ACTIVE: Mở phiên trò chuyện trong 12 giây. Trong 12 giây này người dùng KHÔNG CẦN gọi lại "Kim Qui",
              nói bất kỳ câu hỏi nào Robot cũng lắng nghe và trả lời ngay.
    - Mỗi khi người dùng hỏi một câu mới trong phiên, Timer 12s tự động RESET lùi lại.
    """

    def __init__(self, session_timeout_s: float = 12.0):
        self.session_timeout_s = session_timeout_s
        self._active_until = 0.0
        self._lock = threading.Lock()

    def trigger_wake_word(self):
        """Kích hoạt Từ Đánh Thức -> Mở phiên trò chuyện XiaoZhi trong 12 giây."""
        with self._lock:
            self._active_until = time.time() + self.session_timeout_s
            logger.info(f"🟢 [XIAOZHI SESSION] Mở phiên trò chuyện active ({self.session_timeout_s:.0f}s). Người dùng không cần gọi lại 'Kim Qui'!")

    def refresh_session(self):
        """Gia hạn thêm 12 giây mỗi khi vừa hoàn thành 1 câu trò chuyện thành công."""
        with self._lock:
            if time.time() < self._active_until:
                self._active_until = time.time() + self.session_timeout_s
                logger.info(f"🔄 [XIAOZHI SESSION] Gia hạn phiên trò chuyện thêm {self.session_timeout_s:.0f}s...")

    def is_active(self) -> bool:
        """Trả về True nếu đang trong Phiên Trò Chuyện Mở."""
        with self._lock:
            return time.time() < self._active_until

    def close_session(self):
        """Chủ động đóng phiên trò chuyện (quay lại trạng thái IDLE)."""
        with self._lock:
            self._active_until = 0.0
            logger.info("🔴 [XIAOZHI SESSION] Đã đóng phiên trò chuyện. Chuyển về trạng thái IDLE chờ Wake Word.")

    def get_status_str(self) -> str:
        with self._lock:
            rem = self._active_until - time.time()
            if rem > 0:
                return f"ACTIVE ({rem:.1f}s còn lại)"
            return "IDLE (Chờ Wake Word)"


# Instance Singleton toàn hệ thống
session_manager = XiaoZhiSessionManager(session_timeout_s=12.0)
