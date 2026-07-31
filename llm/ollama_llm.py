import json
import requests
from config.settings import OLLAMA_URL, OLLAMA_MODEL
from utils.logger import get_logger

logger = get_logger("OllamaLLM")


class OllamaLLM:
    """
    Module giao tiếp với Ollama LLM Server (Offline 100% trên PC Windows).
    Mặc định sử dụng mô hình 'qwen2.5:3b' chuyên biệt trả lời tiếng Việt ngắn gọn.
    """

    SYSTEM_PROMPT = (
        "BẮT BUỘC: Bạn tên là Kim Qui (Kim Quy), một Robot AI thông minh và thân thiện. "
        "BẠN CHỈ ĐƯỢC TRẢ LỜI 100% BẰNG TIẾNG VIỆT. "
        "TUYỆT ĐỐI KHÔNG TRẢ LỜI BẰNG TIẾNG ANH HOẶC TIẾNG TRUNG. "
        "Hãy trả lời một cách tự nhiên, ngắn gọn, lịch sự và rõ ràng bằng tiếng Việt."
    )

    def __init__(self, base_url: str = OLLAMA_URL, model: str = OLLAMA_MODEL):
        self.base_url = base_url.rstrip('/')
        self.generate_url = f"{self.base_url}/api/generate"
        self.model = model

    def is_available(self) -> bool:
        """Kiểm tra Ollama Server có đang chạy không."""
        try:
            res = requests.get(f"{self.base_url}/api/tags", timeout=1.5)
            return res.status_code == 200
        except Exception:
            return False

    def generate_response(self, prompt: str) -> str:
        """Gửi prompt tới Ollama LLM và nhận câu trả lời."""
        if not prompt or not prompt.strip():
            return ""

        payload = {
            "model": self.model,
            "prompt": f"System: {self.SYSTEM_PROMPT}\nUser: {prompt}\nAssistant:",
            "stream": False,
            "options": {
                "temperature": 0.5,
                "num_predict": 120  # Trả lời mượt mà linh hoạt (ngắn hoặc chi tiết)
            }
        }

        logger.info(f"[LLM] Prompting Ollama ({self.model}): '{prompt}'")
        try:
            response = requests.post(
                self.generate_url,
                json=payload,
                timeout=10.0
            )

            if response.status_code == 200:
                data = response.json()
                reply = data.get("response", "").strip()
                logger.info(f"[LLM] Response: '{reply}'")
                return reply
            else:
                logger.warning(f"[LLM] Error HTTP {response.status_code}: {response.text}")
                return "Xin lỗi, tôi chưa hiểu ý bạn."

        except requests.exceptions.ConnectionError:
            logger.error("[LLM] Không thể kết nối tới Ollama Server (Hãy chắc chắn bạn đã chạy 'ollama run qwen2.5:3b').")
            return "Tôi chưa sẵn sàng kết nối Ollama."
        except Exception as e:
            logger.error(f"[LLM] Unexpected error: {e}")
            return "Tôi gặp sự cố khi xử lý câu hỏi."
