import re
import unicodedata
from typing import Optional


class CommandNormalizer:
    """
    Chuẩn hóa câu lệnh giọng nói tiếng Việt từ Whisper STT
    thành 1 trong 5 lệnh chuẩn duy nhất gửi xuống Raspberry Pi / ROS2:
    - "đi thẳng"
    - "đi lùi"
    - "rẽ trái"
    - "rẽ phải"
    - "dừng"

    Nếu không xác định rõ ý định di chuyển/dừng robot -> trả về None.
    """

    COMMAND_FORWARD = "đi thẳng"
    COMMAND_BACKWARD = "đi lùi"
    COMMAND_LEFT = "rẽ trái"
    COMMAND_RIGHT = "rẽ phải"
    COMMAND_ROTATE_LEFT = "xoay trái"
    COMMAND_ROTATE_RIGHT = "xoay phải"
    COMMAND_CHEO_TRAI = "chéo trái"
    COMMAND_CHEO_PHAI = "chéo phải"
    COMMAND_LUI_CHEO_TRAI = "lùi chéo trái"
    COMMAND_LUI_CHEO_PHAI = "lùi chéo phải"
    COMMAND_XOAY_TRON = "xoay tròn"
    COMMAND_STOP = "dừng"

    def __init__(self):
        pass

    @staticmethod
    def remove_accents(text: str) -> str:
        """Chuyển đổi chuỗi tiếng Việt có dấu thành không dấu (ASCII)."""
        if not text:
            return ""
        text = text.replace("đ", "d").replace("Đ", "D")
        nfkd = unicodedata.normalize("NFKD", text)
        ascii_text = "".join([c for c in nfkd if not unicodedata.combining(c)])
        return ascii_text

    def _clean_text(self, text: str) -> tuple[str, str]:
        """Bỏ ký tự đặc biệt, đưa về chữ thường và trả về dạng (có_dấu, không_dấu)."""
        raw = text.strip().lower()
        cleaned = re.sub(r"[^\w\s]", " ", raw)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        ascii_str = self.remove_accents(cleaned)
        return cleaned, ascii_str

    def normalize(self, text: str) -> Optional[str]:
        if not text or not isinstance(text, str):
            return None

        cleaned, ascii_str = self._clean_text(text)
        if not cleaned:
            return None

        # 1. ĐỘ ƯU TIÊN CAO NHẤT: DỪNG (STOP)
        if self._check_stop(cleaned, ascii_str):
            return self.COMMAND_STOP

        # 2. KIỂM TRA LÙI CHÉO TRÁI / LÙI CHÉO PHẢI
        if "lui cheo trai" in ascii_str or "lui trai" in ascii_str:
            return self.COMMAND_LUI_CHEO_TRAI
        if "lui cheo phai" in ascii_str or "lui phai" in ascii_str:
            return self.COMMAND_LUI_CHEO_PHAI

        # 3. KIỂM TRA ĐI CHÉO TRÁI / CHÉO PHẢI
        if "cheo trai" in ascii_str or "tien trai" in ascii_str:
            return self.COMMAND_CHEO_TRAI
        if "cheo phai" in ascii_str or "tien phai" in ascii_str:
            return self.COMMAND_CHEO_PHAI

        # 4. KIỂM TRA XOAY TRÁI / XOAY PHẢI / XOAY TRÒN 360
        if "xoay trai" in ascii_str or "quay trai" in ascii_str:
            return self.COMMAND_ROTATE_LEFT
        if "xoay phai" in ascii_str or "quay phai" in ascii_str:
            return self.COMMAND_ROTATE_RIGHT
        if any(x in ascii_str for x in ["xoay tron", "vong tron", "quay 360", "xoay 360", "quay tron"]):
            return self.COMMAND_XOAY_TRON

        # 2. LỌC CÁC CÂU QUAN SÁT / GIAO TIẾP KHÔNG PHẢI LỆNH
        # Kiểm tra lần lượt các hướng di chuyển
        is_backward = self._check_backward(cleaned, ascii_str)
        is_left = self._check_left(cleaned, ascii_str)
        is_right = self._check_right(cleaned, ascii_str)
        is_forward = self._check_forward(cleaned, ascii_str)

        # Trả về kết quả phù hợp (xử lý xung đột nếu có)
        if is_backward and not (is_left or is_right or is_forward):
            return self.COMMAND_BACKWARD
        if is_left and not (is_right or is_backward or is_forward):
            return self.COMMAND_LEFT
        if is_right and not (is_left or is_backward or is_forward):
            return self.COMMAND_RIGHT
        if is_forward and not (is_left or is_right or is_backward):
            return self.COMMAND_FORWARD

        # Nếu có duy nhất 1 ý định nổi bật:
        intents = []
        if is_backward:
            intents.append(self.COMMAND_BACKWARD)
        if is_left:
            intents.append(self.COMMAND_LEFT)
        if is_right:
            intents.append(self.COMMAND_RIGHT)
        if is_forward:
            intents.append(self.COMMAND_FORWARD)

        if len(intents) == 1:
            return intents[0]

        return None

    def _check_stop(self, cleaned: str, ascii_str: str) -> bool:
        """Kiểm tra ý định dừng robot."""
        stop_patterns = r"\b(dừng|dung|đứng|ngừng|ngung|ngưng|tắt|tat|thôi|thoi)\b"
        if re.search(stop_patterns, ascii_str) or re.search(stop_patterns, cleaned):
            return True
        if "dung lai" in ascii_str or "dung xe" in ascii_str or "đứng lại" in cleaned:
            return True
        return False

    def _check_backward(self, cleaned: str, ascii_str: str) -> bool:
        """Kiểm tra ý định đi lùi."""
        if re.search(r"\b(lui|lùi)\b", ascii_str) or re.search(r"\b(lui|lùi)\b", cleaned):
            return True
        if re.search(r"\b(di|tien|đi|tiến)\s+.*(sau|nguoc lai|ngược lại)\b", ascii_str):
            return True
        return False

    def _check_left(self, cleaned: str, ascii_str: str) -> bool:
        """Kiểm tra ý định rẽ trái."""
        if "cheo trai" in ascii_str or "tien trai" in ascii_str:
            return False  # Đi chéo trái xử lý riêng
        left_words = r"\b(trai|trái|chai|chái)\b"
        if not re.search(left_words, ascii_str):
            return False
        turn_verbs = r"\b(re|rẽ|queo|quẹo|quay|xoay|di|đi|sang|qua|luon|lượn)\b"
        if re.search(turn_verbs, ascii_str):
            return True
        return False

    def _check_right(self, cleaned: str, ascii_str: str) -> bool:
        """Kiểm tra ý định rẽ phải."""
        if "cheo phai" in ascii_str or "tien phai" in ascii_str:
            return False  # Đi chéo phải xử lý riêng
        right_words = r"\b(phai|phải|pai)\b"
        if not re.search(right_words, ascii_str):
            return False
        turn_verbs = r"\b(re|rẽ|queo|quẹo|quay|xoay|di|đi|sang|qua|luon|lượn)\b"
        if re.search(turn_verbs, ascii_str):
            return True
        return False

    def _check_forward(self, cleaned: str, ascii_str: str) -> bool:
        """Kiểm tra ý định đi thẳng."""
        # Tránh trùng với lùi
        if re.search(r"\b(lui|lùi|sau|nguoc lai|ngược lại)\b", ascii_str):
            return False
        # Tránh trùng với rẽ trái/phải/chéo
        if re.search(r"\b(re|rẽ|queo|quẹo|quay|xoay|trai|trái|chai|chái|phai|phải|pai)\b", ascii_str):
            return False

        # Các cụm từ chỉ hướng tiến thẳng
        if re.search(r"\b(di thang|đi thẳng|tien len|tiến lên|di len|đi lên|di toi|đi tới|chay thang|chạy thẳng|tien toi|tiến tới|di tiep|đi tiếp)\b", ascii_str) or re.search(r"\b(đi thẳng|tiến lên|đi lên|đi tới|chạy thẳng|tiến tới)\b", cleaned):
            return True

        # Từ đơn "tiến" hoặc "thẳng" độc lập
        if re.search(r"\b(tien|tiến|thang|thẳng)\b", ascii_str):
            return True

        return False
