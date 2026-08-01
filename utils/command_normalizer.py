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
    COMMAND_CHEO_TRAI = "chéo trái"
    COMMAND_CHEO_PHAI = "chéo phải"
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

        # 1.5 KIỂM TRA ĐI CHÉO TRÁI / CHÉO PHẢI
        if "cheo trai" in ascii_str or "cheo trai" in cleaned or "tien trai" in ascii_str:
            return self.COMMAND_CHEO_TRAI
        if "cheo phai" in ascii_str or "cheo phai" in cleaned or "tien phai" in ascii_str:
            return self.COMMAND_CHEO_PHAI

        # 1.6 KIỂM TRA XOAY VÒNG TRÒN / XOAY 360 ĐỘ
        if any(x in ascii_str for x in ["xoay tron", "vong tron", "quay 360", "xoay 360", "quay tron", "xoay qua", "xoay vong"]):
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
        # 1. Các cụm từ dừng rõ ràng
        stop_keywords = [
            "dừng", "dung", "đứng lại", "dung lai", "ngừng", "ngung",
            "đừng đi", "dung di", "không đi nữa", "khong di nua", "ngưng"
        ]

        # Kiểm tra từ/cụm từ dừng bằng Regex (khớp từ hoàn chỉnh)
        pattern = r"\b(dừng|dung|đứng|ngừng|ngung|ngưng)\b"
        if re.search(pattern, ascii_str) or re.search(pattern, cleaned):
            # Cần đảm bảo không phải từ ghép không liên quan
            # Ví dụ: "đứng" phải là "đứng lại", "đứng ngay", "robot đứng", v.v.
            # Nếu chỉ có "đứng" hoặc "dừng", hoặc "dừng lại", "đứng lại"
            if re.search(r"\b(dung|dừng|ngung|ngừng|ngưng)\b", ascii_str):
                return True
            if re.search(r"\b(dung lai|đứng lại)\b", ascii_str) or re.search(r"\b(dung lai|đứng lại)\b", cleaned):
                return True

        if re.search(r"\b(dung di|đừng đi|khong di nua|không đi nữa)\b", ascii_str):
            return True

        if ascii_str in ["thoi", "dung thoi"] or cleaned in ["thôi", "dừng thôi"]:
            return True

        return False

    def _check_backward(self, cleaned: str, ascii_str: str) -> bool:
        """Kiểm tra ý định đi lùi."""
        # Từ khóa chính: lùi / lui
        if re.search(r"\b(lui|lùi)\b", ascii_str):
            return True

        # Cụm từ: đi về phía sau, đi ra phía sau, tiến ngược lại
        if re.search(r"\b(di|tien|đi|tiến)\s+.*(sau|nguoc lai|ngược lại)\b", ascii_str):
            return True

        return False

    def _check_left(self, cleaned: str, ascii_str: str) -> bool:
        """Kiểm tra ý định rẽ trái."""
        # Từ chỉ hướng trái: trái, trai, chái
        if not re.search(r"\b(trai|trái|chai|chái)\b", ascii_str):
            return False

        # Lọc câu quan sát: "bên trái có...", "bên trái là..." mà không có động từ chuyển hướng
        if re.search(r"\b(ben|bên)\s+(trai|trái|chai|chái)\s+(co|có|la|là|dang|đang)\b", ascii_str):
            if not re.search(r"\b(re|rẽ|queo|quẹo|quay|di|đi|chuyen|chuyển)\b", ascii_str):
                return False

        # Động từ điều hướng trái: rẽ/re, quẹo/queo, quay, đi sang, chuyển sang, sang
        turn_verbs = r"\b(re|rẽ|queo|quẹo|quay|di sang|đi sang|chuyen sang|chuyển sang|sang)\b"
        if re.search(turn_verbs, ascii_str):
            return True

        return False

    def _check_right(self, cleaned: str, ascii_str: str) -> bool:
        """Kiểm tra ý định rẽ phải."""
        # Từ chỉ hướng phải: phải, phai, pai
        if not re.search(r"\b(phai|phải|pai)\b", ascii_str):
            return False

        # Lọc câu quan sát: "bên phải có...", "bên phải là..." mà không có động từ chuyển hướng
        if re.search(r"\b(ben|bên)\s+(phai|phải|pai)\s+(co|có|la|là|dang|đang)\b", ascii_str):
            if not re.search(r"\b(re|rẽ|queo|quẹo|quay|di|đi|chuyen|chuyển)\b", ascii_str):
                return False

        # Động từ điều hướng phải: rẽ/re, quẹo/queo, quay, đi sang, chuyển sang, sang
        turn_verbs = r"\b(re|rẽ|queo|quẹo|quay|di sang|đi sang|chuyen sang|chuyển sang|sang)\b"
        if re.search(turn_verbs, ascii_str):
            return True

        return False

    def _check_forward(self, cleaned: str, ascii_str: str) -> bool:
        """Kiểm tra ý định đi thẳng."""
        # Không trùng với lùi (lui, sau, ngược lại)
        if re.search(r"\b(lui|lùi|sau|nguoc lai|ngược lại)\b", ascii_str):
            return False

        # Không trùng với rẽ trái/phải
        if re.search(r"\b(re|rẽ|queo|quẹo|quay|trai|trái|chai|chái|phai|phải|pai)\b", ascii_str):
            return False

        # Lọc câu quan sát: "phía trước có người/cửa..." nếu không có động từ di chuyển
        if re.search(r"\b(phia truoc|phía trước|truoc|trước)\s+(co|có|la|là|dang|đang)\b", ascii_str):
            if not re.search(r"\b(di|đi|tien|tiến|chay|chạy)\b", ascii_str):
                return False

        # Các cụm từ chỉ hướng tiến về phía trước (kể cả lỗi Whisper: thang, than, thẳn, thăng)
        forward_qualifiers = r"\b(thang|thẳng|than|thẳn|thăng|toi|tới|tiep|tiếp|len|lên|phia truoc|phía trước|truoc|trước)\b"
        movement_verbs = r"\b(di|đi|tien|tiến|chay|chạy)\b"

        # Kết hợp Động từ di chuyển + Từ chỉ hướng tiến
        if re.search(movement_verbs, ascii_str) and re.search(forward_qualifiers, ascii_str):
            return True

        # "tiếp tục đi"
        if re.search(r"\btiep tuc\s+di\b", ascii_str) or re.search(r"\btiếp tục\s+đi\b", cleaned):
            return True

        return False
