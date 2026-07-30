import cv2
import numpy as np
from config.settings import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS
from utils.logger import get_logger

logger = get_logger("CameraStream")

class CameraStream:
    """Quản lý kết nối & thu nhận luồng hình ảnh từ Camera/Webcam."""

    def __init__(self, camera_index: int = CAMERA_INDEX):
        self.camera_index = camera_index
        self.cap = None

    def start(self) -> bool:
        """Mở thiết bị camera."""
        logger.info(f"Đang mở Camera index {self.camera_index}...")
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            logger.error(f"Không thể mở Camera index {self.camera_index}!")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        logger.info("Mở Camera thành công!")
        return True

    def read_frame(self) -> Tuple[bool, np.ndarray]:
        """Đọc 1 khung hình từ camera."""
        if self.cap is None or not self.cap.isOpened():
            return False, None
        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        """Giải phóng tài nguyên camera."""
        if self.cap is not None:
            self.cap.release()
            cv2.destroyAllWindows()
            logger.info("Đã đóng Camera.")
