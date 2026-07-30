import cv2
import numpy as np
from typing import Tuple, Optional
from config.settings import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS
from utils.logger import get_logger

logger = get_logger("VisionCamera")

class Camera:
    """
    Quản lý kết nối thiết bị Camera/Webcam độc lập khỏi YOLO detector và ROS.
    """

    def __init__(self, camera_index: int = CAMERA_INDEX, width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT, fps: int = TARGET_FPS):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        """Mở thiết bị camera và thiết lập thông số."""
        logger.info(f"[VISION] Opening camera index {self.camera_index}...")
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                logger.error(f"[VISION] Camera failed to open (index {self.camera_index})")
                return False

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            logger.info(f"[VISION] Camera opened successfully (index {self.camera_index}, {self.width}x{self.height})")
            return True
        except Exception as e:
            logger.error(f"[VISION] Camera failed with exception: {e}")
            return False

    def is_opened(self) -> bool:
        """Kiểm tra camera có đang hoạt động hay không."""
        return self.cap is not None and self.cap.isOpened()

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Đọc 1 khung hình từ camera."""
        if not self.is_opened():
            return False, None
        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                return False, None
            return True, frame
        except Exception as e:
            logger.error(f"[VISION] Error reading frame from camera: {e}")
            return False, None

    def release(self):
        """Giải phóng tài nguyên camera."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                logger.warning(f"[VISION] Error releasing camera: {e}")
            self.cap = None
            cv2.destroyAllWindows()
            logger.info("[VISION] Camera released.")

# Alias để giữ tính tương thích ngược với code cũ
CameraStream = Camera
