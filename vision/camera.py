import cv2
import time
import threading
import urllib.request
import numpy as np
from typing import Tuple, Optional, Union
from config.settings import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS
from utils.logger import get_logger

logger = get_logger("VisionCamera")

class Camera:
    """
    Quản lý kết nối thiết bị Camera/Webcam/IP Stream độc lập khỏi YOLO detector và ROS.
    Tự động hỗ trợ cả USB Webcam lẫn HTTP MJPEG Stream từ Raspberry Pi với độ trễ 0ms.
    """

    def __init__(self, camera_index: Union[int, str] = CAMERA_INDEX, width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT, fps: int = TARGET_FPS):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._running: bool = False
        self._is_http: bool = isinstance(camera_index, str) and (camera_index.startswith("http://") or camera_index.startswith("https://"))
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    def open(self) -> bool:
        """Mở thiết bị camera và khởi chạy luồng ngầm đọc frame."""
        logger.info(f"[VISION] Opening camera index {self.camera_index} (HTTP: {self._is_http})...")
        
        self._running = True

        if self._is_http:
            # Danh sách URL thử nghiệm tự động khi IP Pi thay đổi dải mạng Wi-Fi
            candidate_urls = [
                str(self.camera_index),
                "http://192.168.60.127:8080/video_feed",
                "http://172.16.68.245:8080/video_feed",
                "http://localhost:8080/video_feed"
            ]

            active_url = str(self.camera_index)
            for test_url in candidate_urls:
                try:
                    req = urllib.request.urlopen(test_url, timeout=1.5)
                    req.close()
                    active_url = test_url
                    logger.info(f"[VISION] HTTP Camera Stream connected successfully ({active_url})")
                    break
                except Exception:
                    pass

            self.camera_index = active_url
            self._thread = threading.Thread(target=self._http_update_loop, daemon=True)
            self._thread.start()

            # Lắng nghe tối đa 3 giây chờ khung hình HTTP đầu tiên nạp vào RAM
            start_wait = time.time()
            while time.time() - start_wait < 3.0:
                if self._latest_frame is not None:
                    break
                time.sleep(0.05)

            return True
        else:
            try:
                # DirectShow backend (cv2.CAP_DSHOW) allows concurrent Microphone audio access on Windows USB webcams
                self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                if not self.cap or not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(self.camera_index)

                if not self.cap or not self.cap.isOpened():
                    logger.error(f"[VISION] Camera failed to open (index {self.camera_index})")
                    self._running = False
                    return False

                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)

                self._thread = threading.Thread(target=self._v4l2_update_loop, daemon=True)
                self._thread.start()

                logger.info(f"[VISION] Camera opened successfully with Async Thread (index {self.camera_index}, {self.width}x{self.height})")
                return True
            except Exception as e:
                logger.error(f"[VISION] Camera failed with exception: {e}")
                self._running = False
                return False

    def _http_update_loop(self):
        """Luồng ngầm đọc luồng MJPEG HTTP từ Pi 4 mượt mà 0ms và tự động đọc khung hình."""
        url = str(self.camera_index)
        while self._running:
            stream = None
            try:
                stream = urllib.request.urlopen(url, timeout=5.0)
                buffer = b''
                while self._running:
                    chunk = stream.read(4096)
                    if not chunk:
                        break
                    buffer += chunk

                    # Tìm mốc bắt đầu (0xffd8) và kết thúc (0xffd9) của khung hình JPEG MỚI NHẤT (Reverse Find - 0ms Latency)
                    end = buffer.rfind(b'\xff\xd9')
                    if end != -1:
                        start = buffer.rfind(b'\xff\xd8', 0, end)
                        if start != -1:
                            jpg = buffer[start:end+2]
                            buffer = buffer[end+2:]
                            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                            if frame is not None:
                                with self._lock:
                                    self._latest_frame = frame
                    elif len(buffer) > 200000:
                        buffer = b''
            except Exception:
                time.sleep(0.3)
            finally:
                if stream:
                    try:
                        stream.close()
                    except Exception:
                        pass

    def _v4l2_update_loop(self):
        """Luồng ngầm liên tục cướp khung hình mới nhất từ cv2.VideoCapture."""
        while self._running and self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self._lock:
                        self._latest_frame = frame
                else:
                    time.sleep(0.01)
            except Exception:
                time.sleep(0.01)

    def is_opened(self) -> bool:
        """Kiểm tra camera có đang hoạt động hay không."""
        return self._running

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Đọc 1 khung hình mới nhất từ camera (Độ trễ 0ms)."""
        if not self.is_opened():
            return False, None
        with self._lock:
            if self._latest_frame is None:
                return False, None
            return True, self._latest_frame.copy()

    def release(self):
        """Giải phóng tài nguyên camera."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
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
