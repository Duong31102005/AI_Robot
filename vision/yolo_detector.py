import torch
import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from config.settings import (
    YOLO_MODEL, YOLO_CONFIDENCE, YOLO_IMAGE_SIZE, YOLO_PERSON_CLASS
)
from utils.logger import get_logger

logger = get_logger("YOLODetector")

class YOLOPersonDetector:
    """
    Module phát hiện người (Person Detection) sử dụng mô hình YOLO11s.
    Chỉ phát hiện COCO class 0 ('person').
    Tách biệt khỏi Camera và ROS communication.
    """

    def __init__(
        self,
        model_name: str = YOLO_MODEL,
        conf_threshold: float = YOLO_CONFIDENCE,
        imgsz: int = YOLO_IMAGE_SIZE,
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.model = None

        # Tự động xác định thiết bị tính toán (GPU / CPU)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"[YOLO] Device: {self.device}")
        self._load_model()

    def _load_model(self):
        """Tải mô hình YOLO11s từ Ultralytics."""
        try:
            from ultralytics import YOLO
            logger.info(f"[YOLO] Loading model '{self.model_name}'...")
            self.model = YOLO(self.model_name)
            # Chuyển model sang thiết bị tương ứng (GPU/CPU)
            self.model.to(self.device)
            logger.info(f"[YOLO] Model loaded: {self.model_name}")
        except Exception as e:
            logger.error(f"[YOLO] Failed to load model '{self.model_name}': {e}")
            self.model = None

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Nhận 1 khung hình OpenCV (BGR) từ bên ngoài và trả về danh sách đối tượng person phát hiện được.
        Chỉ lọc class 0 ('person').
        """
        if self.model is None or frame is None:
            return []

        try:
            # Chạy inference trên class 0 ('person')
            results = self.model(
                frame,
                verbose=False,
                conf=self.conf_threshold,
                imgsz=self.imgsz,
                classes=[YOLO_PERSON_CLASS],
                device=self.device
            )

            detections: List[Dict[str, Any]] = []

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])

                    w = x2 - x1
                    h = y2 - y1
                    cx = x1 + w // 2
                    cy = y1 + h // 2

                    det = {
                        "class_id": cls_id,
                        "class_name": "person",
                        "confidence": round(conf, 4),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "center_x": cx,
                        "center_y": cy,
                        "width": w,
                        "height": h
                    }
                    detections.append(det)

            return detections

        except Exception as e:
            logger.error(f"[YOLO] Inference error: {e}")
            return []
