import numpy as np
from typing import List, Dict, Any
from vision.yolo_detector import YOLOPersonDetector
from vision.person_tracker import select_target, calculate_person_position, draw_debug_overlay
from config.settings import YOLO_MODEL, YOLO_CONFIDENCE, VISION_DEBUG
from utils.logger import get_logger

logger = get_logger("PersonDetector")

class PersonDetector:
    """
    Wrapper giữ tính tương thích ngược với PersonDetector cũ,
    sử dụng mô hình YOLO11s từ YOLOPersonDetector mới bên dưới.
    """

    def __init__(self, confidence: float = YOLO_CONFIDENCE, model_name: str = YOLO_MODEL):
        self.yolo = YOLOPersonDetector(model_name=model_name, conf_threshold=confidence)

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Phát hiện người và bổ sung thông số position/distance cho tương thích ngược."""
        raw_detections = self.yolo.detect(frame)
        h, w = frame.shape[:2]

        formatted_detections = []
        for det in raw_detections:
            cx, cy = det["center_x"], det["center_y"]
            bw, bh = det["width"], det["height"]
            error_x, pos = calculate_person_position(det, w)

            height_ratio = bh / float(h)
            if height_ratio < 0.35:
                dist = "FAR"
            elif height_ratio > 0.75:
                dist = "CLOSE"
            else:
                dist = "OPTIMAL"

            formatted_det = {
                "bbox": (det["x1"], det["y1"], bw, bh),
                "center": (cx, cy),
                "offset_x": cx - (w // 2),
                "error_x": error_x,
                "confidence": det["confidence"],
                "position": pos,
                "distance": dist,
                # Giữ nguyên cấu trúc YOLO mới
                **det
            }
            formatted_detections.append(formatted_det)

        return formatted_detections

    def draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        target = select_target(detections)
        return draw_debug_overlay(frame, detections, target=target, show_debug=VISION_DEBUG)
