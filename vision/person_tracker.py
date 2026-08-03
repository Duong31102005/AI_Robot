import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from config.settings import VISION_DEBUG
from utils.logger import get_logger

logger = get_logger("PersonTracker")

def select_target(detections: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Lựa chọn người mục tiêu tốt nhất từ danh sách detections.
    Mặc định: chọn người có diện tích bounding box (width * height) lớn nhất (gần robot nhất).
    """
    if not detections:
        return None

    best_target = max(detections, key=lambda d: d["width"] * d["height"])
    return best_target

def calculate_person_position(target: Dict[str, Any], frame_width: int) -> Tuple[float, str]:
    """
    Tính vị trí tương quan chuẩn hóa (error_x) của mục tiêu so với tâm màn hình.
    - error_x < 0: Người ở bên trái (error_x = -1.0 ở sát mép trái)
    - error_x > 0: Người ở bên phải (error_x = 1.0 ở sát mép phải)
    - error_x ≈ 0: Người ở giữa màn hình

    Trả về: (error_x, position_label)
    """
    center_x = target["center_x"]
    image_center_x = frame_width / 2.0

    if image_center_x <= 0:
        return 0.0, "CENTER"

    error_x = (center_x - image_center_x) / image_center_x
    error_x = max(-1.0, min(1.0, round(error_x, 4)))

    margin = 0.15  # Vùng 30% giữa màn hình coi như CENTER
    if error_x < -margin:
        pos = "LEFT"
    elif error_x > margin:
        pos = "RIGHT"
    else:
        pos = "CENTER"

    return error_x, pos

def draw_debug_overlay(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    target: Optional[Dict[str, Any]] = None,
    fps: Optional[float] = None,
    show_debug: bool = VISION_DEBUG
) -> np.ndarray:
    """
    Hiển thị thông tin Debug chuẩn:
    - Bounding Box người
    - Confidence
    - Center Point
    - FPS
    - Số người detected
    """
    if not show_debug or frame is None:
        return frame

    output_frame = frame.copy()
    h, w = output_frame.shape[:2]
    image_center_x = w // 2

    # 1. Vẽ đường trung tâm dọc
    cv2.line(output_frame, (image_center_x, 0), (image_center_x, h), (255, 255, 0), 1)

    # 2. Vẽ danh sách tất cả người phát hiện được
    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        cx, cy = det["center_x"], det["center_y"]
        conf = det["confidence"]

        is_target = (target is not None and det == target)
        color = (0, 255, 0) if is_target else (0, 215, 255) # Xanh lá cho target, Vàng nhạt cho người khác
        thickness = 3 if is_target else 1

        # Khung BBox & Tâm
        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, thickness)
        cv2.circle(output_frame, (cx, cy), 5, (0, 0, 255), -1)

        # Label hiển thị
        class_name = det.get("class_name", "person").upper()
        label = f"{class_name} {conf:.2f} center=({cx},{cy})"
        if is_target:
            label += " [TARGET]"

        cv2.putText(output_frame, label, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 3. Hiển thị thông tin FPS & Số lượng đối tượng góc trên trái
    info_y = 30
    if fps is not None:
        cv2.putText(output_frame, f"FPS: {fps:.1f}", (15, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        info_y += 30

    cv2.putText(output_frame, f"Objects: {len(detections)}", (15, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    info_y += 30

    if target is not None:
        error_x, pos = calculate_person_position(target, w)
        cv2.putText(output_frame, f"Target pos: {pos} (err_x: {error_x:+.2f})", (15, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return output_frame
