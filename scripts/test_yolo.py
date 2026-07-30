import os
import sys
import time
import cv2

# Đảm bảo import các module từ thư mục gốc dự án
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.camera import Camera
from vision.yolo_detector import YOLOPersonDetector
from vision.person_tracker import select_target, draw_debug_overlay
from utils.logger import get_logger

logger = get_logger("TestYOLO")

def main():
    logger.info("=== BẮT ĐẦU KIỂM TRA MÔ HÌNH YOLO11s PERSON DETECTION ===")

    # 1. Khởi tạo Camera
    camera = Camera()
    if not camera.open():
        logger.error("[VISION] Không thể khởi tạo Camera. Đóng chương trình test.")
        sys.exit(1)

    # 2. Khởi tạo YOLO11s Detector
    detector = YOLOPersonDetector()

    logger.info("Chương trình đang chạy. Nhấn 'q' hoặc ESC trên cửa sổ video để thoát.")

    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = camera.read_frame()
            if not ret or frame is None:
                logger.error("[VISION] Đọc khung hình thất bại!")
                break

            # Tính toán FPS
            curr_time = time.time()
            time_diff = curr_time - prev_time
            if time_diff > 0:
                fps = 1.0 / time_diff
            prev_time = curr_time

            # 3. Detect người qua YOLO11s
            detections = detector.detect(frame)

            if detections:
                logger.info(f"[YOLO] Person detected: {len(detections)}")
            else:
                logger.info("[YOLO] No person detected")

            # 4. Lựa chọn target
            target = select_target(detections)

            # 5. Vẽ giao diện Debug
            debug_frame = draw_debug_overlay(frame, detections, target=target, fps=fps, show_debug=True)

            # 6. Hiển thị khung hình
            cv2.imshow("Test YOLO11s Person Detection - Press 'q' to Quit", debug_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                logger.info("Đã nhận phím thoát.")
                break

    except KeyboardInterrupt:
        logger.info("Người dùng ngắt chương trình bằng Ctrl+C.")
    except Exception as e:
        logger.error(f"Xảy ra lỗi trong quá trình thực thi: {e}")
    finally:
        camera.release()
        logger.info("=== KẾT THÚC KIỂM TRA YOLO11s ===")

if __name__ == "__main__":
    main()
