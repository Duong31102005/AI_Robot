import os
import sys
import time
import cv2
import numpy as np

# Đảm bảo import các module từ thư mục gốc dự án
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.camera import Camera
from vision.yolo_detector import YOLOPersonDetector
from vision.person_tracker import select_target, draw_debug_overlay, calculate_person_position
from utils.logger import get_logger

logger = get_logger("TestObstacles")

OBSTACLE_CLASSES = [
    "person",
    "bicycle",
    "motorcycle",
    "car",
    "truck",
    "bus",
    "chair",
    "bench",
    "couch",
    "dining table",
    "suitcase",
    "backpack",
    "dog",
    "cat"
]

def main():
    logger.info("=== BẮT ĐẦU KIỂM TRA MÔ HÌNH YOLO26m OBSTACLE DETECTION ===")
    logger.info(f"Obstacle classes to detect: {OBSTACLE_CLASSES}")

    # 1. Khởi tạo Camera
    camera = Camera()
    camera_ok = False
    if camera.open():
        camera_ok = True
    else:
        logger.error("[VISION] Không thể khởi tạo Camera ban đầu.")

    # Đợi tối đa 3 giây cho khung hình đầu tiên (đặc biệt cần thiết cho HTTP stream)
    first_frame_ok = False
    if camera_ok:
        logger.info("Đang đợi khung hình đầu tiên từ camera (tối đa 3 giây)...")
        start_wait = time.time()
        while time.time() - start_wait < 3.0:
            ret, frame = camera.read_frame()
            if ret and frame is not None:
                first_frame_ok = True
                break
            time.sleep(0.1)

    # Thử fallback sang webcam cục bộ (index 0) nếu HTTP stream không hoạt động/lỗi
    if not first_frame_ok and camera_ok and isinstance(camera.camera_index, str) and camera.camera_index.startswith("http"):
        logger.warning("[VISION] HTTP stream không phản hồi. Đang thử kết nối Webcam cục bộ (index 0)...")
        camera.release()
        camera = Camera(camera_index=0)
        if camera.open():
            start_wait = time.time()
            while time.time() - start_wait < 3.0:
                ret, frame = camera.read_frame()
                if ret and frame is not None:
                    first_frame_ok = True
                    break
                time.sleep(0.1)

    # Chế độ chạy giả lập Mock Frame nếu không có camera nào hoạt động
    using_mock = False
    if not first_frame_ok:
        logger.warning("[VISION] Không tìm thấy nguồn camera hoạt động! Tự động chuyển sang chế độ GIẢ LẬP (Mock Frame) để chạy test...")
        using_mock = True
        if camera:
            camera.release()
    else:
        logger.info("[VISION] Kết nối camera thành công.")

    # 2. Khởi tạo YOLO26m Detector với danh sách chướng ngại vật
    detector = YOLOPersonDetector(model_name="yolo26m.pt", classes=OBSTACLE_CLASSES)

    logger.info("Chương trình đang chạy. Nhấn 'q' hoặc ESC trên cửa sổ video để thoát.")

    prev_time = time.time()
    fps = 0.0
    static_frame_count = 0

    try:
        while True:
            if using_mock:
                # Tạo khung hình giả lập màu đen 640x480
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "MOCK CAMERA MODE - NO HARDWARE DETECTED", (30, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                ret = True
            else:
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

            # 3. Phát hiện chướng ngại vật qua YOLO26m hoặc dùng mock detections
            if using_mock:
                # Giả lập phát hiện chướng ngại vật để vẽ debug overlay
                detections = [
                    {
                        "class_id": 0,
                        "class_name": "person",
                        "confidence": 0.92,
                        "x1": 150, "y1": 100, "x2": 300, "y2": 400,
                        "center_x": 225, "center_y": 250,
                        "width": 150, "height": 300
                    },
                    {
                        "class_id": 2,
                        "class_name": "car",
                        "confidence": 0.88,
                        "x1": 350, "y1": 200, "x2": 580, "y2": 380,
                        "center_x": 465, "center_y": 290,
                        "width": 230, "height": 180
                    }
                ]
            else:
                detections = detector.detect(frame)

            if detections and not using_mock:
                logger.info(f"[YOLO] Detected {len(detections)} obstacles")
                for det in detections:
                    logger.debug(f"  - {det['class_name']} ({det['confidence']:.2f}): bbox=({det['x1']},{det['y1']},{det['x2']},{det['y2']})")
            elif not detections and not using_mock:
                logger.info("[YOLO] No obstacles detected")

            # 4. Lựa chọn target (Chọn vật thể có diện tích lớn nhất)
            target = select_target(detections)
            h, w = frame.shape[:2]

            delivery_status = "HOAT_DONG_BINH_THUONG"
            if target is not None:
                error_x, pos = calculate_person_position(target, w)
                height_ratio = target["height"] / float(h)

                if height_ratio > 0.85:
                    delivery_status = "CANH_BAO_VAT_CAN_GAN"
                elif target.get("class_name") == "person" and 0.35 <= height_ratio <= 0.85 and pos == "CENTER":
                    delivery_status = "DA_DEN_DIEM_GIAO_HANG"
                else:
                    delivery_status = "DANG_DI_CHUYEN_GIAO_HANG"

            if delivery_status == "CANH_BAO_VAT_CAN_GAN":
                logger.warning(f"[TEST SAFETY ALERT] Vật cản '{target.get('class_name')}' quá gần! Kích hoạt lệnh DỪNG XE.")

            # 5. Vẽ giao diện Debug
            debug_frame = draw_debug_overlay(frame, detections, target=target, fps=fps, show_debug=True)

            # Bổ sung khung viền cảnh báo đỏ/xanh lá và chữ thông tin trạng thái
            if delivery_status == "CANH_BAO_VAT_CAN_GAN":
                cv2.rectangle(debug_frame, (0, 0), (w, h), (0, 0, 255), 10)  # Viền đỏ dày báo động
                cv2.putText(debug_frame, f"!!! CANH BAO !!! DUNG XE (VAT CAN: {target.get('class_name').upper()})", 
                            (15, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            elif delivery_status == "DA_DEN_DIEM_GIAO_HANG":
                cv2.rectangle(debug_frame, (0, 0), (w, h), (0, 255, 0), 10)  # Viền xanh lá
                cv2.putText(debug_frame, "DA DEN DIEM GIAO HANG (PERSON)", 
                            (15, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

            cv2.putText(debug_frame, f"STATUS: {delivery_status}", (15, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 6. Hiển thị khung hình (Hỗ trợ chạy không có giao diện)
            SHOW_POPUP = os.getenv("SHOW_POPUP", "True").lower() in ("true", "1", "yes")
            if SHOW_POPUP:
                try:
                    cv2.imshow("Test YOLO26m Obstacle Detection - Press 'q' to Quit", debug_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('q'), 27):
                        logger.info("Đã nhận phím thoát.")
                        break
                except Exception as e:
                    logger.warning(f"Không thể hiển thị cửa sổ GUI: {e}. Chuyển sang chế độ chạy ngầm (Headless).")
                    os.environ["SHOW_POPUP"] = "False"
            
            # Nếu chạy ngầm, giới hạn 15 frames test rồi tự thoát để tránh lặp vô tận khi kiểm thử tự động
            if not SHOW_POPUP or os.getenv("SHOW_POPUP") == "False":
                static_frame_count += 1
                if static_frame_count >= 15:
                    logger.info("Đã hoàn thành kiểm tra 15 frames ở chế độ chạy ngầm. Tự động thoát.")
                    break
                time.sleep(0.1)

    except KeyboardInterrupt:
        logger.info("Người dùng ngắt chương trình bằng Ctrl+C.")
    except Exception as e:
        logger.error(f"Xảy ra lỗi trong quá trình thực thi: {e}")
    finally:
        if not using_mock:
            camera.release()
        logger.info("=== KẾT THÚC KIỂM TRA YOLO26m ===")

if __name__ == "__main__":
    main()
