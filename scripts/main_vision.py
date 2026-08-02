import os
import sys
import time
import cv2
import threading

# Đảm bảo import các module từ thư mục gốc dự án
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.camera import Camera
from vision.yolo_detector import YOLOPersonDetector
from vision.person_tracker import select_target, calculate_person_position, draw_debug_overlay
from vision.yolo_stream_server import start_yolo_stream_server, update_yolo_frame
from communication.pi_client import PiClient
from config.settings import VISION_DEBUG, SEND_COMMAND_INTERVAL, DRY_RUN
from utils.logger import get_logger

logger = get_logger("MainVision")

def main():
    logger.info("--- KHỞI CHẠY HỆ THỐNG PHÁT HIỆN & THEO DÕI NGƯỜI DÙNG YOLO11s + RASPBERRY PI ---")

    # Khởi chạy HTTP Streamer phát Video YOLO AI đè khung nhận diện lên Web (Cổng 5050)
    start_yolo_stream_server(5050)

    # 1. Khởi tạo Camera
    camera = Camera()
    if not camera.open():
        logger.error("[VISION] Camera failed to open. Dừng hệ thống Vision.")
        sys.exit(1)

    # 2. Khởi tạo YOLO11s Person Detector & Pi Client
    detector = YOLOPersonDetector()
    pi_client = PiClient()

    # Kiểm tra kết nối Pi ban đầu
    pi_client.test_connection()

    logger.info("Hệ thống đã sẵn sàng. Nhấn 'q' hoặc ESC trên cửa sổ video để thoát.")

    prev_time = time.time()
    last_send_time = 0.0
    last_sent_action = ""
    fps = 0.0
    frame_count = 0
    cached_detections = []
    cached_target = None

    try:
        while True:
            ret, frame = camera.read_frame()
            if not ret or frame is None:
                logger.error("[VISION] Đọc khung hình từ Camera thất bại.")
                break

            h, w = frame.shape[:2]
            frame_count += 1

            # Tính toán FPS
            curr_time = time.time()
            time_diff = curr_time - prev_time
            if time_diff > 0:
                fps = 1.0 / time_diff
            prev_time = curr_time

            # 3. Phát hiện người qua YOLO11n (Chạy AI mỗi 2 frame để tối ưu FPS cực mượt)
            if frame_count % 2 == 0 or not cached_detections:
                cached_detections = detector.detect(frame)
                cached_target = select_target(cached_detections)

            detections = cached_detections
            target = cached_target

            # Gửi nhận diện YOLO lên ROS2 qua cổng 8001 theo khoảng thời gian (tránh tạo 30 thread/giây gây lag GIL)
            if detections and (curr_time - last_send_time) >= SEND_COMMAND_INTERVAL:
                threading.Thread(target=pi_client.send_detections, args=(detections,), daemon=True).start()

            # 4. Lựa chọn target duy nhất (người gần robot nhất - BBox lớn nhất)
            target = select_target(detections)

            action_text = "giu_nguyen"
            if target is not None:
                # Tính vị trí tương quan error_x (-1.0 đến 1.0)
                error_x, pos = calculate_person_position(target, w)

                # Ước tính khoảng cách từ height của target bbox
                height_ratio = target["height"] / float(h)
                if height_ratio < 0.35:
                    dist = "FAR"
                elif height_ratio > 0.75:
                    dist = "CLOSE"
                else:
                    dist = "OPTIMAL"

                # Xác định hành động điều hướng cho Robot
                if pos == "LEFT":
                    action_text = "quay_trai"
                elif pos == "RIGHT":
                    action_text = "quay_phai"
                elif dist == "FAR":
                    action_text = "tiens_len"
                elif dist == "CLOSE":
                    action_text = "lui_lai"
                else:
                    action_text = "giu_nguyen"
            else:
                # Không phát hiện người -> Dừng robot
                action_text = "giu_nguyen"

            # 5. Gửi lệnh HTTP tới Raspberry Pi theo khoảng thời gian SEND_COMMAND_INTERVAL (Async Thread ngầm không làm lag video)
            if (curr_time - last_send_time) >= SEND_COMMAND_INTERVAL:
                # Gửi lệnh tới Pi (hoặc khi lệnh thay đổi)
                if action_text != last_sent_action or (curr_time - last_send_time) >= 0.8:
                    threading.Thread(target=pi_client.send_command, args=(action_text,), daemon=True).start()
                    last_sent_action = action_text
                    last_send_time = curr_time

            # 6. Vẽ giao diện Debug trên màn hình
            debug_frame = draw_debug_overlay(frame, detections, target=target, fps=fps, show_debug=VISION_DEBUG)

            # Bổ sung thông tin Robot Status & Pi Connection Status
            status_color = (0, 255, 0) if pi_client.is_connected() else (0, 0, 255)
            conn_str = "DRY_RUN" if DRY_RUN else ("CONNECTED" if pi_client.is_connected() else "DISCONNECTED")

            cv2.putText(debug_frame, f"PI: {conn_str} | ACTION: {action_text.upper()}", (15, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            # Đẩy khung hình đã vẽ nhận diện YOLO AI lên Stream Server (Cổng 5050)
            update_yolo_frame(debug_frame)

            SHOW_POPUP = os.getenv("SHOW_POPUP", "True").lower() in ("true", "1", "yes")
            if SHOW_POPUP:
                try:
                    cv2.imshow("Robot AI Server - YOLO11s Person Detection & Tracking", debug_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('q'), 27):
                        break
                except Exception:
                    time.sleep(0.03)
            else:
                time.sleep(0.03)

    except KeyboardInterrupt:
        logger.info("Dừng chương trình Vision.")
    except Exception as e:
        logger.error(f"Xảy ra lỗi trong luồng Vision: {e}")
    finally:
        # Khi thoát, gửi lệnh dừng khẩn cấp cho Pi
        pi_client.send_command("giu_nguyen")
        camera.release()
        logger.info("Đã đóng luồng Vision và dừng Robot.")

if __name__ == "__main__":
    main()
