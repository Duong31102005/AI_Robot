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
from config.settings import VISION_DEBUG, SEND_COMMAND_INTERVAL, DRY_RUN, OBSTACLE_CLASSES, YOLO_MODEL
from utils.logger import get_logger

logger = get_logger("MainVision")

def main():
    logger.info("--- KHỞI CHẠY HỆ THỐNG ROBOT GIAO HÀNG (DELIVERY AI VISION) YOLO + RASPBERRY PI ---")

    # Khởi chạy HTTP Streamer phát Video YOLO AI đè khung nhận diện lên Web (Cổng 5050)
    start_yolo_stream_server(5050)

    # 1. Khởi tạo Camera
    camera = Camera()
    if not camera.open():
        logger.error("[VISION] Camera failed to open. Dừng hệ thống Vision.")
        sys.exit(1)

    # 2. Khởi tạo YOLO Detector (Hỗ trợ phát hiện chướng ngại vật) & Pi Client
    detector = YOLOPersonDetector(model_name=YOLO_MODEL, classes=OBSTACLE_CLASSES)
    pi_client = PiClient()

    # Kiểm tra kết nối Pi ban đầu
    pi_client.test_connection()

    logger.info("Hệ thống Vision Robot Giao Hàng đã sẵn sàng (Chế độ Cảnh báo Chướng ngại & Nhận diện Điểm Giao).")

    prev_time = time.time()
    last_send_time = 0.0
    last_sent_status = ""
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

            # 3. Chạy AI YOLO11s nhận diện đối tượng & chướng ngại vật (Chạy mỗi 2 frame để mượt 30+ FPS)
            if frame_count % 2 == 0 or not cached_detections:
                cached_detections = detector.detect(frame)
                cached_target = select_target(cached_detections)

            detections = cached_detections
            target = cached_target

            # Gửi danh sách vật thể YOLO nhận dạng được lên ROS2/Pi qua cổng 8001
            if detections and (curr_time - last_send_time) >= SEND_COMMAND_INTERVAL:
                threading.Thread(target=pi_client.send_detections, args=(detections,), daemon=True).start()

            # 4. Logic Robot Giao Hàng (Delivery Status):
            # SLAM/LiDAR nắm quyền điều khiển bánh xe chính. Camera kiểm tra chướng ngại vật & điểm giao hàng.
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

            # 5. Gửi trạng thái AI Giao hàng & Cảnh báo giọng nói ra Loa Robot nếu quá gần hoặc đến điểm giao
            if (curr_time - last_send_time) >= SEND_COMMAND_INTERVAL:
                if delivery_status != last_sent_status:
                    if delivery_status == "CANH_BAO_VAT_CAN_GAN":
                        logger.warning("[SAFETY ALERT] Cảnh báo chướng ngại vật quá gần! Phát loa xin nhường đường...")
                        warn_text = "Xin lỗi, vui lòng nhường đường cho robot giao hàng, xin cảm ơn!"
                        threading.Thread(target=pi_client.send_tts, args=(warn_text,), daemon=True).start()
                        threading.Thread(target=pi_client.send_command, args=("dung",), daemon=True).start()

                    elif delivery_status == "DA_DEN_DIEM_GIAO_HANG":
                        logger.info("[DELIVERY SUCCESS] Đã đến điểm giao hàng! Phát loa thông báo...")
                        arrived_text = "Dạ, Kim Qui đã mang đồ đến điểm giao hàng, xin vui lòng nhận hàng!"
                        threading.Thread(target=pi_client.send_tts, args=(arrived_text,), daemon=True).start()
                        threading.Thread(target=pi_client.send_command, args=("dung",), daemon=True).start()

                    last_sent_status = delivery_status
                    last_send_time = curr_time

            # 6. Vẽ giao diện Debug trên màn hình (Hiển thị nhãn Giao Hàng & Bounding Box)
            debug_frame = draw_debug_overlay(frame, detections, target=target, fps=fps, show_debug=VISION_DEBUG)

            # Bổ sung thông tin Robot Delivery Status & Pi Connection Status
            status_color = (0, 255, 0) if pi_client.is_connected() else (0, 0, 255)
            conn_str = "DRY_RUN" if DRY_RUN else ("CONNECTED" if pi_client.is_connected() else "DISCONNECTED")

            cv2.putText(debug_frame, f"PI: {conn_str} | DELIVERY STATUS: {delivery_status}", (15, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

            # Đẩy khung hình đã vẽ nhận diện YOLO AI lên Stream Server (Cổng 5050)
            update_yolo_frame(debug_frame)

            SHOW_POPUP = os.getenv("SHOW_POPUP", "True").lower() in ("true", "1", "yes")
            if SHOW_POPUP:
                try:
                    cv2.imshow("Robot Delivery AI - YOLO11s Object & Target Perception", debug_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('q'), 27):
                        break
                except Exception:
                    time.sleep(0.03)
            else:
                time.sleep(0.03)

    except KeyboardInterrupt:
        logger.info("Dừng chương trình Delivery Vision.")
    except Exception as e:
        logger.error(f"Xảy ra lỗi trong luồng Delivery Vision: {e}")
    finally:
        camera.release()
        logger.info("Đã đóng luồng Delivery Vision.")

if __name__ == "__main__":
    main()
