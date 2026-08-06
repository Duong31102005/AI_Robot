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
    last_tts_warn_time = 0.0
    obstacle_start_time = 0.0
    last_sent_status = ""
    fps = 0.0
    frame_count = 0
    cached_detections = []
    cached_target = None

    try:
        while True:
            ret, frame = camera.read_frame()
            if not ret or frame is None:
                time.sleep(0.02)
                continue

            h, w = frame.shape[:2]
            frame_count += 1

            # Tính toán FPS
            curr_time = time.time()
            time_diff = curr_time - prev_time
            if time_diff > 0:
                fps = 1.0 / time_diff
            prev_time = curr_time

            # 3. Chạy AI YOLO11s nhận diện đối tượng & chướng ngại vật (Chạy mỗi 2 frame để mượt 40+ FPS)
            if frame_count % 2 == 0 or not cached_detections:
                cached_detections = detector.detect(frame)
                cached_target = select_target(cached_detections)

            detections = cached_detections
            target = cached_target

            # Gửi danh sách vật thể YOLO nhận dạng được lên Web/Pi (Mỗi 1.5s để tránh nghẽn mạng)
            if detections and (curr_time - last_send_time) >= 1.5:
                last_send_time = curr_time
                threading.Thread(target=pi_client.send_detections, args=(detections,), daemon=True).start()

            # 3.5 TÍCH HỢP VISION INTELLIGENCE (QR CODE SCANNER & HAND GESTURE CONTROL)
            from vision.vision_intelligence import vision_intelligence
            vision_intelligence.current_frame = frame

            # a. Quét mã QR Code Giao Hàng
            scanned_qr = vision_intelligence.scan_qr_code(frame)
            if scanned_qr:
                qr_text = f"Xác nhận thành công mã Q R giao hàng {scanned_qr}! Mời bạn nhận hàng ạ!"
                logger.info(f"📦 [DELIVERY QR SUCCESS] Quét thành công: '{scanned_qr}'")
                threading.Thread(target=pi_client.send_tts, args=(qr_text,), daemon=True).start()

            # b. Nhận diện cử chỉ tay (✋ Stop / ✌️ Go) trên vùng thân người mục tiêu
            gesture_cmd = vision_intelligence.detect_hand_gesture(frame, target=target)
            if gesture_cmd:
                g_text = f"Kim Qui đã nhận cử chỉ tay {gesture_cmd}"
                logger.info(f"✋ [GESTURE CONTROL] Thực thi cử chỉ: '{gesture_cmd}'")
                threading.Thread(target=pi_client.send_command, args=(gesture_cmd,), daemon=True).start()
                threading.Thread(target=pi_client.send_tts, args=(g_text,), daemon=True).start()

            # c. Nhận diện khuôn mặt & Chào hỏi tự động
            face_greeting = vision_intelligence.detect_face_and_greet(frame)
            if face_greeting:
                logger.info(f"😊 [FACE GREETING] Chào khách hàng: '{face_greeting}'")
                threading.Thread(target=pi_client.send_tts, args=(face_greeting,), daemon=True).start()

            # d. Nhận diện Cửa đóng / Thang Máy & Tự Động Dừng Xe + Phát Loa Nhờ Giúp Đỡ
            assist_text = vision_intelligence.detect_door_or_elevator_and_assist(frame, detections)
            if assist_text:
                logger.info(f"🚪 [DOOR/ELEVATOR ASSISTANCE] Dừng xe & phát loa: '{assist_text}'")
                threading.Thread(target=pi_client.send_command, args=("dung",), daemon=True).start()
                threading.Thread(target=pi_client.send_tts, args=(assist_text,), daemon=True).start()

            # 4. Logic Robot Giao Hàng & Bộ Lọc Cảnh Báo Vật Cản Duy Trì 2.5 Giây (2.5s Persistence Obstacle Filter):
            delivery_status = "DANG_DI_CHUYEN_GIAO_HANG"
            if target is not None:
                error_x, pos = calculate_person_position(target, w)
                height_ratio = target["height"] / float(h)

                # Vật cản phải chiếm > 85% chiều cao (< 40cm sát mặt camera), đúng trung tâm và CHẮN LIÊN TỤC TRÊN 2.5 GIÂY!
                if height_ratio >= 0.85 and abs(error_x) <= 0.30:
                    if obstacle_start_time == 0.0:
                        obstacle_start_time = curr_time
                    elif (curr_time - obstacle_start_time) >= 2.5:
                        delivery_status = "CANH_BAO_VAT_CAN_GAN"
                else:
                    obstacle_start_time = 0.0

            # 5. Gửi cảnh báo giọng nói ra Loa Robot nếu vật cản CHẮN LIÊN TỤC TRÊN 2.5 GIÂY (Cooldown 10s)
            if (curr_time - last_send_time) >= SEND_COMMAND_INTERVAL:
                if delivery_status == "CANH_BAO_VAT_CAN_GAN" and (curr_time - last_tts_warn_time) > 10.0:
                    last_tts_warn_time = curr_time
                    logger.warning("[SAFETY ALERT] Cảnh báo vật cản chắn mặt liên tục >2.5s! Phát loa xin nhường đường...")
                    warn_text = "Phía trước có vật cản chắn đường quá lâu, xin vui lòng tránh đường cho robot giao hàng, xin cảm ơn!"
                    threading.Thread(target=pi_client.send_tts, args=(warn_text,), daemon=True).start()

                last_sent_status = delivery_status
                last_send_time = curr_time

            # 6. Vẽ giao diện Debug trên màn hình (Hiển thị nhãn Giao Hàng & Bounding Box)
            debug_frame = draw_debug_overlay(frame, detections, target=target, fps=fps, show_debug=VISION_DEBUG)
            debug_frame = vision_intelligence.draw_intelligence_hud(debug_frame)

            # Thêm nhãn QR Code nếu vừa quét
            if vision_intelligence.last_scanned_qr:
                cv2.putText(debug_frame, f"QR VALIDATED: {vision_intelligence.last_scanned_qr}", (15, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

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
