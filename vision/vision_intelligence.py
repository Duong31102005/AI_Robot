import cv2
import time
import base64
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from utils.logger import get_logger

logger = get_logger("VisionIntelligence")


class VisionIntelligence:
    """
    Module Tích hợp Các Tính Năng Thị Giác AI Cao Cấp Đa Tầng cho Robot Otio / Kim Qui:
    1. QR Code Scanner (Xác thực mã nhận hàng + Hiển thị khung phát sáng)
    2. Hand Gesture Perception (Cử chỉ tay Dừng ✋ / Tiến ✌️ + Giao diện HUD)
    3. Face Detection & Personal Greeting (Nhận diện khuôn mặt & Chào hỏi khách hàng)
    4. Interactive Robot Expression HUD (Mặt biểu cảm kỹ thuật số AI)
    5. Multi-modal Vision Scene Memory (Ghi nhớ bối cảnh thị giác nối tiếp)
    """

    def __init__(self):
        # 1. Khởi tạo OpenCV QR Code Detector (0ms delay)
        self.qr_detector = cv2.QRCodeDetector()
        self.last_qr_time = 0.0
        self.last_scanned_qr = ""
        self.last_qr_bbox = None

        # 2. Khởi tạo OpenCV Face Cascade Detector
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except Exception:
            self.face_cascade = None
        self.last_face_greeting_time = 0.0

        # 3. Khởi tạo Gesture Perception & HUD
        self.last_gesture_time = 0.0
        self.last_detected_gesture = ""
        self.gesture_display_until = 0.0

        # 4. Trạng thái Biểu cảm Robot AI (Expression Engine)
        self.current_expression = "(◕‿◕) HAPPY"
        self.expression_color = (0, 255, 0)

        # 5. Ghi nhớ bối cảnh thị giác (Scene Memory)
        self.current_frame: Optional[np.ndarray] = None
        self.last_scene_description = ""

    # --- 1. FACE DETECTION & PERSONALIZED GREETING (Nhận diện khuôn mặt & Chào hỏi) ---
    def detect_face_and_greet(self, frame: np.ndarray) -> Optional[str]:
        """Phát hiện khuôn mặt trước camera và tạo lời chào thân thiện (Cooldown 15s)."""
        if frame is None or self.face_cascade is None:
            return None
        now = time.time()
        if now - self.last_face_greeting_time < 15.0:
            return None

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))
            if len(faces) > 0:
                self.last_face_greeting_time = now
                logger.info(f"😊 [FACE DETECTED] Phát hiện {len(faces)} khuôn mặt trước Camera Robot!")
                return "Dạ, Kim Qui xin chào bạn! Rất vui được gặp bạn ạ!"
        except Exception:
            pass
        return None

    # --- 2. QR CODE SCANNER (Xác thực giao hàng 0ms + Khung phát sáng) ---
    def scan_qr_code(self, frame: np.ndarray) -> Optional[str]:
        """Quét mã QR Code hiển thị trước Camera. Trả về nội dung mã nếu phát hiện."""
        if frame is None:
            return None
        now = time.time()

        try:
            data, bbox, _ = self.qr_detector.detectAndDecode(frame)
            if bbox is not None and len(bbox) > 0:
                self.last_qr_bbox = bbox[0]
            else:
                self.last_qr_bbox = None

            if data and data.strip():
                if data != self.last_scanned_qr or (now - self.last_qr_time > 5.0):
                    self.last_scanned_qr = data
                    self.last_qr_time = now
                    logger.info(f"📦 [VISION QR SCAN] Quét thành công mã QR Giao Hàng: '{data}'")
                    return data
        except Exception:
            pass
        return None

    # --- 3. HAND GESTURE CONTROL (Cử chỉ tay ✋ Stop / ✌️ Go + HUD Overlay) ---
    def detect_hand_gesture(self, frame: np.ndarray) -> Optional[str]:
        """
        Nhận diện cử chỉ tay đơn giản qua phân tích đường viền (Hand Contour Perception):
        - ✋ STOP: Bàn tay xòe trước camera -> Trả về "dừng"
        - ✌️ FORWARD: Hai ngón tay / Giơ tay -> Trả về "đi thẳng"
        """
        if frame is None:
            return None
        now = time.time()
        if now - self.last_gesture_time < 2.5:
            return None

        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)

            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)

            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                max_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(max_contour)
                h, w = frame.shape[:2]

                # Nếu vùng bàn tay chiếm > 4% màn hình trước camera
                if area > (w * h * 0.04):
                    hull = cv2.convexHull(max_contour, returnPoints=False)
                    defects = cv2.convexDefects(max_contour, hull)

                    if defects is not None:
                        finger_count = 0
                        for i in range(defects.shape[0]):
                            s, e, f, d = defects[i, 0]
                            start = tuple(max_contour[s][0])
                            end = tuple(max_contour[e][0])
                            far = tuple(max_contour[f][0])
                            a = np.linalg.norm(np.array(start) - np.array(far))
                            b = np.linalg.norm(np.array(end) - np.array(far))
                            c = np.linalg.norm(np.array(start) - np.array(end))
                            angle = np.arccos((a**2 + b**2 - c**2) / (2 * a * b + 1e-5))
                            if angle <= np.pi / 2 and d > 12000:
                                finger_count += 1

                        self.last_gesture_time = now
                        self.gesture_display_until = now + 3.0

                        if finger_count >= 3:
                            self.last_detected_gesture = "✋ GESTURE: STOP (DỪNG)"
                            logger.info("✋ [GESTURE DETECTED] Cử chỉ Bàn Tay Xòe (OPEN PALM) -> Lệnh DỪNG XE!")
                            return "dừng"
                        elif finger_count in (1, 2):
                            self.last_detected_gesture = "✌️ GESTURE: GO (TIẾN)"
                            logger.info("✌️ [GESTURE DETECTED] Cử chỉ Giơ Tay Tiến (V-SIGN) -> Lệnh ĐI THẲNG!")
                            return "đi thẳng"
        except Exception:
            pass
        return None

    # --- 4. DRAW HIGH-TECH HUD OVERLAY & ROBOT EXPRESSION EMOJI ---
    def draw_intelligence_hud(self, debug_frame: np.ndarray) -> np.ndarray:
        """Vẽ khung phát sáng QR Code, Banner Cử chỉ tay & Mặt biểu cảm Robot AI."""
        if debug_frame is None:
            return debug_frame

        h, w = debug_frame.shape[:2]
        now = time.time()

        # a. Vẽ Biểu cảm Robot AI (Expression Emoji) góc trên bên phải
        cv2.putText(debug_frame, f"AI: {self.current_expression}", (w - 240, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, self.expression_color, 2)

        # b. Vẽ khung phát sáng QR Code
        if self.last_qr_bbox is not None and len(self.last_qr_bbox) >= 4:
            pts = np.int32(self.last_qr_bbox)
            cv2.polylines(debug_frame, [pts], isClosed=True, color=(255, 255, 0), thickness=3)
            cv2.putText(debug_frame, f"QR VALIDATED: {self.last_scanned_qr}", (pts[0][0], max(pts[0][1] - 10, 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # c. Vẽ HUD Banner Cử chỉ tay (Hand Gesture HUD)
        if now < self.gesture_display_until and self.last_detected_gesture:
            overlay = debug_frame.copy()
            cv2.rectangle(overlay, (w // 2 - 180, 10), (w // 2 + 180, 55), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, debug_frame, 0.4, 0, debug_frame)
            cv2.rectangle(debug_frame, (w // 2 - 180, 10), (w // 2 + 180, 55), (0, 255, 255), 2)

            text_color = (0, 255, 0) if "GO" in self.last_detected_gesture else (0, 0, 255)
            cv2.putText(debug_frame, self.last_detected_gesture, (w // 2 - 160, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

        return debug_frame

    # --- 5. SCENE ENCODER FOR VLM ---
    @staticmethod
    def encode_frame_to_base64(frame: np.ndarray, quality: int = 75) -> Optional[str]:
        """Mã hóa khung hình OpenCV BGR thành chuỗi Base64 JPEG để gửi lên Vision Cloud AI."""
        if frame is None:
            return None
        try:
            small = cv2.resize(frame, (480, 360))
            _, buffer = cv2.imencode('.jpg', small, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            logger.error(f"[VISION ENCODER] Lỗi mã hóa Base64: {e}")
            return None


# Instance Singleton toàn hệ thống
vision_intelligence = VisionIntelligence()
