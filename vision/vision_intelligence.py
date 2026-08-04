import cv2
import time
import base64
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from utils.logger import get_logger

logger = get_logger("VisionIntelligence")

# Nạp động Google MediaPipe 3D Hand Landmark Engine
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except Exception:
    MP_AVAILABLE = False


class VisionIntelligence:
    """
    Module Tích hợp Các Tính Năng Thị Giác AI Cao Cấp Đa Tầng cho Robot Otio / Kim Qui:
    1. Google MediaPipe 3D Hand Landmark AI Detector (Cử chỉ tay 2 ngón ✌️ Tiến / 5 ngón ✋ Dừng)
    2. QR Code Scanner (Xác thực mã nhận hàng + Hiển thị khung phát sáng)
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

        # 2. Khởi tạo Google MediaPipe 3D Hand Skeleton Landmark Engine
        self.hands_detector = None
        if MP_AVAILABLE:
            try:
                self.mp_hands = mp.solutions.hands
                self.hands_detector = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.45,
                    min_tracking_confidence=0.45
                )
                self.mp_draw = mp.solutions.drawing_utils
                logger.info("🟢 [MEDIAPIPE AI] Khởi tạo Google MediaPipe 3D Hand Skeleton Engine thành công!")
            except Exception as e:
                logger.warning(f"⚠️ [MEDIAPIPE AI] Lỗi khởi tạo MediaPipe: {e}")

        # 3. Khởi tạo OpenCV Face Cascade Detector
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except Exception:
            self.face_cascade = None
        self.last_face_greeting_time = 0.0

        # 4. Khởi tạo Gesture Perception & HUD
        self.last_gesture_time = 0.0
        self.last_detected_gesture = ""
        self.gesture_display_until = 0.0

        # 5. Trạng thái Biểu cảm Robot AI (Expression Engine)
        self.current_expression = "(◕‿◕) HAPPY"
        self.expression_color = (0, 255, 0)

        # 6. Ghi nhớ bối cảnh thị giác (Scene Memory)
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

    # --- 3. GOOGLE MEDIAPIPE 3D HAND SKELETON GESTURE DETECTOR ---
    def detect_hand_gesture(self, frame: np.ndarray, target: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Nhận diện cử chỉ tay 3D qua Google MediaPipe Keypoints (21 Khớp xương ngón tay):
        - ✌️ FORWARD (2 ngón tay giơ V-Sign / Giơ ngón trỏ + ngón giữa): Lệnh "đi thẳng"
        - ✋ STOP (5 ngón tay bàn tay xòe): Lệnh "dừng"
        """
        if frame is None:
            return None
        now = time.time()
        if now - self.last_gesture_time < 1.5:
            return None

        # 🌟 ƯU TIÊN 1: DÙNG GOOGLE MEDIAPIPE 3D HAND SKELETON AI DETECTOR
        if self.hands_detector is not None:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands_detector.process(rgb_frame)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Vẽ bộ xương 21 khớp tay 3D trực tiếp lên màn hình
                        self.mp_draw.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                            self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                            self.mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2)
                        )

                        lm = hand_landmarks.landmark
                        # Trạng thái duỗi ngón tay chuẩn xác 360 độ (Khoảng cách từ Cổ tay lm[0] tới Đỉnh ngón vs Khớp PIP)
                        wrist_x, wrist_y = lm[0].x, lm[0].y
                        def is_ext(tip_i, pip_i):
                            d_tip = np.hypot(lm[tip_i].x - wrist_x, lm[tip_i].y - wrist_y)
                            d_pip = np.hypot(lm[pip_i].x - wrist_x, lm[pip_i].y - wrist_y)
                            return d_tip > (d_pip * 1.12)

                        index_up = is_ext(8, 6)
                        middle_up = is_ext(12, 10)
                        ring_up = is_ext(16, 14)
                        pinky_up = is_ext(20, 18)

                        self.last_gesture_time = now
                        self.gesture_display_until = now + 3.0

                        # a. Cử chỉ ✌️ V-Sign / 2 Ngón tay duỗi (bất kể xoay dọc hay xoay ngang)
                        if index_up and middle_up and (not ring_up) and (not pinky_up):
                            self.last_detected_gesture = "✌️ GESTURE: GO (TIẾN)"
                            logger.info("✌️ [MEDIAPIPE 3D AI] Nhận diện Cử chỉ 2 Ngón Tay (V-SIGN 360°) -> Lệnh ĐI THẲNG!")
                            return "đi thẳng"

                        # b. Cử chỉ ✋ Bàn tay xòe (Cả 4 ngón tay duỗi)
                        elif index_up and middle_up and ring_up and pinky_up:
                            self.last_detected_gesture = "✋ GESTURE: STOP (DỪNG)"
                            logger.info("✋ [MEDIAPIPE 3D AI] Nhận diện Cử chỉ Bàn Tay Xòe (OPEN PALM 360°) -> Lệnh DỪNG XE!")
                            return "dừng"

                        # c. Trường hợp giơ ngón trỏ hoặc ngón giữa
                        elif (index_up or middle_up) and (not ring_up) and (not pinky_up):
                            self.last_detected_gesture = "✌️ GESTURE: GO (TIẾN)"
                            logger.info("✌️ [MEDIAPIPE 3D AI] Nhận diện Cử chỉ Giơ Ngón Tay -> Lệnh ĐI THẲNG!")
                            return "đi thẳng"
            except Exception as e:
                logger.error(f"[MEDIAPIPE EXEC] Lỗi suy luận MediaPipe: {e}")

        # 🌟 ƯU TIÊN 2: BÀN THỪA KHÁC (OPENCV HSV CONTOUR FALLBACK)
        try:
            h, w = frame.shape[:2]
            if target and ("x1" in target or "bbox" in target):
                if "bbox" in target:
                    tx1, ty1, tx2, ty2 = target["bbox"]
                else:
                    tx1, ty1, tx2, ty2 = target["x1"], target["y1"], target["x2"], target["y2"]

                crop_y1 = max(0, ty1)
                crop_y2 = min(h, int(ty1 + (ty2 - ty1) * 0.48))
                crop_x1 = max(0, int(tx1 - (tx2 - tx1) * 0.15))
                crop_x2 = min(w, int(tx2 + (tx2 - tx1) * 0.15))
                roi = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            else:
                roi = frame[int(h * 0.05):int(h * 0.5), int(w * 0.15):int(w * 0.85)]

            if roi is None or roi.size == 0:
                return None

            roi_h, roi_w = roi.shape[:2]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            lower_skin = np.array([0, 30, 60], dtype=np.uint8)
            upper_skin = np.array([25, 255, 255], dtype=np.uint8)

            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)

            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                valid_contours = [c for c in contours if cv2.contourArea(c) > (roi_w * roi_h * 0.015)]
                if not valid_contours:
                    return None

                max_contour = max(valid_contours, key=cv2.contourArea)
                area = cv2.contourArea(max_contour)

                if area > (roi_w * roi_h * 0.015):
                    hull = cv2.convexHull(max_contour, returnPoints=False)
                    defects = cv2.convexDefects(max_contour, hull)

                    finger_count = 0
                    if defects is not None:
                        for i in range(defects.shape[0]):
                            s, e, f, d = defects[i, 0]
                            start = tuple(max_contour[s][0])
                            end = tuple(max_contour[e][0])
                            far = tuple(max_contour[f][0])
                            a = np.linalg.norm(np.array(start) - np.array(far))
                            b = np.linalg.norm(np.array(end) - np.array(far))
                            c = np.linalg.norm(np.array(start) - np.array(end))
                            angle = np.arccos((a**2 + b**2 - c**2) / (2 * a * b + 1e-5))
                            if angle <= np.pi / 2 and d > 4000:
                                finger_count += 1

                    self.last_gesture_time = now
                    self.gesture_display_until = now + 3.0

                    if finger_count >= 3:
                        self.last_detected_gesture = "✋ GESTURE: STOP (DỪNG)"
                        logger.info("✋ [GESTURE DETECTED] Cử chỉ Bàn Tay Xòe (OPEN PALM) -> Lệnh DỪNG XE!")
                        return "dừng"
                    elif finger_count in (1, 2) or (finger_count == 0 and area > (roi_w * roi_h * 0.02)):
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
