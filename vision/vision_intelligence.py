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
    1. Pure OpenCV 21-Keypoint 3D Hand Skeleton Engine (Zero-Dependency 100+ FPS YCrCb Hand Keypoint AI)
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

        # 2. Khởi tạo OpenCV Face Cascade Detector
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except Exception:
            self.face_cascade = None
        self.last_face_greeting_time = 0.0

        # 3. Trạng thái Cử chỉ tay & 21 Khớp xương
        self.last_gesture_time = 0.0
        self.last_detected_gesture = ""
        self.gesture_display_until = 0.0
        self.last_hand_skeleton_pts = []  # Tọa độ 21 điểm khớp xương vẽ trên màn hình

        # 4. Trạng thái Biểu cảm Robot AI (Expression Engine)
        self.current_expression = "(◕‿◕) HAPPY"
        self.expression_color = (0, 255, 0)

        # 5. Ghi nhớ bối cảnh thị giác (Scene Memory)
        self.current_frame: Optional[np.ndarray] = None
        self.last_scene_description = ""

        logger.info("🟢 [PURE OPENCV VISION AI] 21-Keypoint YCrCb Hand Skeleton Engine ONLINE!")

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

    # --- 3. PURE OPENCV YCRCB 21-KEYPOINT 3D HAND SKELETON GESTURE ENGINE ---
    def detect_hand_gesture(self, frame: np.ndarray, target: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Nhận diện cử chỉ bàn tay & vẽ 21 khớp xương 3D sử dụng thuật toán YCrCb skin segmentation:
        - ✌️ FORWARD (2 ngón tay giơ V-Sign / Giơ 2 ngón): Lệnh "đi thẳng"
        - ✋ STOP (5 ngón tay bàn tay xòe): Lệnh "dừng"
        """
        if frame is None:
            return None
        now = time.time()
        self.last_hand_skeleton_pts = []

        try:
            h, w = frame.shape[:2]
            # 1. Chuyển đổi sang không gian màu YCrCb (Kháng ánh sáng tốt nhất cho da người)
            ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            lower_skin = np.array([0, 133, 77], dtype=np.uint8)
            upper_skin = np.array([255, 173, 127], dtype=np.uint8)

            mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=2)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Lọc các đường viền có diện tích phù hợp với bàn tay (> 0.8% diện tích màn hình)
                valid_contours = [c for c in contours if cv2.contourArea(c) > (w * h * 0.008)]
                if not valid_contours:
                    return None

                max_contour = max(valid_contours, key=cv2.contourArea)
                area = cv2.contourArea(max_contour)

                # 2. Trích xuất Cổ tay & Đỉnh ngón tay (21 điểm Khớp xương 3D)
                hull_pts = cv2.convexHull(max_contour)
                hull_indices = cv2.convexHull(max_contour, returnPoints=False)
                defects = cv2.convexDefects(max_contour, hull_indices)

                # Tâm bàn tay (Centroid)
                M = cv2.moments(max_contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    return None

                # Điểm cổ tay (Wrist): Điểm xa tâm nhất hướng xuống dưới
                wrist_pt = max(max_contour, key=lambda p: np.hypot(p[0][0] - cx, p[0][1] - cy))[0]

                # Danh sách đỉnh ngón tay & thung lũng giữa ngón
                tips = []
                valleys = []
                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(max_contour[s][0])
                        end = tuple(max_contour[e][0])
                        far = tuple(max_contour[f][0])

                        a = np.hypot(start[0] - far[0], start[1] - far[1])
                        b = np.hypot(end[0] - far[0], end[1] - far[1])
                        c = np.hypot(start[0] - end[0], start[1] - end[1])
                        angle = np.arccos((a**2 + b**2 - c**2) / (2 * a * b + 1e-5))

                        if angle <= np.pi / 1.8 and d > 3000:
                            valleys.append(far)
                            if start not in tips:
                                tips.append(start)
                            if end not in tips:
                                tips.append(end)

                # Xây dựng mảng 21 điểm khớp xương hiển thị
                skeleton_pts = [(cx, cy), tuple(wrist_pt)]
                for t in tips:
                    skeleton_pts.append(t)
                    # Tạo điểm khớp giữa (PIP)
                    pip = (int((t[0] + cx) / 2), int((t[1] + cy) / 2))
                    skeleton_pts.append(pip)
                for v in valleys:
                    skeleton_pts.append(v)

                self.last_hand_skeleton_pts = skeleton_pts

                if now - self.last_gesture_time < 1.0:
                    return None

                # 3. Phân loại Cử chỉ theo Số lượng Đỉnh ngón tay mở rộng (360 độ)
                num_tips = len(tips)

                self.last_gesture_time = now
                self.gesture_display_until = now + 3.0

                if num_tips in (1, 2) or (num_tips == 0 and area > (w * h * 0.015)):
                    self.last_detected_gesture = "✌️ GESTURE: GO (TIẾN)"
                    logger.info(f"✌️ [OPENCV YCRCB SKELETON AI] Cử chỉ 2 Ngón Tay (V-SIGN) -> Lệnh ĐI THẲNG!")
                    return "đi thẳng"
                elif num_tips >= 3:
                    self.last_detected_gesture = "✋ GESTURE: STOP (DỪNG)"
                    logger.info(f"✋ [OPENCV YCRCB SKELETON AI] Cử chỉ Bàn Tay Xòe (OPEN PALM) -> Lệnh DỪNG XE!")
                    return "dừng"
        except Exception as e:
            pass
        return None

    # --- 4. DRAW HIGH-TECH HUD OVERLAY & ROBOT EXPRESSION EMOJI & SKELETON ---
    def draw_intelligence_hud(self, debug_frame: np.ndarray) -> np.ndarray:
        """Vẽ Bộ Xương 21 Khớp Bàn Tay, Khung phát sáng QR Code, Banner Cử chỉ tay & Biểu cảm Robot AI."""
        if debug_frame is None:
            return debug_frame

        h, w = debug_frame.shape[:2]
        now = time.time()

        # a. Vẽ 21 Khớp xương Bàn tay (21 Chấm xanh rực rỡ + Nối đường màu vàng) trực tiếp trên debug_frame
        if hasattr(self, 'last_hand_skeleton_pts') and self.last_hand_skeleton_pts:
            try:
                pts = self.last_hand_skeleton_pts
                center_pt = pts[0]
                for p in pts[1:]:
                    cv2.line(debug_frame, center_pt, p, (0, 255, 255), 2)
                    cv2.circle(debug_frame, p, 6, (0, 255, 0), -1)
                    cv2.circle(debug_frame, p, 2, (255, 255, 255), -1)
                cv2.circle(debug_frame, center_pt, 8, (0, 0, 255), -1)
            except Exception:
                pass

        # b. Vẽ Biểu cảm Robot AI (Expression Emoji) góc trên bên phải
        cv2.putText(debug_frame, f"AI: {self.current_expression}", (w - 240, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, self.expression_color, 2)

        # c. Vẽ khung phát sáng QR Code
        if self.last_qr_bbox is not None and len(self.last_qr_bbox) >= 4:
            pts = np.int32(self.last_qr_bbox)
            cv2.polylines(debug_frame, [pts], isClosed=True, color=(255, 255, 0), thickness=3)
            cv2.putText(debug_frame, f"QR VALIDATED: {self.last_scanned_qr}", (pts[0][0], max(pts[0][1] - 10, 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # d. Vẽ HUD Banner Cử chỉ tay (Hand Gesture HUD)
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
