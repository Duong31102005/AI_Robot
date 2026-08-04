# 🚀 HƯỚNG DẪN KHỞI CHẠY TỔNG THỂ HỆ THỐNG ROBOT KIM QUI
> **Dự án:** Robot Giao Hàng & Tự Hành Kim Qui (Make in Vietnam, Made by Đại Nam)  
> **Nhóm phát triển:** Galacticos - Khoa Công nghệ Thông tin - Trường Đại học Đại Nam  
> **Thầy Mentor:** Đỗ Quang Thơ  
> **Thành viên:** Nguyễn Thế Phương Nam, Nguyễn Hoàng Dương, Lê Duy A, Đỗ Duy Văn  

---

## 📌 1. TỔNG QUAN KIẾN TRÚC VÀ CÁC PHÂN HỆ

Hệ thống bao gồm 3 phân hệ chính hoạt động giao tiếp đa luồng thời gian thực:

1. **Raspberry Pi 4 (`192.168.61.135`)**: Điều khiển phần cứng ROS2 Humble, đọc RPLidar C1, điều khiển động cơ ESP32 qua Serial, phát luồng Video MJPEG (8080) và thu âm Micro UDP (5000).
2. **PC AI Server (`robot_ai_server`)**: Xử lý trí tuệ nhân tạo gồm PhoWhisper STT (Khử nhiễu DSP 80Hz), ShopAIKey Cloud LLM (`gpt-4o-mini` / `gemini-1.5-flash`), và YOLO11s Vision (30+ vật thể tiếng Việt).
3. **PC Web Platform (`Robot_web`)**: Giao diện trung tâm giám sát & điều khiển gồm FastAPI Backend (8000), React Frontend (5173), Lidar 2D Visualizer và Live Stream Video.

---

## 🚀 2. THỨ TỰ KHỞI CHẠY TOÀN BỘ HỆ THỐNG

### ────────── PHẦN A: TRÊN RASPBERRY PI 4 ──────────

#### 📥 Bước A0: Cập nhật code mới nhất (Chạy 1 lần)
```bash
cd ~/robot_main
git pull
source install/setup.bash
```

#### 🟢 Terminal 1 (Pi): Khởi chạy phần cứng Lidar C1
```bash
cd ~/robot_main
source install/setup.bash
ros2 launch robot_bringup lidar.launch.py serial_port:=/dev/ttyUSB0
```
*(Yêu cầu: Đèn Lidar C1 quay tròn, log báo `SLLidar health status : 0 (OK)`)*

#### 🟢 Terminal 2 (Pi): Khởi chạy Mạch Động cơ ESP32 & Cảm biến
```bash
cd ~/robot_main
source install/setup.bash
ros2 launch robot_serial robot_serial.launch.py
```
*(Yêu cầu: Đã cắm dây USB ESP32 vào cổng `/dev/ttyUSB1` và bật công tắc Pin 12V)*

#### 🟢 Terminal 3 (Pi): Khởi chạy HTTP Bridge (Cổng 8001) & Video/Audio Streamer
```bash
cd ~/robot_main
source install/setup.bash
ros2 launch robot_ai robot_all.launch.py
```
*(Yêu cầu: Mở sẵn HTTP Cổng 8001 nhận lệnh, Cổng Video 8080 và Cổng Micro UDP 5000)*

---

### ────────── PHẦN B: TRÊN MÁY TÍNH PC ──────────

#### 🔵 Terminal 1 (PC): Khởi chạy Web Backend (FastAPI - Cổng 8000)
```powershell
cd f:\Robot_web\backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
*(Yêu cầu: Màn hình báo `INFO: Application startup complete.`)*

#### 🔵 Terminal 2 (PC): Khởi chạy Web Frontend (React - Cổng 5173)
```powershell
cd f:\Robot_web\frontend
npm run dev
```
*(Yêu cầu: Trình duyệt mở được `http://localhost:5173`)*

#### 🔵 Terminal 3 (PC): Khởi chạy AI Server Hợp Nhất (STT + LLM + YOLO Vision)
```powershell
cd c:\Users\dh933\Desktop\robot_ai_server
python main.py 5
```
*(Hoặc gõ `python main.py` rồi chọn số **5** để bật hợp nhất toàn bộ AI)*

---

## 📋 3. BẢNG TRA CỨU DỊCH VỤ & CỔNG MẠNG (PORT REFERENCE)

| Thiết bị | Dịch vụ / Module | Cổng / URL | Chức năng |
| :--- | :--- | :--- | :--- |
| **Raspberry Pi** | HTTP Bridge Server | `http://192.168.61.135:8001` | Nhận lệnh di chuyển `/command`, phát giọng đọc `/tts`, kéo Lidar `/scan` |
| **Raspberry Pi** | Video Stream Server | `http://192.168.61.135:8080/video_feed` | Phát luồng Video MJPEG 640x480 @ 30FPS |
| **Raspberry Pi** | Micro Audio Streamer| `UDP Port 5000` | Truyền gói âm thanh 16kHz PCM mượt mà |
| **PC Server** | AI Vision Streamer | `http://localhost:5050/video_feed` | Phát luồng Video YOLO11s nhận diện Bàn/Ghế/Người tiếng Việt |
| **PC Server** | STT WebSocket | `ws://localhost:8000/ws/stt` | Phát phụ đề chạy chữ thời gian thực kiểu YouTube |
| **PC Server** | Web Backend API | `http://localhost:8000` | API điều khiển, kết nối Lidar, lưu NoSQL Database |
| **PC Server** | Web Dashboard UI | `http://localhost:5173` | Giao diện quản trị, xem Cam, xem Lidar 2D, lái xe Joystick |

---

## 🛠️ 4. HƯỚNG DẪN THAO TÁC TRÊN GIAO DIỆN WEB (`http://localhost:5173`)

1. **Trang Dashboard (`/`)**: Xem tổng quan trạng thái Pin, nhiệt độ, kết nối WiFi.
2. **Trang Control (`/control`)**: Lái xe bằng cần Joystick (Tiến, lùi, rẽ, xoay 360 độ, Dừng khẩn cấp).
3. **Trang Camera (`/camera`)**: Xem trực tiếp góc nhìn AI Vision của Robot (Khung vuông nhận diện Bàn, Ghế, Người...).
4. **Trang LiDAR (`/lidar`)**: Xem trực quan bản đồ 2D LaserScan mượt mà với 360 điểm quét ở trạng thái **`ONLINE`**.
5. **Trang Trợ lý AI (`/ai`)**: Trò chuyện giọng nói với Kim Qui, xem phụ đề thời gian thực và lịch sử trò chuyện.

---

## ❓ 5. XỬ LÝ SỰ CỐ THƯỜNG GẶP (TROUBLESHOOTING)

* **Lỗi bánh xe không quay**:
  * Kiểm tra xem công tắc nguồn Pin 12V đã bật chưa.
  * Gõ `ls -l /dev/ttyUSB*` trên Pi đảm bảo xuất hiện cả `/dev/ttyUSB0` (Lidar) và `/dev/ttyUSB1` (ESP32).
* **Lỗi Lidar trên Web báo OFFLINE**:
  * Đảm bảo Terminal 1 trên Pi đang chạy lệnh `lidar.launch.py` và Terminal 3 đang chạy `robot_all.launch.py`.
  * Nhấn `F5` trên web để tải lại cache.
* **Lỗi Cam bị giật hoặc ngắt Mic**:
  * Hệ thống đã ép nén `MJPG` tự động, kiểm tra dây cắm USB UGREEN chắc chắn vào cổng USB 3.0 trên Pi.
