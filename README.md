# 🤖 Robot AI Server - Hệ thống Thị giác máy tính & Giọng nói AI cho Robot

Hệ thống trung tâm máy chủ AI cho robot giao tiếp tự hành, kết hợp mô hình **YOLO11s** (Phát hiện & Theo dõi con người thời gian thực) và **OpenAI Whisper** (Nhận dạng giọng nói tiếng Việt), kết nối thông suốt tới **Raspberry Pi ROS 2 Controller**.

---

## 🌟 Tính năng chính

- 👁️ **YOLO11s Person Detection & Tracking**:
  - Tải và chạy mô hình `yolo11s.pt` chuyên biệt lọc đối tượng con người (`class_id = 0`).
  - Tự động phát hiện thiết bị tính toán (**GPU/CUDA** hoặc **CPU**).
  - Tự động chọn đối tượng mục tiêu chính (*Target Selection*) theo diện tích bounding box lớn nhất (người ở gần robot nhất).
  - Tính toán vị trí tương quan chuẩn hóa $error\_x \in [-1.0, 1.0]$ hỗ trợ điều hướng xoay robot face-to-face.
- 🎙️ **Speech-to-Text (Whisper STT)**:
  - Ghi âm qua Micro và chuyển đổi giọng nói tiếng Việt thành văn bản bằng mô hình OpenAI Whisper offline.
- 📡 **Giao tiếp Raspberry Pi & ROS 2**:
  - Gửi lệnh HTTP JSON realtime tới Raspberry Pi qua cổng `8001`.
  - Đồng bộ lệnh với `http_bridge_node` và `command_node` (ROS 2) trên Raspberry Pi.
  - Tích hợp cơ chế **Safety Watchdog** và khoảng thời gian gửi lệnh an toàn (0.3s) chống trôi robot.

---

## 📐 Kiến trúc hệ thống tổng thể

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PC / AI SERVER (robot_ai_server)                      │
│                                                                             │
│  [Webcam/Camera] ──> [YOLO11s Inference] ──> [Target Selector & error_x]    │
│                                                          │                  │
│                                              HTTP POST /command             │
│                                              {"text": "quay_trai"|...}      │
└──────────────────────────────────────────────────────────┬──────────────────┘
                                                           │
                               HTTP http://192.168.61.135:8001/command
                                                           │
┌──────────────────────────────────────────────────────────▼──────────────────┐
│                   RASPBERRY PI 4 (f:\robot_main - ROS 2)                    │
│                                                                             │
│  http_bridge_node (port 8001) ──> ROS 2 Topic (/robot/command)              │
│                                            │                                │
│                                            ▼                                │
│                                      command_node                           │
│                                    (Watchdog 1.0s)                          │
│                                            │                                │
│                                            ▼                                │
│                                   ROS 2 Topic (/cmd_vel)                    │
│                                            │                                │
│                                            ▼                                │
│                                      robot_serial                           │
└────────────────────────────────────────────┬────────────────────────────────┘
                                             │
                                     UART Serial (<CMD,lin,ang>)
                                             │
┌────────────────────────────────────────────▼────────────────────────────────┐
│                          ESP32 + ROBOT MOTOR HARDWARE                       │
│  ESP32 Controller ──> Dual Motor Driver ──> Động cơ di chuyển               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Cấu trúc thư mục dự án

```text
robot_ai_server/
├── main.py                  # Entrypoint chính (Interactive CLI Menu)
├── config/
│   ├── __init__.py
│   └── settings.py          # Quản lý cấu hình (IP Pi, Port, Model, Camera Index, DRY_RUN)
├── vision/                  # Module Xử lý hình ảnh & Thị giác máy tính
│   ├── __init__.py
│   ├── camera.py            # Lớp quản lý Camera/Webcam an toàn
│   ├── yolo_detector.py      # Lớp YOLOPersonDetector chạy YOLO11s (class 0)
│   ├── person_tracker.py     # Chọn Target & tính error_x, vẽ giao diện Debug
│   └── camera_stream.py     # Wrapper tương thích ngược
├── audio/                   # Module Nhận dạng giọng nói
│   ├── __init__.py
│   └── whisper_stt.py       # Thu âm & nhận diện giọng nói tiếng Việt bằng Whisper
├── communication/           # Module Giao tiếp mạng
│   ├── __init__.py
│   ├── pi_client.py         # Client gửi lệnh HTTP POST tới Raspberry Pi
│   └── ros_command_node.py  # ROS 2 Command Node
├── scripts/                 # Thư mục chứa các kịch bản thực thi lẻ
│   ├── main_vision.py       # Script chạy phát hiện & theo dõi người realtime
│   ├── main_stt.py          # Script thu âm giọng nói & gửi lệnh cho Pi
│   ├── test_yolo.py         # Script kiểm tra độc lập YOLO11s (Camera + FPS)
│   ├── test_pi_connection.py# Script kiểm tra kết nối mạng HTTP tới Pi
│   ├── send_command.py      # Script gửi lệnh văn bản thủ công
│   └── server.py            # ROS 2 server node wrapper
└── utils/
    ├── __init__.py
    └── logger.py            # Quản lý ghi log chuẩn hệ thống
```

---

## 🛠️ Yêu cầu & Cài đặt môi trường

### 1. Yêu cầu hệ thống:
- **Python**: 3.10+ (Đã thử nghiệm thành công trên Python 3.14)
- **Thiết bị**: Webcam USB hoặc IP Camera.

### 2. Cài đặt thư viện phụ thuộc:
```bash
# Tạo môi trường ảo (khuyên dùng)
python -m venv .venv
source .venv/bin/activate  # Trên Linux/macOS
# hoặc .venv\Scripts\activate trên Windows

# Cài đặt các thư viện cần thiết
pip install ultralytics opencv-python torch torchvision requests sounddevice scipy openai-whisper
```

---

## ⚙️ Cấu hình dự án (`config/settings.py`)

Tệp `config/settings.py` cho phép dễ dàng tùy chỉnh các thông số hệ thống:

```python
# IP và Cổng Raspberry Pi Server
PI_IP = "192.168.61.135"
PI_PORT = 8001

# Chế độ chạy thử không gửi lệnh động cơ (Safety test)
DRY_RUN = False 
SEND_COMMAND_INTERVAL = 0.3  # Khoảng thời gian gửi lệnh (giây)

# Cấu hình Vision & YOLO11s
CAMERA_INDEX = 0
YOLO_MODEL = "yolo11s.pt"
YOLO_CONFIDENCE = 0.45
YOLO_IMAGE_SIZE = 640
YOLO_PERSON_CLASS = 0        # Chỉ phát hiện con người
VISION_DEBUG = True
```

---

## 🚀 Hướng dẫn khởi chạy

### 1. Khởi chạy Menu trung tâm:
```bash
python main.py
```
Menu sẽ hiển thị các lựa chọn:
- `1`: Chạy Vision (Phát hiện & Theo dõi người Face-to-Face)
- `2`: Chạy STT (Nhận dạng giọng nói tiếng Việt bằng Whisper)
- `3`: Chạy ROS 2 Command Node
- `4`: Gửi lệnh văn bản thử nghiệm

### 2. Khởi chạy trực tiếp từng kịch bản (Scripts):

- **Phát hiện & Theo dõi người realtime (Camera + YOLO11s + Gửi lệnh Pi)**:
  ```bash
  python scripts/main_vision.py
  ```

- **Kiểm tra kết nối mạng HTTP tới Raspberry Pi**:
  ```bash
  python scripts/test_pi_connection.py
  ```

- **Kiểm tra độc lập mô hình YOLO11s (Webcam debug overlay)**:
  ```bash
  python scripts/test_yolo.py
  ```

- **Nhận dạng giọng nói Whisper STT**:
  ```bash
  python scripts/main_stt.py
  ```

---

## 🛡️ Cơ chế An toàn (Safety & Failsafe)

1. **Raspberry Pi Watchdog**: Node `command_node` trên Raspberry Pi được cài đặt watchdog 1.0s. Nếu không nhận được lệnh điều khiển mới từ AI Server trong quá 1.0s, Pi sẽ tự động phát lệnh dừng (`Twist(0,0)`) để chống hiện tượng robot tự trôi khi mất mạng.
2. **Auto Emergency Stop**: Khi tắt kịch bản Vision hoặc khi đối tượng người di chuyển ra khỏi tầm nhìn (`target is None`), AI Server sẽ lập tức phát lệnh `giu_nguyen` để dừng robot an toàn.

---

## 📜 Giấy phép (License)
Dự án được phát hành theo giấy phép **MIT License**.
