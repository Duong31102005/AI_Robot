import os

# --- Raspberry Pi Server Configuration ---
PI_IP = os.getenv("PI_IP", "192.168.61.135")
PI_PORT = int(os.getenv("PI_PORT", 8000))
PI_COMMAND_URL = f"http://{PI_IP}:{PI_PORT}/command"

# --- Movement & Safety Configuration ---
DRY_RUN = os.getenv("DRY_RUN", "False").lower() in ("true", "1", "yes")
SEND_COMMAND_INTERVAL = float(os.getenv("SEND_COMMAND_INTERVAL", 0.3))  # Giây giữa các lần gửi lệnh HTTP

# --- Vision & Camera Configuration ---
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", f"http://{PI_IP}:8080/video_feed")
try:
    CAMERA_INDEX = int(CAMERA_SOURCE)
except ValueError:
    CAMERA_INDEX = CAMERA_SOURCE  # IP Stream URL string (e.g. http://192.168.61.135:8080/video_feed)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30

# --- YOLO Configuration ---
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolo11n.pt")
YOLO_MODEL_NAME = YOLO_MODEL  # Tương thích ngược
YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", 0.35))
CONFIDENCE_THRESHOLD = YOLO_CONFIDENCE  # Tương thích ngược
YOLO_IMAGE_SIZE = int(os.getenv("YOLO_IMAGE_SIZE", 320))
YOLO_PERSON_CLASS = 0  # Class ID của 'person' trong COCO dataset
VISION_DEBUG = os.getenv("VISION_DEBUG", "True").lower() in ("true", "1", "yes")

OBSTACLE_CLASSES = [
    # 👥 Con người & Động vật
    "person", "dog", "cat",
    
    # 🪑 Nội thất Bàn Ghế & Đồ gia dụng
    "chair", "couch", "dining table", "bed", "bench", "potted plant", "vase",
    
    # 💻 Thiết bị Điện tử & Công nghệ
    "tv", "laptop", "mouse", "keyboard", "cell phone", "remote", "clock",
    
    # 📦 Hành lý & Đồ tư trang
    "backpack", "handbag", "suitcase", "umbrella",
    
    # 🍱 Đồ ăn / Thức uống Giao hàng (Delivery)
    "bottle", "cup", "wine glass", "bowl", "apple", "sandwich",
    
    # 🚗 Phương tiện & Vật cản ngoài đường
    "car", "motorcycle", "bicycle", "bus", "truck", "stop sign"
]

# --- Audio & Speech Recognition Engine Configuration ---
SAMPLE_RATE = 16000
RECORD_SECONDS = 5
AUDIO_OUTPUT_PATH = "input.wav"
LANGUAGE = "vi"

# Engine selection: PhoWhisper Large (VinAI PhoWhisper-large CTranslate2)
STT_ENGINE = os.getenv("STT_ENGINE", "parakeet").lower()
STT_VAD = os.getenv("STT_VAD", "silero").lower()  # "silero" (mặc định cho độ chính xác cao) hoặc "webrtc"
STT_USE_GPU = os.getenv("STT_USE_GPU", "True").lower() in ("true", "1", "yes")
STT_CPU_THREADS = int(os.getenv("STT_CPU_THREADS", 4))

# PhoWhisper Optimized Decode Parameters (UGREEN Webcam Mic Accuracy Boost)
STT_BEAM_SIZE = int(os.getenv("STT_BEAM_SIZE", 5))
STT_BEST_OF = int(os.getenv("STT_BEST_OF", 5))
STT_TEMPERATURE = float(os.getenv("STT_TEMPERATURE", 0.0))
STT_PATIENCE = float(os.getenv("STT_PATIENCE", 1.0))
STT_CONDITION_ON_PREVIOUS_TEXT = False

PHOWHISPER_MODEL_NAME = os.getenv("PHOWHISPER_MODEL_NAME", "diepho/PhoWhisper-small-ct2")
MOONSHINE_MODEL_NAME = os.getenv("MOONSHINE_MODEL_NAME", "onnx-community/moonshine-tiny-vi-ONNX")
WHISPER_TURBO_MODEL_NAME = os.getenv("WHISPER_TURBO_MODEL_NAME", "deepdml/faster-whisper-large-v3-turbo")
WHISPER_MODEL_SIZE = PHOWHISPER_MODEL_NAME  # Tương thích ngược

# Streaming Realtime YouTube-like Subtitle Settings
STT_STREAMING_ENABLED = os.getenv("STT_STREAMING_ENABLED", "True").lower() in ("true", "1", "yes")
PARTIAL_INTERVAL_MS = int(os.getenv("PARTIAL_INTERVAL_MS", 200))  # Cập nhật chữ tạm thời mỗi 200ms

# --- VAD & Continuous Speech Detection Configuration ---
STT_MIN_CONFIDENCE = float(os.getenv("STT_MIN_CONFIDENCE", 0.50))          # Ngưỡng độ tin cậy tối thiểu 50% (chuẩn hóa tiếng Việt)
VAD_RMS_THRESHOLD = float(os.getenv("VAD_RMS_THRESHOLD", 0.015))          # Ngưỡng âm lượng RMS: Bắt nói gần Mic, loại tiếng xa/ồn
VAD_SILENCE_DURATION = float(os.getenv("VAD_SILENCE_DURATION", 0.7))       # Dừng ngay sau 0.7s im lặng để phản hồi cực nhanh kiểu XiaoZhi
VAD_MIN_SPEECH_DURATION = float(os.getenv("VAD_MIN_SPEECH_DURATION", 0.3)) # Lọc tiếng gõ phím/tiếng động ngắn (<0.3s)
VAD_MAX_SPEECH_DURATION = float(os.getenv("VAD_MAX_SPEECH_DURATION", 8.0))  # Thời gian nói tối đa
VAD_PRE_ROLL = float(os.getenv("VAD_PRE_ROLL", 0.2))                       # Đệm trước tiếng nói (0.2s)
VAD_POST_ROLL = float(os.getenv("VAD_POST_ROLL", 0.2))                     # Đệm sau tiếng nói (0.2s)
VAD_NOISE_MULTIPLIER = float(os.getenv("VAD_NOISE_MULTIPLIER", 3.5))       # Gấp 3.5 lần Noise Floor khi calibrate
CALIBRATION_DURATION = float(os.getenv("CALIBRATION_DURATION", 1.0))       # Thời gian đo tiếng ồn ban đầu (giây)

# --- Offline LLM (Ollama) & TTS Configuration ---
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
ENABLE_LLM_CHAT = os.getenv("ENABLE_LLM_CHAT", "True").lower() in ("true", "1", "yes")
ENABLE_TTS_SPEAKER = os.getenv("ENABLE_TTS_SPEAKER", "True").lower() in ("true", "1", "yes")

# --- ShopAIKey Cloud LLM Configuration ---
SHOPAIKEY_API_KEY = os.getenv("SHOPAIKEY_API_KEY", "sk-j7Oux1kadgbj4FF2oM2jXKyLsO8p3O6khSmFNMJh8LNgwxuI")
SHOPAIKEY_BASE_URL = os.getenv("SHOPAIKEY_BASE_URL", "https://api.shopaikey.com/v1")
SHOPAIKEY_MODEL = os.getenv("SHOPAIKEY_MODEL", "gpt-4o-mini")
