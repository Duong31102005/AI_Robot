import os

# --- Raspberry Pi Server Configuration ---
PI_IP = os.getenv("PI_IP", "192.168.61.135")
PI_PORT = int(os.getenv("PI_PORT", 8001))
PI_COMMAND_URL = f"http://{PI_IP}:{PI_PORT}/command"

# --- Movement & Safety Configuration ---
DRY_RUN = os.getenv("DRY_RUN", "False").lower() in ("true", "1", "yes")
SEND_COMMAND_INTERVAL = float(os.getenv("SEND_COMMAND_INTERVAL", 0.3))  # Giây giữa các lần gửi lệnh HTTP

# --- Vision & Camera Configuration ---
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "http://192.168.61.135:8080/video_feed")
try:
    CAMERA_INDEX = int(CAMERA_SOURCE)
except ValueError:
    CAMERA_INDEX = CAMERA_SOURCE  # IP Stream URL string (e.g. http://192.168.61.135:8080/video_feed)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30

# --- YOLO11s Person Detection Configuration ---
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolo11s.pt")
YOLO_MODEL_NAME = YOLO_MODEL  # Tương thích ngược
YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", 0.45))
CONFIDENCE_THRESHOLD = YOLO_CONFIDENCE  # Tương thích ngược
YOLO_IMAGE_SIZE = int(os.getenv("YOLO_IMAGE_SIZE", 640))
YOLO_PERSON_CLASS = 0  # Class ID của 'person' trong COCO dataset
VISION_DEBUG = os.getenv("VISION_DEBUG", "True").lower() in ("true", "1", "yes")

# --- Audio & Speech Recognition Configuration ---
SAMPLE_RATE = 16000
RECORD_SECONDS = 5
WHISPER_MODEL_SIZE = "base"
AUDIO_OUTPUT_PATH = "input.wav"
LANGUAGE = "vi"

# --- VAD & Continuous Speech Detection Configuration ---
VAD_RMS_THRESHOLD = float(os.getenv("VAD_RMS_THRESHOLD", 0.00003))         # Ngưỡng năng lượng RMS thích ứng với Mic máy tính
VAD_SILENCE_DURATION = float(os.getenv("VAD_SILENCE_DURATION", 0.4))       # Thời gian im lặng ngắt câu siêu nhanh (0.4s)
VAD_MIN_SPEECH_DURATION = float(os.getenv("VAD_MIN_SPEECH_DURATION", 0.25))# Thời gian nói tối thiểu (giây)
VAD_MAX_SPEECH_DURATION = float(os.getenv("VAD_MAX_SPEECH_DURATION", 5.0)) # Thời gian nói tối đa (giây)
VAD_PRE_ROLL = float(os.getenv("VAD_PRE_ROLL", 0.15))                      # Đệm trước tiếng nói (giây)
VAD_NOISE_MULTIPLIER = float(os.getenv("VAD_NOISE_MULTIPLIER", 2.5))       # Gấp 2.5 lần Noise Floor khi calibrate
CALIBRATION_DURATION = float(os.getenv("CALIBRATION_DURATION", 1.0))       # Thời gian đo tiếng ồn ban đầu (giây)

# --- Whisper Anti-Hallucination Configuration ---
WHISPER_NO_SPEECH_THRESHOLD = float(os.getenv("WHISPER_NO_SPEECH_THRESHOLD", 0.6))
WHISPER_TEMPERATURE = float(os.getenv("WHISPER_TEMPERATURE", 0.0))

# --- Offline LLM (Ollama) & TTS Configuration ---
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
ENABLE_LLM_CHAT = os.getenv("ENABLE_LLM_CHAT", "True").lower() in ("true", "1", "yes")
ENABLE_TTS_SPEAKER = os.getenv("ENABLE_TTS_SPEAKER", "True").lower() in ("true", "1", "yes")


