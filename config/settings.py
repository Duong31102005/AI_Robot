import os

# --- Raspberry Pi Server Configuration ---
PI_IP = os.getenv("PI_IP", "192.168.61.135")
PI_PORT = int(os.getenv("PI_PORT", 8001))
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

# --- YOLO11s Person Detection Configuration ---
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolo11s.pt")
YOLO_MODEL_NAME = YOLO_MODEL  # Tương thích ngược
YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", 0.45))
CONFIDENCE_THRESHOLD = YOLO_CONFIDENCE  # Tương thích ngược
YOLO_IMAGE_SIZE = int(os.getenv("YOLO_IMAGE_SIZE", 640))
YOLO_PERSON_CLASS = 0  # Class ID của 'person' trong COCO dataset
VISION_DEBUG = os.getenv("VISION_DEBUG", "True").lower() in ("true", "1", "yes")

# --- Audio & Speech Recognition Engine Configuration ---
SAMPLE_RATE = 16000
RECORD_SECONDS = 5
AUDIO_OUTPUT_PATH = "input.wav"
LANGUAGE = "vi"

# Engine selection: PhoWhisper Large (VinAI PhoWhisper-large CTranslate2)
STT_ENGINE = os.getenv("STT_ENGINE", "phowhisper").lower()
STT_VAD = os.getenv("STT_VAD", "silero").lower()  # "silero" (mặc định cho độ chính xác cao) hoặc "webrtc"
STT_USE_GPU = os.getenv("STT_USE_GPU", "True").lower() in ("true", "1", "yes")
STT_CPU_THREADS = int(os.getenv("STT_CPU_THREADS", 4))

# PhoWhisper Optimized Decode Parameters (UGREEN Webcam Mic Accuracy Boost)
STT_BEAM_SIZE = int(os.getenv("STT_BEAM_SIZE", 10))
STT_BEST_OF = int(os.getenv("STT_BEST_OF", 10))
STT_TEMPERATURE = float(os.getenv("STT_TEMPERATURE", 0.0))
STT_PATIENCE = float(os.getenv("STT_PATIENCE", 2.0))
STT_CONDITION_ON_PREVIOUS_TEXT = False

PHOWHISPER_MODEL_NAME = os.getenv("PHOWHISPER_MODEL_NAME", "diepho/PhoWhisper-small-ct2")
MOONSHINE_MODEL_NAME = os.getenv("MOONSHINE_MODEL_NAME", "onnx-community/moonshine-tiny-vi-ONNX")
WHISPER_MODEL_SIZE = PHOWHISPER_MODEL_NAME  # Tương thích ngược

# --- VAD & Continuous Speech Detection Configuration ---
VAD_RMS_THRESHOLD = float(os.getenv("VAD_RMS_THRESHOLD", 0.003))          # Ngưỡng năng lượng RMS thích ứng với Mic máy tính
VAD_SILENCE_DURATION = float(os.getenv("VAD_SILENCE_DURATION", 0.8))       # Thời gian im lặng 700-900ms (0.8s)
VAD_MIN_SPEECH_DURATION = float(os.getenv("VAD_MIN_SPEECH_DURATION", 0.2)) # Thời gian nói tối thiểu (giây)
VAD_MAX_SPEECH_DURATION = float(os.getenv("VAD_MAX_SPEECH_DURATION", 10.0))# Thời gian nói tối đa (giây)
VAD_PRE_ROLL = float(os.getenv("VAD_PRE_ROLL", 0.4))                       # Đệm trước tiếng nói (0.4s)
VAD_POST_ROLL = float(os.getenv("VAD_POST_ROLL", 0.4))                     # Đệm sau tiếng nói (0.4s)
VAD_NOISE_MULTIPLIER = float(os.getenv("VAD_NOISE_MULTIPLIER", 2.5))       # Gấp 2.5 lần Noise Floor khi calibrate
CALIBRATION_DURATION = float(os.getenv("CALIBRATION_DURATION", 1.0))       # Thời gian đo tiếng ồn ban đầu (giây)

# --- Offline LLM (Ollama) & TTS Configuration ---
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
ENABLE_LLM_CHAT = os.getenv("ENABLE_LLM_CHAT", "True").lower() in ("true", "1", "yes")
ENABLE_TTS_SPEAKER = os.getenv("ENABLE_TTS_SPEAKER", "True").lower() in ("true", "1", "yes")


