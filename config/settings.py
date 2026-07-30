import os

# --- Raspberry Pi Server Configuration ---
PI_IP = os.getenv("PI_IP", "192.168.61.135")
PI_PORT = int(os.getenv("PI_PORT", 8001))
PI_COMMAND_URL = f"http://{PI_IP}:{PI_PORT}/command"

# --- Movement & Safety Configuration ---
DRY_RUN = os.getenv("DRY_RUN", "False").lower() in ("true", "1", "yes")
SEND_COMMAND_INTERVAL = float(os.getenv("SEND_COMMAND_INTERVAL", 0.3))  # Giây giữa các lần gửi lệnh HTTP

# --- Vision & Camera Configuration ---
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", 0))
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
