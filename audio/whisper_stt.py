import sounddevice as sd
from scipy.io.wavfile import write
import whisper
from config.settings import (
    SAMPLE_RATE, RECORD_SECONDS, WHISPER_MODEL_SIZE,
    AUDIO_OUTPUT_PATH, LANGUAGE
)
from utils.logger import get_logger

logger = get_logger("WhisperSTT")

class WhisperSTT:
    """Module thu âm qua Micro và chuyển đổi giọng nói thành văn bản sử dụng OpenAI Whisper."""

    def __init__(self, model_size: str = WHISPER_MODEL_SIZE):
        logger.info(f"Đang tải mô hình Whisper ('{model_size}')...")
        self.model = whisper.load_model(model_size)
        logger.info("Mô hình Whisper đã sẵn sàng.")

    def record_and_transcribe(self, duration: int = RECORD_SECONDS, output_file: str = AUDIO_OUTPUT_PATH) -> str:
        """Thu âm giọng nói trong N giây và nhận dạng văn bản tiếng Việt."""
        logger.info("================================================")
        logger.info(f"Đang ghi âm trong {duration} giây... (hãy nói vào mic)")
        logger.info("================================================")

        audio = sd.rec(
            int(duration * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16"
        )
        sd.wait()

        write(output_file, SAMPLE_RATE, audio)
        logger.info("Đang xử lý nhận dạng giọng nói với Whisper...")

        result = self.model.transcribe(
            output_file,
            language=LANGUAGE
        )

        text = result.get("text", "").strip()
        logger.info(f"Kết quả nhận dạng: '{text}'")
        return text
