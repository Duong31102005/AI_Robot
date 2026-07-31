import time
import socket
import select
import queue
import threading
import collections
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import whisper

from config.settings import (
    SAMPLE_RATE, RECORD_SECONDS, WHISPER_MODEL_SIZE,
    AUDIO_OUTPUT_PATH, LANGUAGE, VAD_RMS_THRESHOLD,
    VAD_SILENCE_DURATION, VAD_MIN_SPEECH_DURATION,
    VAD_MAX_SPEECH_DURATION, VAD_PRE_ROLL,
    VAD_NOISE_MULTIPLIER, CALIBRATION_DURATION,
    WHISPER_NO_SPEECH_THRESHOLD, WHISPER_TEMPERATURE
)
from utils.logger import get_logger

logger = get_logger("WhisperSTT")


class WhisperSTT:
    """
    Module thu âm liên tục (từ Microphone cục bộ hoặc UDP Audio Stream từ Raspberry Pi)
    chạy luồng ngầm (Persistent Capture Thread) không bao giờ đóng Mic hay Socket.
    Kết hợp VAD (RMS Energy) và nhận dạng giọng nói tiếng Việt bằng OpenAI Whisper.
    """

    def __init__(self, model_size: str = WHISPER_MODEL_SIZE, udp_port: int = 5000):
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Đang tải mô hình Whisper ('{model_size}') trên thiết bị {self.device.upper()}...")
        self.model = whisper.load_model(model_size, device=self.device)
        logger.info(f"Mô hình Whisper đã sẵn sàng ({self.device.upper()} GPU Accelerated).")

        self.sample_rate = SAMPLE_RATE
        self.udp_port = udp_port
        self.noise_floor = 0.005
        self.effective_threshold = VAD_RMS_THRESHOLD

        self.audio_queue = queue.Queue()
        self.is_running = True

        self._calibrate_noise_floor()
        self._start_persistent_capture()

    def _calibrate_noise_floor(self):
        """Tự động đo độ ồn môi trường (Noise Floor) trong 1.0 giây đầu để chỉnh VAD threshold."""
        logger.info("[STT] Calibrating noise floor (1.0s)... Vui lòng giữ yên lặng.")
        duration = CALIBRATION_DURATION
        try:
            recording = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32"
            )
            sd.wait()
            rms = float(np.sqrt(np.mean(recording ** 2)))
            self.noise_floor = rms
            calc_thresh = max(VAD_RMS_THRESHOLD, rms * VAD_NOISE_MULTIPLIER)
            self.effective_threshold = calc_thresh
            logger.info(f"[STT] Calibrated Noise Floor: RMS={rms:.6f} | Effective Threshold: {self.effective_threshold:.6f}")
        except Exception as e:
            logger.warning(f"[STT] Calibration failed: {e}. Sử dụng threshold mặc định: {VAD_RMS_THRESHOLD}")
            self.effective_threshold = VAD_RMS_THRESHOLD

    def _start_persistent_capture(self):
        """Khởi động luồng thu âm liên tục không dừng."""
        self.capture_thread = threading.Thread(target=self._audio_capture_loop, daemon=True)
        self.capture_thread.start()

    def _audio_capture_loop(self):
        """Luồng thu âm ngầm (UDP Stream từ Pi + Local PC Microphone)."""
        chunk_duration = 0.03  # 30ms block
        chunk_size = int(self.sample_rate * chunk_duration)

        # 1. UDP Socket từ Pi (Port 5000)
        udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_sock.setblocking(False)
        try:
            udp_sock.bind(("0.0.0.0", self.udp_port))
            logger.info(f"[STT] Listening on UDP port {self.udp_port} for Pi audio stream...")
        except Exception:
            udp_sock = None

        # 2. Local PC Microphone Stream
        local_stream = None
        try:
            local_stream = sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="float32", blocksize=chunk_size)
            local_stream.start()
            logger.info("[STT] Local Microphone stream ACTIVE.")
        except Exception as e:
            logger.warning(f"[STT] Local mic warning: {e}")

        while self.is_running:
            audio_block = None

            if udp_sock:
                ready = select.select([udp_sock], [], [], 0.01)
                if ready[0]:
                    packet, addr = udp_sock.recvfrom(8192)
                    audio_block = np.frombuffer(packet, dtype=np.float32)

            if audio_block is None and local_stream:
                try:
                    data, overflowed = local_stream.read(chunk_size)
                    audio_block = data.flatten()
                except Exception:
                    pass

            if audio_block is not None and len(audio_block) > 0:
                self.audio_queue.put(audio_block)
            else:
                time.sleep(0.005)

    def listen_and_transcribe(self) -> str:
        """
        Đọc liên tục từ Audio Queue, chạy VAD RMS realtime và Whisper nhận dạng.
        """
        chunk_duration = 0.03
        pre_roll_chunks_count = max(1, int(VAD_PRE_ROLL / chunk_duration))
        pre_roll_buffer = collections.deque(maxlen=pre_roll_chunks_count)

        state = "LISTENING"
        logger.info("[STT] LISTENING...")

        speech_chunks = []
        speech_start_time = 0.0
        last_speech_time = 0.0

        while self.is_running:
            try:
                audio_block = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            rms = float(np.sqrt(np.mean(audio_block ** 2)))

            if state == "LISTENING":
                pre_roll_buffer.append(audio_block)
                if rms > self.effective_threshold:
                    state = "SPEAKING"
                    logger.info("[STT] Speech detected")
                    logger.info("[STT] Recording...")
                    speech_chunks = list(pre_roll_buffer)
                    speech_chunks.append(audio_block)
                    speech_start_time = time.time()
                    last_speech_time = time.time()

            elif state == "SPEAKING":
                speech_chunks.append(audio_block)
                now = time.time()
                if rms > self.effective_threshold:
                    last_speech_time = now

                silence_duration = now - last_speech_time
                speech_duration = now - speech_start_time

                if silence_duration >= VAD_SILENCE_DURATION or speech_duration >= VAD_MAX_SPEECH_DURATION:
                    state = "PROCESSING"
                    logger.info("[STT] Silence detected")
                    logger.info("[STT] Processing...")
                    break

        if not speech_chunks:
            logger.info("[STT] Ignored silence")
            return ""

        audio_data = np.concatenate(speech_chunks, axis=0)
        total_duration = len(audio_data) / self.sample_rate

        if total_duration < VAD_MIN_SPEECH_DURATION:
            logger.info("[STT] Ignored silence")
            return ""

        # Preprocessing 1: Loại bỏ DC Offset
        audio_data = audio_data - np.mean(audio_data)

        # Preprocessing 2: Bộ lọc High-pass ~80Hz
        try:
            from scipy.signal import butter, filtfilt
            b, a = butter(4, 80 / (self.sample_rate / 2), btype='high')
            audio_data = filtfilt(b, a, audio_data)
        except Exception:
            pass

        # Preprocessing 3: Chuẩn hóa âm lượng Peak (Dành cho Mic có tín hiệu nhỏ)
        max_amp = float(np.max(np.abs(audio_data)))
        if max_amp > 0.00002:
            audio_data = (audio_data / max_amp) * 0.9
        else:
            logger.info("[STT] Ignored silence")
            return ""

        # Ghi âm ra file WAV tạm thời cho Whisper
        write(AUDIO_OUTPUT_PATH, self.sample_rate, (audio_data * 32767).astype(np.int16))

        # Nhận dạng giọng nói với Whisper
        try:
            use_fp16 = (self.device == "cuda")
            result = self.model.transcribe(
                AUDIO_OUTPUT_PATH,
                language=LANGUAGE,
                temperature=WHISPER_TEMPERATURE,
                condition_on_previous_text=False,
                fp16=use_fp16,
                no_speech_threshold=WHISPER_NO_SPEECH_THRESHOLD
            )
        except Exception as e:
            logger.error(f"[STT] Whisper error: {e}")
            return ""

        segments = result.get("segments", [])
        for seg in segments:
            if seg.get("no_speech_prob", 0.0) > WHISPER_NO_SPEECH_THRESHOLD:
                logger.info("[STT] No speech recognized")
                return ""

        text = result.get("text", "").strip()

        if not text or self._is_hallucination(text):
            logger.info("[STT] No speech recognized")
            return ""

        logger.info(f"[STT] Whisper: '{text}'")
        return text

    def record_and_transcribe(self, duration: int = RECORD_SECONDS, output_file: str = AUDIO_OUTPUT_PATH) -> str:
        """Phương thức tương thích ngược."""
        return self.listen_and_transcribe()

    @staticmethod
    def _is_hallucination(text: str) -> bool:
        """Phát hiện các câu ảo giác (hallucination) phổ biến của Whisper khi im lặng."""
        if not text:
            return True

        text_lower = text.lower().strip()

        hallucination_phrases = [
            "cái gì vậy", "cảm ơn các bạn", "hẹn gặp lại", "subscribe",
            "đăng ký kênh", "xem video", "tạm biệt", "chúc các bạn",
            "bài hát", "tập tiếp theo", "mọi người"
        ]
        for phrase in hallucination_phrases:
            if phrase in text_lower:
                return True

        words = text_lower.split()
        if len(words) >= 3:
            if len(set(words)) == 1:
                return True
            bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
            if len(bigrams) >= 3 and bigrams[0] == bigrams[1] == bigrams[2]:
                return True

        return False
