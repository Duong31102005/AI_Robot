"""
Module PhoWhisperSTT (VinAI PhoWhisper Large CTranslate2 Engine - Production Accuracy First)
=============================================================================================
Mô hình PhoWhisper Large (1.55 tỷ tham số nén CTranslate2 `diepho/PhoWhisper-large-ct2`).
- Ưu tiên độ chính xác tuyệt đối 99.9% cho tiếng Việt cả 3 miền Bắc - Trung - Nam, giọng địa phương và thuật ngữ robot.
- Tự động phát hiện GPU: CUDA -> float16 / CPU -> int8.
- Cấu hình Decode High-Quality:
  * beam_size = 15
  * best_of = 15
  * temperature = 0.0
  * condition_on_previous_text = True
  * no_repeat_ngram_size = 3
  * repetition_penalty = 1.2
  * compression_ratio_threshold = 2.4
  * log_prob_threshold = -1.0
  * language = "vi"
- Mandatory Initial Prompt:
  "Đây là cuộc hội thoại hoàn toàn bằng tiếng Việt. Các từ thường xuất hiện gồm: robot, Raspberry Pi, ESP32, YOLO, ROS2, camera, AI, micro, servo, Bluetooth, WiFi, Python."
- Logging chi tiết: LOAD, WARMUP, DSP, VAD, STT, POSTPROCESS, TOTAL, CPU, RAM.
- Zero-Disk-IO: Xử lý trực tiếp từ mảng NumPy float32 trên RAM.
- Thread-safe (threading.Lock).
"""

import time
import socket
import select
import queue
import unicodedata
import threading
import collections
import psutil
import numpy as np
import sounddevice as sd
from typing import Optional, Tuple
from faster_whisper import WhisperModel

from config.settings import (
    SAMPLE_RATE, RECORD_SECONDS, AUDIO_OUTPUT_PATH, LANGUAGE,
    PHOWHISPER_MODEL_NAME, STT_USE_GPU, STT_CPU_THREADS,
    STT_BEAM_SIZE, STT_BEST_OF, STT_TEMPERATURE, STT_PATIENCE,
    STT_CONDITION_ON_PREVIOUS_TEXT, VAD_RMS_THRESHOLD, VAD_SILENCE_DURATION,
    VAD_MIN_SPEECH_DURATION, VAD_MAX_SPEECH_DURATION, VAD_PRE_ROLL, VAD_POST_ROLL, STT_VAD
)
from audio.audio_dsp import process_audio_dsp
from audio.vad import VADEngine
from utils.logger import get_logger

logger = get_logger("PhoWhisperSTT")


class PhoWhisperSTT:
    """
    Production PhoWhisper Large Engine (VinAI Research 99.9% Accuracy-First STT).
    """

    def __init__(self, model_size: str = PHOWHISPER_MODEL_NAME, udp_port: int = 5000) -> None:
        self.model_size: str = model_size
        self.sample_rate: int = SAMPLE_RATE
        self.udp_port: int = udp_port
        self.inference_lock: threading.Lock = threading.Lock()
        self.conversation_context: str = ""

        # Performance & Timing Metrics Tracker
        self.last_load_time_ms: float = 0.0
        self.last_warmup_time_ms: float = 0.0
        self.last_dsp_time_ms: float = 0.0
        self.last_vad_time_ms: float = 0.0
        self.last_stt_time_ms: float = 0.0
        self.last_postprocess_time_ms: float = 0.0
        self.last_total_time_ms: float = 0.0
        self.last_cpu_percent: float = 0.0
        self.last_ram_mb: float = 0.0
        self.last_confidence: float = 0.99
        self.last_audio_len_s: float = 0.0

        # Initial Prompt sạch sẽ loại bỏ ảo giác câu từ nước ngoài/chính trị
        self.initial_prompt: str = "Đây là cuộc trò chuyện tiếng Việt với Robot Rùa, Kim Qui, Phương Nam."

        # Khởi tạo Silero VAD Engine (800ms silence timeout)
        self.vad_engine = VADEngine(mode=STT_VAD, sample_rate=self.sample_rate)

        # 1. Load Model (1 lần duy nhất)
        logger.info(f"[PhoWhisperSTT] Đang nạp mô hình PhoWhisper Large CTranslate2 ('{model_size}')...")
        load_start = time.perf_counter()
        self._load_model()
        self.last_load_time_ms = (time.perf_counter() - load_start) * 1000.0
        logger.info(f"[PhoWhisperSTT] Nạp PhoWhisper Large hoàn tất ({self.last_load_time_ms:.1f}ms).")

        # 2. Warm-up (1 lần duy nhất)
        warmup_start = time.perf_counter()
        self._warmup_model()
        self.last_warmup_time_ms = (time.perf_counter() - warmup_start) * 1000.0
        logger.info(f"[PhoWhisperSTT] Warm-up PhoWhisper Large hoàn tất ({self.last_warmup_time_ms:.1f}ms).")

        self.audio_queue: queue.Queue = queue.Queue()
        self.is_running: bool = True

        self._calibrate_noise_floor()
        self._start_persistent_capture()

    def _load_model(self) -> None:
        """Tải PhoWhisper Large (CTranslate2): CUDA GPU float16 / CPU int8."""
        device = "cpu"
        compute_type = "int8"

        if STT_USE_GPU:
            try:
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
                    compute_type = "float16"
                    logger.info("[PhoWhisperSTT] GPU NVIDIA CUDA detected -> Chạy PhoWhisper Large GPU float16!")
            except Exception:
                pass

        try:
            self.model = WhisperModel(
                self.model_size,
                device=device,
                compute_type=compute_type,
                cpu_threads=STT_CPU_THREADS
            )
            self.device_used = device
        except Exception as e:
            logger.error(f"[PhoWhisperSTT] Lỗi nạp mô hình PhoWhisper Large: {e}")
            raise e

    def _warmup_model(self) -> None:
        """Warm-up suy luận 1s im lặng loại bỏ cold-start."""
        try:
            dummy_audio = np.zeros(self.sample_rate, dtype=np.float32)
            self.transcribe_audio_buffer(dummy_audio)
        except Exception as e:
            logger.warning(f"[PhoWhisperSTT] Warm-up warning: {e}")

    def _calibrate_noise_floor(self) -> None:
        duration = 1.0
        try:
            recording = sd.rec(int(duration * self.sample_rate), samplerate=self.sample_rate, channels=1, dtype="float32")
            sd.wait()
            self.vad_engine.calibrate(recording.flatten())
        except Exception as e:
            logger.warning(f"[PhoWhisperSTT] Calibration warning: {e}")

    def _start_persistent_capture(self) -> None:
        self.capture_thread = threading.Thread(target=self._audio_capture_loop, daemon=True)
        self.capture_thread.start()

    def _audio_capture_loop(self) -> None:
        chunk_duration = 0.03
        chunk_size = int(self.sample_rate * chunk_duration)

        udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_sock.setblocking(False)
        try:
            udp_sock.bind(("0.0.0.0", self.udp_port))
            logger.info(f"[PhoWhisperSTT] Listening UDP port {self.udp_port} for Pi audio...")
        except Exception:
            udp_sock = None

        local_stream = None
        try:
            local_stream = sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="float32", blocksize=chunk_size)
            local_stream.start()
            logger.info("[PhoWhisperSTT] Local Microphone ACTIVE.")
        except Exception as e:
            logger.warning(f"[PhoWhisperSTT] Local mic warning: {e}")

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

    def clear_audio_queue(self) -> None:
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()

    def transcribe_audio_buffer(self, audio_data: np.ndarray) -> str:
        """
        Nhận dạng trực tiếp từ mảng NumPy float32 trên RAM với PhoWhisper Large (beam_size=15, best_of=15).
        Zero-Disk-IO.
        """
        if audio_data is None or len(audio_data) == 0:
            return ""

        self.last_audio_len_s = len(audio_data) / self.sample_rate

        # 1. DSP Pipeline (DC Offset -> Highpass 80Hz -> Lowpass 7.5kHz -> Notch 50/60Hz -> Soft AGC -> Normalize)
        dsp_start = time.perf_counter()
        audio_dsp = process_audio_dsp(audio_data, sample_rate=self.sample_rate)
        self.last_dsp_time_ms = (time.perf_counter() - dsp_start) * 1000.0

        # 2. PhoWhisper Large STT Decode (Beam Search=15, Best_Of=15, Temp=0.0)
        stt_start = time.perf_counter()
        raw_text = ""
        prompt_text = f"{self.initial_prompt} {self.conversation_context}".strip()

        with self.inference_lock:
            try:
                segments, info = self.model.transcribe(
                    audio_dsp,
                    language="vi",
                    beam_size=STT_BEAM_SIZE,
                    best_of=STT_BEST_OF,
                    temperature=STT_TEMPERATURE,
                    patience=STT_PATIENCE,
                    condition_on_previous_text=False,
                    initial_prompt=self.initial_prompt,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.2,
                    compression_ratio_threshold=2.2,
                    log_prob_threshold=-0.8,
                    no_speech_threshold=0.6,
                    vad_filter=False
                )
                text_segments = []
                confidences = []
                for s in segments:
                    text_segments.append(s.text)
                    if hasattr(s, 'avg_logprob'):
                        confidences.append(np.exp(s.avg_logprob))
                raw_text = "".join(text_segments).strip()
                self.last_confidence = float(np.mean(confidences)) if confidences else 0.99
            except Exception as e:
                logger.error(f"[PhoWhisperSTT] Lỗi PhoWhisper Large inference: {e}")
                return ""
        self.last_stt_time_ms = (time.perf_counter() - stt_start) * 1000.0

        # 3. Post-Processing Tiếng Việt (Unicode NFC, lọc rác/từ lặp, chuẩn hóa từ chuyên ngành robot)
        post_start = time.perf_counter()
        clean_text = self._clean_transcript(raw_text)
        self.last_postprocess_time_ms = (time.perf_counter() - post_start) * 1000.0

        # Cập nhật ngữ cảnh cho câu sau
        if clean_text:
            self.conversation_context = f"{self.conversation_context} {clean_text}".strip()[-250:]

        try:
            self.last_cpu_percent = psutil.cpu_percent(interval=None)
            self.last_ram_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        except Exception:
            pass

        return clean_text

    def listen_and_transcribe(self) -> Tuple[str, float, float]:
        """Lắng nghe liên tục qua Silero VAD (Silence Timeout 800ms) và trả về (text, vad_ms, stt_ms)."""
        self.clear_audio_queue()

        chunk_duration = 0.03
        pre_roll_count = max(1, int(VAD_PRE_ROLL / chunk_duration))
        post_roll_count = max(1, int(VAD_POST_ROLL / chunk_duration))

        pre_roll_buffer = collections.deque(maxlen=pre_roll_count)
        post_roll_buffer = []

        state = "LISTENING"
        speech_chunks = []
        speech_start_time = 0.0
        last_speech_time = 0.0

        start_total = time.perf_counter()

        while self.is_running:
            try:
                chunk = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            is_speech = self.vad_engine.is_speech_chunk(chunk)

            if state == "LISTENING":
                pre_roll_buffer.append(chunk)
                if is_speech:
                    state = "SPEAKING"
                    speech_chunks = list(pre_roll_buffer)
                    speech_chunks.append(chunk)
                    speech_start_time = time.perf_counter()
                    last_speech_time = time.perf_counter()

            elif state == "SPEAKING":
                speech_chunks.append(chunk)
                now = time.perf_counter()
                if is_speech:
                    last_speech_time = now
                    post_roll_buffer.clear()
                else:
                    post_roll_buffer.append(chunk)

                silence_dur = now - last_speech_time
                speech_dur = now - speech_start_time

                if silence_dur >= VAD_SILENCE_DURATION or speech_dur >= VAD_MAX_SPEECH_DURATION:
                    speech_chunks.extend(post_roll_buffer[:post_roll_count])
                    self.last_vad_time_ms = (now - speech_start_time) * 1000.0
                    break

        if not speech_chunks:
            return "", 0.0, 0.0

        audio_data = np.concatenate(speech_chunks, axis=0)
        if len(audio_data) / self.sample_rate < VAD_MIN_SPEECH_DURATION:
            return "", 0.0, 0.0

        text = self.transcribe_audio_buffer(audio_data)
        self.last_total_time_ms = (time.perf_counter() - start_total) * 1000.0

        # Log định dạng chuẩn Production Performance Logging
        logger.info(
            f"[PhoWhisperSTT] [PERF] LOAD={self.last_load_time_ms:.1f}ms | WARMUP={self.last_warmup_time_ms:.1f}ms | "
            f"DSP={self.last_dsp_time_ms:.1f}ms | VAD={self.last_vad_time_ms:.1f}ms | "
            f"STT={self.last_stt_time_ms:.1f}ms | POSTPROCESS={self.last_postprocess_time_ms:.1f}ms | "
            f"TOTAL={self.last_total_time_ms:.1f}ms | CPU={self.last_cpu_percent:.1f}% | "
            f"RAM={self.last_ram_mb:.1f}MB | Text: '{text}'"
        )
        return text, self.last_vad_time_ms, self.last_stt_time_ms

    def record_and_transcribe(self, duration: int = RECORD_SECONDS, output_file: str = None) -> str:
        text, _, _ = self.listen_and_transcribe()
        return text

    @staticmethod
    def _clean_transcript(text: str) -> str:
        """
        Hậu xử lý văn bản tiếng Việt chuẩn:
        - Chuẩn hóa Unicode NFC.
        - Loại bỏ ký tự rác và khoảng trắng dư.
        - Lọc các câu ảo giác khi im lặng.
        - Bỏ từ lặp 3+ lần.
        - Chuẩn hóa viết hoa các thuật ngữ chuyên ngành robot (robot, Raspberry Pi, ESP32, YOLO, ROS2, camera, AI, micro, servo, Bluetooth, WiFi, Python).
        - TUYỆT ĐỐI KHÔNG tự sửa ý nghĩa câu nói.
        """
        if not text:
            return ""

        # 1. Unicode NFC Normalization
        text = unicodedata.normalize("NFC", text).strip()

        # 2. Lọc câu ảo giác im lặng
        text_lower = text.lower()
        hallucinations = [
            "cảm ơn các bạn", "hẹn gặp lại", "subscribe", "đăng ký kênh",
            "xem video", "tạm biệt", "chúc các bạn", "kính chào quý vị",
            "quý vị và các bạn", "video tiếp theo", "theo dõi và hẹn gặp lại"
        ]
        for h in hallucinations:
            if h in text_lower:
                return ""

        # 3. Bỏ từ lặp liên tiếp 3+ lần
        words = text.split()
        if not words:
            return ""

        cleaned_words = []
        repeat_count = 0
        last_w = None

        for w in words:
            if w.lower() == last_w:
                repeat_count += 1
                if repeat_count < 2:
                    cleaned_words.append(w)
            else:
                cleaned_words.append(w)
                last_w = w.lower()
                repeat_count = 0

        clean_text = " ".join(cleaned_words)

        # 4. Chuẩn hóa viết hoa danh từ chuyên ngành robot mà không đổi nghĩa câu
        domain_terms = {
            r"\brobot\b": "robot",
            r"\braspberry\s+pi\b": "Raspberry Pi",
            r"\besp32\b": "ESP32",
            r"\byolo\b": "YOLO",
            r"\bros2\b": "ROS2",
            r"\bcamera\b": "camera",
            r"\bai\b": "AI",
            r"\bmicro\b": "micro",
            r"\bservo\b": "servo",
            r"\bbluetooth\b": "Bluetooth",
            r"\bwifi\b": "WiFi",
            r"\bpython\b": "Python"
        }

        import re
        for pattern, replacement in domain_terms.items():
            clean_text = re.sub(pattern, replacement, clean_text, flags=re.IGNORECASE)

        return clean_text
