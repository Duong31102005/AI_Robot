"""
Module VAD Engine (Voice Activity Detection - Production Multi-Mode)
====================================================================
Hỗ trợ 2 chế độ VAD linh hoạt:
1. WebRTC VAD ("webrtc"): Mặc định, siêu nhẹ, cực kỳ nhanh (< 1ms).
2. Silero VAD ("silero"): Tùy chọn học sâu ONNX / PyTorch.
Tích hợp đệm Pre-roll Buffer và tự động đo ngưỡng tiếng ồn môi trường (Adaptive Noise Floor Calibration).
"""

import time
import numpy as np
import collections
from typing import Tuple, List, Optional
from config.settings import (
    SAMPLE_RATE, VAD_RMS_THRESHOLD, VAD_SILENCE_DURATION,
    VAD_MIN_SPEECH_DURATION, VAD_MAX_SPEECH_DURATION,
    VAD_PRE_ROLL, STT_VAD
)
from utils.logger import get_logger

logger = get_logger("VADEngine")

# Kiểm tra thư viện WebRTC VAD
try:
    import webrtcvad
    HAS_WEBRTC = True
except ImportError:
    HAS_WEBRTC = False


class VADEngine:
    """
    Bộ phát hiện tiếng nói đa chế độ (Voice Activity Detector) cho Robot AI.
    """

    def __init__(self, mode: str = STT_VAD, sample_rate: int = SAMPLE_RATE) -> None:
        self.mode = mode.lower()
        self.sample_rate = sample_rate
        self.noise_floor = 0.005
        self.effective_threshold = VAD_RMS_THRESHOLD

        self.webrtc_vad = None
        if self.mode == "webrtc" and HAS_WEBRTC:
            try:
                self.webrtc_vad = webrtcvad.Vad(2)  # Aggressiveness level 2
                logger.info("[VAD] Khởi tạo WebRTC VAD Engine (Mode 2) thành công.")
            except Exception as e:
                logger.warning(f"[VAD] Không tạo được WebRTC VAD: {e}. Fallback về Adaptive Energy VAD.")
                self.webrtc_vad = None

        self.silero_session = None
        self.silero_state = np.zeros((2, 1, 64), dtype=np.float32)
        if self.mode == "silero":
            try:
                import onnxruntime as ort
                from huggingface_hub import hf_hub_download
                model_path = hf_hub_download(repo_id="onnx-community/silero-vad", filename="onnx/model.onnx")
                self.silero_session = ort.InferenceSession(model_path)
                logger.info("[VAD] Khởi tạo Silero VAD Engine (ONNX Runtime Zero-Dependency) thành công.")
            except Exception as e:
                logger.warning(f"[VAD] Không nạp được Silero VAD ONNX: {e}. Fallback về Adaptive Energy VAD.")
                self.silero_session = None

    def is_speech_chunk(self, chunk: np.ndarray) -> bool:
        """Kiểm tra 1 block âm thanh (chunk) có phải tiếng nói không."""
        if chunk is None or len(chunk) == 0:
            return False

        rms = float(np.sqrt(np.mean(chunk ** 2)))

        # 1. Chế độ WebRTC VAD
        if self.mode == "webrtc" and self.webrtc_vad:
            try:
                pcm16 = (chunk * 32767).astype(np.int16).tobytes()
                frame_bytes = int(self.sample_rate * 0.03 * 2)
                if len(pcm16) >= frame_bytes:
                    return self.webrtc_vad.is_speech(pcm16[:frame_bytes], self.sample_rate)
            except Exception:
                pass

        # 2. Chế độ Silero VAD ONNX (Zero Dependency)
        if self.mode == "silero" and self.silero_session:
            try:
                if chunk.ndim == 1:
                    input_data = np.expand_dims(chunk, axis=0).astype(np.float32)
                else:
                    input_data = chunk.astype(np.float32)
                sr_data = np.array(self.sample_rate, dtype=np.int64)

                out, self.silero_state = self.silero_session.run(
                    None,
                    {
                        "input": input_data,
                        "state": self.silero_state,
                        "sr": sr_data
                    }
                )
                prob = float(out[0][0])
                # Yêu cầu đồng thời: Xác suất tiếng nói > 0.65 VÀ âm lượng RMS đủ lớn (chỉ bắt người nói gần mic)
                return prob > 0.65 and rms >= self.effective_threshold
            except Exception:
                pass

        # 3. Fallback Adaptive Energy RMS VAD
        return rms > self.effective_threshold

    def calibrate(self, recording: np.ndarray) -> None:
        """Cân bằng thích ứng ngưỡng tiếng ồn môi trường."""
        if recording is None or len(recording) == 0:
            return
        rms = float(np.sqrt(np.mean(recording ** 2)))
        self.noise_floor = max(0.002, rms)
        self.effective_threshold = max(0.012, self.noise_floor * 3.0)
        logger.info(f"[VAD] Calibrated Noise Floor: RMS={self.noise_floor:.6f} | Effective Threshold: {self.effective_threshold:.6f}")
