"""
Module Audio DSP Pipeline (Digital Signal Processing - High Quality Production Grade)
====================================================================================
Bao gồm các bộ lọc xử lý tín hiệu âm thanh đầu vào chất lượng cao trước khi đưa sang STT:
- DC Offset Removal (Loại bỏ độ lệch dòng một chiều)
- Highpass Filter (80Hz Butterworth - triệt tiêu tiếng ù quạt/máy tính/tiếng gió)
- Lowpass Filter (7500Hz - triệt tiêu nhiễu rít tần số cao)
- Notch Filter 50/60Hz (Triệt tiêu tiếng ù điện lưới)
- Adaptive Spectral Noise Reduction (Khử nhiễu nền thích ứng)
- Soft AGC (Automatic Gain Control - chỉnh âm lượng tự nhiên êm dịu)
- Anti-clipping Limiter (chống xé/vỡ tiếng)
- Peak Normalization (chuẩn hóa âm lượng an toàn)
"""

import numpy as np
from typing import Tuple

try:
    from scipy.signal import butter, filtfilt, iirnotch
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def remove_dc_offset(audio_data: np.ndarray) -> np.ndarray:
    """Loại bỏ độ lệch dòng một chiều (DC Offset)."""
    if audio_data is None or len(audio_data) == 0:
        return audio_data
    return audio_data - np.mean(audio_data)


def highpass_filter(audio_data: np.ndarray, cutoff: float = 80.0, sample_rate: int = 16000) -> np.ndarray:
    """Bộ lọc thông cao High-pass Butterworth 80Hz triệt tiêu âm tần số thấp (tiếng ù quạt, rung bàn)."""
    if not HAS_SCIPY or len(audio_data) < 32:
        return audio_data
    try:
        nyquist = sample_rate / 2.0
        norm_cutoff = cutoff / nyquist
        b, a = butter(4, norm_cutoff, btype='high')
        return filtfilt(b, a, audio_data)
    except Exception:
        return audio_data


def lowpass_filter(audio_data: np.ndarray, cutoff: float = 7500.0, sample_rate: int = 16000) -> np.ndarray:
    """Bộ lọc thông thấp Low-pass Butterworth 7.5kHz triệt tiêu nhiễu rít tần số cao."""
    if not HAS_SCIPY or len(audio_data) < 32:
        return audio_data
    try:
        nyquist = sample_rate / 2.0
        norm_cutoff = min(0.99, cutoff / nyquist)
        b, a = butter(4, norm_cutoff, btype='low')
        return filtfilt(b, a, audio_data)
    except Exception:
        return audio_data


def hum_removal_filter(audio_data: np.ndarray, freqs: Tuple[float, ...] = (50.0, 60.0), sample_rate: int = 16000) -> np.ndarray:
    """Bộ lọc Notch triệt tiêu tiếng ù điện lưới 50Hz và 60Hz."""
    if not HAS_SCIPY or len(audio_data) < 32:
        return audio_data
    processed = audio_data
    nyquist = sample_rate / 2.0
    for f in freqs:
        try:
            b, a = iirnotch(f / nyquist, 30.0)
            processed = filtfilt(b, a, processed)
        except Exception:
            pass
    return processed


def adaptive_noise_reduction(audio_data: np.ndarray) -> np.ndarray:
    """Khử nhiễu nền thích ứng dựa theo phổ biên độ (Spectral Subtraction Noise Reduction)."""
    if audio_data is None or len(audio_data) < 256:
        return audio_data
    try:
        fft = np.fft.rfft(audio_data)
        magnitude = np.abs(fft)
        phase = np.angle(fft)

        # Lấy 10% phổ năng lượng nhỏ nhất làm tiếng ồn nền
        noise_floor = np.percentile(magnitude, 10)
        clean_magnitude = np.maximum(magnitude - noise_floor * 0.8, 0.0)

        clean_fft = clean_magnitude * np.exp(1j * phase)
        return np.fft.irfft(clean_fft, n=len(audio_data)).astype(np.float32)
    except Exception:
        return audio_data


def soft_agc(audio_data: np.ndarray, target_rms: float = 0.12) -> np.ndarray:
    """
    Automatic Gain Control (AGC) êm dịu điều chỉnh âm lượng giọng nói nhỏ về mức chuẩn
    mà không gây khuếch đại nhiễu nền hoặc xé tiếng.
    """
    if audio_data is None or len(audio_data) == 0:
        return audio_data

    rms = float(np.sqrt(np.mean(audio_data ** 2)))
    if rms < 0.005:  # Nhiễu im lặng, không tác động
        return audio_data

    # Giới hạn hệ số khuếch đại 0.5x - 3.0x
    gain = min(3.0, max(0.5, target_rms / rms))
    return audio_data * gain


def anti_clip(audio_data: np.ndarray, threshold: float = 0.95) -> np.ndarray:
    """Bộ nén giới hạn biên độ chống vỡ tiếng/xé tiếng (Anti-clipping Limiter)."""
    if audio_data is None or len(audio_data) == 0:
        return audio_data
    return np.clip(audio_data, -threshold, threshold)


def normalize(audio_data: np.ndarray, target_peak: float = 0.85) -> np.ndarray:
    """Chuẩn hóa âm lượng dựa theo đỉnh sóng (Peak Normalization)."""
    if audio_data is None or len(audio_data) == 0:
        return audio_data

    max_amp = float(np.max(np.abs(audio_data)))
    if max_amp > 1.0:
        return audio_data / max_amp
    elif 0.01 < max_amp < 0.5:
        return (audio_data / max_amp) * target_peak
    return audio_data


def process_audio_dsp(audio_data: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """
    Chuỗi xử lý DSP chất lượng cao tiền xử lý âm thanh trước khi đưa vào STT:
    DC Offset -> Highpass 80Hz -> Lowpass 7.5kHz -> Notch 50/60Hz -> Noise Reduction -> Soft AGC -> Anti-clip -> Normalize
    """
    if audio_data is None or len(audio_data) == 0:
        return audio_data

    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32768.0

    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    # 1. DC Offset Removal
    processed = remove_dc_offset(audio_data)

    # 2. High-pass Filter (80Hz)
    processed = highpass_filter(processed, cutoff=80.0, sample_rate=sample_rate)

    # 3. Low-pass Filter (7.5kHz)
    processed = lowpass_filter(processed, cutoff=7500.0, sample_rate=sample_rate)

    # 4. Notch Filter (50/60Hz)
    processed = hum_removal_filter(processed, freqs=(50.0, 60.0), sample_rate=sample_rate)

    # 5. Adaptive Noise Reduction
    processed = adaptive_noise_reduction(processed)

    # 6. Soft AGC
    processed = soft_agc(processed, target_rms=0.12)

    # 7. Anti Clipping & Peak Normalization
    processed = anti_clip(processed, threshold=0.95)

    return processed.astype(np.float32)
