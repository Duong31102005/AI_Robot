"""
Module Streaming STT (YouTube-like Realtime Subtitles)
======================================================
Cung cấp khả năng nhận dạng giọng nói liên tục theo thời gian thực (Streaming):
- Partial Text: Cập nhật chữ nhảy trực tiếp mỗi 250ms khi người dùng đang nói.
- Final Text: Chốt câu hoàn chỉnh khi VAD phát hiện im lặng.
- Độ trễ cực thấp (< 300ms) trên GPU CUDA / CTranslate2.
"""

import time
import queue
import collections
import numpy as np
from typing import Callable, Optional, Tuple

from audio.phowhisper_stt import PhoWhisperSTT
from config.settings import (
    PHOWHISPER_MODEL_NAME, VAD_PRE_ROLL, VAD_POST_ROLL, VAD_SILENCE_DURATION,
    VAD_MIN_SPEECH_DURATION, VAD_MAX_SPEECH_DURATION,
    PARTIAL_INTERVAL_MS
)
from utils.logger import get_logger

logger = get_logger("WhisperStreamingSTT")


class WhisperStreamingSTT(PhoWhisperSTT):
    """
    Streaming STT Engine nâng cấp hỗ trợ hiển thị phụ đề trực tiếp kiểu YouTube.
    """

    def __init__(self, model_size: str = PHOWHISPER_MODEL_NAME) -> None:
        if not model_size:
            model_size = PHOWHISPER_MODEL_NAME
        super().__init__(model_size=model_size)
        self.partial_interval_s = PARTIAL_INTERVAL_MS / 1000.0

    def listen_and_stream(
        self,
        on_partial: Optional[Callable[[str], None]] = None,
        on_final: Optional[Callable[[str, float, float], None]] = None
    ) -> Tuple[str, float, float]:
        """
        Lắng nghe luồng audio liên tục:
        - Gọi on_partial(text) mỗi 250ms khi đang nói.
        - Trả về (final_text, vad_ms, stt_ms) và gọi on_final khi kết thúc câu.
        """
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
        last_partial_time = 0.0

        start_total = time.perf_counter()

        while self.is_running:
            try:
                chunk = self.audio_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            is_speech = self.vad_engine.is_speech_chunk(chunk)
            now = time.perf_counter()

            if state == "LISTENING":
                pre_roll_buffer.append(chunk)
                if is_speech:
                    state = "SPEAKING"
                    speech_chunks = list(pre_roll_buffer)
                    speech_chunks.append(chunk)
                    speech_start_time = now
                    last_speech_time = now
                    last_partial_time = now

            elif state == "SPEAKING":
                speech_chunks.append(chunk)
                if is_speech:
                    last_speech_time = now
                    post_roll_buffer.clear()
                else:
                    post_roll_buffer.append(chunk)

                silence_dur = now - last_speech_time
                speech_dur = now - speech_start_time

                # Cập nhật Partial Subtitle mỗi 250ms (chữ bay thời gian thực như YouTube)
                if on_partial and (now - last_partial_time >= self.partial_interval_s):
                    last_partial_time = now
                    if len(speech_chunks) > 10:
                        partial_data = np.concatenate(speech_chunks, axis=0)
                        partial_text = self.transcribe_audio_buffer(partial_data)
                        if partial_text and partial_text.strip():
                            on_partial(partial_text.strip())

                if silence_dur >= VAD_SILENCE_DURATION or speech_dur >= VAD_MAX_SPEECH_DURATION:
                    speech_chunks.extend(post_roll_buffer[:post_roll_count])
                    self.last_vad_time_ms = (now - speech_start_time) * 1000.0
                    break

        if not speech_chunks:
            return "", 0.0, 0.0

        audio_data = np.concatenate(speech_chunks, axis=0)
        if len(audio_data) / self.sample_rate < VAD_MIN_SPEECH_DURATION:
            return "", 0.0, 0.0

        final_text = self.transcribe_audio_buffer(audio_data)
        self.last_total_time_ms = (time.perf_counter() - start_total) * 1000.0

        if final_text and final_text.strip():
            logger.info(
                f"[WhisperStreamingSTT] 🎯 [FINAL] VAD={self.last_vad_time_ms:.0f}ms | STT={self.last_stt_time_ms:.0f}ms | Text: '{final_text}'"
            )
            if on_final:
                on_final(final_text, self.last_vad_time_ms, self.last_stt_time_ms)

        return final_text, self.last_vad_time_ms, self.last_stt_time_ms
