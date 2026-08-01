"""
Module MoonshineSTT (100% Backward Compatible Wrapper -> PhoWhisper Large Backend)
===================================================================================
Wrapper tương thích ngược 100% cho tất cả các module Robot Main đang import MoonshineSTT.
Tự động chuyển tiếp toàn bộ cuộc gọi sang PhoWhisper Large Backend (Accuracy First).
"""

from audio.phowhisper_stt import PhoWhisperSTT

# Alias wrapper tương thích ngược 100%
MoonshineSTT = PhoWhisperSTT

__all__ = ["MoonshineSTT"]
