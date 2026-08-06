"""
Module STT WebSocket Broadcaster (Realtime JSON API for External Apps)
=====================================================================
Phát sóng kết quả STT dạng JSON thời gian thực (Partial / Final) qua WebSocket:
- Endpoint: ws://0.0.0.0:8000/ws/stt
- Hỗ trợ kết nối song song từ Web Frontend, Electron, C#, Python hoặc ROS2.
"""

import json
import asyncio
import threading
import time
from typing import Set, Optional

from utils.logger import get_logger

logger = get_logger("STTWebSocket")

_connected_clients: Set[object] = set()
_loop: Optional[asyncio.AbstractEventLoop] = None


def broadcast_stt_event(event_type: str, text: str, latency_ms: float = 0.0, confidence: float = 0.98, noise_level: float = 0.02) -> None:
    """
    Hàm phát sóng sự kiện STT (Partial / Final) dạng JSON cho toàn bộ WebSocket Clients.
    """
    if not text or not text.strip():
        return

    payload = json.dumps({
        "type": event_type,  # "partial" hoặc "final"
        "text": text.strip(),
        "confidence": round(confidence, 2),
        "noise_level": round(noise_level, 3),
        "speech_detected": True,
        "latency_ms": round(latency_ms, 1),
        "timestamp": round(time.time(), 3)
    }, ensure_ascii=False)

    if _loop and _loop.is_running():
        asyncio.run_coroutine_threadsafe(_async_broadcast(payload), _loop)


async def _async_broadcast(message: str) -> None:
    if not _connected_clients:
        return
    disconnected = set()
    for client in list(_connected_clients):
        try:
            await client.send_text(message)
        except Exception:
            disconnected.add(client)
    for client in disconnected:
        _connected_clients.discard(client)


def register_ws_client(client: object) -> None:
    _connected_clients.add(client)
    logger.info(f"[STTWebSocket] Client connected. Total active clients: {len(_connected_clients)}")


def unregister_ws_client(client: object) -> None:
    _connected_clients.discard(client)
    logger.info(f"[STTWebSocket] Client disconnected. Total active clients: {len(_connected_clients)}")


def set_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    global _loop
    _loop = loop
