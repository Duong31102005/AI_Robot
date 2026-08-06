import cv2
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from utils.logger import get_logger

logger = get_logger("YOLOStreamServer")

_latest_annotated_frame: bytes | None = None
_lock = threading.Lock()


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Threaded HTTP Server for serving YOLO AI Video stream."""
    daemon_threads = True


class YOLOStreamHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.end_headers()

    def do_GET(self):
        if self.path.startswith('/video_feed') or self.path == '/':
            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', '*')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            while True:
                try:
                    with _lock:
                        frame_bytes = _latest_annotated_frame
                    if frame_bytes is None:
                        time.sleep(0.04)
                        continue

                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(frame_bytes)))
                    self.end_headers()
                    self.wfile.write(frame_bytes)
                    self.wfile.write(b'\r\n')
                    time.sleep(0.033)  # ~30 FPS
                except (BrokenPipeError, ConnectionResetError):
                    break
                except Exception:
                    break
    def do_POST(self):
        """Xử lý câu hỏi Text từ Web UI và đẩy qua luồng Log Terminal + LLM + TTS Speaker."""
        if self.path in ['/chat', '/api/ai/chat', '/ai/chat']:
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length).decode('utf-8')
                import json
                data = json.loads(post_data) if post_data else {}
                question = data.get("question") or data.get("prompt") or data.get("text") or ""

                from scripts.main_stt import process_text_prompt
                answer = process_text_prompt(question)

                self.send_response(200)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                response_json = json.dumps({"question": question, "answer": answer, "reply": answer}, ensure_ascii=False)
                self.wfile.write(response_json.encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(str(e).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress standard HTTP request logging to keep output clean
        return


_latest_bgr_frame = None
_latest_annotated_frame: bytes | None = None
_lock = threading.Lock()
_encode_thread_started = False

def _async_jpeg_encoder_loop():
    global _latest_annotated_frame
    while True:
        try:
            with _lock:
                frame = _latest_bgr_frame
            if frame is not None:
                ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ret:
                    _latest_annotated_frame = jpeg.tobytes()
            time.sleep(0.025)  # ~40 FPS background encoder
        except Exception:
            time.sleep(0.03)

def update_yolo_frame(frame_bgr) -> None:
    """Cập nhật khung hình nhận diện (0.1ms instant copy, async JPEG encode ngầm)."""
    global _latest_bgr_frame, _encode_thread_started
    if not _encode_thread_started:
        _encode_thread_started = True
        threading.Thread(target=_async_jpeg_encoder_loop, daemon=True).start()

    if frame_bgr is not None:
        with _lock:
            _latest_bgr_frame = frame_bgr


def start_yolo_stream_server(port: int = 5050) -> None:
    """Khởi chạy HTTP Server phát luồng Video YOLO AI trên cổng 5050."""
    def _run():
        try:
            server = ThreadedHTTPServer(('0.0.0.0', port), YOLOStreamHandler)
            logger.info(f"🟢 [YOLO Stream] Server ONLINE at http://localhost:{port}/video_feed")
            server.serve_forever()
        except Exception as e:
            logger.error(f"❌ Failed to start YOLO Stream Server on port {port}: {e}")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
