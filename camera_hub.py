"""
JARVIS Camera Hub - Single Camera Resource Manager
===================================================
Provides a thread-safe singleton wrapper around OpenCV VideoCapture(0)
so both gesture control and voice-vision AI can access camera frames
without device lock conflicts on Windows.
"""

import threading
import time
import logging
import cv2
from typing import Optional

log = logging.getLogger("jarvis.camera")

class CameraHub:
    _instance: Optional["CameraHub"] = None
    _lock = threading.Lock()

    def __init__(self, camera_idx: int = 0):
        self.camera_idx = camera_idx
        self.cap: Optional[cv2.VideoCapture] = None
        self.current_frame = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.frame_lock = threading.Lock()
        self.last_capture_time = 0.0

    @classmethod
    def get_instance(cls, camera_idx: int = 0) -> "CameraHub":
        with cls._lock:
            if cls._instance is None:
                cls._instance = CameraHub(camera_idx)
            return cls._instance

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        try:
            self.cap = cv2.VideoCapture(self.camera_idx, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap.release()
                self.cap = cv2.VideoCapture(self.camera_idx)
            
            if not self.cap.isOpened():
                log.warning(f"Camera {self.camera_idx} could not be opened - vision running in fallback mode.")
                self.cap = None
                return

            log.info(f"Camera Hub active on index {self.camera_idx}.")
            while self.running and self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self.frame_lock:
                        self.current_frame = frame
                        self.last_capture_time = time.time()
                time.sleep(0.03) # ~30 FPS
        except Exception as e:
            log.warning(f"Camera Hub error: {e}")
        finally:
            if self.cap:
                self.cap.release()
                self.cap = None

    def get_frame(self):
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            return None

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

camera_hub = CameraHub.get_instance()
