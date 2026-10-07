"""
JARVIS gesture control — pointer, pinch click, fist-drag, two-finger scroll.
Designed to consume frames from CameraHub so it does not open the webcam itself.
"""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Callable, Optional

import cv2
import numpy as np

log = logging.getLogger("jarvis.gesture")

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")


class OneEuroFilter:
    def __init__(self, min_cutoff=0.8, beta=0.015, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x, t):
        if self.t_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            self.t_prev = t
            return x
        dt = max(t - self.t_prev, 0.001)
        self.t_prev = t
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


def is_fist_closed(landmarks) -> bool:
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    folded = sum(1 for tip, pip in zip(tips, pips) if landmarks[tip].y > landmarks[pip].y)
    return folded >= 3


def is_two_finger_scroll(landmarks) -> bool:
    index_open = landmarks[8].y < landmarks[6].y
    middle_open = landmarks[12].y < landmarks[10].y
    ring_folded = landmarks[16].y > landmarks[14].y
    pinky_folded = landmarks[20].y > landmarks[18].y
    return index_open and middle_open and ring_folded and pinky_folded


def get_hand_scale(landmarks, frame_w, frame_h) -> float:
    w_x, w_y = landmarks[0].x * frame_w, landmarks[0].y * frame_h
    m_x, m_y = landmarks[9].x * frame_w, landmarks[9].y * frame_h
    return max(math.hypot(w_x - m_x, w_y - m_y), 1.0)


def run_gesture_loop(
    stop_flag,
    get_frame: Callable[[], Optional[np.ndarray]],
    show_preview: bool = True,
) -> None:
    try:
        import pyautogui
        import mediapipe as mp
    except ImportError as e:
        log.warning("Gesture disabled — missing dependency: %s", e)
        return

    if not os.path.exists(MODEL_PATH):
        url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        )
        log.info("Downloading hand landmarker model...")
        try:
            import urllib.request
            urllib.request.urlretrieve(url, MODEL_PATH)
        except Exception as e:
            log.warning("Gesture disabled — could not download model (%s).", e)
            return

    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0
    screen_w, screen_h = pyautogui.size()

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    filter_x = OneEuroFilter(min_cutoff=0.8, beta=0.015)
    filter_y = OneEuroFilter(min_cutoff=0.8, beta=0.015)
    is_dragging = False
    left_pinch_cooldown = 0.0
    right_pinch_cooldown = 0.0
    prev_scroll_y = None

    log.info("Gesture system online (shared camera).")

    try:
        with HandLandmarker.create_from_options(options) as landmarker:
            while not (stop_flag.is_set() if hasattr(stop_flag, "is_set") else False):
                frame = get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                current_time = time.time()
                timestamp_ms = int(current_time * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                status_text = "Tracking"
                status_color = (255, 200, 0)

                if result.hand_landmarks:
                    lm = result.hand_landmarks[0]
                    palm_scale = get_hand_scale(lm, w, h)
                    track_point = lm[5]
                    pad = 0.15
                    norm_x = max(0.0, min(1.0, (track_point.x - pad) / (1.0 - 2 * pad)))
                    norm_y = max(0.0, min(1.0, (track_point.y - pad) / (1.0 - 2 * pad)))
                    smooth_x = filter_x.filter(norm_x * screen_w, current_time)
                    smooth_y = filter_y.filter(norm_y * screen_h, current_time)

                    fist_detected = is_fist_closed(lm)
                    if fist_detected:
                        if not is_dragging:
                            pyautogui.mouseDown()
                            is_dragging = True
                        status_text = "DRAGGING"
                        status_color = (0, 0, 255)
                        pyautogui.moveTo(smooth_x, smooth_y)
                    else:
                        if is_dragging:
                            pyautogui.mouseUp()
                            is_dragging = False

                    if not is_dragging and is_two_finger_scroll(lm):
                        curr_scroll_y = (lm[8].y + lm[12].y) / 2.0 * h
                        if prev_scroll_y is not None:
                            scroll_delta = prev_scroll_y - curr_scroll_y
                            if abs(scroll_delta) > 4:
                                scroll_amount = int(scroll_delta * 3.5)
                                pyautogui.scroll(scroll_amount)
                                status_text = f"SCROLL {'UP' if scroll_amount > 0 else 'DOWN'}"
                                status_color = (0, 255, 255)
                        prev_scroll_y = curr_scroll_y
                    else:
                        prev_scroll_y = None
                        if not is_dragging:
                            pyautogui.moveTo(smooth_x, smooth_y)

                    if not is_dragging and prev_scroll_y is None:
                        ix, iy = lm[8].x * w, lm[8].y * h
                        tx, ty = lm[4].x * w, lm[4].y * h
                        mx, my = lm[12].x * w, lm[12].y * h
                        left_pinch_ratio = math.hypot(ix - tx, iy - ty) / palm_scale
                        right_pinch_ratio = math.hypot(mx - tx, my - ty) / palm_scale
                        if show_preview:
                            cv2.line(frame, (int(ix), int(iy)), (int(tx), int(ty)), (255, 180, 0), 2)
                            cv2.line(frame, (int(mx), int(my)), (int(tx), int(ty)), (180, 255, 0), 1)
                        if left_pinch_ratio < 0.28 and (current_time - left_pinch_cooldown) > 0.35:
                            pyautogui.click()
                            left_pinch_cooldown = current_time
                            status_text = "LEFT CLICK"
                            status_color = (0, 255, 0)
                        elif right_pinch_ratio < 0.28 and (current_time - right_pinch_cooldown) > 0.35:
                            pyautogui.rightClick()
                            right_pinch_cooldown = current_time
                            status_text = "RIGHT CLICK"
                            status_color = (255, 0, 255)
                    kx, ky = int(track_point.x * w), int(track_point.y * h)
                    cv2.circle(frame, (kx, ky), 7, status_color, cv2.FILLED)

                if show_preview:
                    cv2.putText(
                        frame, f"JARVIS GESTURE: {status_text}", (16, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2,
                    )
                    cv2.imshow("JARVIS Gesture", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                else:
                    time.sleep(0.001)
    except Exception as e:
        log.warning("Gesture loop ended: %s", e)
    finally:
        if is_dragging:
            try:
                pyautogui.mouseUp()
            except Exception:
                pass
        if show_preview:
            try:
                cv2.destroyWindow("JARVIS Gesture")
            except Exception:
                pass


def main():
    from camera_hub import CameraHub

    hub = CameraHub()
    if not hub.start():
        raise SystemExit("No camera.")
    stop = __import__("threading").Event()
    try:
        run_gesture_loop(stop, hub.get_frame, show_preview=True)
    finally:
        stop.set()
        hub.stop()


if __name__ == "__main__":
    main()
