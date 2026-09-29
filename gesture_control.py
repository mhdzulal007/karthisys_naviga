"""
JARVIS Ultra-Smooth Gesture Control System (Iron Man Parity)
============================================================
Features:
- One-Euro Signal Filter (1€ Filter): Zero tremor jitter when still + 0ms lag when moving.
- Real-time Hardware Video Timestamps: Eliminates MediaPipe video drift stutter.
- Scale-Invariant Normalized Pinch Detection: Distance-invariant clicks at any distance.
- Full Stark Gesture Suite:
  • Move Pointer  : Index Knuckle (MCP 5) tracked with 1€ dynamic filter
  • Left Click    : Thumb + Index Tip Pinch
  • Right Click   : Thumb + Middle Tip Pinch
  • Drag / Select : Closed Fist Hold (Release to drop)
  • Scroll        : Two Fingers Extended (Index + Middle) moved Up / Down
"""

import math
import os
import time
import cv2
import numpy as np
import pyautogui
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")

# Performance & PyAutoGUI Configuration
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0  # Eliminates built-in PyAutoGUI execution delay
screen_w, screen_h = pyautogui.size()


# --------------------------------------------------------------------------
# One-Euro Filter (1€ Filter) for Zero Jitter + Zero Motion Lag
# --------------------------------------------------------------------------

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


filter_x = OneEuroFilter(min_cutoff=0.8, beta=0.015)
filter_y = OneEuroFilter(min_cutoff=0.8, beta=0.015)


# --------------------------------------------------------------------------
# Gesture Recognition Helpers
# --------------------------------------------------------------------------

def is_fist_closed(landmarks) -> bool:
    """Checks if index, middle, ring, and pinky tips are folded below PIP joints."""
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    folded = sum(1 for tip, pip in zip(tips, pips) if landmarks[tip].y > landmarks[pip].y)
    return folded >= 3


def is_two_finger_scroll(landmarks) -> bool:
    """Checks if Index and Middle fingers are extended up while Ring & Pinky are folded."""
    index_open = landmarks[8].y < landmarks[6].y
    middle_open = landmarks[12].y < landmarks[10].y
    ring_folded = landmarks[16].y > landmarks[14].y
    pinky_folded = landmarks[20].y > landmarks[18].y
    return index_open and middle_open and ring_folded and pinky_folded


def get_hand_scale(landmarks, frame_w, frame_h) -> float:
    """Calculates palm size reference (Wrist 0 to Middle MCP 9) for scale-invariant distance ratio."""
    w_x, w_y = landmarks[0].x * frame_w, landmarks[0].y * frame_h
    m_x, m_y = landmarks[9].x * frame_w, landmarks[9].y * frame_h
    return max(math.hypot(w_x - m_x, w_y - m_y), 1.0)


# --------------------------------------------------------------------------
# Main Camera & Tracking Loop
# --------------------------------------------------------------------------

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7,
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# State tracking variables
is_dragging = False
left_pinch_cooldown = 0.0
right_pinch_cooldown = 0.0
prev_scroll_y = None

print("\n" + "=" * 60)
print("  JARVIS ULTRA-SMOOTH GESTURE SYSTEM ONLINE (1€ Filter Active)")
print("=" * 60)
print("• Move Pointer : Move hand (tracked via Index MCP Knuckle)")
print("• Left Click   : Quick Pinch (Thumb + Index tip)")
print("• Right Click  : Quick Pinch (Thumb + Middle tip)")
print("• Drag/Select  : Close Fist to hold, Open Hand to release")
print("• Scroll       : Extend 2 Fingers (Index + Middle) up/down")
print("• Exit         : Press 'q' in the camera window\n")

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Precise real-time hardware timestamp
        current_time = time.time()
        timestamp_ms = int(current_time * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        status_text = "Tracking (1€ Smooth)"
        status_color = (255, 200, 0)  # Stark Holographic Cyan/Gold

        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            palm_scale = get_hand_scale(lm, w, h)

            # 1. Track Index MCP Knuckle (Landmark 5) to prevent cursor dip during clicks
            track_point = lm[5]

            # 15% edge margin padding for easy screen corner reaching
            pad = 0.15
            norm_x = max(0.0, min(1.0, (track_point.x - pad) / (1.0 - 2 * pad)))
            norm_y = max(0.0, min(1.0, (track_point.y - pad) / (1.0 - 2 * pad)))

            raw_target_x = norm_x * screen_w
            raw_target_y = norm_y * screen_h

            # One-Euro Dynamic Signal Filtering
            smooth_x = filter_x.filter(raw_target_x, current_time)
            smooth_y = filter_y.filter(raw_target_y, current_time)

            # 2. Fist Check (Drag / Select Mode)
            fist_detected = is_fist_closed(lm)
            if fist_detected:
                if not is_dragging:
                    pyautogui.mouseDown()
                    is_dragging = True
                status_text = "DRAGGING / SELECTING (FIST)"
                status_color = (0, 0, 255)
                pyautogui.moveTo(smooth_x, smooth_y)
            else:
                if is_dragging:
                    pyautogui.mouseUp()
                    is_dragging = False

            # 3. Two-Finger Smooth Scrolling Gesture
            if not is_dragging and is_two_finger_scroll(lm):
                curr_scroll_y = (lm[8].y + lm[12].y) / 2.0 * h
                if prev_scroll_y is not None:
                    scroll_delta = prev_scroll_y - curr_scroll_y
                    if abs(scroll_delta) > 4:
                        scroll_amount = int(scroll_delta * 3.5)
                        pyautogui.scroll(scroll_amount)
                        status_text = f"SCROLLING ({'UP' if scroll_amount > 0 else 'DOWN'})"
                        status_color = (0, 255, 255)
                prev_scroll_y = curr_scroll_y
            else:
                prev_scroll_y = None
                if not is_dragging:
                    pyautogui.moveTo(smooth_x, smooth_y)

            # 4. Scale-Invariant Left & Right Pinch Click Detection
            if not is_dragging and prev_scroll_y is None:
                # Left Click Pinch (Thumb 4 + Index Tip 8)
                ix, iy = lm[8].x * w, lm[8].y * h
                tx, ty = lm[4].x * w, lm[4].y * h
                left_pinch_ratio = math.hypot(ix - tx, iy - ty) / palm_scale

                # Right Click Pinch (Thumb 4 + Middle Tip 12)
                mx, my = lm[12].x * w, lm[12].y * h
                right_pinch_ratio = math.hypot(mx - tx, my - ty) / palm_scale

                # Draw pinch lines
                cv2.line(frame, (int(ix), int(iy)), (int(tx), int(ty)), (255, 180, 0), 2)
                cv2.line(frame, (int(mx), int(my)), (int(tx), int(ty)), (180, 255, 0), 1)

                # Scale-invariant pinch ratio threshold (0.28)
                if left_pinch_ratio < 0.28 and (current_time - left_pinch_cooldown) > 0.35:
                    pyautogui.click()
                    left_pinch_cooldown = current_time
                    status_text = "LEFT CLICK"
                    status_color = (0, 255, 0)
                    cv2.circle(frame, (int(ix), int(iy)), 12, (0, 255, 0), cv2.FILLED)

                elif right_pinch_ratio < 0.28 and (current_time - right_pinch_cooldown) > 0.35:
                    pyautogui.rightClick()
                    right_pinch_cooldown = current_time
                    status_text = "RIGHT CLICK"
                    status_color = (255, 0, 255)
                    cv2.circle(frame, (int(mx), int(my)), 12, (255, 0, 255), cv2.FILLED)

            # Draw visual knuckle anchor indicator
            kx, ky = int(track_point.x * w), int(track_point.y * h)
            cv2.circle(frame, (kx, ky), 7, status_color, cv2.FILLED)

        # High-tech HUD Overlay
        cv2.putText(frame, f"JARVIS GESTURE HUD: {status_text}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

        cv2.imshow("JARVIS Ultra-Smooth Gesture System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if is_dragging:
    pyautogui.mouseUp()

cap.release()
cv2.destroyAllWindows()