import os
import cv2
import math
import time
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
pyautogui.PAUSE = 0  # Eliminates the built-in 0.1s PyAutoGUI execution delay
screen_w, screen_h = pyautogui.size()

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)

# Smoothness & Sensitivity Configuration
prev_x, prev_y = screen_w // 2, screen_h // 2
BASE_SMOOTHING = 0.35  # Lower = smoother, higher = snappy
DEADZONE_PIXELS = 3    # Ignores microscopic hand tremor

# State tracking
is_dragging = False
pinch_cooldown = 0.0

def is_fist_closed(landmarks):
    """
    Checks if index, middle, ring, and pinky tips are folded 
    below their respective lower knuckle (PIP) joints.
    """
    # Landmark indices: Tips = [8, 12, 16, 20], PIP joints = [6, 10, 14, 18]
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    
    folded_fingers = 0
    for tip_idx, pip_idx in zip(tips, pips):
        if landmarks[tip_idx].y > landmarks[pip_idx].y:
            folded_fingers += 1
            
    return folded_fingers >= 3

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)
timestamp_ms = 0

print("\n--- Gesture Control Active ---")
print("• Move Pointer : Move hand (tracked via index base)")
print("• Click        : Quick Pinch (Thumb + Index tip)")
print("• Drag/Select  : Close Fist to hold, Open Hand to release")
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

        timestamp_ms += 16  # ~60 fps steps
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        status_text = "Tracking"
        status_color = (0, 255, 0)

        if result.hand_landmarks:
            lm = result.hand_landmarks[0]

            # 1. Use Landmark 5 (Index MCP knuckle) or Landmark 9 (Palm center) for movement
            # This completely avoids cursor dip during pinches
            track_point = lm[5]
            
            # Map tracking area to active screen (with 15% edge padding for easy corner reach)
            pad_x = 0.15
            pad_y = 0.15
            norm_x = (track_point.x - pad_x) / (1.0 - 2 * pad_x)
            norm_y = (track_point.y - pad_y) / (1.0 - 2 * pad_y)
            norm_x = max(0.0, min(1.0, norm_x))
            norm_y = max(0.0, min(1.0, norm_y))

            target_x = norm_x * screen_w
            target_y = norm_y * screen_h

            # Adaptive smoothing & deadzone calculation
            delta_x = target_x - prev_x
            delta_y = target_y - prev_y
            dist = math.hypot(delta_x, delta_y)

            if dist > DEADZONE_PIXELS:
                # Dynamic response: faster movements get less smoothing to stay snappy
                dynamic_smoothing = min(0.85, BASE_SMOOTHING + (dist / 1200))
                curr_x = prev_x + delta_x * dynamic_smoothing
                curr_y = prev_y + delta_y * dynamic_smoothing
                pyautogui.moveTo(curr_x, curr_y)
                prev_x, prev_y = curr_x, curr_y

            # 2. Check for Fist (Drag / Select mode)
            fist_detected = is_fist_closed(lm)

            if fist_detected:
                if not is_dragging:
                    pyautogui.mouseDown()
                    is_dragging = True
                status_text = "DRAGGING / SELECTING (FIST)"
                status_color = (0, 0, 255)
            else:
                if is_dragging:
                    pyautogui.mouseUp()
                    is_dragging = False

            # 3. Check for Quick Pinch Click (Only when hand is open / not dragging)
            if not is_dragging:
                # Normalized distance scaled to frame size
                ix, iy = int(lm[8].x * w), int(lm[8].y * h)
                tx, ty = int(lm[4].x * w), int(lm[4].y * h)
                pinch_dist = math.hypot(ix - tx, iy - ty)

                # Visual pinch indicator line
                cv2.line(frame, (ix, iy), (tx, ty), (255, 200, 0), 1)

                current_time = time.time()
                # Threshold ~30px, with 0.3s cooldown to prevent double clicks
                if pinch_dist < 32 and (current_time - pinch_cooldown) > 0.35:
                    pyautogui.click()
                    pinch_cooldown = current_time
                    status_text = "CLICK"
                    status_color = (255, 100, 0)
                    cv2.circle(frame, (ix, iy), 10, (255, 100, 0), cv2.FILLED)

            # Draw visual knuckle anchor
            kx, ky = int(track_point.x * w), int(track_point.y * h)
            cv2.circle(frame, (kx, ky), 6, status_color, cv2.FILLED)

        # On-screen HUD
        cv2.putText(frame, f"State: {status_text}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        cv2.imshow("Jarvis Gesture Controller (Upgraded)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if is_dragging:
    pyautogui.mouseUp()

cap.release()
cv2.destroyAllWindows()