"""
JARVIS One-Click Master Launcher
================================
Launches both JARVIS Multimodal Voice Assistant (jarvis_live.py)
and Gesture Control System (gesture_control.py) simultaneously.

Usage:
  python start_jarvis.py
"""

import sys
import os
import subprocess
import time

def main():
    print("=" * 65)
    print("      JARVIS STARK SYSTEM LAUNCHER")
    print("=" * 65)
    print("• Voice Assistant  : Active (jarvis_live.py)")
    print("• Gesture Controller: Active (gesture_control.py)")
    print("• Press Ctrl+C in this terminal to shut down both systems.")
    print("=" * 65 + "\n")

    python_exe = sys.executable

    # Launch Gesture Controller
    print("[1/2] Launching Gesture Controller...")
    gesture_proc = subprocess.Popen([python_exe, "gesture_control.py"])

    time.sleep(1)

    # Launch Voice & Vision Assistant
    print("[2/2] Launching Voice & Vision Assistant...")
    voice_proc = subprocess.Popen([python_exe, "jarvis_live.py"])

    try:
        voice_proc.wait()
        gesture_proc.wait()
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Terminating JARVIS systems... Goodbye, sir.")
        voice_proc.terminate()
        gesture_proc.terminate()

if __name__ == "__main__":
    main()
