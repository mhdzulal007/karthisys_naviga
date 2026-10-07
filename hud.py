"""Always-on-top status strip: DORMANT / LISTENING / SPEAKING / THINKING."""

from __future__ import annotations

import logging
import threading
import tkinter as tk
from typing import Optional

log = logging.getLogger("jarvis.hud")


class StatusHUD:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._mode = "DORMANT"
        self._subtitle = "Say Hey Jarvis"
        self._lock = threading.Lock()
        self._root: Optional[tk.Tk] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if not self.enabled:
            return
        self._thread = threading.Thread(target=self._run, name="jarvis-hud", daemon=True)
        self._thread.start()

    def set(self, mode: str, subtitle: str = "") -> None:
        with self._lock:
            self._mode = mode
            self._subtitle = subtitle

    def _run(self) -> None:
        try:
            root = tk.Tk()
        except Exception as e:
            log.warning("HUD unavailable (%s).", e)
            return
        self._root = root
        root.title("JARVIS")
        root.attributes("-topmost", True)
        root.resizable(False, False)
        try:
            root.attributes("-alpha", 0.92)
        except Exception:
            pass
        root.configure(bg="#0b0f14")
        root.geometry("320x56+24+24")
        root.overrideredirect(True)

        mode_lbl = tk.Label(
            root, text="JARVIS  DORMANT", fg="#7ee0ff", bg="#0b0f14",
            font=("Segoe UI", 11, "bold"),
        )
        sub_lbl = tk.Label(
            root, text="Say Hey Jarvis", fg="#8a93a3", bg="#0b0f14",
            font=("Segoe UI", 9),
        )
        mode_lbl.pack(anchor="w", padx=14, pady=(8, 0))
        sub_lbl.pack(anchor="w", padx=14, pady=(0, 8))

        def tick():
            with self._lock:
                mode = self._mode
                sub = self._subtitle
            colors = {
                "DORMANT": "#6b7380",
                "LISTENING": "#7ee0ff",
                "SPEAKING": "#d4b106",
                "THINKING": "#c084fc",
            }
            mode_lbl.config(text=f"JARVIS  {mode}", fg=colors.get(mode, "#7ee0ff"))
            sub_lbl.config(text=sub or "")
            root.after(80, tick)

        tick()
        try:
            root.mainloop()
        except Exception:
            pass

    def stop(self) -> None:
        if self._root is not None:
            try:
                self._root.after(0, self._root.destroy)
            except Exception:
                pass
