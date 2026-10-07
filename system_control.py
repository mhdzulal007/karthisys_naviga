"""
JARVIS system control — Windows automation with an allowlist, real volume,
honest Spotify behavior, and actual CPU/RAM diagnostics.
"""

from __future__ import annotations

import ctypes
import datetime
import os
import platform
import re
import subprocess
import urllib.parse
import webbrowser
from typing import Dict, Optional

APP_COMMANDS: Dict[str, str] = {
    "spotify": "spotify",
    "notepad": "notepad",
    "calculator": "calc",
    "calc": "calc",
    "chrome": "chrome",
    "google chrome": "chrome",
    "edge": "msedge",
    "microsoft edge": "msedge",
    "code": "code",
    "vs code": "code",
    "vscode": "code",
    "terminal": "wt",
    "cmd": "cmd",
    "command prompt": "cmd",
    "powershell": "powershell",
    "explorer": "explorer",
    "file explorer": "explorer",
    "task manager": "taskmgr",
    "paint": "mspaint",
}

EXE_MAP: Dict[str, str] = {
    "notepad": "notepad.exe",
    "spotify": "spotify.exe",
    "chrome": "chrome.exe",
    "google chrome": "chrome.exe",
    "edge": "msedge.exe",
    "microsoft edge": "msedge.exe",
    "calculator": "calc.exe",
    "calc": "Calculator.exe",
    "code": "code.exe",
    "vs code": "code.exe",
    "vscode": "code.exe",
    "paint": "mspaint.exe",
    "task manager": "Taskmgr.exe",
}

ALLOWED_CMD_RE = re.compile(
    r"^(dir|ls|pwd|whoami|hostname|ipconfig|getmac|systeminfo|ver|"
    r"get-date|get-process|get-computerinfo|get-netipaddress|"
    r"get-childitem|get-location|echo|ping)\b",
    re.IGNORECASE,
)

DANGEROUS_PATTERNS = [
    "rmdir /s", "rd /s", "del /f /s", "format", "diskpart",
    "shutdown", "restart-computer", "stop-computer", "reg delete",
    "drop database", "sudo rm", "rm -rf", "invoke-webrequest",
    "invoke-expression", "iex ", "downloadstring", "start-bitstransfer",
    "net user", "net localgroup", "schtasks", "bcdedit",
]

VK_VOLUME_MUTE = 0xAD
VK_VOLUME_DOWN = 0xAE
VK_VOLUME_UP = 0xAF
VK_MEDIA_PLAY_PAUSE = 0xB3
KEYEVENTF_KEYUP = 0x0002


def _press_key(vk_code: int, count: int = 1) -> None:
    user32 = ctypes.windll.user32
    for _ in range(count):
        user32.keybd_event(vk_code, 0, 0, 0)
        user32.keybd_event(vk_code, 0, KEYEVENTF_KEYUP, 0)


def _endpoint_volume():
    try:
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        return interface.QueryInterface(IAudioEndpointVolume)
    except Exception:
        return None


def launch_app(app_name: str) -> str:
    clean_name = app_name.strip().lower()
    cmd = APP_COMMANDS.get(clean_name)
    if cmd is None:
        allowed = ", ".join(sorted(set(APP_COMMANDS)))
        return f"Unknown app '{app_name}'. Allowed: {allowed}."
    try:
        os.startfile(cmd)
        return f"Launched '{app_name}'."
    except Exception as err:
        return f"Failed to launch '{app_name}': {err}"


def close_app(app_name: str) -> str:
    clean_name = app_name.strip().lower()
    exe_name = EXE_MAP.get(clean_name)
    if exe_name is None:
        allowed = ", ".join(sorted(set(EXE_MAP)))
        return f"Unknown app '{app_name}'. Allowed: {allowed}."
    try:
        res = subprocess.run(["taskkill", "/f", "/im", exe_name], capture_output=True, text=True)
        if res.returncode == 0:
            return f"Closed '{app_name}'."
        return f"Could not close '{app_name}' ({(res.stderr or 'not running').strip()})."
    except Exception as err:
        return f"Error closing '{app_name}': {err}"


def play_spotify(query: str) -> str:
    encoded_query = urllib.parse.quote(query.strip())
    spotify_uri = f"spotify:search:{encoded_query}"
    try:
        os.startfile(spotify_uri)
        _press_key(VK_MEDIA_PLAY_PAUSE, 1)
        return (
            f"Opened Spotify search for '{query}' and sent play. "
            "If the wrong track starts, pick it in Spotify."
        )
    except Exception:
        web_url = f"https://open.spotify.com/search/{encoded_query}"
        webbrowser.open(web_url)
        return f"Opened Spotify web search for '{query}'."


def open_url(url: str) -> str:
    clean_url = (url or "").strip()
    lower_url = clean_url.lower()

    if lower_url in ("google", "google.com"):
        clean_url = "https://www.google.com"
    elif lower_url in ("youtube", "youtube.com"):
        clean_url = "https://www.youtube.com"
    elif lower_url.startswith("youtube ") or "youtube" in lower_url:
        search_q = lower_url.replace("youtube", "").replace("play", "").strip()
        if search_q:
            clean_url = f"https://www.youtube.com/results?search_query={urllib.parse.quote_plus(search_q)}"
        else:
            clean_url = "https://www.youtube.com"
    elif not (clean_url.startswith("http://") or clean_url.startswith("https://")):
        if " " in clean_url or "." not in clean_url:
            q = urllib.parse.quote_plus(clean_url)
            clean_url = f"https://www.google.com/search?q={q}"
        else:
            clean_url = "https://" + clean_url

    try:
        webbrowser.open(clean_url)
        return f"Opened {clean_url}."
    except Exception as err:
        return f"Error opening URL: {err}"


def control_volume(action: str, level: Optional[int] = None) -> str:
    act = (action or "").strip().lower()
    if level is not None and not isinstance(level, int):
        try:
            level = int(float(str(level).replace("%", "").strip()))
        except (TypeError, ValueError):
            level = None
    vol = _endpoint_volume()

    if act in ("set", "level", "percent"):
        if level is None:
            return "Need a volume level from 0 to 100."
        pct = max(0, min(100, int(float(level))))
        if vol is not None:
            vol.SetMasterVolumeLevelScalar(pct / 100.0, None)
            return f"Volume set to {pct}%."
        return "Core Audio unavailable — say 'volume up' or 'volume down' instead."

    if act in ("up", "louder", "increase"):
        if vol is not None:
            cur = vol.GetMasterVolumeLevelScalar()
            nxt = min(1.0, cur + 0.1)
            vol.SetMasterVolumeLevelScalar(nxt, None)
            return f"Volume {int(nxt * 100)}%."
        _press_key(VK_VOLUME_UP, 4)
        return "Increased volume."
    if act in ("down", "quieter", "decrease"):
        if vol is not None:
            cur = vol.GetMasterVolumeLevelScalar()
            nxt = max(0.0, cur - 0.1)
            vol.SetMasterVolumeLevelScalar(nxt, None)
            return f"Volume {int(nxt * 100)}%."
        _press_key(VK_VOLUME_DOWN, 4)
        return "Decreased volume."
    if act == "mute":
        if vol is not None:
            vol.SetMute(1, None)
            return "Muted."
        _press_key(VK_VOLUME_MUTE, 1)
        return "Toggled mute (hardware key fallback)."
    if act == "unmute":
        if vol is not None:
            vol.SetMute(0, None)
            return "Unmuted."
        _press_key(VK_VOLUME_MUTE, 1)
        return "Toggled mute (hardware key fallback)."
    if act in ("status", "get", "level"):
        if vol is not None:
            pct = int(vol.GetMasterVolumeLevelScalar() * 100)
            muted = bool(vol.GetMute())
            return f"Volume {pct}%, muted={muted}."
        return "Volume status unavailable without Core Audio."
    return "Unknown volume action. Use up, down, mute, unmute, or set with level 0-100."


def run_cmd(command: str) -> str:
    clean_cmd = command.strip()
    cmd_lower = clean_cmd.lower()
    if any(pat in cmd_lower for pat in DANGEROUS_PATTERNS):
        return "Rejected: command matches a blocked dangerous pattern."
    if not ALLOWED_CMD_RE.match(clean_cmd):
        return (
            "Rejected: command is not on the diagnostic allowlist "
            "(dir, ipconfig, ping, Get-Process, Get-Date, systeminfo, whoami, …)."
        )
    try:
        res = subprocess.run(
            ["powershell", "-NoProfile", "-Command", clean_cmd],
            capture_output=True,
            text=True,
            timeout=8,
        )
        out = (res.stdout or res.stderr or "No output.").strip()
        if len(out) > 1500:
            out = out[:1500] + "\n...[truncated]"
        return out
    except subprocess.TimeoutExpired:
        return f"Command timed out: {command}"
    except Exception as err:
        return f"Error executing command: {err}"


def get_system_info() -> str:
    now = datetime.datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
    lines = [
        f"Time: {now}",
        f"Host: {platform.node()}",
        f"OS: {platform.system()} {platform.release()} ({platform.machine()})",
    ]
    try:
        import psutil

        lines.append(f"CPU: {psutil.cpu_percent(interval=0.15):.0f}%")
        mem = psutil.virtual_memory()
        lines.append(f"RAM: {mem.percent:.0f}% used ({mem.used // (1024**3)}/{mem.total // (1024**3)} GB)")
        batt = psutil.sensors_battery()
        if batt is not None:
            plug = "plugged in" if batt.power_plugged else "on battery"
            lines.append(f"Battery: {batt.percent:.0f}% ({plug})")
    except Exception:
        lines.append("psutil not installed — CPU/RAM/battery unavailable.")
    return "\n".join(lines)


def _grab_screen_win32():
    try:
        import ctypes
        import ctypes.wintypes
        from PIL import Image

        user32 = ctypes.windll.user32
        gdi32 = ctypes.windll.gdi32

        w = user32.GetSystemMetrics(0)
        h = user32.GetSystemMetrics(1)

        hdc_screen = user32.GetDC(0)
        hdc_mem = gdi32.CreateCompatibleDC(hdc_screen)
        hbmp = gdi32.CreateCompatibleBitmap(hdc_screen, w, h)
        old_bmp = gdi32.SelectObject(hdc_mem, hbmp)

        gdi32.BitBlt(hdc_mem, 0, 0, w, h, hdc_screen, 0, 0, 0x00CC0020)

        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [
                ("biSize", ctypes.wintypes.DWORD),
                ("biWidth", ctypes.c_long),
                ("biHeight", ctypes.c_long),
                ("biPlanes", ctypes.wintypes.WORD),
                ("biBitCount", ctypes.wintypes.WORD),
                ("biCompression", ctypes.wintypes.DWORD),
                ("biSizeImage", ctypes.wintypes.DWORD),
                ("biXPelsPerMeter", ctypes.c_long),
                ("biYPelsPerMeter", ctypes.c_long),
                ("biClrUsed", ctypes.wintypes.DWORD),
                ("biClrImportant", ctypes.wintypes.DWORD),
            ]

        bmi = BITMAPINFOHEADER()
        bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.biWidth = w
        bmi.biHeight = -h
        bmi.biPlanes = 1
        bmi.biBitCount = 32
        bmi.biCompression = 0

        buffer = ctypes.create_string_buffer(w * h * 4)
        gdi32.GetDIBits(hdc_mem, hbmp, 0, h, buffer, ctypes.byref(bmi), 0)

        gdi32.SelectObject(hdc_mem, old_bmp)
        gdi32.DeleteObject(hbmp)
        gdi32.DeleteDC(hdc_mem)
        user32.ReleaseDC(0, hdc_screen)

        img = Image.frombytes("RGBA", (w, h), buffer.raw, "raw", "BGRA")
        return img.convert("RGB")
    except Exception:
        import pyautogui
        return pyautogui.screenshot()


def get_desktop_screenshot_jpeg(width: int = 1024, quality: int = 70) -> bytes:
    import io
    from PIL import Image

    img = _grab_screen_win32()
    w, h = img.size
    target_h = int(width * (h / float(w)))
    small = img.resize((width, target_h), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    small.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def take_desktop_screenshot() -> str:
    try:
        screenshot_dir = os.path.abspath("./scratch")
        os.makedirs(screenshot_dir, exist_ok=True)
        filepath = os.path.join(screenshot_dir, "desktop_screenshot.png")

        img = _grab_screen_win32()
        img.save(filepath)
        return f"Desktop screen screenshot captured ({img.width}x{img.height}px) and transmitted to vision stream."
    except Exception as err:
        return f"Could not capture screenshot: {err}"


def remember_fact(fact: str) -> str:
    from workspace_manager import workspace

    existing = ""
    try:
        existing = workspace.read_file("memory.md")
        if existing.startswith("Error:"):
            existing = ""
    except Exception:
        existing = ""
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    blob = (existing + f"\n- [{stamp}] {fact.strip()}\n").strip() + "\n"
    return workspace.write_file("memory.md", blob)
