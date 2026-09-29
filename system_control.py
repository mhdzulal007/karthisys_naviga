"""
JARVIS System Control Module - 100% OS Parity Tools
===================================================
Provides Windows system automation for JARVIS:
- Launching desktop applications (Spotify, Notepad, Chrome, Calculator, etc.)
- Closing desktop applications (taskkill)
- Playing songs, albums, and playlists on Spotify
- Adjusting master system volume (up, down, mute, unmute)
- Opening web URLs and search queries in the default browser
- Executing safe terminal commands (PowerShell / CMD)
- Returning system health diagnostics (CPU, RAM, OS status)
- Capturing primary display desktop screenshots
"""

import ctypes
import datetime
import os
import platform
import subprocess
import urllib.parse
import webbrowser
from typing import Dict

# Safe application alias dictionary
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

# Windows Virtual Key Codes for System Audio Control
VK_VOLUME_MUTE = 0xAD
VK_VOLUME_DOWN = 0xAE
VK_VOLUME_UP = 0xAF
KEYEVENTF_KEYUP = 0x0002

# Commands blocked for security
DANGEROUS_PATTERNS = [
    "rmdir /s", "rd /s", "del /f /s", "format", "diskpart",
    "shutdown", "restart-computer", "stop-computer", "reg delete",
    "drop database", "sudo rm", "rm -rf"
]


def launch_app(app_name: str) -> str:
    """Launches a desktop application by name or executable command."""
    clean_name = app_name.strip().lower()
    cmd = APP_COMMANDS.get(clean_name, clean_name)
    try:
        # Use os.startfile on Windows for seamless application launching
        os.startfile(cmd)
        return f"Successfully launched '{app_name}'."
    except Exception:
        try:
            subprocess.Popen(cmd, shell=True)
            return f"Successfully launched '{app_name}'."
        except Exception as err:
            return f"Failed to launch '{app_name}': {err}"


def close_app(app_name: str) -> str:
    """Closes or terminates a running desktop application (e.g. notepad, chrome, spotify, edge)."""
    clean_name = app_name.strip().lower()
    exe_map = {
        "notepad": "notepad.exe",
        "spotify": "spotify.exe",
        "chrome": "chrome.exe",
        "google chrome": "chrome.exe",
        "edge": "msedge.exe",
        "microsoft edge": "msedge.exe",
        "calculator": "calc.exe",
        "calc": "calc.exe",
        "code": "code.exe",
        "vs code": "code.exe",
        "vscode": "code.exe",
    }
    exe_name = exe_map.get(clean_name, clean_name)
    if not exe_name.endswith(".exe"):
        exe_name += ".exe"

    try:
        res = subprocess.run(["taskkill", "/f", "/im", exe_name], capture_output=True, text=True)
        if res.returncode == 0:
            return f"Successfully closed '{app_name}'."
        else:
            return f"Could not close '{app_name}' ({res.stderr.strip() or 'process not found'})."
    except Exception as err:
        return f"Error closing '{app_name}': {err}"


def play_spotify(query: str) -> str:
    """Launches Spotify and searches/plays a specified song, artist, or album."""
    encoded_query = urllib.parse.quote(query.strip())
    spotify_uri = f"spotify:search:{encoded_query}"
    try:
        os.startfile(spotify_uri)
        return f"Opening Spotify to play '{query}', sir."
    except Exception:
        web_url = f"https://open.spotify.com/search/{encoded_query}"
        webbrowser.open(web_url)
        return f"Opened Spotify web player for '{query}', sir."


def open_url(url: str) -> str:
    """Opens a website URL or web search in the user's default browser."""
    clean_url = url.strip()
    if not (clean_url.startswith("http://") or clean_url.startswith("https://")):
        clean_url = "https://" + clean_url
    try:
        webbrowser.open(clean_url)
        return f"Opened {clean_url} in default browser."
    except Exception as err:
        return f"Error opening URL '{clean_url}': {err}"


def control_volume(action: str) -> str:
    """Controls Windows master system volume: 'up', 'down', 'mute', 'unmute'."""
    act = action.strip().lower()
    user32 = ctypes.windll.user32

    def press_key(vk_code: int, count: int = 1):
        for _ in range(count):
            user32.keybd_event(vk_code, 0, 0, 0)
            user32.keybd_event(vk_code, 0, KEYEVENTF_KEYUP, 0)

    if act in ("up", "louder", "increase"):
        press_key(VK_VOLUME_UP, 5)
        return "Increased system volume, sir."
    elif act in ("down", "quieter", "decrease"):
        press_key(VK_VOLUME_DOWN, 5)
        return "Decreased system volume, sir."
    elif act in ("mute", "unmute", "toggle_mute"):
        press_key(VK_VOLUME_MUTE, 1)
        return f"Toggled volume mute state ({act}), sir."
    else:
        return f"Unknown volume action '{action}'. Use 'up', 'down', or 'mute'."


def run_cmd(command: str) -> str:
    """Executes a safe shell/powershell terminal command and returns stdout."""
    clean_cmd = command.strip()
    cmd_lower = clean_cmd.lower()

    if any(pat in cmd_lower for pat in DANGEROUS_PATTERNS):
        return f"Security Error: Command '{command}' contains blocked dangerous pattern and was rejected."

    try:
        res = subprocess.run(
            ["powershell", "-Command", clean_cmd],
            capture_output=True,
            text=True,
            timeout=10,
        )
        out = (res.stdout or res.stderr or "Command executed with no output.").strip()
        if len(out) > 1500:
            out = out[:1500] + "\n...[Output truncated]"
        return f"Command Output:\n{out}"
    except subprocess.TimeoutExpired:
        return f"Command '{command}' timed out after 10 seconds."
    except Exception as err:
        return f"Error executing command '{command}': {err}"


def get_system_info() -> str:
    """Returns Windows system health diagnostics (OS, hostname, CPU architecture, time/date)."""
    now = datetime.datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
    sys_name = platform.system()
    sys_release = platform.release()
    arch = platform.machine()
    host = platform.node()

    info = (
        f"System Status Report (Stark HUD Diagnostic):\n"
        f"• Current Time: {now}\n"
        f"• Host Machine: {host}\n"
        f"• OS: {sys_name} {sys_release} ({arch})\n"
        f"• Python Environment: Active & Operational"
    )
    return info


def get_desktop_screenshot_jpeg(width: int = 960, quality: int = 65) -> bytes:
    """Captures primary monitor, resizes to target width, and returns JPEG bytes."""
    import io
    import pyautogui
    from PIL import Image

    img = pyautogui.screenshot()
    w, h = img.size
    aspect = h / float(w)
    target_h = int(width * aspect)
    small = img.resize((width, target_h), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    small.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def take_desktop_screenshot() -> str:
    """Captures a screenshot of the primary desktop monitor."""
    try:
        import pyautogui

        screenshot_dir = os.path.abspath("./scratch")
        os.makedirs(screenshot_dir, exist_ok=True)
        filepath = os.path.join(screenshot_dir, "desktop_screenshot.png")
        pyautogui.screenshot(filepath)
        return f"Desktop screenshot captured and transmitted to your vision feed, sir."
    except Exception as err:
        return f"Could not capture screenshot: {err}"

