"""
JARVIS - Production Voice Assistant on Gemini Live
====================================================

Changes vs. the original prototype (see inline comments marked [FIX]):

  1. Removed the manual mic-zeroing "echo gate" during playback. The Live
     model (gemini-3.1-flash-live-preview) natively supports barge-in, so
     zeroing your own mic input while JARVIS talks was actively preventing
     you from interrupting him. Mic audio now streams continuously while
     awake, same as talking to Gemini/Alexa/Siri.
  2. Camera frames are now gated on the awake/asleep state, same as audio.
     Previously vision was streamed to the API 24/7 even while dormant.
  3. Playback buffer is now a proper thread-safe object (lock-guarded),
     instead of a bare bytearray mutated from two different threads.
  4. Interrupting a response now clears the *already-buffered* playback
     audio too, not just the pending async queue - so JARVIS actually
     stops talking immediately on interruption instead of finishing
     whatever was already in the pipe.
  5. The whole connect+run cycle is wrapped in an auto-reconnect loop
     with exponential backoff, so a dropped WebSocket / preview-model
     hiccup doesn't kill the whole process.
  6. Fatal, unrecoverable problems (bad API key, no microphone/speaker
     hardware) are distinguished from recoverable ones (network blips)
     so we don't retry-loop forever against a problem retrying can't fix.
  7. A failure in the camera task no longer takes down mic/audio/receive -
     vision degrades gracefully if no webcam is present.
  8. Blocking file I/O (tool calls) now runs in a thread executor instead
     of on the event loop.
  9. Replaced `print` debugging with the `logging` module (console +
     rotating file handler).
  10. All the tunables that used to be hardcoded module constants are now
      a `Config` dataclass, overridable via environment variables.

Requires: google-genai, opencv-python, numpy, sounddevice, pillow,
          openwakeword (optional but recommended)
"""

from __future__ import annotations

import asyncio
import io
import logging
import logging.handlers
import os
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import sounddevice as sd
from PIL import Image

from google import genai
from google.genai import types

from workspace_manager import workspace
import system_control


try:
    import openwakeword
    from openwakeword.model import Model as WakeWordModel
    _OWW_AVAILABLE = True
except ImportError:
    _OWW_AVAILABLE = False


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------

class FatalError(Exception):
    """Raised for problems that reconnecting will never fix: bad
    credentials, missing audio hardware, etc. Stops the program instead
    of retry-looping forever."""


def _looks_like_auth_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(tok in text for tok in (
        "api key not valid", "api_key", "permission_denied",
        "unauthenticated", "401", "403", "invalid api key",
    ))


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

DEFAULT_SLEEP_PHRASES = [
    "go to sleep", "goodbye jarvis", "that will be all", "that'll be all",
    "power down", "sleep now", "stand down", "that's all for now",
]

DEFAULT_SYSTEM_PROMPT = """
You are JARVIS, an elite personal AI assistant to Tony Stark.
Persona & Capability Guidelines:
- Unfailingly calm, refined British cadence, polite, dry wit. Address the user as 'Sir'.
- ULTRA-FAST & FLUID RESPONSES: Keep answers extremely concise, crisp, and direct (1 short sentence when possible, 2 sentences maximum).
- Begin speaking immediately without introductory filler, fluff, or pleasantries.
- FULL WINDOWS OS AUTHORITY: You possess real-time tools to control the host computer (`launch_app`, `close_app`, `play_spotify`, `control_volume`, `run_cmd`, `open_url`, `take_desktop_screenshot`, `get_system_info`).
- CRITICAL DIRECTIVE: NEVER say you are "just a language model" or that you cannot control the computer. You have active function tools for full PC automation. Whenever asked to open or close apps, play music, change volume, open websites, or execute commands, ALWAYS call the corresponding tool function immediately!
- If told to go to sleep or stand down, give a brief 3-word acknowledgement (e.g. "As you wish, sir.") before falling silent.
"""





@dataclass
class Config:
    api_key: str

    model_name: str = "gemini-3.1-flash-live-preview"
    voice_name: str = "Charon"

    channels: int = 1
    sample_rate_in: int = 16000
    sample_rate_out: int = 24000
    block_size: int = 512
    playback_prebuffer_blocks: int = 2

    wake_score_threshold: float = 0.5
    wake_echo_guard_s: float = 0.2

    start_of_speech_sensitivity: str = "HIGH"
    end_of_speech_sensitivity: str = "HIGH"
    prefix_padding_ms: int = 20
    silence_duration_ms: int = 200

    inactivity_sleep_s: int = 90
    sleep_phrases: List[str] = field(default_factory=lambda: list(DEFAULT_SLEEP_PHRASES))

    camera_enabled: bool = True
    camera_interval_s: float = 3.5
    camera_width: int = 320
    camera_height: int = 180
    camera_jpeg_quality: int = 30


    reconnect_initial_delay_s: float = 2.0
    reconnect_max_delay_s: float = 30.0

    tool_timeout_s: float = 10.0

    log_level: str = "INFO"
    log_file: Optional[str] = "jarvis.log"

    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    @classmethod
    def from_env(cls) -> "Config":
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise FatalError(
                "GEMINI_API_KEY environment variable is not set. "
                "Set it before starting JARVIS."
            )

        def env_str(name: str, default: str) -> str:
            return os.environ.get(name, default)

        def env_float(name: str, default: float) -> float:
            v = os.environ.get(name)
            return float(v) if v is not None else default

        def env_int(name: str, default: int) -> int:
            v = os.environ.get(name)
            return int(v) if v is not None else default

        def env_bool(name: str, default: bool) -> bool:
            v = os.environ.get(name)
            if v is None:
                return default
            return v.strip().lower() not in ("0", "false", "no", "off")

        cfg = cls(api_key=api_key)
        cfg.model_name = env_str("JARVIS_MODEL", cfg.model_name)
        cfg.voice_name = env_str("JARVIS_VOICE", cfg.voice_name)
        cfg.wake_score_threshold = env_float("JARVIS_WAKE_THRESHOLD", cfg.wake_score_threshold)
        cfg.silence_duration_ms = env_int("JARVIS_SILENCE_MS", cfg.silence_duration_ms)
        cfg.inactivity_sleep_s = env_int("JARVIS_INACTIVITY_S", cfg.inactivity_sleep_s)
        cfg.camera_enabled = env_bool("JARVIS_CAMERA_ENABLED", cfg.camera_enabled)
        cfg.camera_interval_s = env_float("JARVIS_CAMERA_INTERVAL_S", cfg.camera_interval_s)
        cfg.log_level = env_str("JARVIS_LOG_LEVEL", cfg.log_level)
        cfg.log_file = env_str("JARVIS_LOG_FILE", cfg.log_file or "") or None
        return cfg


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

def setup_logging(cfg: Config) -> logging.Logger:
    logger = logging.getLogger("jarvis")
    logger.setLevel(cfg.log_level)
    logger.handlers.clear()
    logger.propagate = False


    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    if cfg.log_file:
        try:
            file_handler = logging.handlers.RotatingFileHandler(
                cfg.log_file, maxBytes=2_000_000, backupCount=3
            )
            file_handler.setFormatter(fmt)
            logger.addHandler(file_handler)
        except OSError as e:
            logger.warning("Could not open log file %s (%s) - console logging only.", cfg.log_file, e)

    return logger


log = logging.getLogger("jarvis")


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------

def switch_project(project_name: str) -> str:
    return workspace.switch_project(project_name)


def write_file(filename: str, content: str) -> str:
    return workspace.write_file(filename, content)


def read_file(filename: str) -> str:
    return workspace.read_file(filename)


def list_files() -> str:
    files = workspace.list_project_files()
    return f"Files in project '{workspace.current_project}': {files}"


TOOL_MAP = {
    "switch_project": switch_project,
    "write_file": write_file,
    "read_file": read_file,
    "list_files": list_files,
    "launch_app": system_control.launch_app,
    "close_app": system_control.close_app,
    "play_spotify": system_control.play_spotify,
    "open_url": system_control.open_url,
    "control_volume": system_control.control_volume,
    "run_cmd": system_control.run_cmd,
    "get_system_info": system_control.get_system_info,
    "take_desktop_screenshot": system_control.take_desktop_screenshot,
}

TOOLS = [
    types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="switch_project",
            description="Switches active project workspace to the specified name. Creates directory if missing.",
            parameters=types.Schema(
                type="OBJECT",
                properties={"project_name": types.Schema(type="STRING", description="Project name")},
                required=["project_name"],
            ),
        ),
        types.FunctionDeclaration(
            name="write_file",
            description="Writes or overwrites a text/markdown file in the active project.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "filename": types.Schema(type="STRING", description="File name"),
                    "content": types.Schema(type="STRING", description="Content to write"),
                },
                required=["filename", "content"],
            ),
        ),
        types.FunctionDeclaration(
            name="read_file",
            description="Reads contents of a file inside the active project workspace.",
            parameters=types.Schema(
                type="OBJECT",
                properties={"filename": types.Schema(type="STRING", description="File name")},
                required=["filename"],
            ),
        ),
        types.FunctionDeclaration(
            name="list_files",
            description="Lists all files contained in the active project folder.",
            parameters=types.Schema(type="OBJECT", properties={}),
        ),
        types.FunctionDeclaration(
            name="launch_app",
            description="Launches a desktop application on Windows (e.g. spotify, chrome, notepad, calc, vscode, cmd).",
            parameters=types.Schema(
                type="OBJECT",
                properties={"app_name": types.Schema(type="STRING", description="Application name or executable command")},
                required=["app_name"],
            ),
        ),
        types.FunctionDeclaration(
            name="close_app",
            description="Closes or terminates a running desktop application on Windows (e.g. notepad, chrome, spotify, edge).",
            parameters=types.Schema(
                type="OBJECT",
                properties={"app_name": types.Schema(type="STRING", description="Application name to close")},
                required=["app_name"],
            ),
        ),

        types.FunctionDeclaration(
            name="play_spotify",
            description="Launches Spotify and searches/plays a specified song, artist, album, or playlist.",
            parameters=types.Schema(
                type="OBJECT",
                properties={"query": types.Schema(type="STRING", description="Song name, artist, or music query to play")},
                required=["query"],
            ),
        ),
        types.FunctionDeclaration(
            name="open_url",
            description="Opens a website URL or web search in the user's default web browser.",
            parameters=types.Schema(
                type="OBJECT",
                properties={"url": types.Schema(type="STRING", description="Website URL or domain")},
                required=["url"],
            ),
        ),
        types.FunctionDeclaration(
            name="control_volume",
            description="Adjusts Windows master system audio volume ('up', 'down', 'mute', 'unmute').",
            parameters=types.Schema(
                type="OBJECT",
                properties={"action": types.Schema(type="STRING", description="Volume action: 'up', 'down', 'mute', 'unmute'")},
                required=["action"],
            ),
        ),
        types.FunctionDeclaration(
            name="run_cmd",
            description="Executes a safe shell / terminal command on Windows and returns stdout output.",
            parameters=types.Schema(
                type="OBJECT",
                properties={"command": types.Schema(type="STRING", description="Terminal command to execute")},
                required=["command"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_system_info",
            description="Returns Windows OS health diagnostics (current time, host name, OS version, CPU, RAM status).",
            parameters=types.Schema(type="OBJECT", properties={}),
        ),
        types.FunctionDeclaration(
            name="take_desktop_screenshot",
            description="Captures a screenshot of the main desktop monitor for inspection.",
            parameters=types.Schema(type="OBJECT", properties={}),
        ),
    ])
]



async def execute_tool(name: str, args: Dict[str, Any], timeout_s: float) -> str:
    """[FIX #8] Runs the (synchronous, blocking) tool function off the event
    loop, with a timeout so a stuck file op can't hang the whole assistant."""
    if name not in TOOL_MAP:
        return f"Unknown tool: {name}"
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(TOOL_MAP[name], **args), timeout=timeout_s
        )
    except asyncio.TimeoutError:
        return f"Error: {name} timed out after {timeout_s}s"
    except Exception as err:
        return f"Error executing {name}: {err}"


# --------------------------------------------------------------------------
# Conversation state
# --------------------------------------------------------------------------

class ConversationState:
    """Tracks whether JARVIS is actively streaming mic audio to Gemini
    (awake) or only running the local wake-word detector (asleep)."""

    def __init__(self, start_awake: bool):
        self.awake_event = asyncio.Event()
        if start_awake:
            self.awake_event.set()
        self.last_active_ts = time.time()
        self.pending_sleep = False


# --------------------------------------------------------------------------
# Audio playback - thread-safe jitter buffer [FIX #3, #4]
# --------------------------------------------------------------------------

class AudioPlayback:
    """Owns the output stream and a lock-guarded jitter buffer. Safe to
    mutate from both the PortAudio callback thread and the asyncio task
    that feeds it, and supports immediate interruption (clears both the
    pending queue AND whatever is already buffered for playback)."""

    def __init__(self, cfg: Config):
        self._cfg = cfg
        self.queue: "asyncio.Queue[bytes]" = asyncio.Queue()
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._started = False
        self._min_prebuffer = cfg.playback_prebuffer_blocks * cfg.block_size * 2
        self.last_playback_time = 0.0

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return self._started or len(self._buf) > 0 or not self.queue.empty()

    def _callback(self, outdata, frames, time_info, status):
        bytes_needed = frames * 2
        with self._lock:
            if not self._started:
                if len(self._buf) < self._min_prebuffer:
                    outdata.fill(0)
                    return
                self._started = True

            if len(self._buf) >= bytes_needed:
                chunk = bytes(self._buf[:bytes_needed])
                del self._buf[:bytes_needed]
                outdata[:, 0] = np.frombuffer(chunk, dtype=np.int16)
                self.last_playback_time = time.time()
            elif len(self._buf) > 0:
                avail = len(self._buf) // 2
                outdata[:avail, 0] = np.frombuffer(bytes(self._buf[:avail * 2]), dtype=np.int16)
                outdata[avail:, 0] = 0
                self._buf.clear()
                self._started = False
                self.last_playback_time = time.time()
            else:
                outdata.fill(0)
                self._started = False

    def interrupt(self):
        """Stop whatever is playing/queued immediately - called when the
        model reports the user has barged in."""
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        with self._lock:
            self._buf.clear()
            self._started = False

    async def run(self, stop_event: asyncio.Event):
        try:
            stream = sd.OutputStream(
                samplerate=self._cfg.sample_rate_out,
                channels=self._cfg.channels,
                dtype="int16",
                blocksize=self._cfg.block_size,
                callback=self._callback,
            )
        except Exception as e:
            raise FatalError(f"Could not open audio output device: {e}") from e

        with stream:
            while not stop_event.is_set():
                try:
                    chunk = await asyncio.wait_for(self.queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
                with self._lock:
                    self._buf.extend(chunk)


# --------------------------------------------------------------------------
# Microphone input [FIX #1: no more echo-zeroing while awake]
# --------------------------------------------------------------------------

async def audio_input_loop(
    session, stop_event: asyncio.Event, state: ConversationState,
    wake_model, playback: AudioPlayback, cfg: Config,
):
    """Streams microphone PCM audio hands-free.

    While asleep: audio never leaves the machine - each chunk only goes
    through the local wake-word model. While awake: streamed straight to
    Gemini continuously. We rely on the Live model's native barge-in
    support rather than gating our own mic - see module docstring.
    """
    loop = asyncio.get_running_loop()
    mic_queue: "asyncio.Queue[np.ndarray]" = asyncio.Queue(maxsize=100)

    def mic_callback(indata, frames, time_info, status):
        if stop_event.is_set():
            return
        data = indata.copy()

        def _enqueue():
            try:
                mic_queue.put_nowait(data)
            except asyncio.QueueFull:
                pass

        try:
            loop.call_soon_threadsafe(_enqueue)
        except RuntimeError:
            pass

    try:
        input_stream = sd.InputStream(
            samplerate=cfg.sample_rate_in,
            channels=cfg.channels,
            dtype="int16",
            blocksize=cfg.block_size,
            callback=mic_callback,
        )
    except Exception as e:
        raise FatalError(f"Could not open microphone: {e}") from e

    with input_stream:
        while not stop_event.is_set():
            try:
                indata = await asyncio.wait_for(mic_queue.get(), timeout=0.5)

                if not state.awake_event.is_set():
                    # ASLEEP: local wake-word check only, nothing sent to the cloud.
                    # Guard against JARVIS's own farewell line re-triggering the
                    # wake word right after he stops talking.
                    just_played = (time.time() - playback.last_playback_time) < cfg.wake_echo_guard_s
                    if wake_model is not None and not just_played:
                        frame = indata.flatten()
                        try:
                            prediction = wake_model.predict(frame)
                        except Exception:
                            prediction = {}
                        if prediction.get("hey_jarvis", 0.0) > cfg.wake_score_threshold:
                            state.awake_event.set()
                            state.last_active_ts = time.time()
                            state.pending_sleep = False
                            log.info("'Hey Jarvis' detected - listening, sir.")
                            try:
                                await session.send_client_content(
                                    turns=[types.Content(role="user", parts=[types.Part.from_text(text="User said 'Hey Jarvis' to wake you up. Greet them briefly out loud, e.g. 'Yes, sir?'")])],
                                    turn_complete=True,
                                )
                            except Exception as e:
                                log.warning("Could not send wake acknowledgment trigger: %s", e)
                    continue

                # AWAKE: stream mic audio to Gemini.
                # 1. Echo Gate: Prevent JARVIS from hearing his own voice coming out of the speakers.
                if playback.is_playing or (time.time() - playback.last_playback_time) < 0.15:
                    await asyncio.sleep(0.005)
                    continue

                # 2. Adaptive VAD Noise Gate: Filter out static background noise / fan hiss
                # so Gemini only receives audio when you are actually speaking into the mic.
                rms = float(np.sqrt(np.mean(indata.astype(np.float32) ** 2)))
                if not hasattr(audio_input_loop, "_noise_floor"):
                    audio_input_loop._noise_floor = 3000.0
                audio_input_loop._noise_floor = 0.95 * audio_input_loop._noise_floor + 0.05 * min(rms, 4000.0)

                vad_threshold = max(2800.0, audio_input_loop._noise_floor * 1.15)
                if rms < vad_threshold:
                    await asyncio.sleep(0.005)
                    continue

                await session.send_realtime_input(
                    audio=types.Blob(
                        data=indata.tobytes(),
                        mime_type="audio/pcm;rate=16000",
                    )
                )



            except asyncio.TimeoutError:
                continue
            except Exception as e:
                if not stop_event.is_set():
                    log.error("Mic stream error: %s", e)
                break


# --------------------------------------------------------------------------
# Camera [FIX #2: gated on awake state, FIX #7: failures are non-fatal]
# --------------------------------------------------------------------------

def _open_camera() -> Optional[cv2.VideoCapture]:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
        return None
    return cap


def _encode_jpeg(frame, cfg: Config) -> bytes:
    small = cv2.resize(frame, (cfg.camera_width, cfg.camera_height))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=cfg.camera_jpeg_quality)
    return buf.getvalue()


async def camera_stream_loop(session, stop_event: asyncio.Event, state: ConversationState, cfg: Config):
    """Sends camera frames only while awake. Any camera failure is logged
    and swallowed here so it never takes down mic/audio/receive tasks -
    vision is a nice-to-have, not core functionality."""
    if not cfg.camera_enabled:
        log.info("Camera disabled by config.")
        return

    loop = asyncio.get_running_loop()
    cap = await loop.run_in_executor(None, _open_camera)
    if cap is None:
        log.warning("No camera detected - continuing without vision.")
        return

    log.info("Camera feed ready.")
    try:
        while not stop_event.is_set():
            if not state.awake_event.is_set():
                await asyncio.sleep(0.25)
                continue

            success, frame = await loop.run_in_executor(None, cap.read)
            if success and frame is not None:
                jpeg_bytes = await loop.run_in_executor(None, _encode_jpeg, frame, cfg)
                try:
                    # [FIX] session.send() is deprecated - use send_realtime_input().
                    await session.send_realtime_input(
                        video=types.Blob(data=jpeg_bytes, mime_type="image/jpeg")
                    )
                except Exception as e:
                    log.warning("Camera send failed, stopping vision for this session: %s", e)
                    break
            await asyncio.sleep(cfg.camera_interval_s)
    except Exception as e:
        log.warning("Camera loop stopped due to error (vision only, non-fatal): %s", e)
    finally:
        cap.release()


# --------------------------------------------------------------------------
# Inactivity watchdog
# --------------------------------------------------------------------------

async def inactivity_watchdog(state: ConversationState, stop_event: asyncio.Event, cfg: Config):
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=5)
        except asyncio.TimeoutError:
            pass
        if stop_event.is_set():
            break
        if state.awake_event.is_set() and (time.time() - state.last_active_ts) > cfg.inactivity_sleep_s:
            state.awake_event.clear()
            state.pending_sleep = False
            log.info("%ss of quiet - dormant. Say 'Hey Jarvis' to wake me, sir.", cfg.inactivity_sleep_s)


# --------------------------------------------------------------------------
# Receive loop
# --------------------------------------------------------------------------

async def receive_loop(
    session, stop_event: asyncio.Event, state: ConversationState,
    playback: AudioPlayback, cfg: Config,
):
    try:
        while not stop_event.is_set():
            async for response in session.receive():
                if stop_event.is_set():
                    break

                server_content = response.server_content
                if server_content is not None:
                    if server_content.interrupted:
                        log.debug("Interrupted by user.")
                        playback.interrupt()  # [FIX #4] clears buffered audio too
                        continue

                    input_transcription = getattr(server_content, "input_transcription", None)
                    if input_transcription is not None and input_transcription.text:
                        text = input_transcription.text
                        log.info("You said: %s", text.strip())
                        state.last_active_ts = time.time()
                        if state.awake_event.is_set() and any(p in text.lower() for p in cfg.sleep_phrases):
                            state.pending_sleep = True

                    model_turn = server_content.model_turn
                    if model_turn is not None:
                        for part in model_turn.parts:
                            if part.text and not getattr(part, "thought", False):
                                log.debug("JARVIS: %s", part.text)
                            if part.inline_data:
                                await playback.queue.put(part.inline_data.data)

                    if server_content.turn_complete:
                        state.last_active_ts = time.time()
                        if state.pending_sleep:
                            state.pending_sleep = False
                            state.awake_event.clear()
                            log.info("Going dormant. Say 'Hey Jarvis' to wake me, sir.")

                tool_call = response.tool_call
                if tool_call is not None:
                    function_responses = []
                    for call in tool_call.function_calls:
                        log.info("Tool call: %s(%s)", call.name, call.args)
                        result = await execute_tool(call.name, call.args, cfg.tool_timeout_s)
                        if call.name == "take_desktop_screenshot":
                            try:
                                jpeg_bytes = await asyncio.to_thread(system_control.get_desktop_screenshot_jpeg)
                                await session.send_realtime_input(
                                    video=types.Blob(data=jpeg_bytes, mime_type="image/jpeg")
                                )
                            except Exception as e:
                                log.warning("Failed streaming desktop screenshot image to session: %s", e)
                        function_responses.append(
                            types.FunctionResponse(name=call.name, id=call.id, response={"result": result})
                        )

                    if function_responses:
                        # [FIX] session.send() is deprecated - use send_tool_response().
                        await session.send_tool_response(function_responses=function_responses)


    except asyncio.CancelledError:
        raise
    except Exception as e:
        log.error("Receive loop exception: %s", e)
        log.debug(traceback.format_exc())
        raise
    finally:
        log.info("Receive loop ended.")
        stop_event.set()


# --------------------------------------------------------------------------
# Wake-word engine setup
# --------------------------------------------------------------------------

def init_wake_word():
    if not _OWW_AVAILABLE:
        log.warning("'openwakeword' not installed (pip install openwakeword). Falling back to always-listening mode.")
        return None
    try:
        log.info("Loading local 'Hey Jarvis' wake-word model...")
        openwakeword.utils.download_models()  # one-time download, cached after
        model = WakeWordModel()
        log.info("Wake-word engine ready. JARVIS starts dormant - say 'Hey Jarvis' to wake him.")
        return model
    except Exception as e:
        log.warning("Could not initialize openWakeWord (%s). Falling back to always-listening mode.", e)
        return None


# --------------------------------------------------------------------------
# One connect+run cycle
# --------------------------------------------------------------------------

async def run_session(cfg: Config, wake_model) -> None:
    client = genai.Client(api_key=cfg.api_key)

    speech_sensitivity_map = {
        "LOW": types.StartSensitivity.START_SENSITIVITY_LOW,
        "HIGH": types.StartSensitivity.START_SENSITIVITY_HIGH,
    }
    end_sensitivity_map = {
        "LOW": types.EndSensitivity.END_SENSITIVITY_LOW,
        "HIGH": types.EndSensitivity.END_SENSITIVITY_HIGH,
    }

    live_config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=cfg.voice_name)
            )
        ),
        system_instruction=types.Content(parts=[types.Part.from_text(text=cfg.system_prompt)]),
        tools=TOOLS,
        input_audio_transcription=types.AudioTranscriptionConfig(),
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                disabled=False,
                start_of_speech_sensitivity=speech_sensitivity_map[cfg.start_of_speech_sensitivity],
                end_of_speech_sensitivity=end_sensitivity_map[cfg.end_of_speech_sensitivity],
                prefix_padding_ms=cfg.prefix_padding_ms,
                silence_duration_ms=cfg.silence_duration_ms,
            )
        ),
    )

    state = ConversationState(start_awake=(wake_model is None))
    playback = AudioPlayback(cfg)

    log.info("Connecting to Jarvis Neural Matrix (%s)...", cfg.model_name)
    try:
        async with client.aio.live.connect(model=cfg.model_name, config=live_config) as session:
            log.info("=" * 60)
            log.info("JARVIS Online.")
            log.info("Dormant - say 'Hey Jarvis' to wake him." if wake_model is not None
                      else "Hands-Free Open Audio Streaming Active.")
            log.info("=" * 60)

            stop_event = asyncio.Event()

            t_in = asyncio.create_task(audio_input_loop(session, stop_event, state, wake_model, playback, cfg))
            t_out = asyncio.create_task(playback.run(stop_event))
            t_cam = asyncio.create_task(camera_stream_loop(session, stop_event, state, cfg))
            t_recv = asyncio.create_task(receive_loop(session, stop_event, state, playback, cfg))
            t_watch = asyncio.create_task(inactivity_watchdog(state, stop_event, cfg))

            all_tasks = [t_in, t_out, t_cam, t_recv, t_watch]
            try:
                done, pending = await asyncio.wait(all_tasks, return_when=asyncio.FIRST_EXCEPTION)
                for t in done:
                    exc = t.exception() if not t.cancelled() else None
                    if exc is not None:
                        if isinstance(exc, FatalError):
                            raise exc
                        log.error("Task error: %s", exc)
            finally:
                stop_event.set()
                for t in all_tasks:
                    t.cancel()
                await asyncio.gather(*all_tasks, return_exceptions=True)
    except FatalError:
        raise
    except Exception as e:
        if _looks_like_auth_error(e):
            raise FatalError(f"Authentication/permission error talking to Gemini: {e}") from e
        raise


# --------------------------------------------------------------------------
# Reconnect loop [FIX #5, #6]
# --------------------------------------------------------------------------

async def run_forever(cfg: Config) -> None:
    wake_model = init_wake_word()
    delay = cfg.reconnect_initial_delay_s

    while True:
        try:
            await run_session(cfg, wake_model)
            log.info("Session ended normally - reconnecting.")
            delay = cfg.reconnect_initial_delay_s
        except FatalError as e:
            log.critical("Unrecoverable error, shutting down: %s", e)
            return
        except (KeyboardInterrupt, asyncio.CancelledError):
            raise
        except Exception as e:
            log.error("Session crashed (%s). Reconnecting in %.0fs...", e, delay)
            log.debug(traceback.format_exc())

        await asyncio.sleep(delay)
        delay = min(delay * 2, cfg.reconnect_max_delay_s)


def main():
    try:
        cfg = Config.from_env()
    except FatalError as e:
        print(f"[FATAL] {e}")
        return

    setup_logging(cfg)
    try:
        asyncio.run(run_forever(cfg))
    except KeyboardInterrupt:
        log.info("Shutting down JARVIS - goodbye, sir.")


if __name__ == "__main__":
    main()