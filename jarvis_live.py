"""
JARVIS - Ultra-Fast Real-Time Voice AI Assistant (Antigravity Core)
=====================================================================
Model: gemini-3.1-flash-live-preview
Voice: Charon
"""

from __future__ import annotations

import asyncio
import datetime
import io
import logging
import logging.handlers
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import sounddevice as sd
from PIL import Image

# Suppress OpenCV C++ MSMF backend warnings on Windows
os.environ["OPENCV_LOG_LEVEL"] = "OFF"

from google import genai
from google.genai import types

sys.path.insert(0, r"C:\Users\Lenovo\jarvis_assistant")
from workspace_manager import workspace
import system_control
import web_intel
from hud import StatusHUD

try:
    import openwakeword
    from openwakeword.model import Model as WakeWordModel
    _OWW_AVAILABLE = True
except ImportError:
    _OWW_AVAILABLE = False


class FatalError(Exception):
    """Raised for non-recoverable hardware/auth issues."""


def _looks_like_auth_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(tok in text for tok in (
        "api key not valid", "api_key", "permission_denied",
        "unauthenticated", "401", "403", "invalid api key",
    ))


DEFAULT_SLEEP_PHRASES = [
    "go to sleep", "goodbye jarvis", "that will be all", "that'll be all",
    "power down", "sleep now", "stand down", "that's all for now",
]

def get_system_prompt() -> str:
    now_str = datetime.datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
    return f"""
You are JARVIS, Tony Stark's elite personal AI assistant with full active control over this Windows PC.

1. CRITICAL REAL-TIME KNOWLEDGE & ACCURACY DIRECTIVE:
- Today's date and time is {now_str}.
- You MUST match the accuracy of ChatGPT and Gemini Web Assistant.
- For ANY query about current leaders/officials (e.g. Chief Minister of Kerala, US President, Prime Minister), real-world facts, news, sports scores, people, places, dates, or weather: ALWAYS invoke the 'web_search' tool immediately before answering! NEVER rely on outdated training memory!

2. MULTIMODAL WEBCAM & DESKTOP SCREEN VISION DIRECTIVE:
- WEBCAM VISION: You continuously receive live video frames from the webcam in real-time. When asked 'Can you see me?', 'What am I holding?', 'What am I wearing?', 'Describe what you see', or visual questions about your webcam view, inspect your live camera feed and describe the exact objects, colors, and items in front of you directly!
- DESKTOP SCREEN VISION: When asked 'What is on my screen?', 'Look at my screen', 'Read this page/code', or asked about anything on the monitor, invoke 'take_desktop_screenshot' immediately to capture and inspect the screen image!

3. MANDATORY HARDWARE & FUNCTION TOOL DIRECTIVES:
- To play music or search songs: call 'play_spotify' or 'open_url' immediately.
- To open Google, YouTube, websites: call 'open_url' immediately.
- To search real-time facts, leaders, news: call 'web_search' immediately.
- To launch applications (Chrome, Notepad, Calc, VS Code, Terminal): call 'launch_app' immediately.
- To adjust volume: call 'control_volume' immediately.
- To check date, time, CPU, RAM, battery: call 'get_system_info' immediately.

STRICT RULE: You are NEVER "just a language model". You are an active PC automation agent. NEVER refuse a user request. You MUST call the corresponding function tool immediately!

Persona & Cadence:
- Speak in a calm, refined British cadence with dry wit and unfailing elegance. Address the user as 'Sir'.
- Keep verbal responses crisp, direct, and short (1-2 sentences maximum).
"""

DEFAULT_SYSTEM_PROMPT = get_system_prompt()


@dataclass
class Config:
    api_key: str

    model_name: str = "gemini-3.1-flash-live-preview"
    voice_name: str = "Charon"

    channels: int = 1
    sample_rate_in: int = 16000
    sample_rate_out: int = 24000
    block_size: int = 384
    playback_prebuffer_blocks: int = 1

    wake_score_threshold: float = 0.35
    wake_echo_guard_s: float = 0.15

    inactivity_sleep_s: int = 90
    sleep_phrases: List[str] = field(default_factory=lambda: list(DEFAULT_SLEEP_PHRASES))

    camera_enabled: bool = True
    camera_interval_s: float = 0.8

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
            raise FatalError("GEMINI_API_KEY environment variable is not set.")

        cfg = cls(api_key=api_key)
        cfg.model_name = os.environ.get("JARVIS_MODEL", cfg.model_name)
        cfg.voice_name = os.environ.get("JARVIS_VOICE", cfg.voice_name)
        return cfg


def setup_logging(cfg: Config) -> logging.Logger:
    logger = logging.getLogger("jarvis")
    logger.setLevel(cfg.log_level)
    logger.handlers.clear()
    logger.propagate = False

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if cfg.log_file:
        try:
            file_handler = logging.handlers.RotatingFileHandler(
                cfg.log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8", errors="replace"
            )
            file_handler.setFormatter(fmt)
            logger.addHandler(file_handler)
        except OSError:
            pass

    return logger


log = logging.getLogger("jarvis")


# Tools Mapping
TOOL_MAP = {
    "launch_app": system_control.launch_app,
    "close_app": system_control.close_app,
    "play_spotify": system_control.play_spotify,
    "open_url": system_control.open_url,
    "control_volume": system_control.control_volume,
    "run_cmd": system_control.run_cmd,
    "get_system_info": system_control.get_system_info,
    "take_desktop_screenshot": system_control.take_desktop_screenshot,
    "web_search": web_intel.web_search,
    "get_weather": web_intel.get_weather,
    "remember_fact": system_control.remember_fact,
    "switch_project": workspace.switch_project,
    "read_project_file": workspace.read_file,
    "write_project_file": workspace.write_file,
    "list_project_files": workspace.list_project_files,
}

TOOLS = [
    types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="launch_app",
            description="Launches a desktop application on Windows (e.g. spotify, chrome, notepad, calc, vscode, cmd, terminal).",
            parameters=types.Schema(
                type="OBJECT",
                properties={"app_name": types.Schema(type="STRING", description="Application name")},
                required=["app_name"],
            ),
        ),
        types.FunctionDeclaration(
            name="close_app",
            description="Closes or terminates a running desktop application on Windows.",
            parameters=types.Schema(
                type="OBJECT",
                properties={"app_name": types.Schema(type="STRING", description="Application name")},
                required=["app_name"],
            ),
        ),
        types.FunctionDeclaration(
            name="play_spotify",
            description="Searches and plays any music, song, track, or artist on Spotify or YouTube. Call this whenever asked to play music or songs.",
            parameters=types.Schema(
                type="OBJECT",
                properties={"query": types.Schema(type="STRING", description="Song name, artist, or music search query")},
                required=["query"],
            ),
        ),
        types.FunctionDeclaration(
            name="open_url",
            description="Opens Google, YouTube, websites, URLs, or browser search queries. Call this whenever asked to open Google, YouTube, or any website.",
            parameters=types.Schema(
                type="OBJECT",
                properties={"url": types.Schema(type="STRING", description="Website URL, domain name, or query (e.g. 'google', 'youtube', 'youtube kalyani song')")},
                required=["url"],
            ),
        ),
        types.FunctionDeclaration(
            name="control_volume",
            description="Adjusts Windows system volume ('up', 'down', 'mute', 'unmute', or 'set').",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "action": types.Schema(type="STRING", description="Volume action: 'up', 'down', 'mute', 'unmute', 'set'"),
                    "level": types.Schema(type="INTEGER", description="Volume level percentage (0-100)"),
                },
                required=["action"],
            ),
        ),
        types.FunctionDeclaration(
            name="run_cmd",
            description="Executes a safe terminal command on Windows.",
            parameters=types.Schema(
                type="OBJECT",
                properties={"command": types.Schema(type="STRING", description="Command string")},
                required=["command"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_system_info",
            description="Returns Windows OS health diagnostics (Time, CPU %, RAM %, Battery status).",
            parameters=types.Schema(type="OBJECT", properties={}),
        ),
        types.FunctionDeclaration(
            name="take_desktop_screenshot",
            description="Captures a screenshot of the main desktop screen and transmits the visual image to JARVIS so you can see what is displayed on the monitor and answer questions about it.",
            parameters=types.Schema(type="OBJECT", properties={}),
        ),
        types.FunctionDeclaration(
            name="web_search",
            description="Performs a real-time web search for any question, factual query, current leaders, CM of Kerala, US President, news, or general knowledge. Call this immediately for any question asking who, what, when, where, or why.",
            parameters=types.Schema(
                type="OBJECT",
                properties={"query": types.Schema(type="STRING", description="Search query string")},
                required=["query"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_weather",
            description="Fetches live real-time weather forecasts for any location or city.",
            parameters=types.Schema(
                type="OBJECT",
                properties={"location": types.Schema(type="STRING", description="City or location name (leave empty for current location)")},
            ),
        ),
        types.FunctionDeclaration(
            name="remember_fact",
            description="Saves a key fact, note, or preference into long-term JARVIS memory.",
            parameters=types.Schema(
                type="OBJECT",
                properties={"fact": types.Schema(type="STRING", description="Fact or note to remember")},
                required=["fact"],
            ),
        ),
        types.FunctionDeclaration(
            name="switch_project",
            description="Switches active workspace project folder.",
            parameters=types.Schema(
                type="OBJECT",
                properties={"project_name": types.Schema(type="STRING", description="Project folder name")},
                required=["project_name"],
            ),
        ),
        types.FunctionDeclaration(
            name="read_project_file",
            description="Reads the contents of a file inside the active project workspace.",
            parameters=types.Schema(
                type="OBJECT",
                properties={"filename": types.Schema(type="STRING", description="Filename to read")},
                required=["filename"],
            ),
        ),
        types.FunctionDeclaration(
            name="write_project_file",
            description="Writes or updates a file inside the active project workspace.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "filename": types.Schema(type="STRING", description="Filename to write"),
                    "content": types.Schema(type="STRING", description="File text content"),
                },
                required=["filename", "content"],
            ),
        ),
        types.FunctionDeclaration(
            name="list_project_files",
            description="Lists all files stored inside the active project workspace.",
            parameters=types.Schema(type="OBJECT", properties={}),
        ),
    ])
]


async def execute_tool(name: str, args: Dict[str, Any], timeout_s: float) -> str:
    func = TOOL_MAP.get(name)
    if not func:
        return f"Unknown tool: {name}"
    try:
        if asyncio.iscoroutinefunction(func):
            return await asyncio.wait_for(func(**args), timeout=timeout_s)
        else:
            return await asyncio.wait_for(asyncio.to_thread(func, **args), timeout=timeout_s)
    except Exception as e:
        return f"Error executing {name}: {e}"


class AudioPlayback:
    def __init__(self, cfg: Config):
        self._cfg = cfg
        self.queue: asyncio.Queue[bytes] = asyncio.Queue()
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
                chunk = bytes(self._buf[:avail * 2])
                outdata[:avail, 0] = np.frombuffer(chunk, dtype=np.int16)
                outdata[avail:, 0] = 0
                self._buf.clear()
                self._started = False
                self.last_playback_time = time.time()
            else:
                outdata.fill(0)
                self._started = False

    def interrupt(self):
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
            raise FatalError(f"Could not open speaker device: {e}") from e

        with stream:
            while not stop_event.is_set():
                try:
                    data = await asyncio.wait_for(self.queue.get(), timeout=0.1)
                    with self._lock:
                        self._buf.extend(data)
                except asyncio.TimeoutError:
                    continue


class ConversationState:
    def __init__(self, start_awake: bool = True):
        self.awake_event = asyncio.Event()
        if start_awake:
            self.awake_event.set()
        self.last_active_ts = time.time()
        self.pending_sleep = False


async def audio_input_loop(
    session, stop_event: asyncio.Event, state: ConversationState,
    wake_model, playback: AudioPlayback, hud: StatusHUD, cfg: Config,
    send_lock: asyncio.Lock,
):
    loop = asyncio.get_running_loop()
    mic_queue: "asyncio.Queue[np.ndarray]" = asyncio.Queue(maxsize=100)

    def mic_callback(indata, frames, time_info, status):
        if stop_event.is_set():
            return
        data = indata.copy()
        try:
            loop.call_soon_threadsafe(mic_queue.put_nowait, data)
        except (asyncio.QueueFull, RuntimeError):
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

    current_hud_mode = None

    with input_stream:
        while not stop_event.is_set():
            try:
                indata = await asyncio.wait_for(mic_queue.get(), timeout=0.5)

                if not state.awake_event.is_set():
                    if current_hud_mode != "DORMANT":
                        hud.set("DORMANT", "Say 'Hey Jarvis'")
                        current_hud_mode = "DORMANT"

                    just_played = (time.time() - playback.last_playback_time) < cfg.wake_echo_guard_s
                    if wake_model is not None and not just_played:
                        frame = indata.flatten()
                        prediction = await asyncio.to_thread(wake_model.predict, frame)
                        if prediction.get("hey_jarvis", 0.0) > cfg.wake_score_threshold:
                            state.awake_event.set()
                            state.last_active_ts = time.time()
                            state.pending_sleep = False
                            log.info("'Hey Jarvis' detected — online and listening, sir.")
                            hud.set("LISTENING", "Listening, sir...")
                            current_hud_mode = "LISTENING"

                            # Drain stale wake-word audio frames so speech processing starts instantly
                            while not mic_queue.empty():
                                try:
                                    mic_queue.get_nowait()
                                except asyncio.QueueEmpty:
                                    break
                    continue

                if current_hud_mode != "LISTENING":
                    hud.set("LISTENING", "Listening...")
                    current_hud_mode = "LISTENING"

                async with send_lock:
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


async def receive_loop(
    session, stop_event: asyncio.Event, state: ConversationState,
    playback: AudioPlayback, hud: StatusHUD, cfg: Config,
    send_lock: asyncio.Lock,
):
    try:
        while not stop_event.is_set():
            async for response in session.receive():
                if stop_event.is_set():
                    break

                server_content = response.server_content
                if server_content is not None:
                    if server_content.interrupted:
                        log.info("[Barge-in] Interrupted by user.")
                        playback.interrupt()
                        hud.set("LISTENING", "Listening...")
                        continue

                    input_transcription = getattr(server_content, "input_transcription", None)
                    if input_transcription is not None and input_transcription.text:
                        text = input_transcription.text.strip()
                        if text:
                            log.info("Sir: %s", text)
                            hud.set("THINKING", f'"{text[:25]}..."' if len(text) > 25 else f'"{text}"')
                            state.last_active_ts = time.time()
                            if state.awake_event.is_set() and any(p in text.lower() for p in cfg.sleep_phrases):
                                state.pending_sleep = True

                    model_turn = server_content.model_turn
                    if model_turn is not None:
                        for part in model_turn.parts:
                            if part.text and not getattr(part, "thought", False):
                                log.info("JARVIS: %s", part.text.strip())
                            if part.inline_data:
                                hud.set("SPEAKING", "JARVIS speaking...")
                                await playback.queue.put(part.inline_data.data)

                    if server_content.turn_complete:
                        state.last_active_ts = time.time()
                        if state.pending_sleep:
                            state.pending_sleep = False
                            state.awake_event.clear()
                            hud.set("DORMANT", "Say 'Hey Jarvis'")
                            log.info("JARVIS entering dormant state, sir.")

                tool_call = response.tool_call
                if tool_call is not None:
                    hud.set("THINKING", "Executing command...")
                    function_responses = []
                    for call in tool_call.function_calls:
                        log.info("Tool call: %s(%s)", call.name, call.args)
                        result = await execute_tool(call.name, call.args, cfg.tool_timeout_s)
                        if call.name == "take_desktop_screenshot":
                            try:
                                jpeg_bytes = await asyncio.to_thread(system_control.get_desktop_screenshot_jpeg)
                                async with send_lock:
                                    await session.send_realtime_input(
                                        video=types.Blob(data=jpeg_bytes, mime_type="image/jpeg")
                                    )
                            except Exception as e:
                                log.warning("Desktop screenshot vision stream failed: %s", e)
                        function_responses.append(
                            types.FunctionResponse(name=call.name, id=call.id, response={"result": result})
                        )

                    if function_responses:
                        async with send_lock:
                            await session.send_tool_response(function_responses=function_responses)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        log.error("Receive loop exception: %s", e)
        raise
    finally:
        log.info("Receive loop ended.")
        stop_event.set()


def init_wake_word():
    if not _OWW_AVAILABLE:
        log.warning("'openwakeword' not installed. Falling back to always-listening mode.")
        return None
    try:
        log.info("Loading 'Hey Jarvis' wake-word model...")
        openwakeword.utils.download_models()
        model = WakeWordModel(wakeword_models=["hey_jarvis"])
        log.info("Wake-word model ready.")
        return model
    except Exception as e:
        log.warning("Could not initialize wake-word engine (%s). Always-listening active.", e)
        return None


async def camera_video_loop(
    session, stop_event: asyncio.Event, state: ConversationState, cfg: Config,
    send_lock: asyncio.Lock,
):
    from camera_hub import camera_hub

    camera_hub.start()
    log.info("Camera Hub vision feed active.")

    try:
        while not stop_event.is_set():
            await asyncio.sleep(cfg.camera_interval_s)
            if stop_event.is_set():
                break

            if state.awake_event.is_set():
                if send_lock.locked():
                    continue

                frame = await asyncio.to_thread(camera_hub.get_frame)
                if frame is not None:
                    try:
                        h, w, _ = frame.shape
                        target_w = 480
                        target_h = int(target_w * (h / float(w)))
                        resized = cv2.resize(frame, (target_w, target_h))
                        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                        _, jpeg_buf = cv2.imencode(".jpg", rgb_frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
                        jpeg_bytes = jpeg_buf.tobytes()

                        async with send_lock:
                            await session.send_realtime_input(
                                video=types.Blob(data=jpeg_bytes, mime_type="image/jpeg")
                            )
                    except Exception as e:
                        log.warning("Camera frame streaming error: %s", e)
    finally:
        camera_hub.stop()
        log.info("Camera Hub vision feed stopped.")


async def run_session(cfg: Config, wake_model, hud: StatusHUD) -> None:
    client = genai.Client(api_key=cfg.api_key)

    live_config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=cfg.voice_name)
            )
        ),
        input_audio_transcription=types.AudioTranscriptionConfig(
            language_codes=["en-US"],
        ),
        system_instruction=types.Content(parts=[types.Part.from_text(text=get_system_prompt())]),
        tools=TOOLS,
    )

    state = ConversationState(start_awake=(wake_model is None))
    playback = AudioPlayback(cfg)
    send_lock = asyncio.Lock()

    log.info("Connecting to Jarvis Matrix (%s)...", cfg.model_name)
    try:
        async with client.aio.live.connect(model=cfg.model_name, config=live_config) as session:
            log.info("=" * 60)
            log.info("JARVIS Online on %s.", cfg.model_name)
            log.info("Dormant — say 'Hey Jarvis'." if wake_model is not None else "Hands-Free Audio Active.")
            log.info("=" * 60)

            stop_event = asyncio.Event()

            tasks = [
                asyncio.create_task(audio_input_loop(session, stop_event, state, wake_model, playback, hud, cfg, send_lock)),
                asyncio.create_task(playback.run(stop_event)),
                asyncio.create_task(receive_loop(session, stop_event, state, playback, hud, cfg, send_lock)),
            ]

            if cfg.camera_enabled:
                tasks.append(asyncio.create_task(camera_video_loop(session, stop_event, state, cfg, send_lock)))

            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
            stop_event.set()

            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

            for t in done:
                if not t.cancelled() and t.exception() is not None:
                    exc = t.exception()
                    log.error("Task error: %s", exc)
                    if _looks_like_auth_error(exc):
                        raise FatalError(str(exc)) from exc
                    raise exc

    except Exception as e:
        if _looks_like_auth_error(e):
            raise FatalError(f"API key error: {e}") from e
        raise


async def run_forever(cfg: Config) -> None:
    wake_model = init_wake_word()
    hud = StatusHUD(enabled=True)
    hud.start()
    delay = cfg.reconnect_initial_delay_s

    while True:
        try:
            await run_session(cfg, wake_model, hud)
            delay = cfg.reconnect_initial_delay_s
            log.info("Session ended — reconnecting.")
        except FatalError as e:
            log.critical("FATAL: %s. Terminating.", e)
            hud.stop()
            sys.exit(1)
        except asyncio.CancelledError:
            log.info("Shutting down JARVIS.")
            hud.stop()
            break
        except Exception as e:
            log.error("Session error (%s). Reconnecting in %ss...", e, int(delay))
            await asyncio.sleep(delay)
            delay = min(delay * 2.0, cfg.reconnect_max_delay_s)


def main():
    try:
        cfg = Config.from_env()
    except FatalError as e:
        print(f"FATAL CONFIG ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    setup_logging(cfg)
    log.info("Starting JARVIS Neural Assistant...")

    try:
        asyncio.run(run_forever(cfg))
    except KeyboardInterrupt:
        log.info("JARVIS shut down by user.")


if __name__ == "__main__":
    main()
