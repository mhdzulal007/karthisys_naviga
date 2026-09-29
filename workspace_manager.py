import json
import datetime
import logging
import os
import threading
from typing import List, Dict, Any

log = logging.getLogger("jarvis.workspace")

# [FIX] Configurable root instead of a hardcoded relative path, so behavior
# doesn't depend on the working directory the process happens to launch from.
BASE_WORKSPACE = os.path.abspath(os.environ.get("JARVIS_WORKSPACE_DIR", "./workspace"))
os.makedirs(BASE_WORKSPACE, exist_ok=True)

MAX_FILE_BYTES = int(os.environ.get("JARVIS_MAX_FILE_BYTES", 2_000_000))  # 2 MB default cap


class WorkspaceManager:
    def __init__(self, base_path: str = BASE_WORKSPACE):
        self.base_path = base_path
        self.current_project = "default"
        # [FIX] Guards current_project + the chat-history read-modify-write.
        # Tool calls run in a thread pool (asyncio.to_thread) upstream, so this
        # state should not assume it's only ever touched from one thread.
        self._lock = threading.Lock()
        self._ensure_project_dir(self.current_project)

    def _sanitize(self, name: str) -> str:
        clean_name = "".join(c for c in name if c.isalnum() or c in ("-", "_")).strip()
        return clean_name or "default"

    def _get_project_path(self, project_name: str) -> str:
        return os.path.join(self.base_path, self._sanitize(project_name))

    def _ensure_project_dir(self, project_name: str) -> str:
        path = self._get_project_path(project_name)
        os.makedirs(path, exist_ok=True)
        return path

    def switch_project(self, project_name: str) -> str:
        """Creates (if needed) and switches to an active project workspace."""
        clean_name = self._sanitize(project_name)
        with self._lock:
            # [FIX] Store the sanitized name, not the raw input, so what we
            # report back always matches the directory actually in use.
            self.current_project = clean_name
            proj_dir = self._ensure_project_dir(clean_name)
            files = self._list_project_files_locked()

        return (
            f"Successfully switched to project '{clean_name}'. "
            f"Directory path: {proj_dir}. "
            f"Existing files: {files}"
        )

    def write_file(self, filename: str, content: str) -> str:
        """Writes or updates a document, plan, or note inside the current active project."""
        if len(content.encode("utf-8", errors="ignore")) > MAX_FILE_BYTES:
            return (
                f"Error: content is larger than the {MAX_FILE_BYTES} byte limit "
                f"for '{filename}' - not written."
            )

        clean_filename = os.path.basename(filename)
        if not clean_filename:
            return "Error: no valid filename given."

        with self._lock:
            proj_dir = self._ensure_project_dir(self.current_project)
            file_path = os.path.join(proj_dir, clean_filename)
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
            except OSError as e:
                log.error("Failed writing %s: %s", file_path, e)
                return f"Error: could not write '{clean_filename}' ({e})."
            project = self.current_project

        return f"File '{clean_filename}' saved successfully in project '{project}'."

    def read_file(self, filename: str) -> str:
        """Reads and returns the contents of a file from the current active project."""
        clean_filename = os.path.basename(filename)
        if not clean_filename:
            return "Error: no valid filename given."

        with self._lock:
            proj_dir = self._ensure_project_dir(self.current_project)
            file_path = os.path.join(proj_dir, clean_filename)
            project = self.current_project

            if not os.path.exists(file_path):
                return f"Error: File '{clean_filename}' does not exist in project '{project}'."

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
            except OSError as e:
                log.error("Failed reading %s: %s", file_path, e)
                return f"Error: could not read '{clean_filename}' ({e})."

    def _list_project_files_locked(self) -> List[str]:
        """Assumes self._lock is already held by the caller."""
        proj_dir = self._ensure_project_dir(self.current_project)
        items = os.listdir(proj_dir)
        return sorted(item for item in items if not item.startswith("."))

    def list_project_files(self) -> List[str]:
        """Returns a sorted list of all documents/files in the active project."""
        with self._lock:
            return self._list_project_files_locked()

    def save_chat_turn(self, role: str, text: str):
        """Appends a conversational turn to the project's persistent history log."""
        with self._lock:
            proj_dir = self._ensure_project_dir(self.current_project)
            history_file = os.path.join(proj_dir, ".chat_history.json")

            history = []
            if os.path.exists(history_file):
                try:
                    with open(history_file, "r", encoding="utf-8") as f:
                        history = json.load(f)
                except (OSError, json.JSONDecodeError) as e:
                    log.warning("Could not read history file %s (%s) - starting fresh.", history_file, e)
                    history = []

            history.append({
                "timestamp": datetime.datetime.now().isoformat(),
                "role": role,
                "text": text,
            })

            try:
                with open(history_file, "w", encoding="utf-8") as f:
                    json.dump(history, f, indent=2)
            except OSError as e:
                log.error("Could not write history file %s: %s", history_file, e)

    def load_project_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieves the last N conversation turns from the active project."""
        with self._lock:
            proj_dir = self._ensure_project_dir(self.current_project)
            history_file = os.path.join(proj_dir, ".chat_history.json")

            if not os.path.exists(history_file):
                return []

            try:
                with open(history_file, "r", encoding="utf-8") as f:
                    history = json.load(f)
                    return history[-limit:]
            except (OSError, json.JSONDecodeError) as e:
                log.warning("Could not read history file %s (%s).", history_file, e)
                return []


# Singleton instance to be imported by the assistant runner
workspace = WorkspaceManager()