"""Small JSONL bridge between the live core and its local web controller.

No model, server or application imports belong here. The core keeps command
polling on its control thread; status/model threads may publish concurrently.
"""

from __future__ import annotations

import codecs
from collections import deque
from copy import deepcopy
import json
import math
import os
import re
import select
import threading
from typing import Any, Callable, Mapping
from .headless_output import JsonOutputWriter


_ACTIONS = frozenset({"reply", "flush", "mark", "stop", "lookup"})
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
_LABEL = re.compile(
    r"^(?:[ \t]*(?:(?:🗣\ufe0f?|🌍|⌨\ufe0f?)[ \t]*)?)"
    r"(?:AR|HE|DE|TR|FR|RU|EN|ES)(?:[~✓]?(?:\[\d+/\d+\])?)"
    r":[ \t]*"
)


def _plain(value: Any, *, label: bool = False) -> str:
    text = _ANSI.sub("", "" if value is None else str(value))
    return _LABEL.sub("", text, count=1) if label else text


def _request_id(value: Any) -> str | int | None:
    if isinstance(value, str) and 0 < len(value) <= 128 and not any(ord(char) < 32 or ord(char) == 127 for char in value):
        return value
    if isinstance(value, int) and not isinstance(value, bool) and abs(value) <= 2**53 - 1:
        return value
    return None


class HeadlessTransport:
    """Bounded line parsing and atomic JSONL output, without TextIO read-ahead.

    ``poll`` is called by one control thread. It processes at most 16 complete
    commands per invocation, including commands already buffered from an earlier
    OS read. It returns True after an acknowledged stop or after EOF is drained.
    ``publish`` may be called by any thread; each state/ack is one complete line.
    """

    def __init__(self, input_fd: int, output_stream, on_command: Callable[[dict], dict], maxline: int = 65536,
                 *, async_output=False, state_interval=0.1, critical_capacity=64):
        if not isinstance(input_fd, int) or input_fd < 0:
            raise ValueError("input_fd must be an open nonnegative file descriptor")
        if not callable(on_command):
            raise TypeError("on_command must be callable")
        if not isinstance(maxline, int) or maxline < 1:
            raise ValueError("maxline must be a positive byte limit")
        self.input_fd = input_fd
        self.output_stream = output_stream
        self.on_command = on_command
        self.maxline = maxline
        self._output_lock = threading.Lock()
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
        self._line_parts: list[str] = []
        self._line_bytes = 0
        self._line_error = ""
        self._ready: deque[tuple[str | None, str]] = deque()
        self._eof = False
        self._stopped = False
        self._output_failed = False
        self._writer = JsonOutputWriter(output_stream, interval=state_interval, capacity=critical_capacity) if async_output else None

    def _emit(self, payload: dict) -> bool:
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
        with self._output_lock:
            if self._output_failed:
                return False
            try:
                self.output_stream.write(encoded)
                self.output_stream.flush()
            except (BrokenPipeError, OSError, ValueError):
                self._output_failed = True
                return False
        return True

    def publish(self, state: Mapping[str, Any]) -> bool:
        packet = {"type": "state", "state": state}
        return self._writer.offer(packet, state=True) if self._writer else self._emit(packet)

    def send_packet(self, packet: dict) -> bool:
        return self._writer.offer(packet) if self._writer else self._emit(packet)

    @property
    def snapshot(self):
        return self._writer.snapshot if self._writer else {"output_failed": self._output_failed}

    def close(self, timeout=1):
        if self._writer:
            self._writer.close(timeout)

    def _ack(self, request_id, accepted: bool, message: str = "") -> None:
        self.send_packet({"type": "ack", "request_id": request_id,
                    "accepted": accepted, "message": str(message)[:4096]})

    def _finish_line(self) -> None:
        if not self._line_error:
            try:
                self._line_parts.append(self._decoder.decode(b"", final=True))
            except UnicodeDecodeError:
                self._line_error = "Command is not valid UTF-8"
        self._ready.append((None if self._line_error else "".join(self._line_parts), self._line_error))
        self._decoder.reset()
        self._line_parts.clear()
        self._line_bytes = 0
        self._line_error = ""

    def _ingest(self, payload: bytes) -> None:
        segments = payload.split(b"\n")
        for index, segment in enumerate(segments):
            self._line_bytes += len(segment)
            if not self._line_error and self._line_bytes > self.maxline:
                self._line_error = f"Command exceeds the {self.maxline}-byte line limit"
                self._line_parts.clear()
                self._decoder.reset()
            if not self._line_error:
                try:
                    decoded = self._decoder.decode(segment)
                    if decoded:
                        self._line_parts.append(decoded)
                except UnicodeDecodeError:
                    self._line_error = "Command is not valid UTF-8"
                    self._line_parts.clear()
                    self._decoder.reset()
            if index < len(segments) - 1:
                self._finish_line()

    def _handle(self, line: str | None, error: str) -> bool:
        if error:
            self._ack(None, False, error)
            return False
        try:
            command = json.loads(line or "")
        except (json.JSONDecodeError, TypeError, ValueError):
            self._ack(None, False, "Expected one JSON command object per line")
            return False
        if not isinstance(command, dict):
            self._ack(None, False, "Command must be a JSON object")
            return False
        request_id = _request_id(command.get("request_id"))
        if request_id is None:
            self._ack(None, False, "request_id must be a nonempty bounded string or safe integer")
            return False
        action = command.get("action")
        if not isinstance(action, str) or len(action) > 32 or action not in _ACTIONS:
            self._ack(request_id, False, "Unsupported action; use reply, flush, mark, lookup, or stop")
            return False
        try:
            result = self.on_command(command)
            if not isinstance(result, dict):
                raise TypeError("Command handler did not return a result object")
            accepted = result.get("accepted", result.get("ok", False)) is True
            message = result.get("message", result.get("error", ""))
        except Exception as exc:
            accepted, message, result = False, f"{type(exc).__name__}: {exc}", {}
        self._ack(request_id, accepted, str(message))
        return action == "stop" or (accepted and result.get("stop") is True)

    def poll(self, timeout: float = 0.05) -> bool:
        if self._stopped or self._output_failed or self._writer and self._writer.failed:
            return True
        if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout < 0:
            raise ValueError("timeout must be a finite nonnegative number")
        if not self._ready and not self._eof:
            try:
                readable = select.select([self.input_fd], [], [], timeout)[0]
                if readable:
                    payload = os.read(self.input_fd, 65536)
                    if payload:
                        self._ingest(payload)
                    else:
                        self._eof = True
                        if self._line_bytes or self._line_parts or self._line_error:
                            self._finish_line()
            except InterruptedError:
                return False
            except (OSError, ValueError):
                self._eof = True
                if self._line_bytes or self._line_parts or self._line_error:
                    self._finish_line()
        for _ in range(16):
            if not self._ready:
                break
            if self._handle(*self._ready.popleft()):
                self._stopped = True
                return True
        if self._eof and not self._ready:
            self._stopped = True
        return self._stopped or self._output_failed


def _panel_messages(ui_state: Mapping[str, Any], side: str, source_language: str, target_language: str) -> list[dict]:
    output: list[dict] = []
    done = ui_state.get(f"{side}_done", [])
    done = done if isinstance(done, (list, tuple)) else []
    first = max(0, len(done) - 100)

    def add(entry: Mapping[str, Any], identity: str, *, finalized: bool) -> None:
        message = {
            "id": identity,
            "source": _plain(entry.get("src", entry.get("source", "")), label=True),
            "translation": _plain(entry.get("dst", entry.get("translation", "")), label=True),
            "source_final": bool(entry.get("src_final", entry.get("source_final", finalized))),
            "translation_final": bool(entry.get("dst_final", entry.get("translation_final", finalized))),
            "source_language": source_language,
            "target_language": target_language,
        }
        if side == "left":
            for key in ("audio_started_at", "audio_duration_seconds"):
                if key in entry:
                    message[key] = entry[key]
        preview = entry.get("source_preview")
        if isinstance(preview, Mapping) and not message["source_final"]:
            text = _plain(preview.get("text", ""))
            stable = _plain(preview.get("stable", ""))
            if not text.startswith(stable):
                stable = ""
            message.update(source=text, source_stable=stable,
                           source_provisional=text[len(stable):],
                           source_inflight=bool(preview.get("inflight")))
        # A refined active entry replaces a same-ID historical entry without
        # inventing a second conversation turn or mutating the application's UI.
        output[:] = [old for old in output if old["id"] != identity]
        output.append(message)

    for index, entry in enumerate(done[first:], start=first):
        if isinstance(entry, Mapping):
            identity = str(entry.get("entry_id", entry.get("id", f"done-{index}")))
            add(entry, identity, finalized=True)
    active = ui_state.get(f"{side}_active", {})
    if not isinstance(active, Mapping):
        active = {}
    order = ui_state.get(f"{side}_order", [])
    order = list(order) if isinstance(order, (list, tuple)) else []
    known = {str(key) for key in order}
    order.extend(key for key in active if str(key) not in known)
    for key in order[-20:]:
        entry = active.get(key, active.get(str(key)))
        if isinstance(entry, Mapping):
            add(entry, str(entry.get("entry_id", entry.get("id", key))), finalized=False)
    return output


def build_web_snapshot(ui_state: Mapping[str, Any], config: Mapping[str, Any], session_id,
                       running: bool, audio_review_mode: str, recording_directory) -> dict:
    """Copy a bounded, presentation-free view while the caller holds its UI lock.

    Logical Unicode is preserved. Only terminal escape sequences and the known
    leading console language label are removed; text, markup and punctuation
    are never reordered, normalized or interpreted as HTML.
    """
    source = str(config.get("source_language", "TR"))
    target = str(config.get("target_language", "EN"))
    stages = ui_state.get("stages", {})
    return {
        "session_id": None if session_id is None else str(session_id),
        "config": deepcopy(dict(config)),
        "stages": deepcopy(dict(stages)) if isinstance(stages, Mapping) else {},
        "audio_review_mode": str(audio_review_mode),
        "recording_directory": None if recording_directory is None else str(recording_directory),
        "transcript": _panel_messages(ui_state, "left", source, target),
        "replies": _panel_messages(ui_state, "right", target, source),
        "status": _plain(ui_state.get("status", "")),
        "running": bool(running),
    }
