"""Own one isolated headless core session for the localhost web interface.

The web server never imports the stateful CLI or executes a model. Structured
JSONL commands and acknowledgments cross stdin/stdout; every new run has fresh
process globals and a generation guard against late messages from an old run.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from queue import Empty, Full, Queue
import signal
import subprocess
import sys
import threading
import time
from typing import Any
from uuid import uuid4

from .selection_requests import SelectionRequests


LANGUAGES = frozenset({"AR", "HE", "DE", "TR", "FR", "RU", "EN", "ES"})
_MAX_PROTOCOL_LINE = 2 * 1024 * 1024


def _close_stream(stream) -> None:
    close = getattr(stream, "close", None)
    if callable(close):
        try:
            close()
        except (OSError, ValueError):
            pass


@dataclass
class _Acknowledgment:
    request_id: str
    event: threading.Event = field(default_factory=threading.Event)
    sent: bool = False
    response: dict | None = None
    lookup_id: str | None = None
    created_at: float = field(default_factory=time.monotonic)


@dataclass
class _Run:
    process: Any
    generation: int
    run_id: str
    commands: Queue = field(default_factory=lambda: Queue(maxsize=32))
    pending: dict[str, _Acknowledgment] = field(default_factory=dict)
    stop: threading.Event = field(default_factory=threading.Event)
    stderr: deque = field(default_factory=lambda: deque(maxlen=40))
    threads: list[threading.Thread] = field(default_factory=list)
    stopping: bool = False


class WebController:
    """One owned subprocess with bounded command/acknowledgment handling."""

    def __init__(self, *, core_path: str | Path | None = None,
                 python_executable: str | None = None, ack_timeout: float = 3,
                 stop_timeout: float = 10, diagnostics_dir: str | Path | None = None,
                 initial_config: dict | None = None, core_arguments: list[str] | None = None):
        self.core_path = Path(core_path or Path(__file__).resolve().parents[1] / "tts-transcribe.py").resolve()
        self.python_executable = python_executable or sys.executable
        self.core_arguments = list(core_arguments or [])
        if any(not isinstance(value, str) or "\x00" in value for value in self.core_arguments):
            raise ValueError("Core arguments must be plain strings")
        if ack_timeout <= 0 or stop_timeout <= 0:
            raise ValueError("Controller timeouts must be positive")
        self.ack_timeout, self.stop_timeout = float(ack_timeout), float(stop_timeout)
        self.diagnostics_dir = Path(diagnostics_dir or self.core_path.parent / "runtime-reports/web-controller")
        self._lock = threading.RLock()
        self._lifecycle = threading.Lock()
        self._run: _Run | None = None
        self._generation = 0
        self._draft, self._draft_version = "", 0
        self._snapshot: dict = {"status": "Stopped", "stages": {}, "transcript": [], "replies": [],
                                "config": self._configuration(initial_config)}
        self._device_error = ""
        self._selections = SelectionRequests()

    def _expire_lookup_commands(self, run: _Run | None) -> None:
        if run is None:
            return
        now = time.monotonic()
        for request_id, pending in list(run.pending.items()):
            if pending.lookup_id is not None and now - pending.created_at >= self.ack_timeout:
                run.pending.pop(request_id, None)
                self._selections.fail(pending.lookup_id, "Selection command was not acknowledged; select the phrase again")

    def state(self) -> dict:
        with self._lock:
            result = deepcopy(self._snapshot)
            run = self._run
            self._expire_lookup_commands(run)
            code = run.process.poll() if run is not None else None
            result.update({"running": run is not None and code is None,
                           "stopping": bool(run and run.stopping and code is None),
                           "controller_run_id": run.run_id if run else None,
                           "returncode": code, "draft": self._draft,
                           "stderr_tail": list(run.stderr) if run else [],
                           "device_error": self._device_error})
            return result

    @staticmethod
    def _configuration(config: dict | None) -> dict:
        if config is None:
            config = {}
        if not isinstance(config, dict):
            raise ValueError("Session configuration must be an object")
        result = {}
        for kind, default in (("source", "TR"), ("target", "EN")):
            value = config.get(kind + "_language", config.get(kind, default))
            if not isinstance(value, str) or value.strip().upper() not in LANGUAGES:
                raise ValueError(f"Unsupported {kind} language")
            result[kind + "_language"] = value.strip().upper()
        if result["source_language"] == result["target_language"]:
            raise ValueError("Source and target languages must differ")
        device = config.get("input_device")
        if isinstance(device, bool) or not isinstance(device, (str, int, type(None))):
            raise ValueError("Input device must be a name, index, or null")
        if isinstance(device, str):
            if "\x00" in device or len(device) > 256:
                raise ValueError("Invalid input device name")
            device = device.strip() or None
        result["input_device"] = device
        debug = config.get("debug", False)
        if not isinstance(debug, bool):
            raise ValueError("Debug must be true or false")
        result["debug"] = debug
        review = config.get("audio_review", "auto")
        if not isinstance(review, str) or review not in {"auto", "off", "issues", "full"}:
            raise ValueError("Unsupported audio review mode")
        result["audio_review"] = review
        return result

    def start(self, config: dict | None = None) -> dict:
        with self._lock:
            seed = deepcopy(self._snapshot.get("config", {}))
        if config is not None and not isinstance(config, dict):
            raise ValueError("Session configuration must be an object")
        incoming = dict(config or {})
        for name in ("source", "target"):
            if name in incoming and name + "_language" not in incoming:
                incoming[name + "_language"] = incoming[name]
        config = {**seed, **incoming}
        config = self._configuration(config)
        if not self._lifecycle.acquire(blocking=False):
            return {**self.state(), "accepted": False, "message": "A session change is already in progress"}
        try:
            with self._lock:
                if self._run is not None and self._run.process.poll() is None:
                    return {**self.state(), "accepted": False, "message": "A session is already running; stop it before starting another"}
                self._run = None
                self._generation += 1
                generation, run_id = self._generation, uuid4().hex
                self._selections.reset(run_id)
                self._draft, self._draft_version = "", self._draft_version + 1
                self._snapshot = {"status": "Starting", "stages": {"capture": "starting", "asr": "loading", "translation": "loading"},
                                  "transcript": [], "replies": [], "config": deepcopy(config), "error": ""}
            command = [self.python_executable, "-u", str(self.core_path), "--ui", "headless",
                       *self.core_arguments,
                       "--source-language", config["source_language"], "--target-language", config["target_language"],
                       "--audio-review", config["audio_review"]]
            if config["input_device"] is not None:
                command.append("--input-device=" + str(config["input_device"]))
            if config["debug"]:
                command.append("--debug")
            environment = os.environ.copy()
            environment["PYTHONUNBUFFERED"] = "1"
            environment["TTS_WEB_RUN_ID"] = run_id
            try:
                process = subprocess.Popen(command, cwd=str(self.core_path.parent), env=environment,
                                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                           text=True, encoding="utf-8", errors="replace", bufsize=1,
                                           start_new_session=True)
            except Exception as exc:
                with self._lock:
                    self._run = None
                    self._selections.reset()
                    self._snapshot.update(status="Failed to start", error=f"{type(exc).__name__}: {exc}")
                return {**self.state(), "accepted": False, "message": str(exc)}
            run = _Run(process, generation, run_id)
            with self._lock:
                self._run = run
            for name, target in (("commands", self._writer), ("state", self._stdout), ("stderr", self._stderr), ("exit", self._waiter)):
                thread = threading.Thread(target=target, args=(run,), name=f"web-core-{name}-{run_id[:6]}", daemon=True)
                run.threads.append(thread)
                thread.start()
            return {**self.state(), "accepted": True, "message": "Session starting"}
        finally:
            self._lifecycle.release()

    def _packet(self, run: _Run, packet: dict) -> None:
        with self._lock:
            if self._run is not run or run.generation != self._generation:
                return
            if packet.get("type") == "state" and isinstance(packet.get("state"), dict):
                # The core snapshot is authoritative, but its 'running' field
                # cannot override the actual subprocess state in state().
                config = self._snapshot.get("config", {})
                self._snapshot = deepcopy(packet["state"])
                self._snapshot.setdefault("config", config)
            elif packet.get("type") == "ack":
                pending = run.pending.pop(str(packet.get("request_id", "")), None)
                if pending is not None:
                    pending.response = {"accepted": packet.get("accepted") is True,
                                        "message": str(packet.get("message", "")),
                                        "request_id": pending.request_id}
                    pending.event.set()
                    if pending.lookup_id is not None and packet.get("accepted") is not True:
                        self._selections.fail(pending.lookup_id, packet.get("message") or "Selection translation was not admitted")
            elif packet.get("type") == "lookup_result":
                self._selections.complete(run.run_id, packet)

    def _stdout(self, run: _Run) -> None:
        stream = run.process.stdout
        try:
            while True:
                line = stream.readline(_MAX_PROTOCOL_LINE + 1)
                if not line:
                    return
                if len(line) > _MAX_PROTOCOL_LINE:
                    while line and not line.endswith("\n"):
                        line = stream.readline(_MAX_PROTOCOL_LINE + 1)
                    continue
                try:
                    packet = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if isinstance(packet, dict):
                    self._packet(run, packet)
        except (OSError, ValueError):
            return
        finally:
            _close_stream(stream)

    def _stderr(self, run: _Run) -> None:
        written, log = 0, None
        try:
            directory = self.diagnostics_dir / run.run_id
            directory.mkdir(parents=True, exist_ok=True)
            log = (directory / "stderr.log").open("w", encoding="utf-8")
        except OSError:
            pass
        try:
            while True:
                line = run.process.stderr.readline(8192)
                if not line:
                    return
                with self._lock:
                    if self._run is run:
                        run.stderr.append(line.rstrip()[-2000:])
                if log is not None and written < 4 * 1024 * 1024:
                    log.write(line)
                    log.flush()
                    written += len(line.encode("utf-8"))
        except (OSError, ValueError):
            return
        finally:
            if log is not None:
                log.close()
            _close_stream(run.process.stderr)

    def _writer(self, run: _Run) -> None:
        try:
            while not run.stop.is_set():
                try:
                    command = run.commands.get(timeout=0.1)
                except Empty:
                    continue
                request_id = command["request_id"]
                with self._lock:
                    pending = run.pending.get(request_id)
                    if self._run is not run or pending is None:
                        continue
                    pending.sent = True
                try:
                    run.process.stdin.write(json.dumps(command, ensure_ascii=False) + "\n")
                    run.process.stdin.flush()
                except (BrokenPipeError, OSError, ValueError) as exc:
                    self._fail_pending(run, f"Core input channel closed: {exc}")
                    return
        finally:
            _close_stream(run.process.stdin)

    def _fail_pending(self, run: _Run, message: str) -> None:
        with self._lock:
            pending, run.pending = list(run.pending.values()), {}
            for item in pending:
                item.response = {"accepted": False, "message": message, "request_id": item.request_id}
                item.event.set()
                if item.lookup_id is not None:
                    self._selections.fail(item.lookup_id, message)

    def _waiter(self, run: _Run) -> None:
        code = run.process.wait()
        run.stop.set()
        self._fail_pending(run, "Session ended before the command was acknowledged")
        with self._lock:
            if self._run is run:
                self._selections.reset()
                self._snapshot["status"] = "Stopped" if run.stopping or code == 0 else "Session exited"
                if code != 0 and not run.stopping:
                    self._snapshot["error"] = f"Core exited with status {code}; see stderr_tail and local worker diagnostics"
        # A crashed parent can leave inference children in its owned session.
        # This group was created exclusively for this subprocess, never for the
        # web server, a terminal, or another independently started CLI session.
        self._signal(run, signal.SIGTERM)

    def _command(self, action: str, *, allow_stopping: bool = False, timeout: float | None = None, **values) -> dict:
        pending = _Acknowledgment(uuid4().hex)
        with self._lock:
            run = self._run
            if run is None or run.process.poll() is not None or (run.stopping and not allow_stopping):
                return {"accepted": False, "message": "No active session accepts this command"}
            run.pending[pending.request_id] = pending
            command = {"action": action, "request_id": pending.request_id, **values}
            try:
                run.commands.put_nowait(command)
            except Full:
                run.pending.pop(pending.request_id, None)
                return {"accepted": False, "message": "Command queue is full; input was not submitted"}
        pending.event.wait(self.ack_timeout if timeout is None else timeout)
        with self._lock:
            if pending.response is not None:
                return pending.response
            run.pending.pop(pending.request_id, None)
            return {"accepted": False, "uncertain": pending.sent,
                    "message": "Acknowledgment timed out; check the session before retrying" if pending.sent else "Command timed out before submission",
                    "request_id": pending.request_id}

    def reply(self, text: str) -> dict:
        if not isinstance(text, str) or not text.strip() or len(text) > 10000:
            raise ValueError("Reply must be nonempty text of at most 10000 characters")
        with self._lock:
            self._draft, self._draft_version = text, self._draft_version + 1
            version = self._draft_version
        result = self._command("reply", text=text)
        if result["accepted"]:
            with self._lock:
                if self._draft_version == version:
                    self._draft = ""
        return result

    def flush(self) -> dict:
        return self._command("flush")

    def selection_translation(self, config: dict) -> dict:
        """Admit optional lookup work without waiting for ACK/model execution."""
        # Validation also applies to callers outside the HTTP layer.
        SelectionRequests.validate(config)
        with self._lock:
            run = self._run
            self._expire_lookup_commands(run)
            if run is None or run.process.poll() is not None or run.stopping:
                return {"accepted": False, "message": "Start a session before translating a selection"}
            try:
                result, send_required, normalized = self._selections.reserve(config)
            except RuntimeError as exc:
                return {"accepted": False, "message": str(exc)}
            if send_required:
                pending = _Acknowledgment(uuid4().hex, lookup_id=result["lookup_id"])
                run.pending[pending.request_id] = pending
                command = {"action": "lookup", "request_id": pending.request_id,
                           "lookup_id": result["lookup_id"], **normalized}
                try:
                    run.commands.put_nowait(command)
                except Full:
                    run.pending.pop(pending.request_id, None)
                    self._selections.fail(result["lookup_id"], "Command queue is full; selection was not submitted")
                    return {**self._selections.result(result["lookup_id"]), "accepted": False,
                            "message": "Command queue is full; selection was not submitted"}
            return {**result, "accepted": True}

    def selection_translation_result(self, lookup_id: str) -> dict | None:
        with self._lock:
            run = self._run
            self._expire_lookup_commands(run)
            if run is None or run.stopping or run.process.poll() is not None:
                return None
            return self._selections.result(lookup_id)

    def mark(self, utterance_id: int | None = None, reason: str = "manual-review") -> dict:
        if utterance_id is not None and (isinstance(utterance_id, bool) or not isinstance(utterance_id, int) or utterance_id < 0):
            raise ValueError("Utterance ID must be a nonnegative integer")
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 1000:
            raise ValueError("Review reason must be nonempty text of at most 1000 characters")
        return self._command("mark", utterance_id=utterance_id, reason=reason)

    @staticmethod
    def _signal(run: _Run, sig) -> None:
        try:
            if os.name == "posix":
                os.killpg(run.process.pid, sig)
            elif run.process.poll() is None:
                run.process.send_signal(sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass

    def stop(self) -> dict:
        if not self._lifecycle.acquire(blocking=False):
            return {**self.state(), "accepted": False, "message": "A session change is already in progress"}
        try:
            with self._lock:
                run = self._run
                if run is None or run.process.poll() is not None:
                    return {**self.state(), "accepted": True, "message": "Session is stopped"}
                run.stopping = True
                self._selections.reset()
            deadline = time.monotonic() + self.stop_timeout
            self._command("stop", allow_stopping=True, timeout=min(self.ack_timeout, self.stop_timeout / 3))
            waits = [(None, max(0, deadline - time.monotonic() - 2)),
                     (signal.SIGINT, 0.6), (signal.SIGTERM, 0.6), (signal.SIGKILL, 0.6)]
            for sig, seconds in waits:
                if sig is not None:
                    self._signal(run, sig)
                try:
                    run.process.wait(timeout=seconds)
                    break
                except subprocess.TimeoutExpired:
                    continue
            run.stop.set()
            self._fail_pending(run, "Session stopped")
            for thread in run.threads:
                if thread is not threading.current_thread():
                    thread.join(timeout=0.1)
            return {**self.state(), "accepted": run.process.poll() is not None,
                    "message": "Session stopped" if run.process.poll() is not None else "Session is still stopping"}
        finally:
            self._lifecycle.release()

    def devices(self) -> list[dict]:
        try:
            import sounddevice as sd
            default = sd.default.device[0]
            devices = [{"id": index, "name": str(info["name"]),
                        "max_input_channels": int(info["max_input_channels"]),
                        "default_samplerate": float(info["default_samplerate"]), "default": index == default}
                       for index, info in enumerate(sd.query_devices()) if info["max_input_channels"] > 0]
            with self._lock:
                self._device_error = ""
            return devices
        except Exception as exc:
            with self._lock:
                self._device_error = f"Unable to enumerate input devices: {exc}"
            return []

    def close(self) -> None:
        self.stop()
        with self._lock:
            run = self._run
        if run is not None:
            # stop() can observe an already-exited child before its stream
            # readers finish. Join their short cleanup so descriptors/logs do
            # not survive controller shutdown or race temporary directories.
            for thread in run.threads:
                if thread is not threading.current_thread():
                    thread.join(timeout=0.2)
