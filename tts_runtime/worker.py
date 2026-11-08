"""Supervised inference processes with bounded, nonblocking parent mailboxes.

The capture/UI threads only call ``submit``. Model initialization, inference,
serialization, IPC, and watchdog checks run elsewhere. Committed work is FIFO;
queued partials may be replaced or evicted. A full committed mailbox rejects
new work explicitly instead of blocking a microphone or keyboard caller.

Initializer and handler functions must be importable multiprocessing-spawn
targets. The initializer configures child-local state, then the handler receives
``(payload, emit)``. ``emit(result)`` may be called repeatedly for streaming
results. Parent callbacks run on a dedicated delivery thread, so a slow UI
callback cannot stop the process watchdog or make submit wait.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import multiprocessing as mp
import pickle
from queue import Empty, Full, Queue
import threading
import time
import traceback
from typing import Any, Callable, Hashable


@dataclass(frozen=True, slots=True)
class StageEvent:
    kind: str
    stage: str
    state: str
    detail: str = ""
    job_id: int | None = None
    key: Hashable | None = None
    partial: bool = False
    background: bool = False
    queue_wait_seconds: float = 0.0
    request_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class _Job:
    job_id: int
    payload: Any
    partial: bool
    key: Hashable | None
    background: bool = False
    queued_at: float = field(default_factory=time.monotonic)


def _stage_process_main(name, initializer, handler, initargs, initkwargs, input_q, output_q):
    """Child owns models; only serialized bytes cross the queue feeder threads."""
    def send(kind, **values):
        # Serializing here makes an unpicklable result a reported job error,
        # rather than an asynchronous Queue feeder failure and silent hang.
        output_q.put(pickle.dumps({"kind": kind, **values}, protocol=pickle.HIGHEST_PROTOCOL))

    try:
        initializer(*initargs, **initkwargs)
    except BaseException:
        send("failed", detail=f"{name} initialization failed:\n{traceback.format_exc(limit=12)}")
        return
    send("ready")
    while True:
        encoded = input_q.get()
        if encoded is None:
            return
        job = pickle.loads(encoded)

        def emit(result):
            send("result", job_id=job.job_id, result=result)

        try:
            handler(job.payload, emit)
        except BaseException:
            # A failed native initializer can poison a process's extension
            # state. Never accept another request after a handler exception.
            send("failed", job_id=job.job_id,
                 detail=f"{name} inference failed:\n{traceback.format_exc(limit=12)}")
            return
        send("done", job_id=job.job_id)


class ProcessStage:
    """One supervised model process and a bounded nonblocking input mailbox.

    ``capacity`` bounds queued jobs; at most one additional job is in flight.
    ``submit`` returns False when closed/failed/full, with a rejected event.
    The caller should visibly retain/retry rejected committed input as needed.
    It must not assume that every accepted job has already finished inference.

    State values are new, loading, ready, busy, failed, and closed. A failed
    stage is terminal: create a fresh instance to retry native initialization.
    ``snapshot`` is always readable even if a parent callback is stuck.
    """

    def __init__(
        self, name: str, initializer: Callable, handler: Callable, *,
        initargs: tuple = (), initkwargs: dict | None = None,
        capacity: int = 32, startup_timeout: float = 180,
        request_timeout: float = 120, on_event: Callable[[StageEvent], None] | None = None,
        on_result: Callable[[Any], None] | None = None,
    ):
        if capacity < 1 or startup_timeout <= 0 or request_timeout <= 0:
            raise ValueError("capacity and worker timeouts must be positive")
        self.name, self.initializer, self.handler = name, initializer, handler
        self.initargs, self.initkwargs = tuple(initargs), dict(initkwargs or {})
        self.capacity = int(capacity)
        self.startup_timeout, self.request_timeout = float(startup_timeout), float(request_timeout)
        self.on_event, self.on_result = on_event, on_result
        self._condition = threading.Condition()
        self._committed: deque[_Job] = deque()
        self._partials: deque[_Job] = deque()
        self._background: deque[_Job] = deque()
        self._active: _Job | None = None
        self._superseded: set[int] = set()
        self._next_id = 0
        self._state, self._error = "new", ""
        self._started_at = self._request_started_at = 0.0
        self._last_request_seconds = self._request_ewma_seconds = 0.0
        self._last_queue_wait_seconds = 0.0
        self._dropped_partials = self._rejected_jobs = self._completed_jobs = 0
        self._dropped_background = 0
        self._callback_errors = 0
        self._last_callback_error = ""
        self._stop = threading.Event()
        self._notifications: Queue = Queue()
        self._pending_deliveries = 0
        self._delivery_limit = max(64, self.capacity * 4)
        self._process = self._input_q = self._output_q = None
        self._controller = self._delivery = None

    @property
    def state(self) -> str:
        with self._condition:
            return self._state

    @property
    def error(self) -> str:
        with self._condition:
            return self._error

    @property
    def pending_count(self) -> int:
        with self._condition:
            return len(self._committed) + len(self._partials) + len(self._background)

    @property
    def snapshot(self) -> dict:
        with self._condition:
            active = self._active
            return {
                "name": self.name, "state": self._state, "error": self._error,
                "pending": len(self._committed) + len(self._partials) + len(self._background),
                "committed_pending": len(self._committed), "partials_pending": len(self._partials),
                "background_pending": len(self._background),
                "inflight_partial": bool(active and active.partial),
                "inflight_background": bool(active and active.background),
                "last_request_seconds": self._last_request_seconds,
                "request_ewma_seconds": self._request_ewma_seconds,
                "last_queue_wait_seconds": self._last_queue_wait_seconds,
                "inflight": active.job_id if active else None,
                "inflight_key": active.key if active else None,
                "dropped_partials": self._dropped_partials,
                "dropped_background": self._dropped_background,
                "rejected_jobs": self._rejected_jobs, "completed_jobs": self._completed_jobs,
                "callback_errors": self._callback_errors,
                "last_callback_error": self._last_callback_error,
                "pending_callbacks": self._pending_deliveries,
                "pid": self._process.pid if self._process is not None else None,
            }

    def _notify(self, kind: str, detail: str = "", job: _Job | None = None) -> None:
        with self._condition:
            event = StageEvent(kind, self.name, self._state, detail,
                               job.job_id if job else None, job.key if job else None,
                               job.partial if job else False, job.background if job else False,
                               self._last_queue_wait_seconds if kind in {"done", "busy"} else 0.0,
                               self._last_request_seconds if kind == "done" else 0.0)
        self._deliver("event", event)

    def _deliver(self, kind: str, payload) -> None:
        if (kind == "event" and self.on_event is None) or (kind == "result" and self.on_result is None):
            return
        with self._condition:
            self._pending_deliveries += 1
        self._notifications.put_nowait((kind, payload))

    def start(self) -> "ProcessStage":
        with self._condition:
            if self._state != "new":
                raise RuntimeError(f"{self.name} has already been started or closed")
            self._state = "loading"
            self._started_at = time.monotonic()
        self._delivery = threading.Thread(target=self._delivery_loop,
                                          name=f"{self.name}-results", daemon=True)
        self._delivery.start()
        self._notify("loading", f"Loading {self.name} model")
        try:
            context = mp.get_context("spawn")
            self._input_q = context.Queue(maxsize=1)
            self._output_q = context.Queue(maxsize=64)
            self._process = context.Process(
                target=_stage_process_main,
                args=(self.name, self.initializer, self.handler, self.initargs, self.initkwargs,
                      self._input_q, self._output_q), name=f"{self.name}-model", daemon=True,
            )
            self._process.start()
        except Exception as exc:
            self._fail(f"Could not start {self.name}: {type(exc).__name__}: {exc}")
            return self
        self._controller = threading.Thread(target=self._controller_loop,
                                            name=f"{self.name}-control", daemon=True)
        self._controller.start()
        return self

    def submit(self, payload, partial: bool = False, key: Hashable | None = None, *, background: bool = False) -> bool:
        """Queue without IPC/serialization/waits in the submitting thread."""
        if key is not None:
            hash(key)
        discarded = []
        with self._condition:
            self._next_id += 1
            if background and partial:
                raise ValueError("Background work and a live partial are separate lanes")
            job = _Job(self._next_id, payload, bool(partial), key, bool(background))
            reason = ""
            if self._state not in {"loading", "ready", "busy"}:
                reason = self._error or f"{self.name} is {self._state}"
            elif background:
                # A hover/selection request must never evict live input. Keep
                # only the newest queued lookup, with explicit discard events.
                if len(self._committed) + len(self._partials) >= self.capacity:
                    reason = "Live work occupies the model mailbox; try the selection again shortly"
                else:
                    discarded.extend(self._background)
                    self._background.clear()
                    self._background.append(job)
                    self._condition.notify_all()
            elif partial and key is not None and (
                any(old.key == key for old in self._committed)
                or self._active is not None and not self._active.partial and self._active.key == key
            ):
                reason = "Committed work already supersedes this partial"
            else:
                if key is not None:
                    kept = deque()
                    for old in self._partials:
                        if old.key == key:
                            discarded.append(old)
                        else:
                            kept.append(old)
                    self._partials = kept
                while len(self._committed) + len(self._partials) + len(self._background) >= self.capacity and self._background:
                    discarded.append(self._background.popleft())
                while len(self._committed) + len(self._partials) + len(self._background) >= self.capacity and self._partials:
                    discarded.append(self._partials.popleft())
                if len(self._committed) + len(self._partials) >= self.capacity:
                    reason = f"{self.name} mailbox is full of committed work ({self.capacity} queued)"
                else:
                    (self._partials if partial else self._committed).append(job)
                    if key is not None and self._active and self._active.partial and self._active.key == key:
                        self._superseded.add(self._active.job_id)
                    self._condition.notify_all()
            self._dropped_partials += sum(not old.background for old in discarded)
            self._dropped_background += sum(old.background for old in discarded)
            if reason:
                self._rejected_jobs += 1
        for old in discarded:
            self._notify("discarded", "Superseded speculative work", old)
        self._notify("rejected" if reason else "queued", reason, job)
        return not reason

    def _fail(self, detail: str) -> None:
        with self._condition:
            if self._state in {"failed", "closed"}:
                return
            self._state, self._error = "failed", detail
            active = self._active
            pending = list(self._committed) + list(self._partials) + list(self._background)
            self._committed.clear()
            self._partials.clear()
            self._background.clear()
            self._active = None
            self._superseded.clear()
            self._rejected_jobs += len(pending)
            self._condition.notify_all()
        self._notify("failed", detail, active)
        for job in pending:
            self._notify("rejected", f"{self.name} failed before this job could run", job)

    def _controller_loop(self) -> None:
        while not self._stop.is_set():
            try:
                encoded = self._output_q.get(timeout=0.025)
            except Empty:
                encoded = None
            except (EOFError, OSError, ValueError) as exc:
                if not self._stop.is_set():
                    self._fail(f"{self.name} result channel closed: {exc}")
                break
            if encoded is not None:
                try:
                    self._receive(pickle.loads(encoded))
                except Exception as exc:
                    self._fail(f"{self.name} result protocol failed: {type(exc).__name__}: {exc}")
            with self._condition:
                state, active = self._state, self._active
                now = time.monotonic()
                if state in {"failed", "closed"}:
                    break
                startup_expired = state == "loading" and now - self._started_at > self.startup_timeout
                request_expired = active is not None and now - self._request_started_at > self.request_timeout
            if startup_expired:
                self._fail(f"{self.name} model initialization exceeded {self.startup_timeout:g}s; check model files and runtime logs")
                break
            if request_expired:
                self._fail(f"{self.name} inference exceeded {self.request_timeout:g}s; model worker was stopped")
                break
            if not self._process.is_alive():
                # A child sends its error before exiting. A final nonblocking
                # drain preserves that detail when the Queue feeder won the race.
                try:
                    while True:
                        self._receive(pickle.loads(self._output_q.get_nowait()))
                except Empty:
                    pass
                with self._condition:
                    state = self._state
                if state != "failed":
                    self._fail(f"{self.name} worker exited unexpectedly (exit code {self._process.exitcode})")
                break
            self._dispatch_one()
        if self.state == "failed" and self._process is not None and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=0.3)

    def _dispatch_one(self) -> None:
        with self._condition:
            if self._state != "ready" or self._active is not None or self._pending_deliveries >= self._delivery_limit:
                return
            if self._committed:
                job = self._committed.popleft()
            elif self._partials:
                job = self._partials.popleft()
            elif self._background:
                job = self._background.popleft()
            else:
                return
            self._active = job
            self._state = "busy"
            self._request_started_at = time.monotonic()
            self._last_queue_wait_seconds = max(0.0, self._request_started_at - job.queued_at)
        try:
            encoded = pickle.dumps(job, protocol=pickle.HIGHEST_PROTOCOL)
            self._input_q.put(encoded, timeout=0.05)
        except Exception as exc:
            self._fail(f"{self.name} could not send inference job: {type(exc).__name__}: {exc}")
            return
        self._notify("busy", "Processing", job)

    def _receive(self, message: dict) -> None:
        kind = message.get("kind")
        if kind == "ready":
            with self._condition:
                if self._state != "loading":
                    return
                self._state = "ready"
            self._notify("ready", f"{self.name} model ready")
        elif kind == "failed":
            self._fail(message.get("detail") or f"{self.name} model worker failed")
        elif kind in {"result", "done"}:
            with self._condition:
                job = self._active
                if job is None or message.get("job_id") != job.job_id:
                    raise RuntimeError("Result does not belong to the in-flight request")
                superseded = job.job_id in self._superseded
                if kind == "done":
                    elapsed = max(0.0, time.monotonic() - self._request_started_at)
                    self._last_request_seconds = elapsed
                    self._request_ewma_seconds = (0.8 * self._request_ewma_seconds + 0.2 * elapsed
                                                 if self._completed_jobs else elapsed)
                    self._active = None
                    self._superseded.discard(job.job_id)
                    self._completed_jobs += 1
                    self._state = "ready"
            if kind == "result" and not superseded:
                self._deliver("result", message["result"])
            elif kind == "done":
                self._notify("discarded" if superseded else "done",
                             "Superseded partial result" if superseded else "Inference completed", job)
        else:
            raise RuntimeError(f"Unknown worker message {kind!r}")

    def _delivery_loop(self) -> None:
        while True:
            item = self._notifications.get()
            if item is None:
                return
            kind, payload = item
            try:
                callback = self.on_event if kind == "event" else self.on_result
                if callback is not None:
                    callback(payload)
            except Exception as exc:
                with self._condition:
                    self._callback_errors += 1
                    self._last_callback_error = f"{type(exc).__name__}: {exc}"
                if kind != "event":
                    self._notify("callback_error", f"{self.name} result callback failed: {exc}")
            finally:
                with self._condition:
                    self._pending_deliveries -= 1

    def close(self, timeout: float = 2.0) -> None:
        """Cancel pending/in-flight work and release the process within a budget."""
        deadline = time.monotonic() + max(0.0, timeout)
        with self._condition:
            if self._state == "closed":
                return
            self._state = "closed"
            active = self._active
            pending = list(self._committed) + list(self._partials) + list(self._background)
            self._committed.clear()
            self._partials.clear()
            self._background.clear()
            self._active = None
            self._superseded.clear()
            self._stop.set()
            self._condition.notify_all()
        for job in pending:
            self._notify("rejected", f"{self.name} closed before this job could run", job)
        self._notify("closed", "Canceled unfinished work" if active or pending else "Stopped", active)
        if self._input_q is not None:
            try:
                self._input_q.put_nowait(None)
            except (Full, OSError, ValueError):
                pass
        if self._process is not None and self._process.pid is not None:
            self._process.join(timeout=min(0.1, max(0.0, deadline - time.monotonic())))
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=min(0.3, max(0.0, deadline - time.monotonic())))
            if self._process.is_alive():
                self._process.kill()
        if self._controller is not None and threading.current_thread() is not self._controller:
            self._controller.join(timeout=min(0.2, max(0.0, deadline - time.monotonic())))
        self._notifications.put_nowait(None)
        if self._delivery is not None and threading.current_thread() is not self._delivery:
            self._delivery.join(timeout=max(0.0, deadline - time.monotonic()))
        for channel in (self._input_q, self._output_q):
            if channel is not None:
                channel.cancel_join_thread()
                channel.close()

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, tb):
        self.close()
