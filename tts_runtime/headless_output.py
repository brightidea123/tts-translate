"""Bounded JSONL output: critical replies first, coalesced status at 10 Hz."""
from collections import deque
from copy import deepcopy
import json
import threading
import time


class JsonOutputWriter:
    def __init__(self, stream, *, interval=0.1, capacity=64):
        if interval <= 0 or capacity < 1:
            raise ValueError("Output interval and capacity must be positive")
        self.stream, self.interval, self.capacity = stream, interval, capacity
        self.condition = threading.Condition()
        self.critical, self.latest = deque(), None
        self.closed, self.failed = False, False
        self.coalesced, self.rejected, self.written = 0, 0, 0
        self.thread = threading.Thread(target=self._run, name="headless-output", daemon=True)
        self.thread.start()

    def offer(self, packet, *, state=False):
        with self.condition:
            if self.closed or self.failed:
                return False
            if state:
                self.coalesced += self.latest is not None
                self.latest = deepcopy(packet)
            elif len(self.critical) >= self.capacity:
                self.rejected += 1
                self.failed = True  # Stop input; never silently lose command replies.
                self.condition.notify_all()
                return False
            else:
                self.critical.append(deepcopy(packet))
            self.condition.notify_all()
        return True

    @property
    def snapshot(self):
        with self.condition:
            return {"state_coalesced": self.coalesced, "critical_rejected": self.rejected,
                    "critical_pending": len(self.critical), "state_pending": self.latest is not None,
                    "packets_written": self.written, "output_failed": self.failed,
                    "writer_alive": self.thread.is_alive()}

    def _run(self):
        next_state = 0.0
        while True:
            with self.condition:
                while True:
                    if self.failed:
                        return
                    if self.critical:
                        packet = self.critical.popleft()
                        break
                    remaining = next_state - time.monotonic()
                    if self.latest is not None and (remaining <= 0 or self.closed):
                        packet, self.latest = self.latest, None
                        next_state = time.monotonic() + self.interval
                        break
                    if self.closed:
                        return
                    self.condition.wait(remaining if self.latest is not None else None)
            try:
                line = json.dumps(packet, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
                self.stream.write(line)
                self.stream.flush()
            except Exception:
                with self.condition:
                    self.failed = True
                return
            with self.condition:
                self.written += 1

    def close(self, timeout=1):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        if threading.current_thread() is not self.thread:
            self.thread.join(max(0, timeout))
