"""Capture-clock timing for pause estimation and provisional text stability."""
from __future__ import annotations

from dataclasses import dataclass


def completed_pause_estimate(previous: float, pause_ms: int, *, base_ms: int, maximum_ms: int, alpha: float) -> float:
    if pause_ms <= 0:
        return previous
    # Brief phoneme closures must not train the deadline below the configured
    # baseline. Only a pause followed by more speech is an observed word gap;
    # an unfinished trailing pause never trains its own deadline.
    sample = min(maximum_ms, max(base_ms, pause_ms))
    return min(maximum_ms, max(base_ms, (1 - alpha) * previous + alpha * sample))


@dataclass
class PartialTextTiming:
    text: str = ""
    capture_end_ns: int | None = None
    changed_capture_end_ns: int | None = None
    changed_monotonic: float | None = None

    def update(self, text: str, capture_end_ns: int | None, now: float) -> None:
        if capture_end_ns is not None and self.capture_end_ns is not None and capture_end_ns < self.capture_end_ns:
            return
        if text != self.text or self.changed_monotonic is None:
            self.changed_capture_end_ns = capture_end_ns
            self.changed_monotonic = now
        # A revision without source provenance keeps its monotonic fallback;
        # a later timestamp cannot supply its missing original audio-time origin.
        self.text = text
        self.capture_end_ns = capture_end_ns

    def age_ms(self, now: float, captured_until_ns: int | None = None) -> float | None:
        if self.changed_monotonic is None:
            return None
        if captured_until_ns is not None and self.changed_capture_end_ns is not None:
            return max(0.0, (captured_until_ns - self.changed_capture_end_ns) / 1_000_000)
        if self.capture_end_ns is not None and self.changed_capture_end_ns is not None:
            return max(0.0, (self.capture_end_ns - self.changed_capture_end_ns) / 1_000_000)
        return max(0.0, (now - self.changed_monotonic) * 1000)
