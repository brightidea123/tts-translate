"""Presentation timing from captured PCM, independent of model latency."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Mapping


def _integer(value, *, minimum: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def capture_clip_timing(provenance: Mapping, wall_clock_offset_ns: int) -> dict:
    timing = {}
    if not isinstance(provenance, Mapping):
        return timing
    offset = provenance.get("capture_wall_clock_offset_ns")
    if isinstance(offset, int) and not isinstance(offset, bool):
        wall_clock_offset_ns = offset
    # Sliding previews carry both their own window and full-phrase provenance.
    # Prefer the latter so the label never moves to the newest window start.
    start = provenance.get("utterance_start_ns", provenance.get("timestamp_start_ns"))
    samples = provenance.get("utterance_samples", provenance.get("samples"))
    rate = provenance.get("sample_rate")
    if _integer(start):
        try:
            timing["audio_started_at"] = datetime.fromtimestamp(
                (start + wall_clock_offset_ns) / 1_000_000_000, timezone.utc,
            ).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        except (ValueError, OverflowError, OSError):
            pass
    if _integer(samples) and _integer(rate, minimum=1):
        timing["audio_duration_seconds"] = samples / rate
    return timing
