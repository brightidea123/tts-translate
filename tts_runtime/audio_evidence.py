"""Bounded asynchronous audio evidence with immutable payloads and source offsets.

Capture callers never open files or wait for the writer. Full-stream recording
is opt-in; normal runs retain only explicitly flagged clips/blobs. Each writer
has its own files, including when ASR and capture processes share a session.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Any
import uuid
import wave


def create_session_directory(base: str | Path) -> Path:
    """Create a fresh session; separate program launches never share filenames."""
    base = Path(base).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    directory = base / f"{stamp}-pid{os.getpid()}-{uuid.uuid4().hex[:10]}"
    directory.mkdir()
    _atomic_json(directory / "session.json", {
        "schema_version": 1, "session_id": directory.name,
        "created_at": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
        "note": "Local diagnostic audio. Accuracy is not inferred from confidence or issue flags.",
    })
    return directory


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip(".-")[:72] or "audio"


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pcm_bytes(pcm) -> bytes:
    view = memoryview(pcm)
    if view.itemsize not in {1, 2}:
        raise ValueError("PCM evidence must be mono Int16 little-endian bytes, not floating-point samples")
    if view.itemsize == 2 and view.format not in {"h", "<h", "=h", "@h"}:
        raise ValueError("PCM evidence requires signed Int16 little-endian samples")
    if view.ndim != 1 and not (view.ndim == 2 and view.shape[1] == 1):
        raise ValueError("PCM evidence must contain exactly one audio channel")
    payload = view.tobytes()
    if len(payload) % 2:
        raise ValueError("PCM evidence has an incomplete Int16 sample")
    return payload


class AudioEvidence:
    """One asynchronous writer with separate bounded capture and clip budgets.

    ``capture`` accepts PCM16LE and absolute input sample offsets. Omitting an
    offset reserves the next sequential interval, including rejected packets.
    A gap/overlap starts a new WAV segment and is explicit in metadata; samples
    from disjoint intervals are never silently concatenated. ``save_clip`` and
    ``save_blob`` require nonempty reasons so normal operation cannot retain
    every inference buffer accidentally. Debug callers can use reason ``debug``.

    The issue budget counts admitted PCM/blob bytes, not JSON/WAV overhead. It
    stops retaining new issue artifacts rather than deleting existing evidence.
    Full debug capture is rotated but not subject to this separate issue budget.
    """

    def __init__(
        self, directory: str | Path, *, role: str = "capture", full_stream: bool = False,
        sample_rate: int = 16000, queue_seconds: float = 10,
        segment_seconds: float = 300, max_clip_bytes: int = 256 * 1024 * 1024,
        queue_clip_bytes: int = 16 * 1024 * 1024, queue_items: int = 512,
    ):
        if sample_rate <= 0 or queue_seconds <= 0 or segment_seconds <= 0:
            raise ValueError("sample rate, queue duration and segment duration must be positive")
        if max_clip_bytes < 0 or queue_clip_bytes < 1 or queue_items < 1:
            raise ValueError("clip budget must be nonnegative and queue limits positive")
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.sample_rate, self.full_stream = int(sample_rate), bool(full_stream)
        self.segment_samples = max(1, round(segment_seconds * self.sample_rate))
        self.max_capture_bytes = max(2, round(queue_seconds * self.sample_rate) * 2)
        self.max_clip_bytes, self.max_queued_clip_bytes = int(max_clip_bytes), int(queue_clip_bytes)
        self.max_queue_items = int(queue_items)
        self.writer_id = f"{_safe_name(role)}-pid{os.getpid()}-{uuid.uuid4().hex[:10]}"
        self.events_path = self.directory / f"{self.writer_id}-events.jsonl"
        self.summary_path = self.directory / f"{self.writer_id}-summary.json"
        self._condition = threading.Condition()
        self._queue: deque[tuple[str, dict]] = deque()
        self._queued_capture_bytes = self._queued_clip_bytes = 0
        self._clip_admitted_bytes = self._active = self._admitted = self._completed = 0
        self._next_capture_sample = self._capture_packets = self._capture_samples = 0
        self._rejected_capture_packets = self._rejected_capture_samples = 0
        self._rejected_artifacts = self._rejected_events = 0
        self._pending_drops: deque[dict] = deque()
        self._closing, self._closed, self._error = False, False, ""
        self._segment = None
        self._segment_number = self._artifact_number = 0
        self._expected_sample: int | None = None
        self._written_capture_samples = self._written_artifacts = 0
        self._thread = threading.Thread(target=self._run, name=f"audio-evidence-{role}", daemon=True)
        self._thread.start()

    @property
    def snapshot(self) -> dict:
        with self._condition:
            return {
                "writer_id": self.writer_id, "directory": str(self.directory),
                "full_stream": self.full_stream, "closed": self._closed, "error": self._error,
                "queued_commands": len(self._queue), "active_commands": self._active,
                "capture_packets": self._capture_packets, "capture_samples": self._capture_samples,
                "written_capture_samples": self._written_capture_samples,
                "rejected_capture_packets": self._rejected_capture_packets,
                "rejected_capture_samples": self._rejected_capture_samples,
                "clip_admitted_bytes": self._clip_admitted_bytes,
                "written_artifacts": self._written_artifacts,
                "rejected_artifacts": self._rejected_artifacts,
                "rejected_events": self._rejected_events,
                "events_path": str(self.events_path), "summary_path": str(self.summary_path),
            }

    def _offer_locked(self, kind: str, item: dict, *, size: int = 0) -> bool:
        if self._closing or self._error or len(self._queue) >= self.max_queue_items:
            return False
        queued = self._queued_capture_bytes if kind == "capture" else self._queued_clip_bytes
        limit = self.max_capture_bytes if kind == "capture" else self.max_queued_clip_bytes
        if size and queued + size > limit:
            return False
        self._queue.append((kind, item))
        if kind == "capture":
            self._queued_capture_bytes += size
        else:
            self._queued_clip_bytes += size
        self._admitted += 1
        self._condition.notify()
        return True

    def _record_drop_locked(self, start: int, end: int, reason: str) -> None:
        if self._pending_drops and self._pending_drops[-1]["end_sample"] == start:
            previous = self._pending_drops[-1]
            previous["end_sample"] = end
            previous["dropped_samples"] += end - start
        else:
            if len(self._pending_drops) >= 64:
                previous = self._pending_drops[-1]
                previous["start_sample"] = min(previous["start_sample"], start)
                previous["end_sample"] = max(previous["end_sample"], end)
                previous["dropped_samples"] += end - start
                previous["range_is_envelope"] = True
            else:
                self._pending_drops.append({"kind": "recording_drop", "start_sample": start,
                                            "end_sample": end, "dropped_samples": end - start,
                                            "reason": reason, "range_is_envelope": False})
        self._condition.notify()

    def capture(self, pcm, *, start_sample: int | None = None, timestamp_ns: int | None = None,
                discontinuity: bool = False) -> bool:
        if not self.full_stream:
            return False
        payload = _pcm_bytes(pcm)
        count = len(payload) // 2
        with self._condition:
            start = self._next_capture_sample if start_sample is None else int(start_sample)
            if start < 0:
                raise ValueError("input sample offset must be nonnegative")
            self._next_capture_sample = start + count
            self._capture_packets += 1
            self._capture_samples += count
            if not count:
                return not self._closing and not self._error
            item = {"pcm": payload, "start_sample": start,
                    "timestamp_ns": int(timestamp_ns) if timestamp_ns is not None else None,
                    "discontinuity": bool(discontinuity)}
            if self._offer_locked("capture", item, size=len(payload)):
                return True
            self._rejected_capture_packets += 1
            self._rejected_capture_samples += count
            self._record_drop_locked(start, start + count, "recording queue full or writer stopping/failed")
            return False

    def _artifact(self, kind: str, payload: bytes, metadata, reasons, label: str,
                  suffix: str, start_sample: int | None, timestamp_ns: int | None) -> bool:
        if isinstance(reasons, str):
            reasons = [reasons]
        reasons = [str(reason) for reason in reasons if str(reason)]
        if not reasons:
            return False
        if start_sample is not None and int(start_sample) < 0:
            raise ValueError("input sample offset must be nonnegative")
        item = {"payload": payload, "metadata": deepcopy(dict(metadata or {})), "reasons": reasons,
                "label": _safe_name(label), "suffix": suffix,
                "start_sample": int(start_sample) if start_sample is not None else None,
                "timestamp_ns": int(timestamp_ns) if timestamp_ns is not None else None}
        with self._condition:
            if self._clip_admitted_bytes + len(payload) > self.max_clip_bytes:
                self._rejected_artifacts += 1
                self._offer_locked("event", {"kind": "artifact_not_saved", "reason": "issue audio byte budget exhausted",
                                             "label": item["label"], "reasons": reasons,
                                             "requested_bytes": len(payload), "metadata": item["metadata"]})
                return False
            if not self._offer_locked(kind, item, size=len(payload)):
                self._rejected_artifacts += 1
                return False
            self._clip_admitted_bytes += len(payload)
            return True

    def save_clip(self, pcm, metadata: dict | None = None, reasons=(), *, label: str = "raw",
                  start_sample: int | None = None, timestamp_ns: int | None = None) -> bool:
        return self._artifact("clip", _pcm_bytes(pcm), metadata, reasons, label, ".wav", start_sample, timestamp_ns)

    def save_blob(self, payload, metadata: dict | None = None, reasons=(), *, label: str = "model-input",
                  suffix: str = ".f32le", start_sample: int | None = None, timestamp_ns: int | None = None) -> bool:
        if not re.fullmatch(r"\.[A-Za-z0-9_-]{1,20}", suffix):
            raise ValueError("blob suffix must be a simple extension such as .f32le")
        return self._artifact("blob", memoryview(payload).tobytes(), metadata, reasons, label,
                              suffix, start_sample, timestamp_ns)

    def event(self, metadata: dict) -> bool:
        with self._condition:
            accepted = self._offer_locked("event", deepcopy(dict(metadata)))
            if not accepted:
                self._rejected_events += 1
            return accepted

    def flush(self, timeout: float = 5) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        with self._condition:
            target = self._admitted
            self._condition.notify()
            while self._completed < target and not self._error:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return not self._error and self._completed >= target

    def close(self, timeout: float = 5) -> bool:
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        if self._thread is not threading.current_thread():
            self._thread.join(max(0.0, timeout))
        return not self._thread.is_alive() and not self._error

    def _emit(self, handle, record: dict) -> None:
        record = {"schema_version": 1, "writer_id": self.writer_id,
                  "recorded_at": datetime.now(timezone.utc).isoformat(), **record}
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        handle.flush()

    def _new_segment(self, start: int, timestamp_ns: int | None) -> None:
        self._segment_number += 1
        path = self.directory / f"{self.writer_id}-stream-{self._segment_number:06d}.wav"
        temporary = path.with_name(path.name + ".part")
        audio = wave.open(str(temporary), "wb")
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(self.sample_rate)
        self._segment = {"path": path, "temporary": temporary, "audio": audio,
                         "start_sample": start, "timestamp_ns": timestamp_ns,
                         "sample_count": 0, "pcm_hash": hashlib.sha256()}

    def _finish_segment(self, events) -> None:
        segment = self._segment
        if segment is None:
            return
        self._segment = None
        segment["audio"].close()
        os.replace(segment["temporary"], segment["path"])
        record = {"schema_version": 1, "writer_id": self.writer_id,
                  "kind": "stream_segment", "audio_file": segment["path"].name,
                  "sample_rate": self.sample_rate, "channels": 1, "sample_width_bytes": 2,
                  "sample_count": segment["sample_count"], "start_sample": segment["start_sample"],
                  "end_sample": segment["start_sample"] + segment["sample_count"],
                  "timestamp_ns": segment["timestamp_ns"],
                  "pcm_sha256": segment["pcm_hash"].hexdigest(),
                  "file_sha256": _sha256_file(segment["path"])}
        _atomic_json(segment["path"].with_suffix(".json"), record)
        self._emit(events, record)

    def _write_capture(self, item: dict, events) -> None:
        data, start = item["pcm"], item["start_sample"]
        if item.get("discontinuity"):
            self._finish_segment(events)
            self._emit(events, {"kind": "stream_discontinuity", "actual_sample": start,
                                "timestamp_ns": item["timestamp_ns"],
                                "reason": "input_device_or_clock_discontinuity", "missing_input_samples_unknown": True})
        if self._expected_sample is not None and start != self._expected_sample:
            self._finish_segment(events)
            self._emit(events, {"kind": "stream_discontinuity", "expected_sample": self._expected_sample,
                                "actual_sample": start, "delta_samples": start - self._expected_sample,
                                "reason": "gap" if start > self._expected_sample else "overlap_or_reordered_input"})
        consumed = 0
        while consumed < len(data):
            position = start + consumed // 2
            timestamp = item["timestamp_ns"]
            if timestamp is not None:
                timestamp += consumed // 2 * 1_000_000_000 // self.sample_rate
            if self._segment is None:
                self._new_segment(position, timestamp)
                self._emit(events, {"kind": "stream_segment_started",
                                    "audio_file": self._segment["path"].name,
                                    "temporary_audio_file": self._segment["temporary"].name,
                                    "start_sample": position, "timestamp_ns": timestamp,
                                    "sample_rate": self.sample_rate})
            segment = self._segment
            take = min(len(data) - consumed, (self.segment_samples - segment["sample_count"]) * 2)
            part = data[consumed:consumed + take]
            # Patch the WAV header on every background write. If a process is
            # killed before orderly close, the .part WAV remains readable up
            # to its last completed write; the start event preserves offsets.
            segment["audio"].writeframes(part)
            segment["pcm_hash"].update(part)
            segment["sample_count"] += take // 2
            self._written_capture_samples += take // 2
            consumed += take
            if segment["sample_count"] == self.segment_samples:
                self._finish_segment(events)
        self._expected_sample = start + len(data) // 2

    def _write_artifact(self, kind: str, item: dict, events) -> None:
        self._artifact_number += 1
        path = self.directory / f"{self.writer_id}-clip-{self._artifact_number:06d}-{item['label']}{item['suffix']}"
        temporary = path.with_name(path.name + ".part")
        if kind == "clip":
            with wave.open(str(temporary), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(self.sample_rate)
                audio.writeframes(item["payload"])
        else:
            temporary.write_bytes(item["payload"])
        os.replace(temporary, path)
        record = {"schema_version": 1, "writer_id": self.writer_id,
                  "kind": "issue_clip" if kind == "clip" else "issue_blob", "artifact_file": path.name,
                  "label": item["label"], "reasons": item["reasons"], "metadata": item["metadata"],
                  "payload_bytes": len(item["payload"]), "payload_sha256": hashlib.sha256(item["payload"]).hexdigest(),
                  "file_sha256": _sha256_file(path), "start_sample": item["start_sample"],
                  "timestamp_ns": item["timestamp_ns"]}
        if kind == "clip":
            count = len(item["payload"]) // 2
            record.update({"sample_rate": self.sample_rate, "channels": 1, "sample_width_bytes": 2,
                           "sample_count": count, "duration_seconds": count / self.sample_rate,
                           "end_sample": item["start_sample"] + count if item["start_sample"] is not None else None})
        _atomic_json(path.with_name(path.name + ".json"), record)
        self._emit(events, record)
        self._written_artifacts += 1

    def _run(self) -> None:
        try:
            with self.events_path.open("x", encoding="utf-8", buffering=1) as events:
                self._emit(events, {"kind": "writer_started", "full_stream": self.full_stream,
                                    "sample_rate": self.sample_rate,
                                    "capture_queue_bytes": self.max_capture_bytes,
                                    "issue_pcm_blob_byte_budget": self.max_clip_bytes})
                while True:
                    with self._condition:
                        while not self._queue and not self._pending_drops and not self._closing:
                            self._condition.wait()
                        drops = list(self._pending_drops)
                        self._pending_drops.clear()
                        if not self._queue:
                            if self._closing:
                                item = None
                            else:
                                item = False
                        else:
                            kind, value = self._queue.popleft()
                            size = len(value.get("pcm", value.get("payload", b"")))
                            if kind == "capture":
                                self._queued_capture_bytes -= size
                            else:
                                self._queued_clip_bytes -= size
                            self._active = 1
                            item = kind, value
                    for drop in drops:
                        self._emit(events, drop)
                    if item is None:
                        break
                    if item is False:
                        continue
                    try:
                        kind, value = item
                        if kind == "capture":
                            self._write_capture(value, events)
                        elif kind in {"clip", "blob"}:
                            self._write_artifact(kind, value, events)
                        else:
                            self._emit(events, value)
                    except Exception as exc:
                        # Publish failure before waking flush waiters. Otherwise
                        # a waiter could see "completed" just before the outer
                        # failure handler records the write error.
                        with self._condition:
                            self._error = f"{type(exc).__name__}: {exc}"
                        raise
                    finally:
                        with self._condition:
                            self._active = 0
                            self._completed += 1
                            self._condition.notify_all()
                self._finish_segment(events)
                self._emit(events, {"kind": "writer_finished", **self.snapshot, "closed": True})
        except Exception as exc:
            with self._condition:
                self._error = f"{type(exc).__name__}: {exc}"
                self._condition.notify_all()
            try:
                if self._segment is not None:
                    self._segment["audio"].close()
            except Exception:
                pass
        finally:
            with self._condition:
                self._closed = True
                self._condition.notify_all()
            try:
                _atomic_json(self.summary_path, self.snapshot)
            except Exception:
                pass
