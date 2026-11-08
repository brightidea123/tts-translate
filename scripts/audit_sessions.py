#!/usr/bin/env python3
"""Audit saved sessions offline, without inference, devices or copied transcripts.

Default: inspect the three latest directories with saved audio and print aggregate
diagnostics. --output explicitly saves a report with per-request diagnostics.
Hashes/placement verify saved artifacts, not recognition accuracy or WER.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_CONTROLS = re.compile(r"<\|[^>]+\|>|<end_of_turn>|</?think\b[^>]*>")


def _hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(directory: Path, name: str) -> Path:
    path = (directory / name).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("Artifact path escapes its recording directory")
    return path


def _pcm(path: Path, start: int = 0, count: int | None = None) -> bytes:
    with wave.open(str(path), "rb") as handle:
        if (handle.getnchannels(), handle.getsampwidth(), handle.getframerate()) != (1, 2, 16000):
            raise ValueError("Expected 16 kHz mono Int16 WAV")
        handle.setpos(start)
        return handle.readframes(handle.getnframes() if count is None else count)


def _plain(text) -> str:
    return " ".join(str(text or "").split())


def _id_order(value):
    return (1, value) if isinstance(value, str) else (0, value or 0)


def _has_saved_audio(directory: Path) -> bool:
    """Skip empty startup/smoke directories without reading every waveform."""
    for pattern in ("*.wav", "*.wav.part"):
        for path in directory.glob(pattern):
            try:
                if path.is_file() and path.stat().st_size > 44:
                    return True
            except OSError:
                continue
    return False


def audit_session(directory: Path, runtime_directory: Path | None = None) -> dict:
    """Inspect one recording directory; never mutate it or import the live core."""
    directory = directory.resolve()
    checks, issues, flags = Counter(), Counter(), Counter()
    errors, streams, requests, inputs, summaries, config = [], [], [], [], [], {}

    def failure(path, exc):
        errors.append({"file": path.name, "error": f"{type(exc).__name__}: {exc}"})

    for path in sorted(directory.glob("*.json")):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(item, dict):
                raise ValueError("Sidecar must be an object")
            kind = item.get("kind")
            if path.name == "session.json":
                config = {key: item.get("config", {}).get(key) for key in (
                    "source_language", "target_language", "whisper_model", "asr_preprocessing", "debug", "profile")}
            elif kind == "stream_segment":
                artifact = _artifact(directory, item["audio_file"])
                pcm = _pcm(artifact)
                checks["stream_files_checked"] += 1
                checks["stream_pcm_hash_mismatches"] += _hash(pcm) != item.get("pcm_sha256")
                checks["stream_file_hash_mismatches"] += _file_hash(artifact) != item.get("file_sha256")
                checks["stream_sample_count_mismatches"] += len(pcm) // 2 != item.get("sample_count")
                streams.append((item, artifact))
            elif kind == "issue_blob" and str(item.get("artifact_file", "")).endswith(".f32le"):
                inputs.append(item)
            elif kind == "issue_clip" and isinstance(item.get("metadata", {}).get("request"), dict):
                metadata = item["metadata"]
                request = metadata["request"]
                pcm_path = _artifact(directory, item["artifact_file"])
                pcm = _pcm(pcm_path)
                digest = _hash(pcm)
                checks["raw_files_checked"] += 1
                checks["raw_payload_hash_mismatches"] += digest != item.get("payload_sha256")
                checks["raw_file_hash_mismatches"] += _file_hash(pcm_path) != item.get("file_sha256")
                checks["request_hash_mismatches"] += digest != request.get("pcm_sha256")
                checks["actual_request_hash_mismatches"] += digest != request.get("actual_pcm_sha256")
                raw_text = _plain(metadata.get("decoder", {}).get("raw_text"))
                text = _plain(metadata.get("text"))
                row_flags = []
                if raw_text and not text:
                    row_flags.append("empty_after_filter")
                elif raw_text and raw_text != text:
                    row_flags.append("changed_after_filter")
                if _CONTROLS.search(text):
                    row_flags.append("model_control_in_output")
                if request.get("capture_gap"):
                    row_flags.append("capture_gap")
                if request.get("source_span_minus_pcm_ns", 0) != 0:
                    row_flags.append("source_span_mismatch")
                issues.update(metadata.get("issues", []))
                flags.update(row_flags)
                requests.append({"request_id": request.get("request_id"), "utterance_id": request.get("utterance_id"),
                                 "partial": bool(metadata.get("partial")), "duration_seconds": len(pcm) / 32000,
                                 "start_ns": request.get("timestamp_start_ns"), "end_ns": request.get("timestamp_end_ns"),
                                 "text_nonempty": bool(text), "text_sha256": _hash(text.casefold().encode()) if text else None,
                                 "flags": row_flags, "issues": metadata.get("issues", []), "artifact": pcm_path.name})
            elif path.name.startswith("capture-") and path.name.endswith("-summary.json"):
                summaries.append({key: item.get(key) for key in (
                    "capture_samples", "written_capture_samples", "rejected_capture_samples", "rejected_artifacts", "error", "closed")})
        except (OSError, ValueError, KeyError, TypeError, wave.Error) as exc:
            failure(path, exc)

    by_request = {row["request_id"]: row for row in requests}
    for row in requests:
        try:
            raw = _pcm(directory / row["artifact"])
            matched = False
            for metadata, path in streams:
                delta = (row["start_ns"] - metadata["timestamp_ns"]) * 16000
                offset = round(delta / 1_000_000_000)
                if offset < 0 or offset + len(raw) // 2 > metadata["sample_count"]:
                    continue
                if abs(delta - offset * 1_000_000_000) > 1_000_000:  # 0.001 sample rounding tolerance
                    continue
                checks["raw_to_stream_compared"] += 1
                checks["raw_to_stream_mismatches"] += raw != _pcm(path, offset, len(raw) // 2)
                matched = True
                break
            checks["raw_unmapped_to_stream"] += not matched
        except (OSError, ValueError, KeyError, TypeError, wave.Error) as exc:
            failure(Path(row["artifact"]), exc)
    for item in inputs:
        try:
            path = _artifact(directory, item["artifact_file"])
            payload = path.read_bytes()
            metadata = item["metadata"]
            digest = _hash(payload)
            checks["model_input_files_checked"] += 1
            checks["model_input_hash_mismatches"] += digest != item.get("payload_sha256") or digest != metadata.get("model_input", {}).get("sha256")
            floats = np.frombuffer(payload, dtype="<f4")
            checks["model_input_nonfinite_files"] += not np.all(np.isfinite(floats))
            raw_row = by_request.get(metadata.get("request", {}).get("request_id"))
            if raw_row and metadata.get("model_input", {}).get("preprocessing") == "raw":
                expected = np.frombuffer(_pcm(directory / raw_row["artifact"]), dtype="<i2").astype(np.float32) / 32768.0
                checks["raw_to_float_compared"] += 1
                checks["raw_to_float_mismatches"] += not np.array_equal(floats, expected)
        except (OSError, ValueError, KeyError, TypeError, wave.Error) as exc:
            failure(Path(str(item.get("artifact_file", "unknown"))), exc)

    groups = defaultdict(list)
    for row in requests:
        groups[row["utterance_id"]].append(row)
    partial_only = sorted((uid for uid, rows in groups.items() if all(row["partial"] for row in rows)), key=_id_order)
    finals = sorted((row for row in requests if not row["partial"]), key=lambda row: (row["start_ns"] or 0))
    repeated, previous = [], {}
    for row in finals:
        digest = row["text_sha256"]
        old = previous.get(digest)
        if digest and old and old["end_ns"] is not None and row["start_ns"] is not None and old["end_ns"] <= row["start_ns"]:
            repeated.append({"previous_utterance_id": old["utterance_id"], "utterance_id": row["utterance_id"],
                             "gap_seconds": (row["start_ns"] - old["end_ns"]) / 1e9})
        if digest:
            previous[digest] = row
    journals = []
    if runtime_directory and runtime_directory.is_dir():
        for path in sorted(runtime_directory.glob("*decision*.jsonl")):
            counts, reasons, ids, records = Counter(), Counter(), set(), []
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    item = json.loads(line)
                    counts[str(item.get("decision", item.get("kind", item.get("event", "unknown"))))] += 1
                    if item.get("reason"):
                        reasons[str(item["reason"])] += 1
                    if isinstance(item.get("utterance_id"), (int, str)):
                        ids.add(item["utterance_id"])
                    records.append({key: item.get(key) for key in (
                        "timestamp", "utterance_id", "request_id", "decision", "reason", "processing_seconds")})
                except (ValueError, AttributeError) as exc:
                    failure(path, exc)
            journals.append({"file": path.name, "decisions": dict(counts), "reasons": dict(reasons),
                             "utterance_ids": sorted(ids, key=_id_order), "records": records})
    return {"session_id": directory.name, "directory": str(directory), "config": config, "capture_summaries": summaries,
            "capture_seconds": sum(item["sample_count"] for item, _ in streams) / 16000,
            "checks": dict(checks), "issue_counts": dict(issues), "filter_flag_counts": dict(flags),
            "request_count": len(requests), "final_count": len(finals), "final_nonempty_count": sum(row["text_nonempty"] for row in finals),
            "partial_without_saved_final_ids": partial_only, "repeated_text_on_disjoint_finals": repeated,
            "decision_journals": journals, "requests": requests, "errors": errors}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", action="append", help="Recording directory or session ID; repeat to select several")
    parser.add_argument("--latest", type=int, default=3, help="Latest sessions with saved audio when --session is omitted (default: 3)")
    parser.add_argument("--include-empty", action="store_true", help="Include empty startup/smoke directories in --latest selection")
    parser.add_argument("--recordings-dir", type=Path, default=ROOT / "recordings")
    parser.add_argument("--runtime-dir", type=Path, default=ROOT / "runtime-reports/sessions")
    parser.add_argument("--output", type=Path, help="Explicitly write the JSON report here")
    args = parser.parse_args(argv)
    if args.latest < 1:
        parser.error("--latest must be positive")
    if args.session:
        directories = [Path(value).expanduser() if Path(value).expanduser().is_dir() else args.recordings_dir / value for value in args.session]
    else:
        candidates = (p for p in args.recordings_dir.iterdir() if p.is_dir() and (args.include_empty or _has_saved_audio(p))) if args.recordings_dir.is_dir() else ()
        directories = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[:args.latest]
    missing = [str(path) for path in directories if not path.is_dir()]
    if missing:
        parser.error("Recording directory does not exist: " + ", ".join(missing))
    audio_sessions = sum(_has_saved_audio(path) for path in directories)
    report = {"schema_version": 1, "model_inference_run": False, "network_accessed": False, "accuracy_measured": False,
              "saved_audio_session_count": audio_sessions,
              "selection_note": "No saved-audio sessions selected; use --include-empty to inspect startup failures." if not audio_sessions else "",
              "limitations": ["Saved artifacts cannot establish recognition accuracy without reference speech/transcripts.",
                              "Partial-only IDs and repeated text are review flags, not proof of lost words or hallucination."],
              "sessions": [audit_session(path, args.runtime_dir / path.name) for path in directories]}
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.expanduser().write_text(encoded, encoding="utf-8")
    summary = {**report, "sessions": [{key: value for key, value in session.items() if key != "requests"}
                                      for session in report["sessions"]]}
    for session in summary["sessions"]:
        session["decision_journals"] = [{key: value for key, value in journal.items() if key != "records"}
                                        for journal in session["decision_journals"]]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return int(any(session["errors"] or any(value for key, value in session["checks"].items()
                                           if key.endswith("_mismatches") or key == "model_input_nonfinite_files")
                   for session in report["sessions"]))


if __name__ == "__main__":
    raise SystemExit(main())
