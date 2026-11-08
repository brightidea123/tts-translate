#!/usr/bin/env python3
"""Inspect saved ASR audio; model replay requires explicit --run.

Examples:
  python scripts/replay_audio.py recordings/session/clip.wav
  python scripts/replay_audio.py clip.wav --run --source HE
  python scripts/replay_audio.py --prepared-f32 model-input.f32le --run --source AR

Inspection does not import model runtimes, capture audio, play sound, or access
the network. Model replay is offline unless --allow-download is explicit.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import wave
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tts_runtime.asr_quality import pcm_quality


def load_audio(path: Path, prepared: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
    data = path.read_bytes()
    if prepared:
        if not data or len(data) % 4:
            raise ValueError("Prepared audio must contain complete little-endian Float32 samples")
        samples = np.frombuffer(data, dtype="<f4")
        if not np.all(np.isfinite(samples)):
            raise ValueError("Prepared audio contains non-finite samples")
        rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
        metrics = {
            "samples": int(samples.size), "sample_rate": 16000,
            "duration_seconds": int(samples.size) / 16000,
            "peak_abs": float(np.max(np.abs(samples))),
            "rms_dbfs": max(-120.0, 20 * math.log10(max(rms, 1e-12))),
            "out_of_range_fraction": float(np.mean(np.abs(samples) > 1)),
            "dc_offset": float(np.mean(samples.astype(np.float64))),
            "accuracy_measured": False,
        }
        format_name = "16kHz mono little-endian Float32 model input"
    else:
        with wave.open(str(path), "rb") as handle:
            if (handle.getframerate(), handle.getnchannels(), handle.getsampwidth(), handle.getcomptype()) != (16000, 1, 2, "NONE"):
                raise ValueError("Replay WAV must be uncompressed 16kHz mono Int16 PCM; no implicit resampling is performed")
            pcm = handle.readframes(handle.getnframes())
            if not pcm:
                raise ValueError("Replay WAV contains no samples")
            if len(pcm) != handle.getnframes() * 2:
                raise ValueError("Replay WAV is truncated")
        samples = np.frombuffer(pcm, dtype="<i2")
        metrics = pcm_quality(samples)
        format_name = "16kHz mono Int16 PCM WAV"
    return samples, {
        "path": str(path.resolve()), "format": format_name,
        "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data),
        "audio_metrics": metrics,
    }


def load_metadata(path: Path, explicit: Path | None = None) -> tuple[dict[str, Any], str | None]:
    sidecar = explicit if explicit is not None else path.with_suffix(".json")
    if not sidecar.is_file():
        if explicit is not None:
            raise FileNotFoundError(f"Metadata file not found: {sidecar}")
        return {}, None
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Audio metadata must be a JSON object")
    return metadata, str(sidecar.resolve())


def infer_source(metadata: dict[str, Any]) -> str:
    candidates = [metadata]
    for field in ("request", "asr", "metadata", "decoder"):
        if isinstance(metadata.get(field), dict):
            candidates.append(metadata[field])
    for item in candidates:
        for key in ("source_language", "source_language_code", "language"):
            if isinstance(item.get(key), str) and item[key].strip():
                return item[key].strip()
    return "TR"


def load_core(source: str, backend: str, model: str | None, preprocessing: str):
    from translation_prompts import normalize_translation_language
    language = normalize_translation_language(source).upper()
    argv = [
        str(ROOT / "tts-transcribe.py"), "--mode", "audio",
        "--source-language", language, "--target-language", "EN" if language != "EN" else "TR",
        "--stt-backend", backend, "--no-session-logs", "--no-warmup",
        "--audio-review", "off", "--asr-preprocessing", preprocessing,
    ]
    if model:
        argv.extend(["--mlx-vlm-model" if backend == "mlx-vlm-gemma" else "--whisper-model", model])
    module_name = "_tts_audio_replay_core"
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "tts-transcribe.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    previous_argv = sys.argv
    try:
        sys.argv = argv
        spec.loader.exec_module(module)
    finally:
        sys.argv = previous_argv
    return module


def replay(samples: np.ndarray, *, prepared: bool, source: str, backend: str,
           model: str | None, preprocessing: str, allow_download: bool) -> dict[str, Any]:
    os.environ["HF_HUB_OFFLINE"] = "0" if allow_download else "1"
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    core = load_core(source, backend, model, preprocessing)
    core.init_stt_runtime_only()
    started = time.perf_counter()
    if prepared:
        text = core.decode_prepared_audio(samples, partial=False, from_ring=True)
    else:
        text = core.transcribe_pcm_i16(samples, partial=False, from_ring=True)
    diagnostics = dict(getattr(core, "last_asr_diagnostics", {}))
    diagnostics.setdefault("decoder", dict(getattr(core, "last_decoder_quality", {})))
    inference_run = diagnostics.get("decode_attempted", True) is True
    return {
        "text": text, "source_language": core.source_language_code(),
        "backend": core.detected_stt_backend,
        "preprocessing": "exact saved Float32; DSP bypassed" if prepared else preprocessing,
        "elapsed_seconds_excluding_initializer": time.perf_counter() - started,
        "model_inference_run": inference_run, "quality_evaluated": False,
        "decoder_diagnostics": diagnostics,
        "note": "Decoder completed" if inference_run else "Input was gated before decoder inference; inspect diagnostics",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=Path, nargs="?", help="Saved 16kHz mono Int16 WAV")
    parser.add_argument("--prepared-f32", type=Path, help="Exact saved little-endian Float32 model-input file, bypassing DSP")
    parser.add_argument("--metadata", type=Path, help="Explicit clip JSON sidecar; default: same basename .json")
    parser.add_argument("--run", action="store_true", help="Explicitly load a model and run transcription")
    parser.add_argument("--source", help="Source language; otherwise inferred from metadata, then TR")
    parser.add_argument("--backend", choices=("mlx-whisper", "mlx-vlm-gemma", "faster-whisper"), default="mlx-whisper")
    parser.add_argument("--asr-model", help="Explicit local model path or Hub ID")
    parser.add_argument("--preprocessing", choices=("raw", "legacy"), default="raw")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--output", type=Path, help="Also save the JSON replay report")
    args = parser.parse_args(argv)
    if bool(args.audio) == bool(args.prepared_f32):
        parser.error("Choose exactly one WAV path or --prepared-f32 PATH")
    path = (args.audio or args.prepared_f32).expanduser()
    report: dict[str, Any] = {"inference_requested": args.run, "model_inference_run": False,
                              "downloads_allowed": bool(args.run and args.allow_download),
                              "quality_evaluated": False, "errors": []}
    try:
        samples, audio_report = load_audio(path, prepared=bool(args.prepared_f32))
        metadata, metadata_path = load_metadata(path, args.metadata.expanduser() if args.metadata else None)
        report.update(audio_report)
        report.update({"metadata": metadata, "metadata_path": metadata_path})
        if args.run:
            result = replay(samples, prepared=bool(args.prepared_f32),
                            source=args.source or infer_source(metadata), backend=args.backend,
                            model=args.asr_model, preprocessing=args.preprocessing,
                            allow_download=args.allow_download)
            report["result"] = result
            report["model_inference_run"] = result["model_inference_run"]
        else:
            report["note"] = "Inspection only; no model loading, network, microphone, playback, or new review recording"
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    report["ok"] = not report["errors"]
    rendered = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    if args.output:
        args.output.expanduser().write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
