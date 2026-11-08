#!/usr/bin/env python3
"""Inspect the local MLX environment without loading models or opening a microphone.

The default includes import checks and a tiny Metal calculation. --metadata-only
does neither. --devices explicitly enables a read-only SoundDevice device query.
No part of this command downloads model weights or tests recognition quality.
Native checks run in separate processes so a failed initialization cannot abort
the doctor or corrupt the next check.
"""

from __future__ import annotations

import argparse
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PACKAGES = ("numpy", "mlx", "mlx-metal", "mlx-whisper", "mlx-lm", "transformers", "sounddevice")
OPTIONAL_PACKAGES = ("mlx-vlm", "mlx-audio", "torch", "numba", "scipy", "huggingface-hub", "sentencepiece", "silero-vad", "onnxruntime")
BASELINE_IMPORTS = ("numpy", "mlx_whisper", "mlx_lm", "sounddevice")
PROBE_TIMEOUT_SECONDS = 30
PROBE_RESULT_PREFIX = "TTS_DOCTOR_RESULT="


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def model_status(path: Path) -> dict[str, Any]:
    exists = path.is_dir()
    weights = sorted(p.name for p in path.glob("*.safetensors") if p.is_file()) if exists else []
    weights += sorted(p.name for p in path.glob("*.npz") if p.is_file()) if exists else []
    config = (path / "config.json").is_file()
    index_path = path / "model.safetensors.index.json"
    index: dict[str, Any] = {"present": index_path.exists(), "valid": None, "shard_files": [], "missing_shards": []}
    weights_complete = bool(weights)
    if index["present"]:
        weights_complete = False
        try:
            with index_path.open(encoding="utf-8") as handle:
                data = json.load(handle)
            weight_map = data.get("weight_map") if isinstance(data, dict) else None
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError("Index must contain a nonempty weight_map object")
            if not all(isinstance(name, str) and name for name in weight_map.values()):
                raise ValueError("Every weight_map value must name a shard file")
            shards = sorted(set(weight_map.values()))
            if any(Path(name).is_absolute() or ".." in Path(name).parts for name in shards):
                raise ValueError("Shard paths must be relative to the model directory")
            index.update({"valid": True, "shard_files": shards, "missing_shards": [name for name in shards if not (path / name).is_file()]})
            weights_complete = not index["missing_shards"]
        except (OSError, UnicodeError, ValueError) as exc:
            index.update({"valid": False, "error": f"{type(exc).__name__}: {exc}"})
    return {
        "path": str(path), "exists": exists, "config_present": config,
        "weight_files": weights, "weight_index": index, "appears_complete": bool(config and weights_complete),
        "validation": "File/index presence only; model load/inference not tested",
    }


def cached_models(profiles_file: Path, models_dir: Path) -> list[dict[str, Any]]:
    if not profiles_file.is_file():
        return []
    with profiles_file.open(encoding="utf-8") as handle:
        profiles = json.load(handle).get("profiles", {})
    references = sorted({stage["model"] for profile in profiles.values() for stage in (profile["asr"], profile["translation"])})
    hub_home = Path(os.environ.get("HF_HUB_CACHE", str(Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))) / "hub"))).expanduser()
    result = []
    for reference in references:
        local = models_dir / reference.rsplit("/", 1)[-1]
        hub_root = hub_home / ("models--" + reference.replace("/", "--")) / "snapshots"
        snapshots = sorted((p for p in hub_root.iterdir() if p.is_dir()), key=lambda p: p.name) if hub_root.is_dir() else []
        result.append({"model": reference, "project_local": model_status(local), "hub_snapshots": [model_status(p) for p in snapshots]})
    return result


def _probe_failure(kind: str, error: str) -> dict[str, Any]:
    if kind == "metal":
        return {"tested": True, "available": False, "compute_ok": False, "error": error}
    if kind == "devices":
        return {"queried": True, "error": error}
    return {"ok": False, "error": error}


def _probe_worker(kind: str, module: str | None = None) -> dict[str, Any]:
    """Run one native operation; only called by an isolated child process."""
    try:
        if kind == "import":
            importlib.import_module(module)
            return {"ok": True}
        if kind == "devices":
            sd = importlib.import_module("sounddevice")
            return {"queried": True, "devices": [dict(device) for device in sd.query_devices()], "default": list(sd.default.device)}
        mx = importlib.import_module("mlx.core")
        available = bool(mx.metal.is_available())
        if not available:
            return {"tested": True, "available": False, "compute_ok": False, "error": "MLX reports Metal unavailable"}
        mx.set_default_device(mx.gpu)
        started = time.perf_counter()
        value = mx.sum(mx.array([1.0, 2.0, 3.0], dtype=mx.float32) ** 2)
        mx.eval(value)
        result = float(value.item())
        return {
            "tested": True, "available": True, "default_device": str(mx.default_device()),
            "compute_ok": result == 14.0, "result": result,
            "elapsed_seconds": time.perf_counter() - started,
            "note": "Tiny correctness probe, not a model throughput benchmark",
        }
    except Exception as exc:
        return _probe_failure(kind, f"{type(exc).__name__}: {exc}")


def isolated_probe(kind: str, module: str | None = None, *, timeout: float = PROBE_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Capture native exceptions, process aborts and hangs without retrying imports."""
    command = [sys.executable, str(Path(__file__).resolve()), "--_worker-probe", kind]
    if module is not None:
        command += ["--_probe-module", module]
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        result = _probe_failure(kind, f"Probe timed out after {timeout:g} seconds")
        return {**result, "returncode": None, "timed_out": True}
    except OSError as exc:
        return {**_probe_failure(kind, f"{type(exc).__name__}: {exc}"), "returncode": None}
    result = None
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(PROBE_RESULT_PREFIX):
            try:
                payload = json.loads(line[len(PROBE_RESULT_PREFIX):])
                if isinstance(payload, dict):
                    result = payload
            except json.JSONDecodeError:
                pass
            break
    if result is None or (completed.returncode != 0 and "error" not in result):
        reason = f"Probe exited with return code {completed.returncode}"
        if completed.returncode < 0:
            reason += f" (signal {-completed.returncode})"
        result = _probe_failure(kind, reason)
    result["returncode"] = completed.returncode
    if completed.stderr.strip():
        result["stderr"] = completed.stderr.strip()[-4000:]
    return result


def metal_probe() -> dict[str, Any]:
    return isolated_probe("metal")


def inspect_runtime(
    *, include_devices: bool = False, probe_metal: bool = True,
    import_backends: bool = True, profiles_file: Path = REPO_ROOT / "configs/model-profiles.json",
    models_dir: Path = REPO_ROOT / "models",
) -> dict[str, Any]:
    # These environment settings prevent an import from making implicit Hub requests.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    versions = {name: package_version(name) for name in BASELINE_PACKAGES + OPTIONAL_PACKAGES}
    missing = [name for name in BASELINE_PACKAGES if versions[name] is None]
    imports: dict[str, Any] = {}
    if import_backends:
        for name in BASELINE_IMPORTS:
            imports[name] = isolated_probe("import", name)
    metal = metal_probe() if probe_metal else {"tested": False, "reason": "Skipped by request"}
    devices: dict[str, Any] = {"queried": False, "reason": "Use --devices to query device metadata; no capture is opened"}
    if include_devices:
        devices = isolated_probe("devices")
    failures = [name for name, result in imports.items() if not result["ok"]]
    supported = sys.version_info >= (3, 12) and platform.system() == "Darwin" and platform.machine() == "arm64"
    report = {
        "python": {"version": platform.python_version(), "executable": sys.executable, "native_target": supported},
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "package_versions": versions, "missing_baseline_packages": missing,
        "backend_imports": imports, "metal": metal, "audio": devices,
        "models": cached_models(profiles_file, models_dir),
        "model_inference": {"run": False, "reason": "Doctor does not load or run speech/translation models"},
        "diagnostics": {"native_check_isolation": "One subprocess per check", "probe_timeout_seconds": PROBE_TIMEOUT_SECONDS},
        "ready_for_model_smoke": supported and not missing and import_backends and not failures and probe_metal and metal.get("compute_ok", False),
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", action="store_true")
    parser.add_argument("--metadata-only", action="store_true", help="Skip imports and Metal computation")
    parser.add_argument("--no-metal", action="store_true", help="Skip only the Metal computation")
    parser.add_argument("--models-dir", type=Path, default=REPO_ROOT / "models")
    parser.add_argument("--profiles-file", type=Path, default=REPO_ROOT / "configs/model-profiles.json")
    parser.add_argument("--json", action="store_true", help="Print structured JSON (also the default output format)")
    parser.add_argument("--_worker-probe", choices=("import", "metal", "devices"), help=argparse.SUPPRESS)
    parser.add_argument("--_probe-module", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._worker_probe:
        result = _probe_worker(args._worker_probe, args._probe_module)
        print(PROBE_RESULT_PREFIX + json.dumps(result))
        return 1 if "error" in result else 0
    try:
        report = inspect_runtime(
            include_devices=args.devices, probe_metal=not (args.no_metal or args.metadata_only),
            import_backends=not args.metadata_only, models_dir=args.models_dir.expanduser(), profiles_file=args.profiles_file.expanduser(),
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ready_for_model_smoke"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
