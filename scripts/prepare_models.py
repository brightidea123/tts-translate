#!/usr/bin/env python3
"""Inspect baseline model assets, or explicitly download their pinned snapshots.

No arguments only report local status. Run --download in a regular terminal to
fetch complete snapshots. --download --metadata-only fetches configuration and
tokenizer assets while leaving existing weight files in place; those existing
weights still require comparison with the pinned repository's checksums.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from tts_runtime.model_assets import LARGE_V3_REPOSITORY, LARGE_V3_REVISION, LARGE_V3_WEIGHTS_SHA256

BASELINE_MODELS = {
    "whisper": {
        "repo_id": LARGE_V3_REPOSITORY,
        "revision": LARGE_V3_REVISION,
        "expected_sha256": {"weights.npz": LARGE_V3_WEIGHTS_SHA256},
    },
    "whisper-turbo": {
        "repo_id": "mlx-community/whisper-large-v3-turbo",
        "revision": "a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb",
    },
    "translation": {
        "repo_id": "mlx-community/translategemma-12b-it-4bit",
        "revision": "f3dcfd54df14672fbcf0731086fb47a797a943ae",
    },
}
WEIGHT_PATTERNS = ["*.safetensors", "*.npz", "*.bin", "*.pt", "*.pth", "*.onnx", "*.gguf"]
WEIGHT_SUFFIXES = {".safetensors", ".npz", ".bin", ".pt", ".pth", ".onnx", ".gguf"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.rglob("*")
                  if path.is_file() and ".cache" not in path.relative_to(directory).parts)


def model_directory(models_dir: Path, model: dict[str, Any]) -> Path:
    return models_dir / model["repo_id"].rsplit("/", 1)[-1]


def local_status(models_dir: Path, names: list[str], metadata_only: bool) -> dict[str, Any]:
    models = {}
    for name in names:
        model = BASELINE_MODELS[name]
        directory = model_directory(models_dir, model)
        files = model_files(directory)
        models[name] = {
            **model, "directory": str(directory), "directory_exists": directory.is_dir(),
            "config_present": (directory / "config.json").is_file(),
            "weight_files": [str(path.relative_to(directory)) for path in files if path.suffix in WEIGHT_SUFFIXES],
            "file_count": len(files),
        }
    return {
        "downloads_run": False, "requested_mode": "metadata-only" if metadata_only else "full-snapshot",
        "models": models, "manifest": str(models_dir / "model-manifest.json"),
        "manifest_present": (models_dir / "model-manifest.json").is_file(),
        "note": "Status only; no network access, model loading, or inference. Use --download to fetch the pinned snapshots.",
    }


def fetch_model(name: str, models_dir: Path, metadata_only: bool, max_workers: int) -> dict[str, Any]:
    # Set before importing Hub: it reads its offline setting during import.
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    snapshot_download = importlib.import_module("huggingface_hub").snapshot_download
    model = BASELINE_MODELS[name]
    directory = model_directory(models_dir, model)
    snapshot_download(
        repo_id=model["repo_id"], revision=model["revision"], local_dir=str(directory),
        ignore_patterns=WEIGHT_PATTERNS if metadata_only else None, max_workers=max_workers,
    )
    files = {}
    for path in model_files(directory):
        relative = str(path.relative_to(directory))
        files[relative] = {
            "size_bytes": path.stat().st_size, "sha256": sha256_file(path),
            "included_in_snapshot_request": not metadata_only or path.suffix not in WEIGHT_SUFFIXES,
        }
    if not files:
        raise RuntimeError(f"Snapshot returned no files for {model['repo_id']}")
    if not metadata_only:
        for filename, expected in model.get("expected_sha256", {}).items():
            if files.get(filename, {}).get("sha256") != expected:
                raise RuntimeError(f"Model file failed the pinned SHA-256 check: {directory / filename}")
    return {
        **model, "directory": str(directory),
        "download_mode": "metadata-only" if metadata_only else "full-snapshot",
        "complete_snapshot_requested": not metadata_only,
        "download_completed_at": datetime.now(timezone.utc).isoformat(), "files": files,
        "integrity_note": (
            "SHA-256 records local file bytes. Existing weights were excluded from this request; "
            "their correspondence to the pinned revision has not been verified."
            if metadata_only else
            "The pinned snapshot download completed. SHA-256 records local file bytes; inference has not been tested."
        ),
    }


def write_manifest(models_dir: Path, downloaded: dict[str, Any]) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / "model-manifest.json"
    previous_models = {}
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("schema_version") != 1 or not isinstance(previous.get("models"), dict):
            raise ValueError(f"Unsupported existing manifest: {path}")
        previous_models = previous["models"]
    payload = {"schema_version": 1, "updated_at": datetime.now(timezone.utc).isoformat(),
               "models": {**previous_models, **downloaded}}
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=models_dir, suffix=".json", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--download", action="store_true", help="Explicitly permit Hub access and write model files")
    action.add_argument("--dry-run", action="store_true", help="Report the local status and pinned download plan (the default)")
    parser.add_argument("--model", choices=tuple(BASELINE_MODELS), action="append", help="Model to prepare; repeat to select multiple (default: whisper large-v3 and translation)")
    parser.add_argument("--models-dir", type=Path, default=REPO_ROOT / "models")
    parser.add_argument("--metadata-only", action="store_true", help="Skip weight files; retain and hash any weights already present")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.max_workers < 1:
        parser.error("--max-workers must be positive")
    models_dir = args.models_dir.expanduser().resolve()
    names = list(dict.fromkeys(args.model or ("whisper", "translation")))
    report = local_status(models_dir, names, args.metadata_only)
    if not args.download:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    report["downloads_run"] = True
    report["model_inference_run"] = False
    report["note"] = "Explicit snapshot download and SHA-256 inventory; model inference was not run."
    completed = {}
    try:
        for name in names:
            completed[name] = fetch_model(name, models_dir, args.metadata_only, args.max_workers)
            # Persist each completed model even if a later download is interrupted.
            write_manifest(models_dir, {name: completed[name]})
        report["models"] = completed
        report["manifest_present"] = True
        report["ok"] = True
    except Exception as exc:
        report["models_completed_this_run"] = list(completed)
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["ok"] = False
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
