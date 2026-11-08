"""Resolve MLX Whisper assets without importing MLX or opening audio devices."""
from __future__ import annotations

import json
from pathlib import Path


LARGE_V3_REPOSITORY = "mlx-community/whisper-large-v3-mlx"
LARGE_V3_REVISION = "49e6aa286ad60c14352c404340ded53710378a11"
LARGE_V3_WEIGHTS_SHA256 = "05ff791ce3630fae47e7c51004e9666204d786246ec07cac6110af768099b40d"


def mlx_whisper_assets_present(directory: Path) -> bool:
    """Check required files; byte integrity is verified by the preparation tool."""
    try:
        config = directory / "config.json"
        if not config.is_file() or not isinstance(json.loads(config.read_text(encoding="utf-8")), dict):
            return False
        return any((directory / name).is_file() and (directory / name).stat().st_size > 0
                   for name in ("weights.safetensors", "weights.npz"))
    except (OSError, ValueError):
        return False


def resolve_mlx_whisper_reference(model_ref: str, models_dir: Path) -> str:
    model_ref = str(model_ref).strip()
    direct = Path(model_ref).expanduser()
    if direct.exists():
        return str(direct.resolve())
    model_ref = model_ref.replace("mlx-community:", "mlx-community/", 1)
    if "/" not in model_ref:
        model_ref = "mlx-community/whisper-" + model_ref
    if model_ref in {"mlx-community/whisper-large-v3", LARGE_V3_REPOSITORY}:
        model_ref = LARGE_V3_REPOSITORY
        candidates = [models_dir / "whisper-large-v3-mlx", models_dir / "whisper-large-v3"]
    elif model_ref.startswith("mlx-community/"):
        candidates = [models_dir / model_ref.rsplit("/", 1)[-1]]
    else:
        candidates = []
    for candidate in candidates:
        if mlx_whisper_assets_present(candidate):
            return str(candidate.resolve())
    return model_ref


def require_cached_mlx_whisper(model_ref: str, *, offline: bool, preparation_command: str) -> str:
    """Resolve an offline Hub snapshot before warmup, with a repair command.

    Never disable offline mode or silently choose a different model. Local-only
    Hub lookup also reuses models downloaded outside this project's directory.
    """
    direct = Path(model_ref).expanduser()
    if direct.exists():
        if mlx_whisper_assets_present(direct):
            return str(direct.resolve())
        raise RuntimeError(f"Incomplete MLX Whisper model at {direct}: config.json and nonempty "
                           f"weights.npz or weights.safetensors are required. Run: {preparation_command}")
    if not offline:
        return model_ref
    try:
        from huggingface_hub import snapshot_download
        directory = Path(snapshot_download(repo_id=model_ref, local_files_only=True))
        if mlx_whisper_assets_present(directory):
            return str(directory.resolve())
    except Exception as exc:
        raise RuntimeError(f"MLX Whisper model {model_ref!r} is not cached completely. "
                           f"Offline mode is enabled. Download it once before starting: {preparation_command}") from exc
    raise RuntimeError(f"Cached MLX Whisper model {model_ref!r} is incomplete. Run: {preparation_command}")
