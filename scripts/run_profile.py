#!/usr/bin/env python3
"""Launch the saved application with an explicit, source-backed model profile.

Example:
    python scripts/run_profile.py baseline --source AR --target HE --dump-config
    python scripts/run_profile.py baseline --source TR --target DE -- --no-warmup

Hub downloads are disabled unless --allow-download is given. Existing local
models under models/<repository-basename> are preferred over their Hub IDs.
Live audio defaults to the bounded ring scheduler, sounddevice capture and
energy VAD. Use --audio-io apple-raw or apple-vpio for the native bridge instead.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from translation_prompts import normalize_translation_language

APPLICATION_LANGUAGES = {"ar", "he", "de", "tr", "fr", "ru", "en", "es"}


def load_profiles(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    if document.get("schema_version") != 1 or not isinstance(document.get("profiles"), dict):
        raise ValueError(f"Unsupported or malformed model profiles: {path}")
    return document


def application_language(value: str) -> str:
    normalized = normalize_translation_language(value)
    if normalized not in APPLICATION_LANGUAGES:
        supported = ", ".join(sorted(code.upper() for code in APPLICATION_LANGUAGES))
        raise ValueError(f"Application language must be one of {supported}; got {value!r}")
    return normalized.upper()


def prefer_local_model(model_ref: str, models_dir: Path, override: str | None = None) -> str:
    if override:
        path = Path(override).expanduser()
        return str(path.resolve()) if path.exists() else override
    candidate = models_dir / model_ref.rstrip("/").rsplit("/", 1)[-1]
    return str(candidate.resolve()) if candidate.is_dir() else model_ref


def build_command(
    profile_name: str,
    source: str,
    target: str,
    *,
    profiles_file: Path = REPO_ROOT / "configs/model-profiles.json",
    models_dir: Path = REPO_ROOT / "models",
    python: str | None = None,
    asr_model: str | None = None,
    translation_model: str | None = None,
    mode: str = "both",
    application_profile: str | None = None,
    audio_io: str = "sounddevice",
    streaming_engine: str = "ring",
    vad_backend: str = "energy",
    input_device: str | None = None,
    dump_config: bool = False,
    extra_args: list[str] | None = None,
) -> list[str]:
    document = load_profiles(profiles_file)
    if profile_name not in document["profiles"]:
        choices = ", ".join(sorted(document["profiles"]))
        raise ValueError(f"Unknown model profile {profile_name!r}; choose {choices}")
    src, dst = application_language(source), application_language(target)
    if src == dst:
        raise ValueError("Source and target languages must differ for the live translator")
    profile = document["profiles"][profile_name]
    asr = profile["asr"]
    translation = profile["translation"]
    if asr["backend"] not in {"mlx-whisper", "mlx-vlm-gemma"}:
        raise ValueError(f"Unsupported ASR backend in profile: {asr['backend']}")
    if translation["backend"] not in {"mlx-lm", "mlx-vlm-gemma", "nllb"}:
        raise ValueError(f"Unsupported translation backend in profile: {translation['backend']}")
    if mode not in {"audio", "text", "both"}:
        raise ValueError(f"Unknown application mode: {mode}")
    if audio_io not in {"auto", "sounddevice", "apple-raw", "apple-vpio"}:
        raise ValueError(f"Unknown audio backend: {audio_io}")
    if streaming_engine not in {"auto", "ring", "legacy"}:
        raise ValueError(f"Unknown streaming engine: {streaming_engine}")
    if vad_backend not in {"auto", "webrtc", "silero", "energy"}:
        raise ValueError(f"Unknown VAD backend: {vad_backend}")
    selected_profile = application_profile or ("accurate" if profile_name == "quality_comparison" else "live")
    command = [
        python or sys.executable,
        str(REPO_ROOT / "tts-transcribe.py"),
        "--mode", mode,
        "--profile", selected_profile,
        "--source-language", src,
        "--target-language", dst,
        "--stt-backend", asr["backend"],
        "--translation-backend", translation["backend"],
        "--translation-model", prefer_local_model(translation["model"], models_dir, translation_model),
        "--audio-io", audio_io,
        "--streaming-engine", streaming_engine,
        "--vad-backend", vad_backend,
    ]
    asr_flag = "--mlx-vlm-model" if asr["backend"] == "mlx-vlm-gemma" else "--whisper-model"
    command.extend([asr_flag, prefer_local_model(asr["model"], models_dir, asr_model)])
    if input_device is not None:
        command.extend(["--input-device", input_device])
    if dump_config:
        command.append("--dump-config")
    command.extend(extra_args or [])
    return command


def main(argv: list[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if "--" in tokens:
        boundary = tokens.index("--")
        tokens, passthrough = tokens[:boundary], tokens[boundary + 1:]
    else:
        passthrough = []
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("profile", nargs="?", default="baseline", help="Model profile from configs/model-profiles.json")
    parser.add_argument("--source", "--source-language", default="TR")
    parser.add_argument("--target", "--target-language", default="EN")
    parser.add_argument("--profiles-file", type=Path, default=REPO_ROOT / "configs/model-profiles.json")
    parser.add_argument("--models-dir", type=Path, default=REPO_ROOT / "models")
    parser.add_argument("--asr-model", help="Explicit local model path or Hub ID")
    parser.add_argument("--translation-model", help="Explicit local model path or Hub ID")
    parser.add_argument("--mode", choices=["audio", "text", "both"], default="both")
    parser.add_argument("--application-profile", choices=["live", "balanced", "stable", "accurate", "m4-max-live"])
    parser.add_argument("--audio-io", choices=["auto", "sounddevice", "apple-raw", "apple-vpio"], default="sounddevice")
    parser.add_argument("--streaming-engine", choices=["auto", "ring", "legacy"], default="ring")
    parser.add_argument("--vad-backend", choices=["auto", "webrtc", "silero", "energy"], default="energy")
    parser.add_argument("--input-device")
    parser.add_argument("--dump-config", action="store_true", help="Ask the core to print effective configuration and exit")
    parser.add_argument("--dry-run", "--print-command", action="store_true", help="Print the launch command without starting the application")
    parser.add_argument("--allow-download", action="store_true", help="Allow the model libraries to fetch missing Hub assets")
    args = parser.parse_args(tokens)
    try:
        command = build_command(
            args.profile, args.source, args.target,
            profiles_file=args.profiles_file.expanduser(), models_dir=args.models_dir.expanduser(),
            asr_model=args.asr_model, translation_model=args.translation_model, mode=args.mode,
            application_profile=args.application_profile, audio_io=args.audio_io,
            streaming_engine=args.streaming_engine, vad_backend=args.vad_backend,
            input_device=args.input_device, dump_config=args.dump_config, extra_args=passthrough,
        )
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(shlex.join(command))
        return 0
    environment = os.environ.copy()
    environment["HF_HUB_OFFLINE"] = "0" if args.allow_download else "1"
    environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
    try:
        return subprocess.run(command, cwd=REPO_ROOT, env=environment, check=False).returncode
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
