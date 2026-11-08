#!/usr/bin/env python3
"""Inspect or explicitly generate six local macOS speech fixtures without playback.

Run --generate to synthesize the public test phrases with installed matching
macOS voices. say creates temporary audio; afconvert produces 16 kHz mono Int16
PCM WAV files in fixtures/generated. No microphone or model inference is used.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any
import wave

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.smoke_models import SPEECH_FIXTURES

PREFERRED_VOICES = {"ar": "Maged", "he": "Carmit", "de": "Anna", "tr": "Yelda", "fr": "Thomas", "ru": "Milena"}


def parse_voices(output: str) -> list[dict[str, str]]:
    voices = []
    for line in output.splitlines():
        match = re.match(r"^(.+?)\s+([A-Za-z]{2,3}[_-][A-Za-z0-9_-]+)\s+#", line)
        if match:
            name, locale = match.groups()
            voices.append({"name": name.strip(), "locale": locale, "language": re.split(r"[_-]", locale)[0].lower()})
    return voices


def choose_voice(language: str, voices: list[dict[str, str]], override: str | None = None) -> dict[str, str]:
    matching = [voice for voice in voices if voice["language"] == language]
    if override:
        matching = [voice for voice in matching if voice["name"] == override]
    if not matching:
        suffix = f" named {override!r}" if override else ""
        raise ValueError(f"No installed {language.upper()} voice{suffix}; install a matching voice in macOS Spoken Content settings")
    preferred = PREFERRED_VOICES[language]
    return next((voice for voice in matching if voice["name"] == preferred), matching[0])


def validate_wav(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as handle:
        if (handle.getnchannels(), handle.getframerate(), handle.getsampwidth(), handle.getcomptype()) != (1, 16000, 2, "NONE"):
            raise ValueError(f"Fixture is not 16 kHz mono Int16 PCM WAV: {path}")
        frames = handle.getnframes()
        if not frames:
            raise ValueError(f"Fixture contains no samples: {path}")
    return {"sample_rate": 16000, "channels": 1, "sample_width_bytes": 2, "frames": frames,
            "duration_seconds": frames / 16000.0, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def generate_fixture(language: str, voice: dict[str, str], output_dir: Path, say: str, afconvert: str) -> dict[str, Any]:
    phrase = SPEECH_FIXTURES[language]
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{language}.wav"
    with tempfile.TemporaryDirectory(prefix=f"fixture-{language}-", dir=output_dir) as temporary:
        work = Path(temporary)
        text = work / "phrase.txt"
        aiff = work / "speech.aiff"
        wav = work / "speech.wav"
        text.write_text(phrase + "\n", encoding="utf-8")
        subprocess.run([say, "-v", voice["name"], "-o", str(aiff), "-f", str(text)], check=True, timeout=90, capture_output=True, text=True)
        subprocess.run([afconvert, "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(wav)], check=True, timeout=30, capture_output=True, text=True)
        audio = validate_wav(wav)
        wav.replace(destination)
    return {"language": language, "text": phrase, "voice": voice["name"], "locale": voice["locale"],
            "path": str(destination), **audio}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generate", action="store_true", help="Explicitly synthesize and replace the selected generated fixtures")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "fixtures" / "generated")
    parser.add_argument("--languages", nargs="+", type=str.lower, choices=tuple(SPEECH_FIXTURES), default=list(SPEECH_FIXTURES))
    parser.add_argument("--voice", action="append", default=[], metavar="LANG=NAME", help="Require an installed matching voice; repeat for more languages")
    args = parser.parse_args(argv)
    overrides = {}
    for entry in args.voice:
        language, separator, name = entry.partition("=")
        language = language.lower()
        if not separator or language not in SPEECH_FIXTURES or not name.strip():
            parser.error("--voice must be LANG=NAME using ar, he, de, tr, fr, or ru")
        overrides[language] = name.strip()
    output_dir = args.output_dir.expanduser().resolve()
    languages = list(dict.fromkeys(args.languages))
    say, afconvert = shutil.which("say"), shutil.which("afconvert")
    report: dict[str, Any] = {
        "generation_run": False, "playback_run": False, "microphone_opened": False,
        "model_inference_run": False, "system": platform.system(), "tools": {"say": say, "afconvert": afconvert},
        "output_dir": str(output_dir),
        "fixtures": [{"language": language, "text": SPEECH_FIXTURES[language],
                      "path": str(output_dir / f"{language}.wav"), "present": (output_dir / f"{language}.wav").is_file()}
                     for language in languages],
    }
    if not args.generate:
        report["note"] = "Status only; use --generate to synthesize with installed matching macOS voices. No audio was generated or played."
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    report["generation_run"] = True
    try:
        if platform.system() != "Darwin" or not say or not afconvert:
            raise RuntimeError("Fixture generation requires macOS with say and afconvert")
        voices_result = subprocess.run([say, "-v", "?"], check=True, capture_output=True, text=True, timeout=15)
        voices = parse_voices(voices_result.stdout)
        # Validate all language directions before creating the first fixture.
        selected = {language: choose_voice(language, voices, overrides.get(language)) for language in languages}
        generated = [generate_fixture(language, selected[language], output_dir, say, afconvert) for language in languages]
        payload = {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
                   "source": "macOS say using installed voices", "fixtures": generated,
                   "note": "Synthetic smoke fixtures; no recognition accuracy or translation-quality evaluation."}
        manifest = output_dir / "fixtures-manifest.json"
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output_dir, suffix=".json", delete=False) as handle:
            temporary = Path(handle.name)
            try:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        temporary.replace(manifest)
        report["fixtures"] = generated
        report["manifest"] = str(manifest)
        report["ok"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            report["command_stderr"] = exc.stderr.strip()
        report["ok"] = False
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
