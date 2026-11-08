#!/usr/bin/env python3
"""Run explicitly requested local-model smoke tests; never open a microphone.

No arguments produce a no-inference diagnostic. Supply --asr-model and --audio
for Whisper, and/or --translation-model with --text or --all-pairs for MLX-LM.
Hub access is offline by default; --allow-download explicitly permits fetching
missing assets. Nonempty outputs are smoke evidence, not measured accuracy.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import time
import wave
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from translation_prompts import build_translation_messages, normalize_translation_language

SPEECH_FIXTURES = {
    "ar": "مرحباً، أريد تأكيد موعدي غداً في الساعة العاشرة صباحاً.",
    "he": "שלום, אני רוצה לאשר את הפגישה שלי מחר בשעה עשר בבוקר.",
    "de": "Hallo, ich möchte meinen Termin morgen um zehn Uhr bestätigen.",
    "tr": "Merhaba, yarın sabah saat ondaki randevumu onaylamak istiyorum.",
    "fr": "Bonjour, je voudrais confirmer mon rendez-vous demain à dix heures du matin.",
    "ru": "Здравствуйте, я хочу подтвердить встречу завтра в десять часов утра.",
}


def translation_cases(source: str, target: str, text: str | None, all_pairs: bool) -> list[tuple[str, str, str]]:
    if all_pairs:
        return [(src, dst, phrase) for src, phrase in SPEECH_FIXTURES.items() for dst in SPEECH_FIXTURES if src != dst]
    src, dst = normalize_translation_language(source), normalize_translation_language(target)
    if src == dst:
        raise ValueError("Choose different source and target languages for the translation smoke test")
    phrase = text if text is not None else SPEECH_FIXTURES.get(src)
    if not isinstance(phrase, str) or not phrase.strip():
        raise ValueError("Supply nonempty --text for a language without a built-in fixture")
    return [(src, dst, phrase)]


def load_pcm_fixture(path: Path):
    with wave.open(str(path), "rb") as handle:
        if (handle.getnchannels(), handle.getframerate(), handle.getsampwidth(), handle.getcomptype()) != (1, 16000, 2, "NONE"):
            raise ValueError("ASR fixture must be uncompressed 16 kHz mono 16-bit PCM WAV")
        frames = handle.readframes(handle.getnframes())
    if not frames:
        raise ValueError("ASR fixture contains no samples")
    import numpy as np
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


def synchronize() -> None:
    mx = importlib.import_module("mlx.core")
    if hasattr(mx, "synchronize"):
        mx.synchronize()


def asr_smoke(model_ref: str, audio: Path, source: str, require_nonempty: bool = False) -> dict[str, Any]:
    samples = load_pcm_fixture(audio)
    source_code = normalize_translation_language(source)
    import mlx_whisper
    synchronize()
    started = time.perf_counter()
    result = mlx_whisper.transcribe(
        samples, path_or_hf_repo=model_ref, language=source_code,
        task="transcribe", condition_on_previous_text=False, word_timestamps=False,
        verbose=False, temperature=0.0,
    )
    synchronize()
    elapsed = time.perf_counter() - started
    text = str(result.get("text", "")).strip()
    return {
        "backend": "mlx-whisper", "model": model_ref, "audio": str(audio),
        "source_language": source_code, "text": text,
        "audio_seconds": len(samples) / 16000.0, "elapsed_seconds": elapsed,
        "real_time_factor_including_first_load": elapsed / (len(samples) / 16000.0),
        "output_nonempty": bool(text), "ok": bool(text) if require_nonempty else True,
        "quality_evaluated": False, "note": "First call includes model initialization; no recognition accuracy benchmark",
    }


def translation_smoke(model_ref: str, cases: list[tuple[str, str, str]], max_tokens: int) -> dict[str, Any]:
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    started = time.perf_counter()
    model, tokenizer = load(model_ref)
    synchronize()
    load_seconds = time.perf_counter() - started
    results = []
    for source, target, text in cases:
        messages = build_translation_messages(text, source, target, model_ref)
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        synchronize()
        started = time.perf_counter()
        pieces = []
        token_count = 0
        for response in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens, sampler=make_sampler(temp=0.0)):
            pieces.append(getattr(response, "text", ""))
            token_count = int(getattr(response, "generation_tokens", token_count))
        synchronize()
        output = "".join(pieces).strip()
        results.append({
            "source": source, "target": target, "input": text, "output": output,
            "elapsed_seconds": time.perf_counter() - started, "generation_tokens": token_count,
            "ok": bool(output), "quality_evaluated": False,
        })
    return {"backend": "mlx-lm", "model": model_ref, "load_seconds": load_seconds, "cases": results,
            "ok": all(case["ok"] for case in results), "note": "One shared model instance; outputs require human language-quality review"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--asr-model", help="Local MLX Whisper directory or cached Hub ID")
    parser.add_argument("--audio", type=Path, help="16 kHz mono Int16 PCM WAV fixture")
    parser.add_argument("--translation-model", help="Local MLX-LM directory or cached Hub ID")
    parser.add_argument("--source", default="TR")
    parser.add_argument("--target", default="HE")
    parser.add_argument("--text")
    parser.add_argument("--all-pairs", action="store_true", help="Explicitly test all 30 directed pairs of the six fixture languages")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--require-nonempty-asr", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--output", type=Path, help="Also save the JSON smoke report")
    args = parser.parse_args(argv)
    if bool(args.asr_model) != bool(args.audio):
        parser.error("--asr-model and --audio must be supplied together")
    if args.all_pairs and not args.translation_model:
        parser.error("--all-pairs requires --translation-model")
    if args.text is not None and not args.translation_model:
        parser.error("--text requires --translation-model")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    os.environ["HF_HUB_OFFLINE"] = "0" if args.allow_download else "1"
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    jobs = []
    if args.asr_model:
        jobs.append(("asr", lambda: asr_smoke(args.asr_model, args.audio.expanduser(), args.source, args.require_nonempty_asr)))
    if args.translation_model:
        jobs.append(("translation", lambda: translation_smoke(
            args.translation_model, translation_cases(args.source, args.target, args.text, args.all_pairs), args.max_tokens)))
    report: dict[str, Any] = {
        "model_inference_requested": bool(jobs), "model_inference_attempted": False,
        "model_inference_run": False, "model_inference_completed": False,
        "stages": {stage: {"attempted": False, "completed": False, "skipped": False} for stage, _ in jobs},
        "downloads_allowed": args.allow_download, "quality_evaluated": False, "results": {}, "errors": [],
        "inference_accounting_note": "Attempted stages include input validation and backend loading. model_inference_run counts only stages that returned inference results.",
    }
    if not report["model_inference_requested"]:
        report["note"] = "No models requested; no model imports, downloads, inference, microphone capture, or playback performed"
    for index, (stage, run_stage) in enumerate(jobs):
        report["model_inference_attempted"] = True
        status = report["stages"][stage]
        status["attempted"] = True
        try:
            result = run_stage()
            report["results"][stage] = result
            status["completed"] = True
            status["passed"] = result["ok"]
            report["model_inference_run"] = True
        except Exception as exc:
            status["error"] = f"{type(exc).__name__}: {exc}"
            report["errors"].append(f"{stage}: {status['error']}")
            # Failed native initialization can leave process-global bindings in
            # an unsafe state. Importing another MLX backend can then abort the
            # interpreter instead of raising a catchable Python exception.
            for pending_stage, _ in jobs[index + 1:]:
                pending = report["stages"][pending_stage]
                pending["skipped"] = True
                pending["skip_reason"] = f"Earlier stage {stage!r} failed; native backend initialization is not retried in the same process"
            break
    report["model_inference_completed"] = bool(jobs) and all(status["completed"] for status in report["stages"].values())
    report["ok"] = not report["errors"] and all(result["ok"] for result in report["results"].values())
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.expanduser().write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
