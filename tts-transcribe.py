#!/usr/bin/env python3
"""
Responsive TR↔EN instant translator for phone-call audio + typed replies.

Pipeline goals:
- keep the audio callback extremely light
- let STT continue even when MT/TTS is still busy
- prefer responsiveness over processing every stale chunk
- keep typed reply flow separate from incoming-call transcription flow
"""

from __future__ import annotations

import argparse
import codecs
import hashlib
from collections import deque
from datetime import datetime, timezone
from contextlib import ExitStack, nullcontext, redirect_stderr, redirect_stdout
import importlib
import importlib.util
import io
import json
import logging
import multiprocessing as mp
from multiprocessing import shared_memory
import os
import re
from functools import lru_cache
from heapq import heapify
import shlex
import select
import shutil
import subprocess
import struct
import sys
import termios
import threading
import time
import tty
import unicodedata
import warnings
import wave
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Full, Queue, PriorityQueue
from tempfile import TemporaryDirectory
from typing import Iterator, Literal

import numpy as np

from translation_prompts import build_translation_messages
from tts_runtime.worker import ProcessStage, StageEvent
from tts_runtime.text_output import clean_model_output, configure_translation_stop_tokens
from tts_runtime.asr_quality import pcm_quality, summarize_whisper_result, issue_reasons, final_preview_disagrees
from tts_runtime.audio_evidence import AudioEvidence, create_session_directory
from tts_runtime.model_assets import resolve_mlx_whisper_reference, require_cached_mlx_whisper, LARGE_V3_REPOSITORY
from tts_runtime.selection_inference import SelectionTask, SelectionResult
from optional_dependencies import optional_dependency


def eprint(*args, **kwargs) -> None:
    """Write diagnostics to stderr so stdout stays reserved for interactive UI or data."""
    kwargs.setdefault("file", sys.stderr)
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    import webrtcvad
except ImportError:
    webrtcvad = None

def _configure_torch_runtime(module) -> None:
    if hasattr(module, "set_float32_matmul_precision"):
        module.set_float32_matmul_precision("high")
    mps = getattr(getattr(module, "backends", None), "mps", None)
    if mps is not None and hasattr(mps, "allow_tf32"):
        mps.allow_tf32 = True


torch = optional_dependency("torch", configure=_configure_torch_runtime)
WhisperModel = optional_dependency("faster_whisper", "WhisperModel")
AutoModelForSeq2SeqLM = optional_dependency("transformers", "AutoModelForSeq2SeqLM")
AutoTokenizer = optional_dependency("transformers", "AutoTokenizer")

mlx_whisper = None
mlx_whisper_import_error: Exception | None = None
mlx_whisper_beam_available: bool | None = None
mlx_whisper_beam_fallback_reported = False
MLX_WHISPER_INSTALLED = importlib.util.find_spec("mlx_whisper") is not None

mlx_vlm_load = None
mlx_vlm_generate = None
mlx_vlm_apply_chat_template = None
mlx_vlm_import_error: Exception | None = None
mlx_vlm_model = None
mlx_vlm_processor = None
mlx_vlm_model_ref = ""
MLX_VLM_INSTALLED = importlib.util.find_spec("mlx_vlm") is not None
mlx_core = None
mlx_core_import_error: Exception | None = None
mlx_thread_state = threading.local()
mlx_vlm_thread_state = threading.local()

mlx_lm_load = None
mlx_lm_stream_generate = None
mlx_lm_import_error: Exception | None = None
mlx_translation_stop_ids: frozenset[int] = frozenset()
runtime_session_id = ""
runtime_session_dir: Path | None = None
runtime_audio_dir: Path | None = None
audio_review_mode = "off"
audio_evidence: AudioEvidence | None = None
asr_request_context: dict = {}
last_asr_diagnostics: dict = {}
last_decoder_quality: dict = {}
last_prepared_audio: np.ndarray | None = None
asr_request_sequence = 0
pending_audio_evidence: dict = {}
active_ring_feeder = None
MLX_LM_INSTALLED = importlib.util.find_spec("mlx_lm") is not None
SILERO_VAD_INSTALLED = importlib.util.find_spec("silero_vad") is not None
ONNXRUNTIME_INSTALLED = importlib.util.find_spec("onnxruntime") is not None

Kokoro = optional_dependency("kokoro_onnx", "Kokoro")

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
warnings.filterwarnings("ignore", message=r"You are using a model of type `gemma4` to instantiate a model of type ``.*")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").addFilter(
    lambda record: record.getMessage() != "NOTE: Redirects are currently not supported in MacOs."
)
@dataclass(frozen=True, slots=True)
class LanguageSpec:
    cli: str
    name: str
    whisper: str
    nllb: str
    kokoro_lang: str
    apple_locale: str
    native_scripts: tuple[str, ...] = ("LATIN",)
    say_voice: str = ""
    edge_voice: str = ""


LANGUAGE_SPECS: dict[str, LanguageSpec] = {
    "AR": LanguageSpec("AR", "Arabic", "ar", "arb_Arab", "ar", "ar-SA", ("ARABIC",), edge_voice="ar-SA-ZariyahNeural"),
    "DE": LanguageSpec("DE", "German", "de", "deu_Latn", "de", "de-DE", edge_voice="de-DE-KatjaNeural"),
    "EN": LanguageSpec("EN", "English", "en", "eng_Latn", "en-us", "en-US", say_voice="Samantha", edge_voice="en-US-JennyNeural"),
    "ES": LanguageSpec("ES", "Spanish", "es", "spa_Latn", "es", "es-ES", edge_voice="es-ES-ElviraNeural"),
    "FR": LanguageSpec("FR", "French", "fr", "fra_Latn", "fr", "fr-FR", edge_voice="fr-FR-DeniseNeural"),
    "HE": LanguageSpec("HE", "Hebrew", "he", "heb_Hebr", "he", "he-IL", ("HEBREW",), edge_voice="he-IL-HilaNeural"),
    "RU": LanguageSpec("RU", "Russian", "ru", "rus_Cyrl", "ru", "ru-RU", ("CYRILLIC",), edge_voice="ru-RU-SvetlanaNeural"),
    "TR": LanguageSpec("TR", "Turkish", "tr", "tur_Latn", "tr", "tr-TR", say_voice="Yelda", edge_voice="tr-TR-EmelNeural"),
}
LANGUAGE_ALIASES = {
    "SP": "ES",
    "SPA": "ES",
    "ES": "ES",
    "ENG": "EN",
    "TUR": "TR",
    "ARA": "AR",
    "ARB": "AR",
    "GER": "DE",
    "DEU": "DE",
    "FRE": "FR",
    "FRA": "FR",
    "HEB": "HE",
    "RUS": "RU",
}
LANGUAGE_CHOICES = tuple(sorted(set(LANGUAGE_SPECS) | set(LANGUAGE_ALIASES)))


def normalize_language_code(value: str) -> str:
    code = str(value).strip().upper()
    code = LANGUAGE_ALIASES.get(code, code)
    if code not in LANGUAGE_SPECS:
        supported = ", ".join(LANGUAGE_CHOICES)
        raise argparse.ArgumentTypeError(f"unsupported language code {value!r}; supported: {supported}")
    return code


def language_spec(code: str) -> LanguageSpec:
    return LANGUAGE_SPECS[normalize_language_code(code)]


def source_language_code() -> str:
    return normalize_language_code(args.source_language)


def target_language_code() -> str:
    return normalize_language_code(args.target_language)


def source_language() -> LanguageSpec:
    return language_spec(args.source_language)


def target_language() -> LanguageSpec:
    return language_spec(args.target_language)

def asr_language_code() -> str:
    """Return the Whisper/ASR language code that must be forced for source audio."""
    return source_language().whisper


def asr_language_kwargs() -> dict[str, str]:
    """Backend kwargs that force transcription language unless auto-detect is explicitly enabled."""
    if not args.force_asr_language:
        return {}
    return {"language": asr_language_code()}


def asr_language_cli_args(flag: str) -> list[str]:
    """CLI args that force transcription language unless auto-detect is explicitly enabled."""
    if not args.force_asr_language:
        return []
    return [flag, asr_language_code()]


def asr_language_display() -> str:
    return asr_language_code() if args.force_asr_language else "auto"



def source_nllb_lang() -> str:
    return source_language().nllb


def target_nllb_lang() -> str:
    return target_language().nllb


def language_name_from_nllb(nllb_code: str) -> str:
    for spec in LANGUAGE_SPECS.values():
        if spec.nllb == nllb_code:
            return spec.name
    return nllb_code


def cli_code_from_nllb(nllb_code: str) -> str:
    for spec in LANGUAGE_SPECS.values():
        if spec.nllb == nllb_code:
            return spec.cli
    return nllb_code[:2].upper()


def reverse_language_code(code: str) -> str:
    code = normalize_language_code(code)
    return source_language_code() if code == target_language_code() else target_language_code()


def active_tts_lang_code(lang_hint: str) -> str:
    try:
        return normalize_language_code(lang_hint)
    except argparse.ArgumentTypeError:
        return "EN" if lang_hint.lower().startswith("en") else "TR"


# ======================
# CLI
# ======================
class StderrArgumentParser(argparse.ArgumentParser):
    """argparse normally writes --help to stdout; this CLI keeps diagnostics on stderr."""

    def print_help(self, file=None) -> None:
        super().print_help(file or sys.stderr)

    def print_usage(self, file=None) -> None:
        super().print_usage(file or sys.stderr)


parser = StderrArgumentParser(description="Live configurable-language call translator")
parser.add_argument("--mode", choices=["audio", "text", "both"], default="both")
parser.add_argument("--source-language", "--src-lang", type=normalize_language_code, default="TR", choices=LANGUAGE_CHOICES, help="spoken/input language code, e.g. TR, AR, EN, HE, DE, SP/ES, RU, FR")
parser.add_argument("--target-language", "--dst-lang", type=normalize_language_code, default="EN", choices=LANGUAGE_CHOICES, help="translation output language code, e.g. EN, TR, AR, HE, DE, SP/ES, RU, FR")
parser.add_argument("--strict-asr-language", action="store_true", default=True, help="reject ASR output that is likely translated/romanized instead of native-script source text")
parser.add_argument("--no-strict-asr-language", dest="strict_asr_language", action="store_false")
parser.add_argument("--force-asr-language", action="store_true", default=True, help="force ASR to use --source-language instead of auto-detecting spoken language")
parser.add_argument("--auto-detect-asr-language", dest="force_asr_language", action="store_false", help="allow ASR language auto-detection; not recommended for live translation")
parser.add_argument("--asr-initial-prompt", default="", help="optional prior-transcript hint for Whisper; leave empty to avoid prompt-language leakage")
parser.add_argument("--allow-english-asr-model", action="store_true", default=False, help="allow .en Whisper models even when source language is not English")
parser.add_argument("--profile", choices=["live", "balanced", "accurate", "stable", "m4-max-live"], default="live", help="bundle of latency/accuracy defaults; stable enables stricter anti-repeat/anti-hallucination gates")
parser.add_argument("--whisper-model", default="large-v3", help="local model path or multilingual Whisper model name")
parser.add_argument("--whisper-compute", default="int8", help="faster-whisper compute type")
parser.add_argument("--stt-backend", choices=["auto", "mlx-vlm-gemma", "mlx-whisper", "dictate-stt", "faster-whisper", "whisper-cli"], default="auto", help="speech-to-text backend")
parser.add_argument("--partial-stt-backend", choices=["same", "auto", "mlx-vlm-gemma", "mlx-whisper", "dictate-stt", "faster-whisper", "whisper-cli", "apple-speech"], default="same", help="optional partial-only ASR backend; apple-speech requires --apple-speech-command")
parser.add_argument("--final-stt-backend", choices=["same", "auto", "mlx-vlm-gemma", "mlx-whisper", "dictate-stt", "faster-whisper", "whisper-cli", "whispercpp-coreml"], default="same", help="optional final-only ASR backend; whispercpp-coreml requires --whispercpp-model")
parser.add_argument("--stt-device", choices=["auto", "cpu", "metal", "ane"], default="auto", help="preferred speech-to-text device")
parser.add_argument("--asr-beam", type=int, default=3, help="higher = more accurate, slower")
parser.add_argument("--partial-asr-beam", type=int, default=1, help="beam size for partial ASR lane")
parser.add_argument("--final-asr-beam", type=int, default=0, help="beam size for final ASR lane; 0 means use --asr-beam")
parser.add_argument("--partial-whisper-model", default="", help="optional faster partial model, e.g. base or small; empty uses --whisper-model")
parser.add_argument("--final-whisper-model", default="", help="optional accurate final model, e.g. large-v3; empty uses --whisper-model")
parser.add_argument("--mlx-vlm-model", default="mlx-community/gemma-4-e4b-it-4bit", help="MLX-VLM audio model for --stt-backend mlx-vlm-gemma")
parser.add_argument("--partial-mlx-vlm-model", default="", help="optional partial-only MLX-VLM audio model; empty uses --mlx-vlm-model")
parser.add_argument("--final-mlx-vlm-model", default="", help="optional final-only MLX-VLM audio model; empty uses --mlx-vlm-model")
parser.add_argument("--mlx-vlm-prompt", default="", help="optional Gemma audio transcription prompt; supports {source_language}, {target_language}, {source_lang}, {target_lang}, and {task_hint}")
parser.add_argument("--mlx-vlm-partial-prompt", default="", help="optional Gemma prompt for partial ASR windows; empty uses --mlx-vlm-prompt/default")
parser.add_argument("--mlx-vlm-final-prompt", default="", help="optional Gemma prompt for final ASR; empty uses --mlx-vlm-prompt/default")
parser.add_argument("--mlx-vlm-max-tokens", type=int, default=256, help="maximum text tokens generated by Gemma audio transcription")
parser.add_argument("--mlx-vlm-temperature", type=float, default=0.0, help="Gemma audio transcription generation temperature")
parser.add_argument("--finalization-hold-ms", type=int, default=140, help="after VAD endpoint, wait this long before finalizing; resumed speech cancels the pending final")
parser.add_argument("--stuck-partial-timeout-ms", type=int, default=6500, help="watchdog timeout: if a displayed partial has no final after this many ms, force-finalize or close it")
parser.add_argument("--inflight-partial-timeout-ms", type=int, default=2200, help="remove UI-only in-flight partial markers if ASR returns no text/result within this many ms")
parser.add_argument("--terminal-lam-alef-divider-compensation", type=int, default=1, help="move the dashboard divider right by this many cells per lam-alef on the same left-panel row; set 0 to disable")
parser.add_argument("--final-transcription-log-debounce-ms", type=int, default=1200, help="wait this long before appending final audio transcription logs so later same-utterance refinements replace earlier ones")
parser.add_argument("--final-dedupe-window-ms", type=int, default=45000, help="suppress exact repeated final text only over proven overlapping source audio within this window; set 0 to disable")
parser.add_argument("--final-dedupe-contained-ratio", type=float, default=0.78, help="legacy reserved option; fuzzy text-only final suppression is disabled")
parser.add_argument("--stuck-partial-action", choices=["finalize", "close"], default="close", help="watchdog action for stale partials: explicitly opt in to finalizing from a partial, or close/remove it")
parser.add_argument("--no-stuck-partial-watchdog", dest="stuck_partial_watchdog", action="store_false", help="disable stuck partial cleanup watchdog")
parser.set_defaults(stuck_partial_watchdog=True)
parser.add_argument("--empty-final-action", choices=["fallback", "close", "keep"], default="close", help="when final ASR is empty after filtering: fallback to last partial, close UI entry, or keep waiting")
parser.add_argument("--final-merge-last-partial", action="store_true", default=False, help="opt in to merging final ASR with an earlier partial; disabled to preserve authoritative final text")
parser.add_argument("--no-final-merge-last-partial", dest="final_merge_last_partial", action="store_false")
parser.add_argument("--final-translation-completeness-mode", choices=["off", "conservative", "aggressive"], default="off", help="how strongly final audio translation should recover details from the last partial")
parser.add_argument("--final-merge-min-final-words", type=int, default=4, help="merge from the last partial when final ASR has fewer words than this")
parser.add_argument("--debug-dropped-asr", action="store_true", default=False, help="log ASR text dropped by sanitizers/language guards")
parser.add_argument("--native-script-min-ratio", type=float, default=0.25, help="minimum native-script letter ratio required for final ASR in strict native-script mode")
parser.add_argument("--partial-native-script-min-ratio", type=float, default=0.20, help="minimum native-script letter ratio required for partial ASR in strict native-script mode")
parser.add_argument("--confidence-ui", action="store_true", default=True, help="show stable prefix and mutable tail separately for partial transcripts")
parser.add_argument("--no-confidence-ui", dest="confidence_ui", action="store_false")
parser.add_argument("--apple-speech-command", default="", help="optional external Apple Speech transcriber command used only with --partial-stt-backend apple-speech")
parser.add_argument("--whispercpp-command", default="whisper-cli", help="whisper.cpp command used only with --final-stt-backend whispercpp-coreml")
parser.add_argument("--whispercpp-model", default="", help="GGML model path for optional whisper.cpp/Core ML final backend")
parser.add_argument("--whispercpp-extra-args", default="", help="extra args for whisper.cpp, space-separated")
parser.add_argument("--max-no-speech-prob", type=float, default=0.75, help="reject STT segments above this no-speech probability")
parser.add_argument("--min-avg-logprob", type=float, default=-1.55, help="reject STT segments below this average logprob")
parser.add_argument("--max-compression-ratio", type=float, default=2.9, help="reject STT segments above this compression ratio")
parser.add_argument("--min-text-chars", type=int, default=2, help="reject transcriptions shorter than this many chars")
parser.add_argument("--min-text-words", type=int, default=1, help="reject transcriptions with fewer than this many words")
parser.add_argument("--apple-vpio-agc", choices=["on", "off"], default="on", help="Apple VoiceProcessingIO automatic gain control")
parser.add_argument("--apple-vpio-bypass-processing", choices=["on", "off"], default="off", help="bypass Apple voice processing while keeping the same native audio unit path")
parser.add_argument("--translation-backend", choices=["auto", "mlx-vlm-gemma", "mlx-lm", "nllb"], default="auto", help="translation backend")
parser.add_argument("--translation-model", default="mlx-community/translategemma-12b-it-4bit")
parser.add_argument("--mt-prompt", default="", help="optional MLX-LM translation prompt/instructions; supports {source_language}, {target_language}, and {text}; if {text} is omitted, it is added to the default translation prompt")
parser.add_argument("--mt-beams", type=int, default=4, help="translation beam size")
parser.add_argument("--mt-process", action="store_true", default=True, help="run translation in a separate process so the UI/coordinator stays responsive")
parser.add_argument("--no-mt-process", dest="mt_process", action="store_false")
parser.add_argument("--startup-timeout", type=float, default=180.0, help="maximum seconds for each model worker to initialize; failures stay visible in the dashboard")
parser.add_argument("--inference-timeout", type=float, default=120.0, help="maximum seconds for one model request before its worker is stopped")
parser.add_argument("--inference-queue-size", type=int, default=16, help="bounded pending requests per model worker; overflow is reported without blocking capture or typing")
parser.add_argument("--mt-queue-size", type=int, default=3, help="max pending translation tasks before stale speculative audio translations are dropped")
parser.add_argument("--mt-chunk-max-chars", type=int, default=1800, help="max source chars per MLX-LM translation request; larger values preserve context, <=0 disables MLX chunking")
parser.add_argument("--stream-mt-deltas", action="store_true", default=True, help="emit display-only MLX-LM translation deltas while chunks are still generating")
parser.add_argument("--no-stream-mt-deltas", dest="stream_mt_deltas", action="store_false")
parser.add_argument("--mt-delta-min-chars", type=int, default=16, help="minimum changed characters before another streaming translation display update")
parser.add_argument("--mt-delta-min-ms", type=int, default=70, help="minimum time between small streaming translation display updates")
parser.add_argument("--tts", choices=["auto", "kokoro", "say", "piper", "edge"], default="auto")
parser.add_argument("--say-voice", default="", help="optional macOS voice override; otherwise select an installed voice matching the spoken language")
parser.add_argument("--edge-voice", default="", help="optional source-language Edge voice override; otherwise use the spoken language's voice")
parser.add_argument("--rate", type=float, default=1.0, help="edge-tts rate multiplier")
parser.add_argument("--kokoro-model", default=os.path.expanduser("~/models/kokoro/kokoro-v1.0.onnx"), help="path to Kokoro ONNX model")
parser.add_argument("--kokoro-voices", default=os.path.expanduser("~/models/kokoro/voices-v1.0.bin"), help="path to Kokoro voices bundle")
parser.add_argument("--kokoro-voice-en", default="af_sarah", help="Kokoro voice for English output")
parser.add_argument("--kokoro-voice-tr", default="", help="Kokoro voice for Turkish output when available")
parser.add_argument("--kokoro-lang-en", default="en-us", help="Kokoro language code for English output")
parser.add_argument("--kokoro-lang-tr", default="tr", help="Kokoro language code for Turkish output")
parser.add_argument("--tts-speed", type=float, default=1.0, help="local TTS speed multiplier")
parser.add_argument("--input-device", help="sounddevice input device name or index")
parser.add_argument("--audio-io", choices=["auto", "sounddevice", "apple-raw", "apple-vpio"], default="sounddevice", help="audio capture/playback backend")
parser.add_argument("--streaming-engine", choices=["auto", "ring", "legacy"], default="ring", help="audio streaming engine: ring uses ordered timestamped PCM frames + priority ASR with any capture backend; legacy uses the older queue pipeline")
parser.add_argument("--apple-bridge-path", default=str(Path(__file__).with_name("apple_vpio_bridge")), help="path to Swift AVAudioEngine/VPIO bridge used for capture + playback")
parser.add_argument("--ring-buffer-seconds", type=float, default=30.0, help="seconds of timestamped audio history to keep in the ring buffer")
parser.add_argument("--translation-stable-only", action="store_true", default=True, help="legacy alias: translate only stable ASR prefix for audio partials")
parser.add_argument("--no-translation-stable-only", dest="translation_stable_only", action="store_false")
parser.add_argument("--partial-translation-policy", choices=["stable", "speculative", "off"], default="stable", help="audio partial MT policy: stable=translate only locked prefix, speculative=display-only MT for changed partials, off=no partial MT")
parser.add_argument("--stable-translation-min-words", type=int, default=3, help="minimum newly-stable words before translating a partial audio prefix")
parser.add_argument("--partial-translation-stabilize-seconds", type=float, default=2.0, help="wait this many seconds without Turkish partial text changes before queueing partial MT")
parser.add_argument("--partial-translation-min-completeness-score", type=int, default=2, help="minimum semantic completeness score before translating a stabilized partial clump")
parser.add_argument("--partial-translation-force-words", type=int, default=10, help="translate a stabilized partial clump once it reaches this many words even without punctuation")
parser.add_argument("--partial-display-policy", choices=["stable", "clump", "latest"], default="stable", help="source panel partial policy: stable=show locked ASR only, clump=show only translation-ready clumps, latest=show speculative tail")
parser.add_argument("--partial-display-min-words", type=int, default=2, help="minimum locked words before a stable partial is shown")
parser.add_argument("--partial-queue-latest-only", action="store_true", default=True, help="keep only the newest partial ASR job per utterance; final ASR still preempts all partials")
parser.add_argument("--no-partial-queue-latest-only", dest="partial_queue_latest_only", action="store_false")
parser.add_argument("--anti-repeat", choices=["auto", "off", "basic", "strict"], default="auto", help="reject/collapse repeated ASR output; auto=strict for --profile stable, basic otherwise")
parser.add_argument("--stable-lock-confirmations", type=int, default=2, help="number of matching partial revisions required before words become stable")
parser.add_argument("--stable-mutable-tail-words", type=int, default=4, help="number of ending words kept mutable even when partials agree")
parser.add_argument("--partial-min-asr-seconds", type=float, default=0.85, help="minimum partial audio duration before ASR is allowed")
parser.add_argument("--final-min-asr-seconds", type=float, default=0.45, help="minimum final audio duration before ASR is allowed")
parser.add_argument("--min-asr-peak", type=int, default=120, help="reject near-flat PCM buffers below this absolute peak")
parser.add_argument("--strict-min-dbfs", type=float, default=-62.0, help="minimum dBFS before ASR in strict anti-repeat mode")
parser.add_argument("--strict-no-speech-prob", type=float, default=0.55, help="stricter no-speech threshold used when anti-repeat is strict")
parser.add_argument("--strict-min-avg-logprob", type=float, default=-1.0, help="stricter average logprob threshold used when anti-repeat is strict")
parser.add_argument("--strict-max-compression-ratio", type=float, default=2.4, help="stricter compression-ratio threshold used when anti-repeat is strict")
parser.add_argument("--known-hallucination-filter", choices=["auto", "off", "on", "strict"], default="auto", help="remove common Whisper subtitle-credit phrases; auto=on with anti-repeat or stable profile; off preserves even those phrases")
parser.add_argument("--extra-hallucination-phrases", default="", help="extra hallucination phrases to remove, separated by ||, e.g. 'phrase one||phrase two'")
parser.add_argument("--hallucination-phrase-file", default="", help="optional UTF-8 text file with one hallucination phrase per line")
parser.add_argument("--disable-vpio-bandpass", action="store_true", default=True, help="skip FFT bandpass in the hot ASR path when using Apple VPIO/ring capture")
parser.add_argument("--enable-vpio-bandpass", dest="disable_vpio_bandpass", action="store_false")
parser.add_argument("--list-devices", action="store_true", help="print audio devices and exit")
parser.add_argument("--save-last-audio", default="", help="path for the last processed microphone chunk; empty disables hot-path disk writes")
parser.add_argument("--output-wav", default="output.wav", help="path for generated reply audio")
parser.add_argument("--warmup", action="store_true", default=True, help="warm up models at startup for faster first response")
parser.add_argument("--no-warmup", dest="warmup", action="store_false")
parser.add_argument("--verbose", action="store_true", help="show runtime status messages")
parser.add_argument("--debug", action="store_true", help="show diagnostics and, by default, save the full incoming audio stream for review")
parser.add_argument("--audio-review", choices=["auto", "off", "issues", "full"], default="auto", help="audio evidence: auto records full input with --debug, otherwise only flagged ASR clips")
parser.add_argument("--recordings-dir", default="recordings", help="base directory for isolated per-session audio evidence")
parser.add_argument("--issue-audio-budget-mb", type=int, default=256, help="per-writer byte budget for issue clips; recording stops visibly at the limit")
parser.add_argument("--asr-preprocessing", choices=["raw", "legacy"], default="raw", help="raw 16 kHz PCM conversion (default) or legacy bandpass/pre-emphasis/gain for comparison")
parser.add_argument("--ui", choices=["terminal", "web", "headless"], default="terminal", help="terminal interface or local responsive web UI; headless is the web backend transport")
parser.add_argument("--web-host", default="127.0.0.1", help="local web UI bind address")
parser.add_argument("--web-port", type=int, default=8765, help="local web UI port")
parser.add_argument("--dump-config", action="store_true", help="print effective config and exit")
parser.add_argument("--self-test", action="store_true", help="run fast local pipeline invariants without opening audio devices or loading live models")
parser.add_argument("--stream-audio-translation", action="store_true", default=True, help="print audio translations chunk-by-chunk when useful")
parser.add_argument("--no-stream-audio-translation", dest="stream_audio_translation", action="store_false")
parser.add_argument("--text-stream-tts", action="store_true", default=True, help="speak typed replies chunk-by-chunk as translation completes")
parser.add_argument("--no-text-stream-tts", dest="text_stream_tts", action="store_false")
parser.add_argument("--speak-audio-translation", action="store_true", default=False, help="speak translated call audio for voice-to-voice operation")
parser.add_argument("--no-speak-audio-translation", dest="speak_audio_translation", action="store_false")
parser.add_argument("--vad-backend", choices=["auto", "webrtc", "silero", "energy"], default="auto", help="speech detector: WebRTC default, optional Silero if installed, or energy-only")
parser.add_argument("--silero-vad-runtime", choices=["auto", "onnx", "torch"], default="auto", help="Silero VAD runtime; ONNX is preferred when onnxruntime is installed")
parser.add_argument("--silero-vad-threshold", type=float, default=0.50, help="speech probability threshold for --vad-backend silero")
parser.add_argument("--vad-hangover-ms", type=int, default=220, help="VAD hangover time in ms")
parser.add_argument("--tail-pad-ms", type=int, default=140, help="Tail padding time in ms")
parser.add_argument("--silence-base-ms", type=int, default=520, help="Base silence duration in ms")
parser.add_argument("--silence-min-ms", type=int, default=320, help="Minimum silence duration in ms")
parser.add_argument("--silence-max-ms", type=int, default=1200, help="Maximum silence duration in ms")
parser.add_argument("--semantic-endpointing", action="store_true", default=True, help="use partial transcript completeness to commit finished utterances earlier")
parser.add_argument("--no-semantic-endpointing", dest="semantic_endpointing", action="store_false")
parser.add_argument("--semantic-commit-score", type=int, default=8, help="commit-confidence score required for early semantic endpointing")
parser.add_argument("--semantic-min-silence-ms", type=int, default=260, help="minimum pause before semantic endpointing may flush a stable complete utterance")
parser.add_argument("--adapt-alpha", type=float, default=0.22, help="Adaptive silence alpha")
parser.add_argument("--noise-gate-db", type=float, default=2.5, help="Noise gate dB above floor")
parser.add_argument("--min-speech-frames", type=int, default=8, help="Minimum speech frames")
parser.add_argument("--min-speech-ratio", type=float, default=0.22, help="Minimum speech ratio")
parser.add_argument("--audio-queue-size", type=int, default=3, help="max pending audio chunks before oldest is dropped")
parser.add_argument("--capture-queue-ms", type=int, default=240, help="max raw 20ms capture frames buffered before oldest frames are dropped")
parser.add_argument("--tts-queue-size", type=int, default=2, help="max pending TTS tasks; typed replies are preserved while stale optional audio TTS may be coalesced")
parser.add_argument("--prefer-latest-audio", action="store_true", default=True, help="drop oldest queued audio when backlog builds")
parser.add_argument("--no-prefer-latest-audio", dest="prefer_latest_audio", action="store_false")
parser.add_argument("--duplex-capture", action="store_true", default=True, help="keep capturing audio while TTS playback is active")
parser.add_argument("--no-duplex-capture", dest="duplex_capture", action="store_false")
parser.add_argument("--streaming-asr-chunk-seconds", type=float, default=3.6, help="flush long active speech into overlapping ASR chunks before the speaker fully stops")
parser.add_argument("--streaming-asr-overlap-ms", type=int, default=320, help="overlap retained between streaming ASR chunks")
parser.add_argument("--partial-asr-window-seconds", type=float, default=2.4, help="maximum audio window sent for partial ASR; final ASR still uses the complete utterance")
parser.add_argument("--partial-translation-min-delta-words", type=int, default=2, help="minimum new/changed words before a partial audio translation is refreshed")
parser.add_argument("--tts-bleed-duck-db", type=float, default=10.0, help="extra dB above noise floor that triggers temporary ducking while TTS is playing")
parser.add_argument("--tts-bleed-hold-ms", type=int, default=280, help="how long to keep ducking likely playback bleed after detection")
parser.add_argument("--playback-guard-ms", type=int, default=900, help="temporarily ignore local mic capture during and just after local TTS playback")
parser.add_argument("--latency-stats", action="store_true", default=True, help="show capture/ASR/MT/TTS latency milestones in the status line")
parser.add_argument("--no-latency-stats", dest="latency_stats", action="store_false")
parser.add_argument("--transcription-log", default="transcription.log", help="append finalized microphone transcription and its translation to this log file")
parser.add_argument("--tts-log", default="tts.log", help="append typed user text and its translation to this log file")
parser.add_argument("--no-session-logs", dest="session_logs", action="store_false", default=True, help="disable transcription.log and tts.log writing")
args = parser.parse_args()
args.source_language = normalize_language_code(args.source_language)
args.target_language = normalize_language_code(args.target_language)
if args.source_language == args.target_language:
    parser.error("--source-language and --target-language must be different")

EXPLICIT_FLAGS = {token.split("=", 1)[0] for token in sys.argv[1:] if token.startswith("--")}
USER_SET_WHISPER_MODEL = "--whisper-model" in EXPLICIT_FLAGS
USER_SET_STT_BACKEND = "--stt-backend" in EXPLICIT_FLAGS
USER_SET_TRANSLATION_MODEL = "--translation-model" in EXPLICIT_FLAGS

PROFILE_DEFAULTS = {
    "live": {
        "whisper_model": "large-v3",
        "translation_model": "mlx-community/translategemma-12b-it-4bit",
        "asr_beam": 2,
        "mt_beams": 3,
        "mt_queue_size": 2,
        "mt_delta_min_chars": 10,
        "mt_delta_min_ms": 50,
        "vad_hangover_ms": 140,
        "tail_pad_ms": 80,
        "silence_base_ms": 360,
        "silence_min_ms": 220,
        "silence_max_ms": 820,
        "semantic_commit_score": 8,
        "semantic_min_silence_ms": 240,
        "adapt_alpha": 0.32,
        "min_speech_frames": 5,
        "min_speech_ratio": 0.16,
        "audio_queue_size": 1,
        "capture_queue_ms": 2000,
        "tts_queue_size": 1,
        "streaming_asr_chunk_seconds": 1.0,
        "streaming_asr_overlap_ms": 240,
        "partial_asr_window_seconds": 6.0,
        "partial_translation_min_delta_words": 2,
        "playback_guard_ms": 500,
    },
    "m4-max-live": {
        "whisper_model": "large-v3",
        "partial_whisper_model": "large-v3",
        "final_whisper_model": "large-v3",
        "mlx_vlm_model": "mlx-community/gemma-4-e4b-it-4bit",
        "stt_backend": "mlx-whisper",
        "translation_backend": "mlx-lm",
        "translation_model": "mlx-community/translategemma-12b-it-4bit",
        "audio_io": "apple-vpio",
        "streaming_engine": "ring",
        "asr_beam": 2,
        "partial_asr_beam": 1,
        "final_asr_beam": 2,
        "mt_beams": 3,
        "mt_process": True,
        "mt_queue_size": 2,
        "tts_queue_size": 1,
        "partial_translation_policy": "stable",
        "partial_translation_stabilize_seconds": 2.0,
        "partial_translation_min_delta_words": 2,
        "stable_translation_min_words": 3,
        "translation_stable_only": True,
        "partial_queue_latest_only": True,
        "streaming_asr_chunk_seconds": 1.0,
        "streaming_asr_overlap_ms": 220,
        "partial_asr_window_seconds": 6.0,
        "vad_hangover_ms": 140,
        "tail_pad_ms": 90,
        "silence_base_ms": 360,
        "silence_min_ms": 220,
        "silence_max_ms": 820,
        "semantic_commit_score": 8,
        "semantic_min_silence_ms": 240,
        "adapt_alpha": 0.32,
        "min_speech_frames": 5,
        "min_speech_ratio": 0.16,
        "capture_queue_ms": 2000,
        "audio_queue_size": 1,
        "playback_guard_ms": 500,
    },
    "balanced": {
        "whisper_model": "large-v3",
        "mlx_vlm_model": "mlx-community/gemma-4-e4b-it-4bit",
        "translation_model": "mlx-community/translategemma-12b-it-4bit",
        "asr_beam": 3,
        "mt_beams": 4,
        "mt_queue_size": 3,
        "mt_delta_min_chars": 16,
        "mt_delta_min_ms": 70,
        "vad_hangover_ms": 180,
        "tail_pad_ms": 100,
        "silence_base_ms": 440,
        "silence_min_ms": 260,
        "silence_max_ms": 900,
        "semantic_commit_score": 9,
        "semantic_min_silence_ms": 280,
        "adapt_alpha": 0.28,
        "min_speech_frames": 6,
        "min_speech_ratio": 0.18,
        "audio_queue_size": 2,
        "capture_queue_ms": 240,
        "tts_queue_size": 1,
        "streaming_asr_chunk_seconds": 1.1,
        "streaming_asr_overlap_ms": 260,
        "partial_asr_window_seconds": 2.8,
        "partial_translation_min_delta_words": 2,
        "playback_guard_ms": 600,
    },
    "stable": {
        "whisper_model": "large-v3",
        "translation_model": "mlx-community/translategemma-12b-it-4bit",
        "asr_beam": 2,
        "mt_beams": 4,
        "mt_queue_size": 2,
        "mt_delta_min_chars": 18,
        "mt_delta_min_ms": 90,
        "vad_hangover_ms": 200,
        "tail_pad_ms": 140,
        "silence_base_ms": 500,
        "silence_min_ms": 300,
        "silence_max_ms": 1100,
        "semantic_commit_score": 10,
        "semantic_min_silence_ms": 320,
        "adapt_alpha": 0.22,
        "min_speech_frames": 7,
        "min_speech_ratio": 0.20,
        "audio_queue_size": 2,
        "capture_queue_ms": 220,
        "tts_queue_size": 1,
        "streaming_asr_chunk_seconds": 1.0,
        "streaming_asr_overlap_ms": 220,
        "partial_asr_window_seconds": 2.6,
        "partial_translation_min_delta_words": 3,
        "partial_translation_policy": "stable",
        "translation_stable_only": True,
        "partial_queue_latest_only": True,
        "max_no_speech_prob": 0.55,
        "min_avg_logprob": -1.0,
        "max_compression_ratio": 2.4,
        "stable_lock_confirmations": 3,
        "stable_mutable_tail_words": 5,
        "partial_min_asr_seconds": 0.95,
        "final_min_asr_seconds": 0.50,
        "strict_min_dbfs": -62.0,
        "strict_no_speech_prob": 0.55,
        "strict_min_avg_logprob": -1.0,
        "strict_max_compression_ratio": 2.4,
        "playback_guard_ms": 650,
    },
    "accurate": {
        "whisper_model": "large-v3",
        "translation_model": "mlx-community/translategemma-12b-it-4bit",
        "asr_beam": 5,
        "mt_beams": 5,
        "mt_queue_size": 5,
        "mt_delta_min_chars": 24,
        "mt_delta_min_ms": 100,
        "vad_hangover_ms": 280,
        "tail_pad_ms": 180,
        "silence_base_ms": 700,
        "silence_min_ms": 420,
        "silence_max_ms": 1500,
        "semantic_commit_score": 10,
        "semantic_min_silence_ms": 380,
        "adapt_alpha": 0.16,
        "audio_queue_size": 4,
        "capture_queue_ms": 360,
        "tts_queue_size": 3,
        "streaming_asr_chunk_seconds": 3.8,
        "streaming_asr_overlap_ms": 420,
        "partial_asr_window_seconds": 6.0,
        "partial_translation_min_delta_words": 3,
        "partial_display_policy": "stable",
        "partial_display_min_words": 2,
        "stable_lock_confirmations": 3,
        "stable_mutable_tail_words": 6,
    },
}


def apply_profile_defaults() -> None:
    for key, value in PROFILE_DEFAULTS.get(args.profile, {}).items():
        flag = f"--{key.replace('_', '-')}"
        neg_flag = f"--no-{key.replace('_', '-')}"
        if flag in EXPLICIT_FLAGS or neg_flag in EXPLICIT_FLAGS:
            continue
        setattr(args, key, value)


apply_profile_defaults()


# ======================
# Audio / detection config
# ======================
SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)
CHANNELS = 1
DTYPE = "int16"

PRE_ROLL_MS = 240
MAX_UTTERANCE_SECONDS = 18.0
MIN_UTTERANCE_SECONDS = 0.55

# Phone-call friendly gating.
VAD_MODE = 2
MIN_DBFS_UTTERANCE = -60.0
MIN_DBFS_UTTERANCE_APPLE_VPIO = -70.0
NOISE_EWMA_ALPHA = 0.06
BPF_LO_HZ = 120.0
BPF_HI_HZ = 3800.0

# Whisper confidence gates.
DEFAULT_MAX_NO_SPEECH_PROB = 0.75
DEFAULT_MIN_AVG_LOGPROB = -1.55
DEFAULT_MAX_COMPRESSION_RATIO = 2.9
DEFAULT_MIN_TEXT_CHARS = 2
DEFAULT_MIN_TEXT_WORDS = 1

# Translation chunking.
NLLB_SRC_TOKEN_BUDGET = 220
NLLB_MAX_NEW_TOKENS = 220
DEFAULT_MLX_TRANSLATION_MODEL = "mlx-community/translategemma-12b-it-4bit"
DEFAULT_NLLB_TRANSLATION_MODEL = "facebook/nllb-200-distilled-600M"
DEFAULT_MLX_VLM_AUDIO_MODEL = "mlx-community/gemma-4-e4b-it-4bit"
def stt_initial_prompt() -> str | None:
    """Return an optional Whisper prior-transcript prompt.

    Whisper initial prompts are not instructions. Using English instructions here can
    bias short partial windows toward English hallucinations, so the default is empty.
    Language/task forcing is done through backend language and task arguments.
    """
    prompt = str(getattr(args, "asr_initial_prompt", "") or "").strip()
    return prompt or None


def asr_initial_prompt_kwargs() -> dict[str, str]:
    prompt = stt_initial_prompt()
    return {"initial_prompt": prompt} if prompt else {}


def asr_initial_prompt_cli_args(flag: str) -> list[str]:
    prompt = stt_initial_prompt()
    return [flag, prompt] if prompt else []


TURKISH_STT_PROMPT = stt_initial_prompt()

PIPER_DIR = Path(os.path.expanduser("~/.local/share/piper"))
VOICE_TR = str(PIPER_DIR / "tr_TR-fahrettin-medium.onnx")

ANSI_RESET = "\033[0m"
ANSI_DIM_GREEN = "\033[2;32m"
ANSI_BRIGHT_WHITE = "\033[97m"
ANSI_DIM_WHITE = "\033[2;37m"


def effective_config() -> dict:
    resolved_backend = resolve_stt_backend() if args.mode in {"audio", "both"} else args.stt_backend
    resolved_device = resolve_stt_device(resolved_backend) if args.mode in {"audio", "both"} else args.stt_device
    resolved_model = default_whisper_model_for_backend(resolved_backend) if args.mode in {"audio", "both"} else args.whisper_model
    return {
        "mode": args.mode,
        "profile": args.profile,
        "source_language": source_language_code(),
        "target_language": target_language_code(),
        "source_language_name": source_language().name,
        "target_language_name": target_language().name,
        "asr_language": asr_language_display(),
        "asr_language_forced": args.force_asr_language,
        "asr_task": "transcribe",
        "asr_initial_prompt_enabled": bool(stt_initial_prompt()),
        "strict_asr_language": args.strict_asr_language,
        "allow_english_asr_model": args.allow_english_asr_model,
        "source_nllb_lang": source_nllb_lang(),
        "target_nllb_lang": target_nllb_lang(),
        "input_device": args.input_device,
        "audio_io": resolve_audio_io_backend() if args.mode in {"audio", "both"} else args.audio_io,
        "streaming_engine": resolve_streaming_engine() if args.mode in {"audio", "both"} else args.streaming_engine,
        "apple_bridge_path": args.apple_bridge_path,
        "warmup": args.warmup,
        "verbose": args.verbose,
        "debug": args.debug,
        "audio_review": args.audio_review,
        "audio_review_mode": resolved_audio_review_mode(),
        "asr_preprocessing": args.asr_preprocessing,
        "session_id": runtime_session_id,
        "ui": args.ui,
        "stream_audio_translation": args.stream_audio_translation,
        "partial_translation_policy": args.partial_translation_policy,
        "anti_repeat": resolved_anti_repeat_level(),
        "stable_lock_confirmations": args.stable_lock_confirmations,
        "stable_mutable_tail_words": args.stable_mutable_tail_words,
        "translation_stable_only": args.translation_stable_only,
        "partial_queue_latest_only": args.partial_queue_latest_only,
        "text_stream_tts": args.text_stream_tts,
        "speak_audio_translation": args.speak_audio_translation,
        "whisper_model": resolved_model,
        "whisper_compute": args.whisper_compute,
        "mlx_vlm_python_installed": MLX_VLM_INSTALLED,
        "mlx_vlm_model": args.mlx_vlm_model,
        "partial_mlx_vlm_model": args.partial_mlx_vlm_model,
        "final_mlx_vlm_model": args.final_mlx_vlm_model,
        "mlx_vlm_prompt_enabled": bool(args.mlx_vlm_prompt or args.mlx_vlm_partial_prompt or args.mlx_vlm_final_prompt),
        "mlx_vlm_max_tokens": args.mlx_vlm_max_tokens,
        "mlx_vlm_temperature": args.mlx_vlm_temperature,
        "max_no_speech_prob": args.max_no_speech_prob,
        "min_avg_logprob": args.min_avg_logprob,
        "max_compression_ratio": args.max_compression_ratio,
        "min_text_chars": args.min_text_chars,
        "min_text_words": args.min_text_words,
        "apple_vpio_agc": args.apple_vpio_agc,
        "apple_vpio_bypass_processing": args.apple_vpio_bypass_processing,
        "stt_backend": args.stt_backend,
        "stt_device": args.stt_device,
        "resolved_stt_backend": resolved_backend,
        "resolved_stt_device": resolved_device,
        "partial_stt_backend": args.partial_stt_backend,
        "final_stt_backend": args.final_stt_backend,
        "partial_whisper_model": args.partial_whisper_model,
        "final_whisper_model": args.final_whisper_model,
        "partial_asr_beam": args.partial_asr_beam,
        "final_asr_beam": args.final_asr_beam or args.asr_beam,
        "partial_asr_source": "current-utterance-window",
        "partial_asr_busy_policy": "drop-while-partial-inflight",
        "final_asr_queue_policy": "preserve-finals-drop-partials",
        "asr_result_queue_policy": "preserve-finals-drop-partials",
        "tts_queue_policy": "preserve-text-drop-stale-audio",
        "mlx_whisper_beam_strategy": "use-if-supported-then-cache-greedy-fallback",
        "finalization_hold_ms": args.finalization_hold_ms,
        "final_dedupe_window_ms": args.final_dedupe_window_ms,
        "final_dedupe_contained_ratio": args.final_dedupe_contained_ratio,
        "translation_backend": args.translation_backend,
        "resolved_translation_backend": resolve_translation_backend(),
        "translation_model": default_translation_model_for_backend(resolve_translation_backend()),
        "mt_prompt_enabled": bool(args.mt_prompt.strip()),
        "asr_beam": args.asr_beam,
        "mt_beams": args.mt_beams,
        "mt_process": args.mt_process,
        "inference_queue_size": args.inference_queue_size,
        "startup_timeout": args.startup_timeout,
        "inference_timeout": args.inference_timeout,
        "model_execution": "supervised processes; capture/UI/results/playback threads",
        "mt_queue_size": args.mt_queue_size,
        "mt_chunk_max_chars": args.mt_chunk_max_chars,
        "stream_mt_deltas": args.stream_mt_deltas,
        "mt_delta_min_chars": args.mt_delta_min_chars,
        "mt_delta_min_ms": args.mt_delta_min_ms,
        "tts": args.tts,
        "vad_hangover_ms": args.vad_hangover_ms,
        "vad_backend": args.vad_backend,
        "resolved_vad_backend": resolve_vad_backend(),
        "silero_vad_python_installed": SILERO_VAD_INSTALLED,
        "silero_vad_runtime": resolve_silero_vad_runtime() if resolve_vad_backend() == "silero" else None,
        "silero_vad_threshold": args.silero_vad_threshold,
        "tail_pad_ms": args.tail_pad_ms,
        "silence_base_ms": args.silence_base_ms,
        "silence_min_ms": args.silence_min_ms,
        "silence_max_ms": args.silence_max_ms,
        "semantic_endpointing": args.semantic_endpointing,
        "semantic_commit_score": args.semantic_commit_score,
        "semantic_min_silence_ms": args.semantic_min_silence_ms,
        "adapt_alpha": args.adapt_alpha,
        "noise_gate_db": args.noise_gate_db,
        "min_speech_frames": args.min_speech_frames,
        "min_speech_ratio": args.min_speech_ratio,
        "audio_queue_size": args.audio_queue_size,
        "capture_queue_ms": args.capture_queue_ms,
        "tts_queue_size": args.tts_queue_size,
        "prefer_latest_audio": args.prefer_latest_audio,
        "duplex_capture": args.duplex_capture,
        "streaming_asr_chunk_seconds": args.streaming_asr_chunk_seconds,
        "streaming_asr_overlap_ms": args.streaming_asr_overlap_ms,
        "partial_asr_window_seconds": args.partial_asr_window_seconds,
        "partial_translation_min_delta_words": args.partial_translation_min_delta_words,
        "stable_translation_min_words": args.stable_translation_min_words,
        "partial_translation_stabilize_seconds": args.partial_translation_stabilize_seconds,
        "partial_translation_min_completeness_score": args.partial_translation_min_completeness_score,
        "partial_translation_force_words": args.partial_translation_force_words,
        "partial_display_policy": args.partial_display_policy,
        "partial_display_min_words": args.partial_display_min_words,
        "final_merge_last_partial": args.final_merge_last_partial,
        "final_translation_completeness_mode": args.final_translation_completeness_mode,
        "final_merge_min_final_words": args.final_merge_min_final_words,
        "debug_dropped_asr": args.debug_dropped_asr,
        "native_script_min_ratio": args.native_script_min_ratio,
        "partial_native_script_min_ratio": args.partial_native_script_min_ratio,
        "tts_bleed_duck_db": args.tts_bleed_duck_db,
        "tts_bleed_hold_ms": args.tts_bleed_hold_ms,
        "playback_guard_ms": args.playback_guard_ms,
        "latency_stats": args.latency_stats,
        "session_logs": args.session_logs,
        "transcription_log": args.transcription_log if args.session_logs else "",
        "tts_log": args.tts_log if args.session_logs else "",
        "mlx_whisper_python_installed": MLX_WHISPER_INSTALLED,
        "mlx_lm_python_installed": MLX_LM_INSTALLED,
    }


def resolved_audio_review_mode() -> str:
    return ("full" if args.debug else "issues") if args.audio_review == "auto" else args.audio_review


def initialize_session_runtime() -> None:
    """Isolate every launch's generated audio, logs and diagnostic artifacts."""
    global runtime_session_id, runtime_session_dir, runtime_audio_dir, audio_evidence, audio_review_mode
    root = Path(__file__).resolve().parent
    audio_review_mode = resolved_audio_review_mode()
    if audio_review_mode != "off":
        runtime_audio_dir = create_session_directory(Path(args.recordings_dir))
        runtime_session_id = runtime_audio_dir.name
        runtime_session_dir = root / "runtime-reports" / "sessions" / runtime_session_id
        runtime_session_dir.mkdir(parents=True, exist_ok=True)
        audio_evidence = AudioEvidence(runtime_audio_dir, role="capture", full_stream=audio_review_mode == "full",
                                       max_clip_bytes=args.issue_audio_budget_mb * 1024 * 1024)
    else:
        runtime_audio_dir = None
        runtime_session_dir = create_session_directory(root / "runtime-reports" / "sessions")
        runtime_session_id = runtime_session_dir.name
    os.environ["TTS_TRANSLATE_SESSION_DIR"] = str(runtime_session_dir)
    os.environ["TTS_TRANSLATE_SESSION_ID"] = runtime_session_id
    os.environ["TTS_TRANSLATE_AUDIO_DIR"] = str(runtime_audio_dir or "")
    os.environ["TTS_TRANSLATE_AUDIO_MODE"] = audio_review_mode
    if "--output-wav" not in EXPLICIT_FLAGS:
        args.output_wav = str(runtime_session_dir / "reply.wav")
    if "--transcription-log" not in EXPLICIT_FLAGS:
        args.transcription_log = str(runtime_session_dir / "transcription.log")
    if "--tts-log" not in EXPLICIT_FLAGS:
        args.tts_log = str(runtime_session_dir / "tts.log")
    metadata = {"session_id": runtime_session_id, "pid": os.getpid(),
                "config": effective_config(), "recording_directory": str(runtime_audio_dir or ""),
                "recording_policy": audio_review_mode, "quality_flags_are_not_accuracy_measurements": True}
    (runtime_session_dir / "session.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if audio_evidence is not None:
        audio_evidence.event({"type": "session", **metadata})


@dataclass(slots=True)
class AudioTask:
    utterance_id: int
    partial: bool
    raw: bytes | None = None
    shm_name: str | None = None
    slot_id: int | None = None
    num_bytes: int = 0


@dataclass(slots=True)
class TranslationTask:
    source: Literal["audio", "text"]
    text: str
    src_lang: str
    beams: int
    target_lang: str
    forced_bos_id: int = 0
    utterance_id: int | None = None
    partial: bool = False
    revision: int = 0


@dataclass(order=True, slots=True)
class PrioritizedTranslationTask:
    priority: tuple[int, int, int]
    task: TranslationTask = field(compare=False)


translation_task_seq = 0
translation_task_seq_lock = threading.Lock()


def translation_priority(task: TranslationTask) -> int:
    if task.source == "text":
        return 0
    if task.source == "audio" and not task.partial:
        return 1
    return 2


def wrap_translation_task(task: TranslationTask) -> PrioritizedTranslationTask:
    global translation_task_seq
    with translation_task_seq_lock:
        translation_task_seq += 1
        seq = translation_task_seq
    return PrioritizedTranslationTask((translation_priority(task), int(time.time() * 1000), seq), task)


def unwrap_translation_task(item) -> TranslationTask:
    return item.task if isinstance(item, PrioritizedTranslationTask) else item


@dataclass(slots=True)
class AsrResult:
    utterance_id: int
    text: str
    partial: bool
    shm_name: str | None = None
    slot_id: int | None = None
    request_id: str = ""
    session_id: str = ""
    pcm_sha256: str = ""
    diagnostics: dict = field(default_factory=dict)


@dataclass(slots=True)
class TranslationResult:
    source: Literal["audio", "text"]
    text: str
    target_lang: str
    utterance_id: int | None = None
    partial: bool = False
    revision: int = 0
    final: bool = True
    stream_delta: bool = False
    delta_index: int = 0
    chunk_index: int = 0
    total_chunks: int = 1
    error: str = ""


@dataclass(slots=True)
class PartialTranslationCandidate:
    text: str
    locked_prefix: str
    revision: int
    stable_text: str
    stable_changed_at: float
    last_sent_text: str = ""


@dataclass(slots=True)
class TtsTask:
    text: str
    wav_out: str
    lang_hint: str = "TR"
    source: Literal["audio", "text"] = "text"
    utterance_id: int | None = None
    revision: int = 0


# ======================
# Terminal setup
# ======================
stdin_fd: int | None = None
stdin_characters: deque[str] = deque()
stdin_decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
headless_line_buffer = ""
headless_transport = None
old_settings = None
ui_lock = threading.Lock()
ui_initialized = False
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
ui_state = {
    "left_done": [],
    "right_done": [],
    "left_active": {},
    "right_active": {},
    "left_order": [],
    "right_order": [],
    "input": "",
    "status": "",
    "stages": {},
}
inflight_partial_markers: dict[int, float] = {}
text_entry_seq = 0


def setup_terminal() -> None:
    global stdin_fd, old_settings
    if args.ui == "headless":
        global headless_transport
        from tts_runtime.web_transport import HeadlessTransport
        stdin_fd = sys.stdin.fileno()
        headless_transport = HeadlessTransport(stdin_fd, sys.stdout, process_headless_action, async_output=True)
        return
    if not sys.stdin.isatty():
        raise RuntimeError("stdin is not a TTY; interactive keyboard mode is required")
    stdin_fd = sys.stdin.fileno()
    stdin_characters.clear()
    stdin_decoder.reset()
    old_settings = termios.tcgetattr(stdin_fd)
    tty.setcbreak(stdin_fd)
    sys.stdout.write("\x1b[?25l")
    sys.stdout.flush()


def restore_terminal() -> None:
    if args.ui == "headless":
        return
    if getattr(render_dashboard, '_alt', False):
        sys.stdout.write("\x1b[?1049l")
        render_dashboard._alt = False
    sys.stdout.write("\x1b[?25h")
    sys.stdout.flush()
    if stdin_fd is not None and old_settings is not None:
        termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old_settings)


def read_key(timeout: float = 0.05) -> str | None:
    if args.ui == "headless":
        return read_headless_command(timeout)
    if stdin_fd is None:
        return None
    if stdin_characters:
        return stdin_characters.popleft()
    if select.select([stdin_fd], [], [], timeout)[0]:
        # TextIO.read(1) may prefetch the remaining pasted text/Enter into its
        # own buffer. select() then sees an empty FD and never drains those
        # characters. Read bytes once, decode UTF-8 incrementally, and drain
        # our own queue before waiting for more OS input.
        raw = os.read(stdin_fd, 4096)
        if not raw:
            return "\x1b"
        stdin_characters.extend(stdin_decoder.decode(raw))
        if stdin_characters:
            return stdin_characters.popleft()
    return None


def ensure_audio_runtime() -> None:
    missing: list[str] = []
    vad_backend = resolve_vad_backend()
    if vad_backend == "webrtc" and webrtcvad is None:
        missing.append("webrtcvad-wheels")
    if vad_backend == "silero":
        if not SILERO_VAD_INSTALLED:
            missing.append("silero-vad")
        if torch is None:
            missing.append("torch")
        if resolve_silero_vad_runtime() == "onnx" and not ONNXRUNTIME_INSTALLED:
            missing.append("onnxruntime")

    audio_backend = resolve_audio_io_backend()
    if audio_backend == "sounddevice" and sd is None:
        missing.append("sounddevice")
    if audio_backend in {"apple-raw", "apple-vpio"}:
        apple_bridge = Path(args.apple_bridge_path).expanduser()
        if not apple_bridge.exists() or not os.access(apple_bridge, os.X_OK):
            missing.append("apple_vpio_bridge (build it with: swiftc apple_vpio_bridge.swift -framework AVFoundation -o apple_vpio_bridge)")

    if missing:
        raise RuntimeError(f"Missing audio dependencies for this mode: {', '.join(missing)}")


def validate_runtime_configuration() -> None:
    problems: list[str] = []
    if args.mode in {"audio", "both"}:
        streaming_engine = resolve_streaming_engine()
        audio_backend = resolve_audio_io_backend()
        if streaming_engine == "ring" and audio_backend not in {"sounddevice", "apple-raw", "apple-vpio"}:
            problems.append("ring streaming requires --audio-io sounddevice, apple-vpio, or apple-raw")
        if args.input_device is not None and audio_backend != "sounddevice":
            problems.append("--input-device requires --audio-io sounddevice; Apple capture uses the system default input")

    if args.partial_stt_backend == "apple-speech":
        if not args.apple_speech_command:
            problems.append("--partial-stt-backend apple-speech requires --apple-speech-command /path/to/apple_speech_file_transcriber")
        else:
            try:
                resolve_stt_command(args.apple_speech_command)
            except RuntimeError as exc:
                problems.append(str(exc))

    if args.final_stt_backend == "whispercpp-coreml":
        if not args.whispercpp_model:
            problems.append("--final-stt-backend whispercpp-coreml requires --whispercpp-model /path/to/ggml-model.bin")
        elif not Path(args.whispercpp_model).expanduser().exists():
            problems.append(f"whisper.cpp model not found: {Path(args.whispercpp_model).expanduser()}")
        try:
            resolve_stt_command(args.whispercpp_command)
        except RuntimeError as exc:
            problems.append(str(exc))

    if args.partial_asr_window_seconds < args.streaming_asr_chunk_seconds:
        problems.append("--partial-asr-window-seconds should be >= --streaming-asr-chunk-seconds")
    if args.finalization_hold_ms < 0:
        problems.append("--finalization-hold-ms must be >= 0")
    if args.mt_queue_size < 1 or args.audio_queue_size < 1 or args.tts_queue_size < 1:
        problems.append("queue sizes must be >= 1")
    if args.inference_queue_size < 1 or args.startup_timeout <= 0 or args.inference_timeout <= 0:
        problems.append("inference queue size and worker timeouts must be positive")
    if args.capture_queue_ms < FRAME_MS:
        problems.append(f"--capture-queue-ms must be >= {FRAME_MS}")
    if not np.isfinite(args.ring_buffer_seconds) or args.ring_buffer_seconds <= 0:
        problems.append("--ring-buffer-seconds must be positive and finite")
    if args.force_asr_language:
        forced_backends = {"mlx-vlm-gemma", "mlx-whisper", "whisper-cli", "faster-whisper", "whispercpp-coreml", "apple-speech"}
        selected = {
            resolve_stt_backend() if args.stt_backend == "auto" else args.stt_backend,
            effective_lane_backend(partial=True),
            effective_lane_backend(partial=False),
        }
        unsupported = sorted(backend for backend in selected if backend not in forced_backends and backend != "same")
        if unsupported:
            problems.append(
                "forced ASR language is not supported by backend(s): "
                + ", ".join(unsupported)
                + "; use mlx-vlm-gemma/mlx-whisper/whisper-cli/faster-whisper/apple-speech"
            )

    selected_stt_backends = {
        resolve_stt_backend() if args.stt_backend == "auto" else args.stt_backend,
        effective_lane_backend(partial=True),
        effective_lane_backend(partial=False),
    }
    if args.mode in {"audio", "both"} and "mlx-vlm-gemma" in selected_stt_backends and not MLX_VLM_INSTALLED:
        problems.append("mlx-vlm-gemma backend requires mlx-vlm. Install it with: pip install -U mlx-vlm")
    if args.translation_backend == "mlx-vlm-gemma" and not MLX_VLM_INSTALLED:
        problems.append("mlx-vlm-gemma translation requires mlx-vlm. Install it with: pip install -U mlx-vlm")

    if problems:
        raise RuntimeError("Invalid runtime configuration:\n- " + "\n- ".join(problems))


def list_audio_devices() -> None:
    audio_backend = resolve_audio_io_backend()
    if audio_backend == "apple-vpio":
        eprint("apple-vpio uses the system duplex route managed by AVAudioEngine voice processing (experimental on this machine).")
        return
    if audio_backend == "apple-raw":
        eprint("apple-raw uses the system duplex route managed by AVAudioEngine without voice processing.")
        return
    ensure_audio_runtime()
    devices = sd.query_devices()
    if len(devices) == 0:
        eprint("No audio devices were reported by SoundDevice in this session.")
        return
    default_input, default_output = sd.default.device
    for idx, dev in enumerate(devices):
        marks = []
        if idx == default_input:
            marks.append("default-in")
        if idx == default_output:
            marks.append("default-out")
        flags = f" [{' '.join(marks)}]" if marks else ""
        print(
            f"{idx}: {dev['name']}{flags} | in={dev['max_input_channels']} out={dev['max_output_channels']} sr={dev['default_samplerate']}"
        )


# ======================
# Helpers
# ======================
_SENT_SPLIT = re.compile(r"(?<=[.!?…;:])\s+")
_CLAUSE_SPLIT = re.compile(r"(?<=[,.!?…;:])\s+")
_WORD_RE = re.compile(r"[\w']+", re.UNICODE)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).strip()
    if not text:
        return ""
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("—", "-").replace("–", "-")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([?.!,;:])", r"\1", text)
    text = re.sub(r"([?.!,;:])([A-Za-zÇĞİÖŞÜçğıöşü])", r"\1 \2", text)
    return text.strip()


def split_text_soft(text: str) -> list[str]:
    text = normalize_text(text)
    return [part.strip() for part in _SENT_SPLIT.split(text) if part.strip()] or ([text] if text else [])


def looks_like_real_text(text: str) -> bool:
    text = normalize_text(text)
    if len(text) < args.min_text_chars:
        return False
    words = _WORD_RE.findall(text)
    if len(words) < args.min_text_words:
        return False
    return True



def resolved_anti_repeat_level() -> str:
    if args.anti_repeat != "auto":
        return args.anti_repeat
    return "strict" if args.profile == "stable" else "basic"


def anti_repeat_enabled(strict_only: bool = False) -> bool:
    level = resolved_anti_repeat_level()
    if strict_only:
        return level == "strict"
    return level in {"basic", "strict"}


def anti_repeat_thresholds() -> tuple[float, float, float]:
    if resolved_anti_repeat_level() == "strict":
        return args.strict_no_speech_prob, args.strict_min_avg_logprob, args.strict_max_compression_ratio
    return args.max_no_speech_prob, args.min_avg_logprob, args.max_compression_ratio


def should_send_to_asr(audio_i16: np.ndarray, *, partial: bool) -> bool:
    if audio_i16.size == 0:
        return False
    duration = audio_i16.size / SAMPLE_RATE
    min_duration = args.partial_min_asr_seconds if partial else args.final_min_asr_seconds
    if anti_repeat_enabled() and duration < min_duration:
        return False
    peak = int(np.max(np.abs(audio_i16.astype(np.int32)))) if audio_i16.size else 0
    if anti_repeat_enabled() and peak < args.min_asr_peak:
        return False
    db = rms_dbfs_i16(audio_i16)
    db_floor = args.strict_min_dbfs if resolved_anti_repeat_level() == "strict" else min_dbfs_utterance_threshold()
    if anti_repeat_enabled() and db < db_floor:
        return False
    return True


def is_repetitive_text(text: str) -> bool:
    text = normalize_text(text).casefold()
    words = text.split()
    if len(words) < 4:
        return False
    for i in range(len(words) - 2):
        if words[i] == words[i + 1] == words[i + 2]:
            return True
    for n in (2, 3):
        if len(words) < n * 3:
            continue
        for i in range(len(words) - (n * 3) + 1):
            a = words[i:i+n]
            if a == words[i+n:i+2*n] == words[i+2*n:i+3*n]:
                return True
    unique_ratio = len(set(words)) / max(1, len(words))
    if len(words) >= 8 and unique_ratio < 0.45:
        return True
    chars = [c for c in text if c.isalpha()]
    if len(chars) >= 24 and len(set(chars)) <= 3:
        return True
    return False


def collapse_repeated_phrases(text: str, max_ngram: int = 5) -> str:
    words = normalize_text(text).split()
    if len(words) < 4:
        return normalize_text(text)
    changed = True
    while changed:
        changed = False
        for n in range(max_ngram, 0, -1):
            i = 0
            new_words: list[str] = []
            while i < len(words):
                phrase = words[i:i + n]
                next_phrase = words[i + n:i + 2 * n]
                if phrase and phrase == next_phrase:
                    new_words.extend(phrase)
                    i += 2 * n
                    changed = True
                    while words[i:i + n] == phrase:
                        i += n
                else:
                    new_words.append(words[i])
                    i += 1
            words = new_words
    return normalize_text(" ".join(words))


def collapse_overlapping_transcript_repetitions(text: str, max_window: int = 18) -> str:
    """
    Remove ASR stutters where a sliding audio window repeats the same span.

    Example shape: "... Diş hekimliği 30 Nisan ... diş hekimliği 30 Nisan ..."
    Exact adjacent repeats are handled above; this catches repeats separated by
    a few noisy bridge words.
    """
    words = normalize_text(text).split()
    if len(words) < 8:
        return normalize_text(text)
    folded = [word.casefold().strip(".,;:!?()[]") for word in words]
    changed = True
    while changed:
        changed = False
        for window in range(min(max_window, len(words) // 2), 2, -1):
            i = 0
            while i + window < len(words):
                left = folded[i:i + window]
                search_end = min(len(words), i + window * 3 + 4)
                best_j = -1
                best_overlap = 0
                for j in range(i + 1, search_end - window + 1):
                    right = folded[j:j + window]
                    overlap = sum(1 for a, b in zip(left, right) if a and a == b)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_j = j
                if best_j > i and best_overlap >= max(3, int(window * 0.72)):
                    # Keep the later span; it usually has more right context.
                    del words[i:best_j]
                    del folded[i:best_j]
                    changed = True
                    break
                i += 1
            if changed:
                break
    return normalize_text(" ".join(words))


DEFAULT_KNOWN_HALLUCINATION_PHRASES = (
    # Turkish subtitle / caption-credit hallucinations commonly produced by Whisper on silence/noise.
    "Altyazı M. K.",
    "Altyazı M.K.",
    "Altyazı M K",
    "Altyazı MK",
    "Altyazı: M. K.",
    "Altyazı: M.K.",
    "Altyazı ekleyen",
    "Altyazı çeviri",
    "Çeviri ve altyazı",
    "Çeviri: M. K.",
    "Çeviri: M.K.",
    "İzlediğiniz için teşekkürler",
    "Beni izlediğiniz için teşekkürler",
    "Abone olmayı unutmayın",
    "Videoyu beğenmeyi unutmayın",
)


def known_hallucination_filter_enabled() -> bool:
    mode = getattr(args, "known_hallucination_filter", "auto")
    if mode == "off":
        return False
    if mode in {"on", "strict"}:
        return True
    return args.profile == "stable" or anti_repeat_enabled()


def known_hallucination_filter_strict() -> bool:
    mode = getattr(args, "known_hallucination_filter", "auto")
    return mode == "strict" or (mode == "auto" and args.profile == "stable" and resolved_anti_repeat_level() == "strict")


def _phrase_to_loose_regex(phrase: str) -> str:
    # Make phrase matching robust to punctuation/spacing variants:
    # "Altyazı M. K." == "altyazi m k" == "Altyazı: M.K"
    chars: list[str] = []
    for ch in phrase.strip():
        if ch.isalnum():
            chars.append(re.escape(ch))
        elif ch.isspace():
            chars.append(r"[\s\W_]*")
        else:
            chars.append(r"[\s\W_]*")
    return "".join(chars)


@lru_cache(maxsize=16)
def known_hallucination_patterns(extra_phrases: str, phrase_file: str) -> tuple[re.Pattern, ...]:
    phrases = [p.strip() for p in DEFAULT_KNOWN_HALLUCINATION_PHRASES if p.strip()]

    if extra_phrases:
        phrases.extend(p.strip() for p in extra_phrases.split("||") if p.strip())

    if phrase_file:
        try:
            for line in Path(phrase_file).expanduser().read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    phrases.append(line)
        except Exception as exc:
            if args.debug:
                log_debug(f"could not read hallucination phrase file: {exc}")

    patterns: list[re.Pattern] = []
    seen: set[str] = set()
    for phrase in phrases:
        key = phrase.casefold()
        if key in seen:
            continue
        seen.add(key)
        loose = _phrase_to_loose_regex(phrase)
        if not loose:
            continue
        # Avoid deleting inside ordinary words; still allow punctuation around the phrase.
        patterns.append(re.compile(rf"(?<!\w){loose}(?!\w)", re.IGNORECASE | re.UNICODE))
    return tuple(patterns)


def remove_known_hallucinations(text: str) -> str:
    text = normalize_text(text)
    if not text or not known_hallucination_filter_enabled():
        return text

    original = text
    for pattern in known_hallucination_patterns(args.extra_hallucination_phrases, args.hallucination_phrase_file):
        text = pattern.sub(" ", text)

    text = normalize_text(text)

    # Extra narrow guard for the common "Altyazı M. K." family even when punctuation is odd.
    text = re.sub(r"(?i)(?<!\w)altyaz[ıi][\s\W_]*m[\s\W_]*k[\s\W_]*(?!\w)", " ", text).strip()
    text = normalize_text(text)

    if args.debug and original != text:
        log_debug(f"removed known hallucination: {original!r} -> {text!r}")
    return text


def is_mostly_known_hallucination(text: str) -> bool:
    cleaned = remove_known_hallucinations(text)
    if not cleaned:
        return True
    if not known_hallucination_filter_strict():
        return False
    before_words = _WORD_RE.findall(normalize_text(text).casefold())
    after_words = _WORD_RE.findall(cleaned.casefold())
    if len(before_words) >= 2 and len(after_words) <= max(1, len(before_words) // 4):
        return True
    return False


THINKING_LEAK_MARKERS = (
    "<|channel>thought",
    "<think",
    "</think",
    "thinking process:",
    "analyze the request",
    "self-correction/constraint check",
    "simulated listening phase",
    "since no audio is provided",
    "the user wants me to listen",
)


def has_model_thinking_artifact(text: str) -> bool:
    folded = normalize_text(text).casefold()
    return any(marker in folded for marker in THINKING_LEAK_MARKERS)


def strip_model_thinking(text: str, *, final: bool = True) -> str:
    """Remove model reasoning wrappers before any generated text can reach UI/logs."""
    text = normalize_text(clean_model_output(text, final=final))
    if not text:
        return ""

    text = re.sub(r"(?is)<think\b[^>]*>.*?</think\s*>", " ", text)
    text = re.sub(r"(?is)<\|channel\>thought\b.*?<channel\|>", " ", text)

    if re.search(r"(?is)<\|channel\>thought\b|<think\b", text):
        final_markers = list(
            re.finditer(
                r"(?is)<\|channel\>(?:final|answer)\b|(?:final answer|final output|transcript|transcription|output|answer)\s*:",
                text,
            )
        )
        if not final_markers:
            return ""
        text = text[final_markers[-1].end():]

    thinking_label = re.search(r"(?i)\bthinking process\s*:", text)
    if thinking_label:
        final_markers = list(
            re.finditer(
                r"(?i)\b(?:final answer|final output|transcript|transcription|output|answer)\s*:",
                text[thinking_label.end():],
            )
        )
        if not final_markers:
            return ""
        text = text[thinking_label.end() + final_markers[-1].end():]

    text = re.sub(r"(?is)<\|channel\>(?:final|answer)\b|<channel\|>", " ", text)
    text = re.sub(r"(?is)</?think\b[^>]*>", " ", text)
    text = re.sub(r"(?is)<\|(?:start|end)_header_id\|>|<\|eot_id\|>", " ", text)
    text = normalize_text(text)
    if has_model_thinking_artifact(text):
        return ""
    return text


def sanitize_asr_text(text: str, *, final: bool) -> str:
    text = strip_model_thinking(text)
    if not text:
        return ""
    if known_hallucination_filter_enabled():
        text = remove_known_hallucinations(text)
        if not text or is_mostly_known_hallucination(text):
            return ""
    if anti_repeat_enabled():
        text = remove_known_hallucinations(text)
        if is_repetitive_text(text) and args.debug:
            # Decoder-loop suspicion is quality evidence, not permission to
            # delete emphasis, corrections, numbers, or whole similar clauses.
            log_debug("ASR repetition detected; preserving decoded words for quality review")
    if not looks_like_real_text(text):
        return ""
    return text


def unicode_script_counts(text: str) -> dict[str, int]:
    """Count Unicode letter scripts used by ASR text."""
    counts: dict[str, int] = {"LETTERS": 0, "LATIN": 0, "ARABIC": 0, "HEBREW": 0, "CYRILLIC": 0}
    for char in text:
        if not char.isalpha():
            continue
        counts["LETTERS"] += 1
        name = unicodedata.name(char, "")
        for script in ("LATIN", "ARABIC", "HEBREW", "CYRILLIC"):
            if script in name:
                counts[script] += 1
                break
    return counts


def debug_log_dropped_asr(reason: str, text: str) -> None:
    """Log dropped ASR text only when explicitly requested."""
    if getattr(args, "debug_dropped_asr", False) or getattr(args, "debug", False):
        log_debug(f"dropped ASR ({reason}): {text!r}")


def sanitize_asr_language_shape(text: str, *, final: bool) -> str:
    """Drop ASR output that is clearly romanized/translated for native-script source languages."""
    text = normalize_text(text)
    if not text or not getattr(args, "strict_asr_language", False):
        return text

    spec = source_language()
    native_scripts = tuple(script.upper() for script in spec.native_scripts)
    if "LATIN" in native_scripts:
        return text

    counts = unicode_script_counts(text)
    letters = counts.get("LETTERS", 0)
    if letters < (4 if final else 6):
        return text

    native_count = sum(counts.get(script, 0) for script in native_scripts)
    latin_count = counts.get("LATIN", 0)
    native_ratio = native_count / max(1, letters)
    latin_ratio = latin_count / max(1, letters)
    configured_ratio = args.native_script_min_ratio if final else args.partial_native_script_min_ratio
    min_native_ratio = max(0.0, min(1.0, float(configured_ratio)))

    if native_count == 0 and latin_ratio >= 0.50:
        debug_log_dropped_asr(
            f"{spec.cli} expected {native_scripts}, got mostly Latin text",
            text,
        )
        return ""

    if native_ratio < min_native_ratio and latin_ratio > native_ratio:
        debug_log_dropped_asr(
            f"{spec.cli} native_ratio={native_ratio:.2f}, latin_ratio={latin_ratio:.2f}, required={min_native_ratio:.2f}",
            text,
        )
        return ""

    return text


class StableWordLock:
    """Locks only words that survive several partial revisions."""
    def __init__(self, confirmations: int = 2, mutable_tail_words: int = 4):
        self.confirmations = max(2, int(confirmations))
        self.mutable_tail_words = max(1, int(mutable_tail_words))
        self.history: list[list[str]] = []
        self.stable_words: list[str] = []

    def update(self, text: str) -> tuple[str, str, float]:
        words = normalize_text(text).split()
        if not words:
            stable = normalize_text(" ".join(self.stable_words))
            return stable, "", 0.0
        self.history.append(words)
        self.history = self.history[-self.confirmations:]
        if len(self.history) < self.confirmations:
            stable = normalize_text(" ".join(self.stable_words))
            mutable = normalize_text(" ".join(words[len(self.stable_words):]))
            ratio = len(self.stable_words) / max(1, len(words))
            return stable, mutable, ratio
        min_len = min(len(h) for h in self.history)
        common_count = 0
        for i in range(min_len):
            token = self.history[0][i]
            if all(h[i] == token for h in self.history):
                common_count += 1
            else:
                break
        lock_until = max(0, common_count - self.mutable_tail_words)
        if lock_until > len(self.stable_words):
            self.stable_words = words[:lock_until]
        stable = normalize_text(" ".join(self.stable_words))
        mutable = normalize_text(" ".join(words[len(self.stable_words):]))
        ratio = len(self.stable_words) / max(1, len(words))
        return stable, mutable, ratio


def pack_by_tokens(sentences: list[str], tokenizer: AutoTokenizer, max_tokens: int) -> list[str]:
    chunks: list[str] = []
    cur = ""
    for sentence in sentences:
        candidate = (cur + " " + sentence).strip() if cur else sentence
        n_tokens = len(tokenizer(candidate, add_special_tokens=False).input_ids)
        if n_tokens <= max_tokens:
            cur = candidate
            continue

        if cur:
            chunks.append(cur)

        ids = tokenizer(sentence, add_special_tokens=False).input_ids
        if len(ids) <= max_tokens:
            cur = sentence
            continue

        for i in range(0, len(ids), max_tokens):
            sub = tokenizer.decode(ids[i:i + max_tokens], skip_special_tokens=True)
            sub = normalize_text(sub)
            if sub:
                chunks.append(sub)
        cur = ""

    if cur:
        chunks.append(cur)
    return chunks


def chunk_for_tts(text: str, max_chars: int = 220) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []

    sentenceish_parts = [part.strip() for part in _CLAUSE_SPLIT.split(text) if part.strip()] or [text]
    chunks: list[str] = []
    buf = ""
    for part in sentenceish_parts:
        if len(buf) + len(part) + 1 <= max_chars:
            buf = (buf + " " + part).strip()
            continue
        if buf:
            chunks.append(buf)
        if len(part) <= max_chars:
            buf = part
            continue
        for i in range(0, len(part), max_chars):
            piece = part[i:i + max_chars].strip()
            if piece:
                chunks.append(piece)
        buf = ""
    if buf:
        chunks.append(buf)
    return chunks


def pack_by_chars(parts: list[str], max_chars: int) -> list[str]:
    """Pack sentence-like parts into larger context chunks for LLM translation."""
    cleaned = [normalize_text(part) for part in parts if normalize_text(part)]
    if not cleaned:
        return []
    if max_chars <= 0:
        return [normalize_text(" ".join(cleaned))]

    chunks: list[str] = []
    buf = ""
    for part in cleaned:
        if not buf:
            buf = part
            continue
        candidate = normalize_text(f"{buf} {part}")
        if len(candidate) <= max_chars:
            buf = candidate
            continue
        chunks.append(buf)
        buf = part

    if buf:
        chunks.append(buf)

    out: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            out.append(chunk)
            continue
        clauses = [part.strip() for part in _CLAUSE_SPLIT.split(chunk) if part.strip()]
        if len(clauses) > 1:
            out.extend(pack_by_chars(clauses, max_chars))
            continue
        for idx in range(0, len(chunk), max_chars):
            piece = chunk[idx:idx + max_chars].strip()
            if piece:
                out.append(piece)
    return out


def longest_suffix_prefix_word_overlap(left_words: list[str], right_words: list[str], max_window: int = 24) -> int:
    left_tail = left_words[-max_window:]
    right_head = right_words[:max_window]
    max_overlap = min(len(left_tail), len(right_head))
    for size in range(max_overlap, 0, -1):
        if left_tail[-size:] == right_head[:size]:
            return size
    return 0


def longest_common_prefix_word_count(left_words: list[str], right_words: list[str]) -> int:
    count = 0
    for left, right in zip(left_words, right_words):
        if left != right:
            break
        count += 1
    return count


def word_sequence_similarity(left_words: list[str], right_words: list[str]) -> float:
    if not left_words and not right_words:
        return 1.0
    if not left_words or not right_words:
        return 0.0
    prefix = longest_common_prefix_word_count(left_words, right_words)
    suffix = longest_suffix_prefix_word_overlap(left_words, right_words)
    shared = max(prefix, suffix)
    return shared / max(1, max(len(left_words), len(right_words)))


def final_dedupe_words(text: str) -> list[str]:
    text = normalize_text(text).casefold()
    text = "".join(
        ch
        for ch in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(ch) and ch != "\u0640"
    )
    return _WORD_RE.findall(text)


def final_transcripts_are_near_duplicates(left: str, right: str) -> bool:
    """Match word-for-word repeats, preserving corrections and negation.

    Text equality alone never proves duplicate audio; the coordinator must also
    verify that the two final requests cover the same source interval.
    """
    left_words = final_dedupe_words(left)
    right_words = final_dedupe_words(right)
    return bool(left_words and left_words == right_words)


def final_transcript_audio_span(result: AsrResult) -> tuple[str, str, int, int] | None:
    """Extract a validated interval in one capture session's source clock."""
    request = result.diagnostics.get("request", {})
    if not isinstance(request, dict) or request.get("capture_gap") is not False:
        return None
    session = request.get("session_id")
    language = request.get("source_language")
    if not isinstance(session, str) or not session or not isinstance(language, str) or not language:
        return None
    if result.session_id and result.session_id != session:
        return None
    values = [request.get(name) for name in
              ("timestamp_start_ns", "timestamp_end_ns", "samples", "sample_rate", "channels")]
    if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
        return None
    start, end, samples, rate, channels = values
    if start < 0 or end <= start or samples <= 0 or rate != SAMPLE_RATE or channels != 1:
        return None
    if abs((end - start) - samples * 1_000_000_000 // rate) > 2 * 1_000_000_000 // rate:
        return None
    return session, language, start, end


def final_transcript_audio_spans_overlap(left: tuple | None, right: tuple | None) -> bool:
    """Require substantial, proven source overlap before suppressing a repeat."""
    if left is None or right is None or left[:2] != right[:2]:
        return False
    overlap = min(left[3], right[3]) - max(left[2], right[2])
    shorter = min(left[3] - left[2], right[3] - right[2])
    return overlap > 0 and overlap * 5 >= shorter * 4


def common_prefix_text(previous: str, current: str) -> str:
    prev_words = normalize_text(previous).split()
    cur_words = normalize_text(current).split()
    prefix_words = longest_common_prefix_word_count(prev_words, cur_words)
    if prefix_words <= 0:
        return ""
    return normalize_text(" ".join(cur_words[:prefix_words]))


def incremental_text_suffix(previous: str, current: str) -> str:
    previous = normalize_text(previous)
    current = normalize_text(current)
    if not current:
        return ""
    if not previous:
        return current
    if current == previous or current in previous:
        return ""
    if current.startswith(previous):
        return normalize_text(current[len(previous):])

    prev_words = previous.split()
    cur_words = current.split()
    overlap = longest_suffix_prefix_word_overlap(prev_words, cur_words)
    if overlap > 0:
        return normalize_text(" ".join(cur_words[overlap:]))
    return current


def merge_partial_transcript(previous: str, current: str) -> str:
    """
    Merge a bounded sliding-window ASR preview into the cumulative partial text.
    Final ASR still re-runs on the complete utterance, so this path optimizes UI/MT
    responsiveness without making speculative text authoritative.
    """
    previous = normalize_text(previous)
    current = normalize_text(current)
    if not previous:
        return current
    if not current or current == previous:
        return previous
    if current.startswith(previous):
        return current
    if previous.endswith(current) or current in previous:
        return previous

    prev_words = previous.split()
    cur_words = current.split()
    overlap = longest_suffix_prefix_word_overlap(prev_words, cur_words, max_window=32)
    if overlap > 0:
        return normalize_text(" ".join([*prev_words, *cur_words[overlap:]]))

    prefix = longest_common_prefix_word_count(prev_words, cur_words)
    if prefix >= max(2, min(len(prev_words), len(cur_words)) // 2):
        return current

    if len(cur_words) <= 10 and len(prev_words) >= 3:
        return normalize_text(f"{previous} {current}")
    return current


def merge_partial_transcript_with_provenance(previous: str, current: str,
                                            previous_span: tuple | None, current_span: tuple | None) -> str:
    """Replace revisions of a growing window; merge only shifted windows.

    An ASR hypothesis for the same source start is a new reading of the whole
    window, including corrected words. Appending it to the older hypothesis
    would turn a correction such as 'five' → 'six' into two spoken requests.
    """
    if previous_span is not None and current_span is not None and previous_span[:3] == current_span[:3]:
        return normalize_text(current if current_span[3] >= previous_span[3] else previous)
    return merge_partial_transcript(previous, current)


def word_count(text: str) -> int:
    """Return a whitespace word count after normalizing transcript text."""
    return len(normalize_text(text).split())


def choose_more_complete_final_text(final_text: str, previous_partial: str) -> str:
    """Preserve final ASR authority while recovering details from the last partial."""
    final_text = normalize_text(final_text)
    previous_partial = normalize_text(previous_partial)
    if not args.final_merge_last_partial or args.final_translation_completeness_mode == "off":
        return final_text
    if not previous_partial:
        return final_text
    if not final_text:
        return previous_partial

    final_words = final_text.split()
    partial_words = previous_partial.split()
    if not final_words or not partial_words:
        return final_text or previous_partial

    mode = args.final_translation_completeness_mode
    min_final_words = max(1, int(args.final_merge_min_final_words))

    if final_text == previous_partial:
        return final_text
    if final_text in previous_partial and (mode == "aggressive" or len(final_words) < min_final_words):
        return previous_partial
    if previous_partial in final_text:
        return final_text

    partial_then_final = merge_partial_transcript(previous_partial, final_text)
    final_then_partial = merge_partial_transcript(final_text, previous_partial)

    candidates = [final_text]
    if partial_then_final:
        candidates.append(partial_then_final)
    if final_then_partial:
        candidates.append(final_then_partial)

    def score(candidate: str) -> tuple[int, int, int]:
        words = candidate.split()
        duplicate_penalty = max(0, len(words) - len(set(words)))
        added_words = max(0, len(words) - len(final_words))
        return (len(words) - duplicate_penalty, -added_words, len(candidate))

    best = max(candidates, key=score)
    added_words = max(0, word_count(best) - word_count(final_text))

    if mode == "aggressive":
        return best

    final_is_short = word_count(final_text) < min_final_words
    partial_extends_final = previous_partial.startswith(final_text) or previous_partial.endswith(final_text)
    reasonable_addition = added_words <= max(8, word_count(final_text))

    if final_is_short or partial_extends_final or reasonable_addition:
        return best
    return final_text


def prepare_final_transcript_for_translation(raw_final_text: str, previous_partial: str) -> str:
    """Sanitize and optionally merge final ASR before committed translation."""
    raw_final_text = normalize_text(raw_final_text)
    previous_partial = normalize_text(previous_partial)

    final_text = sanitize_asr_text(raw_final_text, final=True)
    if not final_text and args.empty_final_action == "fallback":
        final_text = sanitize_asr_text(previous_partial, final=True)
        if args.debug and final_text:
            log_debug("empty final ASR; falling back to last partial")

    if not final_text:
        return ""

    if args.final_merge_last_partial and previous_partial:
        merged = choose_more_complete_final_text(final_text, previous_partial)
        merged = sanitize_asr_text(merged, final=True)
        if merged and merged != final_text:
            if args.debug:
                log_debug(f"final ASR merged with last partial: final={final_text!r} partial={previous_partial!r} merged={merged!r}")
            return merged

    return final_text


def find_command(cmd: str) -> str | None:
    direct = shutil.which(cmd)
    if direct:
        return direct
    local = Path(sys.executable).resolve().parent / cmd
    if local.exists() and os.access(local, os.X_OK):
        return str(local)
    return None


def have(cmd: str) -> bool:
    return find_command(cmd) is not None


def is_macos() -> bool:
    return sys.platform == "darwin"


def torch_mps_available() -> bool:
    return bool(torch is not None and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def resolve_audio_io_backend() -> str:
    if args.audio_io != "auto":
        return args.audio_io
    if args.input_device is not None:
        return "sounddevice"
    if is_apple_silicon() and Path(__file__).with_name("apple_vpio_bridge").exists():
        return "apple-vpio"
    return "sounddevice"


def sounddevice_input_device() -> str | int | None:
    device = args.input_device
    if isinstance(device, str) and re.fullmatch(r"-?\d+", device.strip()):
        return int(device)
    return device


def resolve_streaming_engine() -> str:
    if args.streaming_engine != "auto":
        return args.streaming_engine
    bridge = Path(args.apple_bridge_path).expanduser()
    if is_apple_silicon() and resolve_audio_io_backend() in {"apple-vpio", "apple-raw"} and bridge.exists() and os.access(bridge, os.X_OK):
        return "ring"
    return "legacy"


def resolve_vad_backend() -> str:
    if args.vad_backend != "auto":
        return args.vad_backend
    if SILERO_VAD_INSTALLED and torch is not None:
        return "silero"
    if webrtcvad is not None:
        return "webrtc"
    return "energy"


def resolve_silero_vad_runtime() -> str:
    if args.silero_vad_runtime != "auto":
        return args.silero_vad_runtime
    return "onnx" if ONNXRUNTIME_INSTALLED else "torch"


def is_apple_silicon() -> bool:
    return is_macos() and os.uname().machine == "arm64"


def prefer_project_model(model_ref: str) -> str:
    """Use the restored local copy of a selected MLX model when it is present."""
    model_ref = str(model_ref).strip()
    direct = Path(model_ref).expanduser()
    if direct.exists():
        return str(direct.resolve())
    if model_ref.startswith("mlx-community/"):
        local = Path(__file__).resolve().parent / "models" / model_ref.rsplit("/", 1)[-1]
        if local.is_dir():
            return str(local)
    return model_ref


def load_pretrained_local_first(loader, model_id: str):
    model_ref = prefer_project_model(model_id)
    allow_download = os.environ.get("HF_HUB_OFFLINE", "1").strip().lower() in {"0", "false", "no"}
    try:
        return loader.from_pretrained(model_ref, local_files_only=not allow_download)
    except (OSError, ValueError) as exc:
        if allow_download:
            raise
        raise RuntimeError(
            f"Cannot load local model {model_ref!r}: {exc}. "
            "Prepare its assets with scripts/prepare_models.py --download, "
            "or explicitly allow Hub access with HF_HUB_OFFLINE=0."
        ) from exc


def ensure_mlx_whisper_loaded():
    global mlx_whisper, mlx_whisper_import_error
    if mlx_whisper is not None:
        return mlx_whisper
    if not MLX_WHISPER_INSTALLED:
        raise RuntimeError("mlx-whisper is not installed")
    try:
        mlx_whisper = importlib.import_module("mlx_whisper")
        return mlx_whisper
    except Exception as exc:
        mlx_whisper_import_error = exc
        raise


def ensure_mlx_vlm_loaded() -> None:
    global mlx_vlm_load, mlx_vlm_generate, mlx_vlm_apply_chat_template, mlx_vlm_import_error
    if mlx_vlm_load is not None and mlx_vlm_generate is not None and mlx_vlm_apply_chat_template is not None:
        return
    if not MLX_VLM_INSTALLED:
        raise RuntimeError("mlx-vlm is not installed. Install it with: pip install -U mlx-vlm")
    try:
        module = importlib.import_module("mlx_vlm")
        prompt_utils = importlib.import_module("mlx_vlm.prompt_utils")
        mlx_vlm_load = module.load
        mlx_vlm_generate = module.generate
        mlx_vlm_apply_chat_template = prompt_utils.apply_chat_template
    except Exception as exc:
        mlx_vlm_import_error = exc
        raise


def ensure_mlx_core_loaded():
    global mlx_core, mlx_core_import_error
    if mlx_core is not None:
        return mlx_core
    try:
        mlx_core = importlib.import_module("mlx.core")
        return mlx_core
    except Exception as exc:
        mlx_core_import_error = exc
        raise


def mlx_thread_stream_context():
    """
    Return an MLX stream context bound to the current Python thread.

    MLX GPU streams are thread-local. Loading a VLM on the main thread and then
    generating in the ASR worker can trigger "There is no Stream(gpu, 0) in
    current thread", so each inference thread gets its own stream context.
    """
    mx = ensure_mlx_core_loaded()
    stream = getattr(mlx_thread_state, "stream", None)
    if stream is None:
        stream = mx.new_thread_local_stream(mx.gpu)
        mlx_thread_state.stream = stream
    return mx.stream(stream)


def ensure_mlx_lm_loaded() -> None:
    global mlx_lm_load, mlx_lm_stream_generate, mlx_lm_import_error
    if mlx_lm_load is not None and mlx_lm_stream_generate is not None:
        return
    if not MLX_LM_INSTALLED:
        raise RuntimeError("mlx-lm is not installed")
    try:
        module = importlib.import_module("mlx_lm")
        mlx_lm_load = module.load
        mlx_lm_stream_generate = module.stream_generate
    except Exception as exc:
        mlx_lm_import_error = exc
        raise


silero_vad_model = None
silero_vad_iterator = None
silero_vad_pending_pcm = bytearray()
silero_vad_last_speech = False


def ensure_silero_vad_loaded():
    global silero_vad_model, silero_vad_iterator
    if silero_vad_model is not None and silero_vad_iterator is not None:
        return silero_vad_model, silero_vad_iterator
    if not SILERO_VAD_INSTALLED:
        raise RuntimeError("silero-vad is not installed")
    if torch is None:
        raise RuntimeError("torch is required for silero-vad")
    module = importlib.import_module("silero_vad")
    silero_vad_model = module.load_silero_vad(onnx=resolve_silero_vad_runtime() == "onnx")
    silero_vad_iterator = module.VADIterator(
        silero_vad_model,
        threshold=args.silero_vad_threshold,
        sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=max(40, min(args.silence_min_ms, args.vad_hangover_ms + 80)),
        speech_pad_ms=max(20, min(args.tail_pad_ms, 120)),
    )
    return silero_vad_model, silero_vad_iterator


def reset_silero_vad_state() -> None:
    global silero_vad_pending_pcm, silero_vad_last_speech
    silero_vad_pending_pcm.clear()
    silero_vad_last_speech = False
    if silero_vad_iterator is not None:
        silero_vad_iterator.reset_states()


def is_silero_speech_frame(frame_i16: np.ndarray) -> bool:
    global silero_vad_pending_pcm, silero_vad_last_speech
    _model, iterator = ensure_silero_vad_loaded()
    frame_i16 = np.asarray(frame_i16, dtype=np.int16).reshape(-1)
    if frame_i16.size:
        silero_vad_pending_pcm.extend(frame_i16.tobytes())

    # Silero's streaming iterator expects ~512-sample windows at 16 kHz. The
    # producer stays at 20 ms/320 samples, so keep pending PCM in a bytearray to
    # avoid allocating a new concatenated float array on every capture frame.
    chunk_bytes = 512 * 2
    while len(silero_vad_pending_pcm) >= chunk_bytes:
        raw = bytes(silero_vad_pending_pcm[:chunk_bytes])
        del silero_vad_pending_pcm[:chunk_bytes]
        chunk = np.frombuffer(raw, dtype=np.int16).astype(np.float32) * (1.0 / 32768.0)
        event = iterator(torch.from_numpy(chunk))
        if event and "start" in event:
            silero_vad_last_speech = True
        elif event and "end" in event:
            silero_vad_last_speech = False
        else:
            silero_vad_last_speech = bool(iterator.triggered)
    return silero_vad_last_speech


def save_wav(path: str | Path, audio_i16: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    path = str(path)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_i16.astype(np.int16).tobytes())


def rms_dbfs(x_f32: np.ndarray) -> float:
    rms = np.sqrt(np.mean(x_f32 * x_f32) + 1e-12)
    return 20.0 * np.log10(rms + 1e-12)


def rms_dbfs_i16(x_i16: np.ndarray) -> float:
    if x_i16.size == 0:
        return -120.0
    x_f32 = x_i16.astype(np.float32) * (1.0 / 32768.0)
    return rms_dbfs(x_f32)


def pre_emphasis(x: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    if x.size < 2:
        return x
    out = np.empty_like(x)
    out[0] = x[0]
    out[1:] = x[1:] - coeff * x[:-1]
    return out


def rms_normalize(x: np.ndarray, target: float = 0.10) -> np.ndarray:
    rms = np.sqrt(np.mean(x * x) + 1e-9)
    if rms < 1e-8:
        return x
    return x * (target / rms)


def bandpass_voice(frame_f32: np.ndarray, sr: int = SAMPLE_RATE, lo: float = BPF_LO_HZ, hi: float = BPF_HI_HZ) -> np.ndarray:
    n = frame_f32.shape[0]
    if n == 0:
        return frame_f32
    spec = np.fft.rfft(frame_f32)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    mask = (freqs >= lo) & (freqs <= hi)
    spec *= mask
    out = np.fft.irfft(spec, n=n)
    return out.astype(np.float32, copy=False)


def compute_adaptive_threshold(pause_est_ms: float, speech_run_ms_local: int) -> int:
    base = 0.75 * pause_est_ms + 0.25 * args.silence_base_ms
    if speech_run_ms_local > 2500:
        base += 140
    elif speech_run_ms_local < 900:
        base -= 60
    return max(args.silence_min_ms, min(args.silence_max_ms, int(base)))


def sentence_completeness_score(text: str) -> int:
    text = normalize_text(text)
    if not text:
        return 0

    lowered = text.casefold()
    words = lowered.split()
    word_count = len(words)
    last_word = words[-1] if words else ""
    first_word = words[0] if words else ""

    final_punct = text.endswith((".", "?", "!"))
    soft_punct = text.endswith((",", ";", ":"))
    incomplete_endings = {
        "ve", "ama", "ya", "veya", "ile", "gibi", "çünkü", "için", "ki", "de", "da",
        "sonra", "önce", "eğer", "şey", "yani", "fakat", "ancak", "olarak", "kadar", "üzere",
        "the", "a", "an", "and", "or", "but", "to", "of", "for", "with", "that", "because",
        "if", "when", "while", "so", "as", "like", "about",
    }
    incomplete_tail_phrases = (
        "istiyorum ki", "şunu istiyorum", "bunu istiyorum", "demek istiyorum", "söylemek istiyorum",
        "i want to", "i would like to", "i need to", "i was going to", "what i mean is",
        "can you", "could you", "would you", "let me",
    )
    complete_tail_phrases = (
        "teşekkürler", "sağ ol", "sağ olun", "tamam", "olur", "evet", "hayır", "görüşürüz",
        "lütfen", "bu kadar", "anladım", "doğru", "yanlış",
        "thank you", "thanks", "okay", "ok", "yes", "no", "please", "that's all", "got it",
    )
    is_complete_short = lowered in complete_tail_phrases or lowered.endswith(complete_tail_phrases)
    openers = {
        "what", "why", "how", "when", "where", "who", "is", "are", "do", "does", "did", "can", "could",
        "ne", "neden", "nasıl", "ne zaman", "nerede", "kim", "mi", "mı", "mu", "mü",
    }

    score = 0
    if word_count >= 3:
        score += 1
    if word_count >= 6:
        score += 1
    if final_punct:
        score += 3
    elif soft_punct:
        score -= 1
    if last_word in incomplete_endings:
        score -= 3
    if lowered.endswith(incomplete_tail_phrases):
        score -= 4
    if word_count <= 2 and not final_punct and not is_complete_short:
        score -= 2
    if first_word in openers and word_count >= 4:
        score += 1
    if is_complete_short:
        score += 4
    if lowered.endswith((" istiyorum", " olacak", " oldu", " lütfen", " teşekkürler", " thanks", " thank you")):
        score += 1
    return score


def semantic_endpoint_threshold(base_ms: int, current_text: str) -> int:
    text = normalize_text(current_text)
    if not text:
        return base_ms

    score = sentence_completeness_score(text)
    word_count = len(text.split())

    if score >= 4:
        base_ms -= 220
    elif score >= 2:
        base_ms -= 120
    elif score <= -2:
        base_ms += 260
    elif score <= 0:
        base_ms += 100

    if word_count >= 10:
        base_ms -= 60
    elif word_count <= 2:
        base_ms += 120

    return max(args.silence_min_ms, min(args.silence_max_ms, int(base_ms)))


def transcript_stability_bias(base_ms: int, stability: int, last_change_ms: float) -> int:
    if stability >= 3:
        base_ms -= 140
    elif stability == 2:
        base_ms -= 80
    elif stability <= 0:
        base_ms += 120

    if last_change_ms <= 220:
        base_ms += 140
    elif last_change_ms <= 420:
        base_ms += 70
    elif last_change_ms >= 900 and stability >= 2:
        base_ms -= 60

    return max(args.silence_min_ms, min(args.silence_max_ms, int(base_ms)))


def commit_confidence_score(current_text: str, stability: int, last_change_ms: float, silence_ms: int, locked_ratio: float = 0.0) -> int:
    text = normalize_text(current_text)
    if not text:
        return 0

    score = 0
    completeness = sentence_completeness_score(text)
    word_count = len(text.split())

    if completeness >= 4:
        score += 4
    elif completeness >= 2:
        score += 2
    elif completeness <= -2:
        score -= 3
    elif completeness <= 0:
        score -= 1

    if stability >= 5:
        score += 3
    elif stability >= 3:
        score += 2
    elif stability >= 1:
        score += 1

    if locked_ratio >= 0.85:
        score += 3
    elif locked_ratio >= 0.65:
        score += 2
    elif locked_ratio >= 0.45:
        score += 1
    elif locked_ratio <= 0.15 and word_count >= 4:
        score -= 2

    if last_change_ms >= 900:
        score += 2
    elif last_change_ms >= 500:
        score += 1
    elif last_change_ms <= 220:
        score -= 2

    if silence_ms >= 700:
        score += 3
    elif silence_ms >= 420:
        score += 2
    elif silence_ms >= 240:
        score += 1

    if word_count <= 2 and silence_ms < 500:
        score -= 2
    elif word_count >= 8:
        score += 1

    return score


def partial_translation_clump(text: str) -> str:
    """
    Return the stabilized source span that is safe to translate speculatively.

    Large Whisper partials can advance in small fragments. Translating every
    fragment makes LLM MT brittle, so partial MT waits for a complete-ish clump,
    a punctuation-bounded prefix, or a sufficiently long stabilized phrase.
    """
    text = normalize_text(text)
    if not text:
        return ""

    min_score = int(args.partial_translation_min_completeness_score)
    force_words = max(1, int(args.partial_translation_force_words))
    words = text.split()
    if sentence_completeness_score(text) >= min_score or len(words) >= force_words:
        return text

    terminal_matches = list(re.finditer(r"[.!?…](?:\s|$)", text))
    if terminal_matches:
        prefix = normalize_text(text[:terminal_matches[-1].end()])
        if prefix and sentence_completeness_score(prefix) >= min_score:
            return prefix

    return ""


def partial_display_source_text(partial_text: str, locked_prefix: str) -> str:
    """
    Return the Turkish span that is stable enough to put on screen.

    Raw Whisper partials are useful for endpointing, but they can be revised
    aggressively. The display path is intentionally stricter so visible text
    and display-only English MT stay close to the eventual final transcript.
    """
    text = normalize_text(partial_text)
    locked = normalize_text(locked_prefix)
    min_words = max(1, int(args.partial_display_min_words))
    policy = args.partial_display_policy

    if policy == "latest":
        return text

    if not locked or word_count(locked) < min_words:
        return ""

    if policy == "clump":
        clump = partial_translation_clump(locked)
        return clump or ""

    return locked


def confidence_adjusted_endpoint_threshold(base_ms: int, commit_confidence: int) -> int:
    if commit_confidence >= 9:
        base_ms -= 220
    elif commit_confidence >= 6:
        base_ms -= 140
    elif commit_confidence >= 4:
        base_ms -= 80
    elif commit_confidence <= 0:
        base_ms += 180
    elif commit_confidence <= 2:
        base_ms += 80
    return max(args.silence_min_ms, min(args.silence_max_ms, int(base_ms)))


def should_semantic_endpoint(current_text: str, stability: int, last_change_ms: float, silence_ms: int, locked_ratio: float) -> bool:
    if not args.semantic_endpointing:
        return False
    if silence_ms < args.semantic_min_silence_ms:
        return False
    text = normalize_text(current_text)
    if not text:
        return False
    completeness = sentence_completeness_score(text)
    if completeness < 3:
        return False
    commit_confidence = commit_confidence_score(text, stability, last_change_ms, silence_ms, locked_ratio)
    if commit_confidence < args.semantic_commit_score:
        return False
    if stability < 2 and locked_ratio < 0.55:
        return False
    return True


def min_dbfs_utterance_threshold() -> float:
    if resolve_audio_io_backend() == "apple-vpio":
        return MIN_DBFS_UTTERANCE_APPLE_VPIO
    return MIN_DBFS_UTTERANCE


def silent_external_output() -> ExitStack:
    stack = ExitStack()
    stack.enter_context(redirect_stdout(io.StringIO()))
    stack.enter_context(redirect_stderr(io.StringIO()))
    return stack


RTL_RANGES = (
    ("\u0590", "\u05ff"),
    ("\u0600", "\u06ff"),
    ("\u0750", "\u077f"),
    ("\u08a0", "\u08ff"),
    ("\ufb50", "\ufdff"),
    ("\ufe70", "\ufeff"),
)


def contains_rtl(text: str) -> bool:
    plain = strip_ansi(text)
    return any(any(start <= ch <= end for start, end in RTL_RANGES) for ch in plain)


def terminal_safe_text(text: str) -> str:
    """
    Return text unchanged for display.

    Arabic shaping must remain intact. The dashboard keeps borders stable by
    rendering fixed columns separately instead of inserting ZWNJ or BiDi control
    characters into user-visible text.
    """
    return text


def color_tr(text: str) -> str:
    text = terminal_safe_text(text)
    return f"{ANSI_DIM_GREEN}{text}{ANSI_RESET}"


def color_en(text: str) -> str:
    text = terminal_safe_text(text)
    return f"{ANSI_BRIGHT_WHITE}{text}{ANSI_RESET}"



def source_label() -> str:
    return source_language_code()


def target_label() -> str:
    return target_language_code()


def color_left_tr(text: str) -> str:
    # Left panel = incoming transcription: Turkish dim green.
    return color_tr(text)


def color_left_en(text: str) -> str:
    # Left panel finalized translation stays visually authoritative.
    return color_en(text)


def color_left_en_partial(text: str) -> str:
    text = terminal_safe_text(text)
    return f"{ANSI_DIM_WHITE}{text}{ANSI_RESET}"


def color_uncertain(text: str) -> str:
    text = terminal_safe_text(text)
    return f"{ANSI_DIM_WHITE}{text}{ANSI_RESET}"


def color_right_en(text: str) -> str:
    # Right panel = typed/source side: reverse colors, English dim green.
    return color_tr(text)


def color_right_tr(text: str) -> str:
    # Right panel = translated output: reverse colors, Turkish bright white.
    return color_en(text)




def format_partial_transcript_for_ui(partial_text: str, locked_prefix: str, *, inflight: bool = False) -> str:
    text = normalize_text(partial_text)
    locked = normalize_text(locked_prefix)
    inflight_note = color_uncertain("… transcribing current audio (quality pending)") if inflight else ""
    if not text:
        body = inflight_note or color_uncertain("… listening")
        return f"🗣️ {source_label()}~: {body}"
    if not args.confidence_ui or not locked or not text.startswith(locked):
        body = color_uncertain(text) if inflight else color_left_tr(text)
        return f"🗣️ {source_label()}~: {body}" + (f"  ·  {inflight_note}" if inflight_note and text else "")
    tail = normalize_text(text[len(locked):])
    tail_text = f"  ·  {color_uncertain('… ' + tail)}" if tail else ""
    inflight_text = f"  ·  {inflight_note}" if inflight_note else ""
    return f"🗣️ {source_label()}✓: {color_left_tr(locked)}{tail_text}{inflight_text}"

def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def strip_for_log(text: str) -> str:
    return normalize_text(strip_ansi(text).replace("\r", " ").replace("\n", " "))


def append_session_log(path: str, payload: dict) -> None:
    if not args.session_logs or not path:
        return
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), **payload}
    try:
        log_path = Path(path).expanduser()
        if log_path.parent and str(log_path.parent) not in {"", "."}:
            log_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False)
        with session_log_lock:
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception as exc:
        if args.debug:
            log_debug(f"failed to write session log {path}: {exc}")


def log_final_transcription(utterance_id: int | None, transcription_text: str, translation_text: str) -> None:
    append_session_log(
        args.transcription_log,
        {
            "type": "audio_transcription",
            "utterance_id": utterance_id,
            "transcription_lang": source_language_code(),
            "translation_lang": target_language_code(),
            "transcription": strip_for_log(transcription_text),
            "translation": strip_for_log(translation_text),
        },
    )




def schedule_final_transcription_log(utterance_id: int, revision: int, transcription_text: str, translation_text: str) -> None:
    """Append only the latest final translation for an utterance after a short settle window."""
    delay = max(0, args.final_transcription_log_debounce_ms) / 1000.0
    with session_log_lock:
        if utterance_id in logged_audio_translation_utterances:
            return
        pending_audio_translation_logs[utterance_id] = (revision, transcription_text, translation_text)

    def _flush() -> None:
        with session_log_lock:
            pending = pending_audio_translation_logs.get(utterance_id)
            if pending != (revision, transcription_text, translation_text):
                return
            if utterance_id in logged_audio_translation_utterances:
                pending_audio_translation_logs.pop(utterance_id, None)
                return
            pending_audio_translation_logs.pop(utterance_id, None)
            logged_audio_translation_utterances.add(utterance_id)
        log_final_transcription(utterance_id, transcription_text, translation_text)

    if delay <= 0:
        _flush()
        return
    timer = threading.Timer(delay, _flush)
    timer.daemon = True
    timer.start()


def log_typed_translation(entry_id: int | None, input_text: str, translation_text: str) -> None:
    append_session_log(
        args.tts_log,
        {
            "type": "typed_translation",
            "entry_id": entry_id,
            "input_lang": target_language_code(),
            "translation_lang": source_language_code(),
            "input": strip_for_log(input_text),
            "translation": strip_for_log(translation_text),
        },
    )


ARABIC_LAM = "\u0644"
ARABIC_ALEF_FORMS = {
    "\u0627",  # alef
    "\u0622",  # alef with madda
    "\u0623",  # alef with hamza above
    "\u0625",  # alef with hamza below
    "\u0671",  # alef wasla
}
ARABIC_LAM_ALEF_PRESENTATION_FORMS = set("\ufef5\ufef6\ufef7\ufef8\ufef9\ufefa\ufefb\ufefc")


def _is_arabic_lam_alef_pair(text: str, index: int) -> bool:
    return (
        index + 1 < len(text)
        and text[index] == ARABIC_LAM
        and text[index + 1] in ARABIC_ALEF_FORMS
    )


def _terminal_cluster_at(text: str, index: int) -> tuple[str, int, int]:
    """Return a terminal-rendered cluster, its display width, and the next index."""
    ch = text[index]
    if _is_arabic_lam_alef_pair(text, index):
        return text[index:index + 2], 2, index + 2
    if ch in ARABIC_LAM_ALEF_PRESENTATION_FORMS:
        return ch, 2, index + 1
    if unicodedata.combining(ch):
        return ch, 0, index + 1
    if unicodedata.category(ch) in {"Cf", "Mn", "Me"}:
        return ch, 0, index + 1
    return ch, 2 if unicodedata.east_asian_width(ch) in {"W", "F"} else 1, index + 1


def count_lam_alef_clusters(text: str) -> int:
    """Count lam-alef clusters for terminal divider compensation without mutating text."""
    plain = strip_ansi(unicodedata.normalize("NFKC", text))
    count = 0
    index = 0
    while index < len(plain):
        if _is_arabic_lam_alef_pair(plain, index):
            count += 1
            index += 2
            continue
        if plain[index] in ARABIC_LAM_ALEF_PRESENTATION_FORMS:
            count += 1
        index += 1
    return count


def display_width(text: str) -> int:
    plain = strip_ansi(text)
    width = 0
    index = 0
    while index < len(plain):
        _, cluster_width, index = _terminal_cluster_at(plain, index)
        width += cluster_width
    return width


def _ansi_tokens(text: str) -> list[str]:
    """Split text into ANSI escape sequences and normal text chunks."""
    return [part for part in re.split(r'(\x1b\[[0-9;]*m)', text) if part]


def _ansi_apply_active(active_codes: list[str]) -> str:
    return "".join(active_codes)


def _ansi_update_active(code: str, active_codes: list[str]) -> None:
    # Reset clears the active style. Other SGR codes replace the active style for
    # our use case (single foreground/intensity color), avoiding duplicated codes.
    if code == ANSI_RESET or code == "\x1b[0m":
        active_codes.clear()
    else:
        active_codes[:] = [code]


def _char_display_width(ch: str) -> int:
    _, cluster_width, _ = _terminal_cluster_at(ch, 0)
    return cluster_width


def truncate_visible(text: str, width: int) -> str:
    """Truncate by visible terminal width while preserving inline ANSI colors."""
    if width <= 0:
        return ""
    if display_width(text) <= width:
        return text
    if width <= 1:
        return "…"

    out: list[str] = []
    active_codes: list[str] = []
    used = 0
    limit = width - 1

    for token in _ansi_tokens(text):
        if ANSI_RE.fullmatch(token):
            out.append(token)
            _ansi_update_active(token, active_codes)
            continue
        index = 0
        while index < len(token):
            cluster, cluster_width, index = _terminal_cluster_at(token, index)
            if used + cluster_width > limit:
                if active_codes:
                    out.append(ANSI_RESET)
                return "".join(out).rstrip() + "…"
            out.append(cluster)
            used += cluster_width

    if active_codes:
        out.append(ANSI_RESET)
    return "".join(out)


def pad_visible(text: str, width: int) -> str:
    clipped = truncate_visible(text, width)
    pad = max(0, width - display_width(clipped))
    return clipped + (" " * pad)


def _wrap_ansi_segment(text: str, width: int) -> list[str]:
    """Wrap a string preserving ANSI color sequences across line breaks."""
    if width <= 0:
        return [""]

    lines: list[str] = []
    cur: list[str] = []
    used = 0
    active_codes: list[str] = []

    def flush_line() -> None:
        nonlocal cur, used
        line = "".join(cur).rstrip()
        if active_codes and line and not line.endswith(ANSI_RESET):
            line += ANSI_RESET
        lines.append(line)
        cur = []
        if active_codes:
            cur.append(_ansi_apply_active(active_codes))
        used = 0

    for token in _ansi_tokens(text):
        if ANSI_RE.fullmatch(token):
            cur.append(token)
            _ansi_update_active(token, active_codes)
            continue

        index = 0
        while index < len(token):
            cluster, cluster_width, index = _terminal_cluster_at(token, index)
            if used + cluster_width > width and used > 0:
                flush_line()
                if cluster.isspace():
                    continue
            cur.append(cluster)
            used += cluster_width

    if cur or not lines:
        line = "".join(cur).rstrip()
        if active_codes and line and not line.endswith(ANSI_RESET):
            line += ANSI_RESET
        lines.append(line)

    return lines or [""]


def wrap_visible(text: str, width: int) -> list[str]:
    """
    Wrap panel text by visible width while preserving inline ANSI colors.

    The old renderer stripped ANSI from any line where the color began after a
    label such as "🗣️ TR:" or "🌍 EN:". That made all panel text appear white.
    This version keeps labels plain and wraps the colored content without
    removing escape sequences.
    """
    if width <= 0:
        return [""]
    if not strip_ansi(text):
        return [""]

    # Most panel lines are "label: colored content". Keep the label visible on
    # the first line and indent continuation lines to the content column.
    raw_label = ""
    raw_content = text
    match = re.match(r'^(.*?:\s+)(.*)$', text, flags=re.DOTALL)
    if match and display_width(match.group(1)) < width:
        raw_label, raw_content = match.groups()

    if not raw_label:
        return _wrap_ansi_segment(text, width)

    indent = " " * display_width(raw_label)
    first_width = max(1, width - display_width(raw_label))
    later_width = max(1, width - display_width(indent))
    content_lines = _wrap_ansi_segment(raw_content, first_width)

    lines: list[str] = []
    for idx, segment in enumerate(content_lines):
        if idx == 0:
            lines.append((raw_label + segment).rstrip())
        else:
            # Re-wrap any over-wide continuation using the later width.
            sublines = _wrap_ansi_segment(segment, later_width)
            for sub in sublines:
                lines.append((indent + sub).rstrip())
    return lines or [""]


def scroll_visible(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if display_width(text) <= width:
        return text
    if width <= 1:
        return "…"
    plain = strip_ansi(text)
    clusters: list[tuple[str, int]] = []
    index = 0
    while index < len(plain):
        cluster, cluster_width, index = _terminal_cluster_at(plain, index)
        clusters.append((cluster, cluster_width))

    out: list[str] = []
    used = 1
    for cluster, cluster_width in reversed(clusters):
        if cluster_width == 0:
            out.append(cluster)
            continue
        if used + cluster_width > width:
            break
        out.append(cluster)
        used += cluster_width
    return "…" + "".join(reversed(out))


def panel_entry_lines(entry: dict[str, str], width: int) -> list[str]:
    lines: list[str] = []
    for field in ("src", "dst"):
        value = entry.get(field, "")
        if value:
            lines.extend(wrap_visible(value, width))
    return lines


def panel_lines(side: str, width: int, max_lines: int) -> list[str]:
    done_key = f"{side}_done"
    active_key = f"{side}_active"
    order_key = f"{side}_order"

    active_lines: list[str] = []
    for entry_id in ui_state[order_key]:
        entry = ui_state[active_key].get(entry_id)
        if entry:
            active_lines.extend(panel_entry_lines(entry, width))

    history_lines: list[str] = []
    for entry in ui_state[done_key]:
        history_lines.extend(panel_entry_lines(entry, width))

    if side == "left" and active_lines:
        visible_active = active_lines[:max_lines]
        remaining = max(0, max_lines - len(visible_active))
        visible_history = history_lines[-remaining:] if remaining else []
        return visible_active + ([""] * max(0, remaining - len(visible_history))) + visible_history

    lines = history_lines + active_lines
    if len(lines) < max_lines:
        return ([""] * (max_lines - len(lines))) + lines
    return lines[-max_lines:]

def _write_dashboard_row(row: int, left: str, divider: str, right: str, half: int, cols: int) -> None:
    """
    Render dashboard columns at fixed cursor positions.

    Some terminals visually pull the center separator left by one cell for each
    lam-alef cluster in RTL text. Compensate at the divider cursor position
    instead of changing the Arabic text.
    """
    compensation = max(0, args.terminal_lam_alef_divider_compensation) * count_lam_alef_clusters(left)
    divider_col = min(cols, half + 1 + compensation)
    right_col = min(cols + 1, divider_col + len(divider))
    right_width = max(0, cols - right_col + 1)
    left_width = max(0, divider_col - 1)

    sys.stdout.write(f"\x1b[{row};1H\x1b[2K")
    sys.stdout.write(f"\x1b[{row};1H{pad_visible(left, left_width)}")
    if divider_col <= cols:
        safe_divider = f"\u200e{divider}\u200e"
        sys.stdout.write(f"\x1b[{row};{divider_col}H{truncate_visible(safe_divider, max(0, cols - divider_col + 1))}")
    if right_width > 0:
        sys.stdout.write(f"\x1b[{row};{right_col}H{pad_visible(right, right_width)}")


def render_dashboard() -> None:
    if args.ui == "headless":
        if headless_transport is not None:
            headless_transport.publish(web_state_snapshot())
        return
    if not ui_initialized:
        return
    size = shutil.get_terminal_size((120, 20))
    cols = size.columns
    rows = max(12, size.lines)
    divider = " │ "
    half = max(20, (cols - len(divider)) // 2)
    body_lines = max(4, rows - 9)
    left_lines = panel_lines("left", half, body_lines)
    right_width = max(20, cols - half - len(divider))
    right_lines = panel_lines("right", right_width, body_lines)
    sys.stdout.write("\x1b[?1049h\x1b[H" if not getattr(render_dashboard, '_alt', False) else "\x1b[H")
    render_dashboard._alt = True

    row = 1
    _write_dashboard_row(row, "MIC / CALL AUDIO", divider, "KEYBOARD / TYPED", half, cols)
    for idx in range(body_lines):
        row += 1
        _write_dashboard_row(row, left_lines[idx], divider, right_lines[idx], half, cols)

    row += 1
    sys.stdout.write(f"\x1b[{row};1H\x1b[2K{'─' * min(cols, half * 2 + len(divider))}")

    row += 1
    input_line = scroll_visible(f"⌨️ {ui_state['input']}", right_width)
    _write_dashboard_row(row, "", divider, input_line, half, cols)

    for stage in ("capture", "asr", "translation"):
        row += 1
        value = ui_state.get("stages", {}).get(stage, "not started")
        sys.stdout.write(f"\x1b[{row};1H\x1b[2K{pad_visible(f'{stage.upper()}: {value}', cols)}")

    row += 1
    sys.stdout.write(f"\x1b[{row};1H\x1b[2K{pad_visible(ui_state['status'], cols)}")

    for clear_row in range(row + 1, rows + 1):
        sys.stdout.write(f"\x1b[{clear_row};1H\x1b[2K")
    sys.stdout.flush()


def init_dashboard(mode_text: list[str]) -> None:
    global ui_initialized
    with ui_lock:
        ui_initialized = True
        ui_state["status"] = " | ".join(mode_text)
        render_dashboard()


def ui_commit_if_ready(side: str, entry_id: int) -> None:
    active_key = f"{side}_active"
    order_key = f"{side}_order"
    done_key = f"{side}_done"
    entry = ui_state[active_key].get(entry_id)
    if not entry:
        return
    if entry.get("src_final") and entry.get("dst_final"):
        ui_state[done_key].append({"entry_id": entry_id, "src": entry.get("src", ""), "dst": entry.get("dst", "")})
        ui_state[active_key].pop(entry_id, None)
        ui_state[order_key] = [x for x in ui_state[order_key] if x != entry_id]


def make_panel_entry() -> dict[str, object]:
    return {
        "src": "",
        "dst": "",
        "src_final": False,
        "dst_final": False,
        "src_revision": None,
        "dst_revision": None,
    }


def ui_source_revision_matches(side: str, entry_id: int, revision: int | None) -> bool:
    if revision is None:
        return True
    active_key = f"{side}_active"
    entry = ui_state[active_key].get(entry_id)
    if not entry:
        return False
    return entry.get("src_revision") == revision


def ui_set_panel_entry(
    side: str,
    entry_id: int,
    field: str,
    text: str,
    final: bool,
    revision: int | None = None,
) -> bool:
    active_key = f"{side}_active"
    order_key = f"{side}_order"
    entry = ui_state[active_key].get(entry_id)

    if field == "dst" and revision is not None:
        if entry is None:
            return False
        if final and entry.get("src_revision") != revision:
            return False
        current_dst_revision = entry.get("dst_revision")
        if not final and isinstance(current_dst_revision, int) and revision < current_dst_revision:
            return False

    if entry is None:
        entry = ui_state[active_key].setdefault(entry_id, make_panel_entry())
        if entry_id not in ui_state[order_key]:
            ui_state[order_key].append(entry_id)
    elif entry_id not in ui_state[order_key]:
        ui_state[order_key].append(entry_id)

    if field == "src":
        source_text_changed = entry.get("src") != text
        if source_text_changed and revision is None:
            entry["dst"] = ""
            entry["dst_final"] = False
            entry["dst_revision"] = None
        if revision is not None:
            entry["src_revision"] = revision

    if field == "dst" and revision is not None:
        entry["dst_revision"] = revision

    entry[field] = text
    entry[f"{field}_final"] = final
    ui_commit_if_ready(side, entry_id)
    return True


def ui_remove_panel_entry(side: str, entry_id: int) -> None:
    """Remove an active panel entry without committing it to history.

    Used by the stuck-partial watchdog when the only available text is empty
    or filtered as hallucination. This keeps unfinished UI entries from
    lingering forever.
    """
    active_key = f"{side}_active"
    order_key = f"{side}_order"
    ui_state[active_key].pop(entry_id, None)
    ui_state[order_key] = [x for x in ui_state[order_key] if x != entry_id]


def ui_update(**kwargs) -> None:
    with ui_lock:
        ui_state.update({k: v for k, v in kwargs.items() if v is not None})
        render_dashboard()


def update_stage_status(stage: str, state: str, detail: str = "") -> None:
    """Persistent stage health, separate from transient progress messages."""
    with ui_lock:
        ui_state.setdefault("stages", {})[stage] = f"{state} | {detail}" if detail else state
        render_dashboard()


def log_verbose(message: str) -> None:
    if args.verbose or args.debug:
        eprint(f"ℹ️ {message}")


def log_debug(message: str) -> None:
    if args.debug:
        eprint(f"🐞 {message}")


def render_input(buffer: str, active: bool = True) -> None:
    ui_update(input=buffer if active else "")


def show_inflight_partial_transcript(utterance_id: int, revision: int | None = None) -> None:
    """
    Surface that audio is being decoded without making it look final.

    These entries are UI-only hints. They are automatically removed if the ASR
    lane produces no text/result, preventing permanent "in-flight" ghosts.
    """
    current_revision = revision if revision is not None else (current_partial_revision or None)
    inflight_partial_markers[utterance_id] = time.perf_counter()
    with ui_lock:
        ui_set_panel_entry(
            "left",
            utterance_id,
            "src",
            format_partial_transcript_for_ui(current_partial_display_text, current_partial_display_locked_prefix, inflight=True),
            final=False,
            revision=current_revision,
        )
        render_dashboard()


def clear_inflight_partial_marker(utterance_id: int) -> None:
    inflight_partial_markers.pop(utterance_id, None)


def sweep_inflight_partial_markers(active_utterances: set[int] | None = None) -> None:
    if not args.stuck_partial_watchdog:
        return
    timeout_s = max(0.3, args.inflight_partial_timeout_ms / 1000.0)
    now = time.perf_counter()
    active_utterances = active_utterances or set()
    expired: list[int] = []
    for utterance_id, started_at in list(inflight_partial_markers.items()):
        if utterance_id in active_utterances:
            continue
        if now - started_at >= timeout_s:
            expired.append(utterance_id)

    if not expired:
        return

    with ui_lock:
        changed = False
        for utterance_id in expired:
            inflight_partial_markers.pop(utterance_id, None)
            entry = ui_state["left_active"].get(utterance_id)
            src = str(entry.get("src", "")) if entry else ""
            if entry and "quality pending" in strip_ansi(src):
                ui_remove_panel_entry("left", utterance_id)
                changed = True
        if changed:
            render_dashboard()


APPLE_AUDIO_COOLDOWN_SECONDS = 3.0


class AppleVPIOBridge:
    def __init__(self, callback, preferred_modes: tuple[str, ...] = ("vpio", "raw")):
        self.callback = callback
        self.preferred_modes = preferred_modes
        self.proc: subprocess.Popen | None = None
        self.reader_thread: threading.Thread | None = None
        self.stderr_thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.chunk_count = 0
        self.mode = preferred_modes[0] if preferred_modes else "raw"
        self.ready_event = threading.Event()
        self.no_input_event = threading.Event()

    def _start_proc(self, mode: str) -> None:
        bridge_path = Path(args.apple_bridge_path).expanduser()
        env = os.environ.copy()
        if args.verbose or args.debug:
            env["APPLE_VPIO_VERBOSE"] = "1"
        if args.debug:
            env["APPLE_VPIO_DEBUG"] = "1"
        env["APPLE_VPIO_MODE"] = mode
        env["APPLE_VPIO_AGC"] = args.apple_vpio_agc
        env["APPLE_VPIO_BYPASS"] = args.apple_vpio_bypass_processing
        self.mode = mode
        self.ready_event.clear()
        self.no_input_event.clear()
        self.proc = subprocess.Popen(
            [str(bridge_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            env=env,
        )
        self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.reader_thread.start()
        self.stderr_thread = threading.Thread(target=self._stderr_loop, daemon=True)
        self.stderr_thread.start()

    def _stop_proc(self) -> None:
        prev_mode = self.mode
        if self.proc is not None:
            try:
                if self.proc.stdin:
                    self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.terminate()
                self.proc.wait(timeout=2)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        self.proc = None
        if prev_mode == "vpio":
            log_verbose(f"waiting {APPLE_AUDIO_COOLDOWN_SECONDS:.1f}s for apple voice-processing audio route to settle")
            time.sleep(APPLE_AUDIO_COOLDOWN_SECONDS)

    def __enter__(self):
        self.stop_event.clear()
        for idx, mode in enumerate(self.preferred_modes):
            self.chunk_count = 0
            self._start_proc(mode)
            self.ready_event.wait(timeout=3.0)
            self.no_input_event.wait(timeout=2.4)
            if not (self.no_input_event.is_set() and self.chunk_count == 0):
                return self
            if idx + 1 < len(self.preferred_modes):
                next_mode = self.preferred_modes[idx + 1]
                log_verbose(f"apple-{mode} startup stalled; retrying with {next_mode} apple audio")
                self._stop_proc()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop_event.set()
        self._stop_proc()

    def play_wav(self, path: str) -> None:
        if self.proc is None or self.proc.stdin is None:
            raise RuntimeError("apple audio bridge is not running")
        payload = Path(path).read_bytes()
        header = len(payload).to_bytes(4, "little", signed=False)
        self.proc.stdin.write(header)
        self.proc.stdin.write(payload)
        self.proc.stdin.flush()

    def _reader_loop(self) -> None:
        if self.proc is None or self.proc.stdout is None:
            return
        chunk_bytes = FRAME_SAMPLES * 2
        pending = bytearray()
        while not self.stop_event.is_set():
            try:
                data = self.proc.stdout.read(chunk_bytes)
            except Exception:
                break
            if not data:
                break
            pending.extend(data)
            while len(pending) >= chunk_bytes:
                chunk = bytes(pending[:chunk_bytes])
                del pending[:chunk_bytes]
                frame_i16 = np.frombuffer(chunk, dtype=np.int16)
                if frame_i16.size == 0:
                    continue
                self.chunk_count += 1
                if args.debug and (self.chunk_count <= 10 or self.chunk_count % 100 == 0):
                    peak = int(np.max(np.abs(frame_i16))) if frame_i16.size else 0
                    log_debug(f"apple-bridge mic chunk#{self.chunk_count} samples={frame_i16.size} peak={peak}")
                self.callback(frame_i16.reshape(-1, 1), int(frame_i16.size), None, None)

    def _stderr_loop(self) -> None:
        if self.proc is None or self.proc.stderr is None:
            return
        for raw in self.proc.stderr:
            if self.stop_event.is_set():
                break
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            if line.startswith("apple-vpio-bridge ready"):
                self.ready_event.set()
                if "mode=" in line:
                    mode = line.split("mode=", 1)[1].strip()
                    log_verbose(f"apple bridge ready ({mode})")
                else:
                    log_verbose("apple bridge ready")
                continue
            if line == "no input frames received within 2s":
                self.no_input_event.set()
                eprint(f"⚠️ {line}")
                continue
            if line.startswith("tap frames=") or line.startswith("input.") or line.startswith("output.") or line.startswith("playback bytes=") or line.startswith("input channel[") or line.startswith("playback finished") or line.startswith("locked input channel=") or line.startswith("input channel layout changed") or line.startswith("reset locked input channel"):
                log_debug(line)
                continue
            eprint(f"⚠️ {line}")


def queue_put_drop_oldest(q: Queue, item, label: str) -> None:
    while True:
        try:
            q.put_nowait(item)
            return
        except Full:
            try:
                dropped = q.get_nowait()
                if isinstance(dropped, AudioTask):
                    cleanup_audio_task_payload(dropped)
                elif isinstance(dropped, AsrResult):
                    cleanup_audio_result_payload(dropped)
                q.task_done()
                eprint(f"\n⏩ Dropped stale {label} task to stay responsive.")
            except Empty:
                time.sleep(0.001)


def release_shared_audio(name: str | None) -> None:
    if not name:
        return
    try:
        shm = shared_memory.SharedMemory(name=name)
    except FileNotFoundError:
        return
    try:
        shm.close()
    finally:
        try:
            shm.unlink()
        except FileNotFoundError:
            pass


def release_audio_slot(slot_id: int | None) -> None:
    if slot_id is None:
        return
    try:
        asr_slot_free.put_nowait(slot_id)
    except Full:
        pass


def cleanup_audio_task_payload(task: AudioTask | None) -> None:
    if task is None:
        return
    slot_id = getattr(task, "slot_id", None)
    if slot_id is not None:
        release_audio_slot(slot_id)
        return
    release_shared_audio(getattr(task, "shm_name", None))


def cleanup_audio_result_payload(result: AsrResult | None) -> None:
    if result is None:
        return
    if result.slot_id is not None:
        release_audio_slot(result.slot_id)
        return
    release_shared_audio(result.shm_name)


def drop_one_partial_asr_result(q: Queue, label: str) -> bool:
    dropped = None
    with q.mutex:
        for index, item in enumerate(q.queue):
            if isinstance(item, AsrResult) and item.partial:
                dropped = item
                del q.queue[index]
                q.unfinished_tasks -= 1
                if q.unfinished_tasks == 0:
                    q.all_tasks_done.notify_all()
                q.not_full.notify()
                break
    if dropped is not None:
        cleanup_audio_result_payload(dropped)
        return True
    return False


def queue_put_asr_result(q: Queue, result: AsrResult, label: str) -> bool:
    while not shutdown_event.is_set():
        try:
            q.put_nowait(result)
            return True
        except Full:
            if drop_one_partial_asr_result(q, label):
                continue
            if result.partial:
                cleanup_audio_result_payload(result)
                if args.debug:
                    log_debug(f"dropped incoming partial {label} result because final ASR backlog is full")
                return False
            try:
                q.put(result, timeout=0.05)
                return True
            except Full:
                if args.debug:
                    log_debug(f"waiting for final {label} result queue capacity")
                continue
    cleanup_audio_result_payload(result)
    return False


def queue_put_drop_oldest_ipc(q, item, label: str) -> None:
    while True:
        try:
            q.put_nowait(item)
            return
        except Full:
            try:
                dropped = q.get_nowait()
                if isinstance(dropped, AudioTask):
                    cleanup_audio_task_payload(dropped)
                elif isinstance(dropped, AsrResult):
                    cleanup_audio_result_payload(dropped)
                eprint(f"\n⏩ Dropped stale {label} task to stay responsive.")
            except Empty:
                time.sleep(0.001)


def queue_put_latest_silent(q: Queue, item) -> None:
    while True:
        try:
            q.put_nowait(item)
            return
        except Full:
            try:
                q.get_nowait()
                q.task_done()
            except Empty:
                return


def queue_put_audio_task(q, task: AudioTask, label: str, *, has_task_done: bool) -> bool:
    while True:
        try:
            q.put_nowait(task)
            return True
        except Full:
            try:
                dropped = q.get_nowait()
            except Empty:
                time.sleep(0.001)
                continue

            if isinstance(dropped, AudioTask) and task.partial and not dropped.partial:
                try:
                    q.put_nowait(dropped)
                except Full:
                    cleanup_audio_task_payload(dropped)
                    if has_task_done:
                        q.task_done()
                    continue
                cleanup_audio_task_payload(task)
                if args.debug:
                    log_debug(f"dropped speculative {label} partial behind queued final")
                return False

            if isinstance(dropped, AudioTask):
                cleanup_audio_task_payload(dropped)
            if has_task_done:
                q.task_done()
            if args.debug:
                log_debug(f"dropped stale {label} task to stay responsive")


# ======================
# Model loading
# ======================
tokenizer: AutoTokenizer | None = None
translator: AutoModelForSeq2SeqLM | None = None
use_mps = False
ID_ENG: int | None = None
ID_TUR: int | None = None
ID_BY_LANG_CODE: dict[str, int] = {}
stt: WhisperModel | None = None
mlx_translation_model = None
mlx_translation_tokenizer = None
kokoro_tts = None
detected_stt_backend = "uninitialized"
detected_stt_device = "uninitialized"
detected_translation_backend = "uninitialized"
translation_lock = threading.Lock()
mlx_vlm_lock = threading.Lock()


session_log_lock = threading.Lock()
audio_final_transcripts: dict[int, str] = {}
typed_input_texts: dict[int, str] = {}
logged_audio_translation_keys: set[tuple[int, int]] = set()
logged_audio_translation_utterances: set[int] = set()
pending_audio_translation_logs: dict[int, tuple[int, str, str]] = {}
logged_typed_translation_keys: set[tuple[int, int]] = set()
final_asr_pending_lock = threading.Lock()
final_asr_pending_utterances: set[int] = set()


def mark_final_asr_pending(utterance_id: int | None) -> None:
    if utterance_id is None:
        return
    with final_asr_pending_lock:
        final_asr_pending_utterances.add(int(utterance_id))


def clear_final_asr_pending(utterance_id: int | None) -> None:
    if utterance_id is None:
        return
    with final_asr_pending_lock:
        final_asr_pending_utterances.discard(int(utterance_id))


def is_final_asr_pending(utterance_id: int | None) -> bool:
    if utterance_id is None:
        return False
    with final_asr_pending_lock:
        return int(utterance_id) in final_asr_pending_utterances



def is_english_only_whisper_model_name(model_name: str) -> bool:
    """Return True when a Whisper model reference is explicitly English-only."""
    value = str(model_name or "").strip().lower()
    if not value:
        return False

    normalized = value.replace("\\", "/")
    stem = normalized.rsplit("/", 1)[-1]
    stem = re.sub(r"^ggml-", "", stem)
    stem = re.sub(r"\.(bin|pt|safetensors|mlmodelc?)$", "", stem)
    stem = stem.replace("mlx-community/whisper-", "")
    stem = stem.replace("openai/whisper-", "")
    stem = stem.replace("whisper-", "")

    english_only_suffixes = (".en", "-en", "_en")
    if stem.endswith(english_only_suffixes):
        return True

    english_only_names = {
        "tiny.en",
        "base.en",
        "small.en",
        "medium.en",
        "ggml-tiny.en",
        "ggml-base.en",
        "ggml-small.en",
        "ggml-medium.en",
    }
    return stem in english_only_names or normalized.endswith(tuple(f"/{name}" for name in english_only_names))

def resolve_whisper_cli_model_path() -> str:
    model = args.whisper_model
    candidate = Path(model).expanduser()
    if candidate.exists():
        return str(candidate)

    names = [f"ggml-{model}.bin"]
    if source_language_code() == "EN" or args.allow_english_asr_model:
        names.append(f"ggml-{model}.en.bin")
    search_dirs = [
        Path.home() / ".cache" / "whisper",
        Path.home() / ".cache" / "whisper.cpp",
        Path.home() / ".local" / "share" / "whisper",
    ]
    for directory in search_dirs:
        for name in names:
            path = directory / name
            if path.exists():
                return str(path)

    raise RuntimeError(
        f"whisper-cli backend requested, but no GGML model file was found for '{model}'. "
        "Pass --whisper-model /path/to/ggml-model.bin or place a ggml model in ~/.cache/whisper."
    )


def resolve_stt_backend() -> str:
    if args.stt_backend != "auto":
        return args.stt_backend
    if is_apple_silicon():
        if MLX_WHISPER_INSTALLED or have("mlx_whisper") or have("mlx-whisper"):
            return "mlx-whisper"
        if is_macos() and have("dictate-stt"):
            return "dictate-stt"
        if is_macos() and have("whisper-cli"):
            return "whisper-cli"
        return "faster-whisper"
    if is_macos() and have("whisper-cli"):
        return "whisper-cli"
    if is_macos() and have("dictate-stt"):
        return "dictate-stt"
    return "faster-whisper"


def resolve_stt_device(backend: str) -> str:
    if args.stt_device != "auto":
        return args.stt_device
    if backend in {"mlx-vlm-gemma", "mlx-whisper"} and is_apple_silicon():
        return "metal"
    if backend == "dictate-stt" and is_macos():
        return "ane"
    if backend == "whisper-cli" and is_macos():
        return "metal"
    return "cpu"


def extract_whisper_cli_text(raw_output: str) -> str:
    ignored_prefixes = (
        "load_backend:",
        "system_info:",
        "main: processing",
        "main: done",
        "whisper_",
        "ggml_",
        "samplerate:",
        "channels:",
        "length:",
        "duration:",
    )
    lines = []
    for line in raw_output.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(ignored_prefixes):
            continue
        lines.append(stripped)
    return normalize_text(" ".join(lines))


def resolve_mlx_whisper_model_ref() -> str:
    return resolve_mlx_whisper_model_ref_for(args.whisper_model)


def resolve_mlx_whisper_model_ref_for(model_name: str) -> str:
    return resolve_mlx_whisper_reference(model_name or args.whisper_model,
                                         Path(__file__).resolve().parent / "models")


def prepare_mlx_whisper_runtime_reference(model_name: str) -> str:
    model_ref = resolve_mlx_whisper_model_ref_for(model_name)
    root = Path(__file__).resolve().parent
    if model_ref == LARGE_V3_REPOSITORY or Path(model_ref).name in {"whisper-large-v3", "whisper-large-v3-mlx"}:
        command = [sys.executable, str(root / "scripts/prepare_models.py"), "--download", "--model", "whisper"]
    else:
        command = ["env", "HF_HUB_OFFLINE=0", sys.executable, "-m", "huggingface_hub.cli.hf", "download", model_ref]
    offline = os.environ.get("HF_HUB_OFFLINE", "1").strip().lower() not in {"0", "false", "no"}
    return require_cached_mlx_whisper(model_ref, offline=offline, preparation_command=shlex.join(command))


def effective_lane_backend(partial: bool) -> str:
    lane = args.partial_stt_backend if partial else args.final_stt_backend
    if lane == "same":
        return detected_stt_backend if detected_stt_backend != "uninitialized" else resolve_stt_backend()
    if lane == "auto":
        return resolve_stt_backend()
    return lane


def effective_lane_model(partial: bool) -> str:
    lane_model = args.partial_whisper_model if partial else args.final_whisper_model
    return lane_model.strip() or args.whisper_model


def validate_asr_model_language(model_name: str, backend: str) -> None:
    if backend == "mlx-vlm-gemma":
        return
    if source_language_code() == "EN" or args.allow_english_asr_model:
        return
    if is_english_only_whisper_model_name(model_name):
        raise RuntimeError(
            f"{backend} model {model_name!r} is English-only, but --source-language is {source_language_code()}. "
            "Use a multilingual Whisper model such as large-v3, medium, small, or pass --allow-english-asr-model knowingly."
        )


def effective_lane_beam(partial: bool) -> int:
    if partial:
        return max(1, args.partial_asr_beam)
    return max(1, args.final_asr_beam or args.asr_beam)


def temporarily_set_attr(obj, name: str, value):
    class _Ctx:
        def __enter__(self_inner):
            self_inner.old = getattr(obj, name)
            setattr(obj, name, value)
        def __exit__(self_inner, exc_type, exc, tb):
            setattr(obj, name, self_inner.old)
    return _Ctx()


def default_whisper_model_for_backend(backend: str) -> str:
    if backend == "mlx-whisper":
        return resolve_mlx_whisper_model_ref_for(args.whisper_model)
    if USER_SET_WHISPER_MODEL:
        return prefer_project_model(args.whisper_model)
    if args.profile == "accurate":
        return "large-v3"
    return args.whisper_model


def resolve_translation_backend() -> str:
    if args.translation_backend != "auto":
        return args.translation_backend
    if is_apple_silicon() and MLX_LM_INSTALLED:
        return "mlx-lm"
    return "nllb"


def default_translation_model_for_backend(backend: str) -> str:
    if USER_SET_TRANSLATION_MODEL:
        return prefer_project_model(args.translation_model)
    if backend == "mlx-vlm-gemma":
        return prefer_project_model(args.mlx_vlm_model or DEFAULT_MLX_VLM_AUDIO_MODEL)
    if backend == "mlx-lm":
        return prefer_project_model(DEFAULT_MLX_TRANSLATION_MODEL)
    return DEFAULT_NLLB_TRANSLATION_MODEL


def active_translation_backend() -> str:
    if detected_translation_backend != "uninitialized":
        return detected_translation_backend
    return resolve_translation_backend()


@lru_cache(maxsize=1)
def available_say_voice_records() -> list[tuple[str, str]]:
    if not is_macos() or not have("say"):
        return []
    try:
        proc = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, check=True)
    except Exception:
        return []

    voices: list[tuple[str, str]] = []
    for line in proc.stdout.splitlines():
        # Voice names can contain spaces and parentheses. The locale is the
        # final field before the sample phrase, not the second word in a name.
        match = re.match(r"^\s*(.+?)\s+([a-z]{2,3}[_-][A-Za-z0-9_-]+)\s+#", line)
        if match:
            voices.append((match.group(1).strip(), match.group(2).replace("-", "_")))
    return voices


def available_say_voices() -> list[str]:
    return [name for name, _locale in available_say_voice_records()]


def pick_say_voice(lang_hint: str = "") -> str:
    code = active_tts_lang_code(lang_hint or source_language_code())
    spec = language_spec(code)
    records = available_say_voice_records()
    voices = [name for name, _locale in records]
    requested = args.say_voice.strip()
    if requested:
        if records and requested not in voices:
            raise RuntimeError(f"Requested macOS voice {requested!r} is not installed")
        return requested
    wanted_locale = spec.apple_locale.replace("-", "_").lower()
    matching = [(name, locale) for name, locale in records
                if locale.split("_", 1)[0].lower() == code.lower()]
    if spec.say_voice and any(name == spec.say_voice for name, _locale in matching):
        return spec.say_voice
    for name, locale in matching:
        if locale.lower() == wanted_locale:
            return name
    if matching:
        return matching[0][0]
    raise RuntimeError(f"No installed macOS voice matches {spec.name} ({code}); install a matching voice or choose another TTS engine")


def resolve_stt_command(preferred: str, alternates: tuple[str, ...] = ()) -> str:
    for name in (preferred, *alternates):
        path = find_command(name)
        if path:
            return path
    raise RuntimeError(f"Required command not found: {preferred}")


def save_prepared_audio_wav(prepared_audio: np.ndarray, tmpdir_prefix: str) -> tuple[TemporaryDirectory, str]:
    tmpdir = TemporaryDirectory(prefix=tmpdir_prefix)
    wav_path = os.path.join(tmpdir.name, "chunk.wav")
    clipped = np.clip(prepared_audio, -1.0, 1.0)
    save_wav(wav_path, (clipped * 32767.0).astype(np.int16))
    return tmpdir, wav_path


def effective_mlx_vlm_model(partial: bool) -> str:
    lane_model = args.partial_mlx_vlm_model if partial else args.final_mlx_vlm_model
    return (lane_model or args.mlx_vlm_model or DEFAULT_MLX_VLM_AUDIO_MODEL).strip()


def default_mlx_vlm_prompt(partial: bool) -> str:
    task_hint = (
        "This is an incomplete live audio window, so transcribe only words clearly present in this audio and do not complete the sentence."
        if partial
        else "This is the complete utterance."
    )
    return (
        "Listen to the audio. {task_hint} The speech language is {source_language}. "
        "Output only the exact words spoken in {source_language}. Preserve names, clinic names, dates, times, and appointment-message wording. "
        "Return the final transcript only. Do not translate. Do not summarize. Do not explain. Do not repeat phrases. If speech is unclear, output exactly EMPTY."
    ).format(source_language=source_language().name, task_hint=task_hint)


def render_mlx_vlm_prompt(partial: bool) -> str:
    template = args.mlx_vlm_partial_prompt if partial else args.mlx_vlm_final_prompt
    template = template or args.mlx_vlm_prompt
    if not template:
        return default_mlx_vlm_prompt(partial)
    task_hint = "incomplete live audio window" if partial else "complete utterance"
    values = {
        "source_language": source_language().name,
        "target_language": target_language().name,
        "source_lang": source_language_code(),
        "target_lang": target_language_code(),
        "task_hint": task_hint,
    }
    try:
        return template.format(**values)
    except KeyError as exc:
        raise RuntimeError(f"Unknown placeholder in --mlx-vlm-prompt: {exc}") from exc


def strip_mlx_vlm_generation(text: str) -> str:
    text = strip_model_thinking(text)
    if not text:
        return ""
    text = re.sub(r"(?is)^```(?:text|txt)?\s*|\s*```$", " ", text)
    text = re.sub(r"(?i)^(transcript|transcription|output|answer)\s*:\s*", "", text.strip())
    return strip_model_thinking(text)


def prompt_echo_words(text: str) -> list[str]:
    return [word.casefold() for word in _WORD_RE.findall(normalize_text(text))]


def is_prompt_echo_text(text: str, prompt: str) -> bool:
    text = normalize_text(text)
    prompt = normalize_text(prompt)
    if not text or not prompt:
        return False

    text_fold = text.casefold()
    prompt_fold = prompt.casefold()
    if text_fold in {"empty", "no clear speech", "unclear", "inaudible"}:
        return True
    if text_fold == prompt_fold or prompt_fold.startswith(text_fold) or text_fold.startswith(prompt_fold):
        return True

    text_words = prompt_echo_words(text)
    prompt_words = prompt_echo_words(prompt)
    if len(text_words) < 4 or not prompt_words:
        return False
    prompt_word_set = set(prompt_words)
    contained = sum(1 for word in text_words if word in prompt_word_set)
    if contained / max(1, len(text_words)) >= 0.80:
        return True

    prefix_len = 0
    for left, right in zip(text_words, prompt_words):
        if left != right:
            break
        prefix_len += 1
    return prefix_len >= min(len(text_words), 8)


def clean_mlx_vlm_transcript_output(text: str, prompt: str, *, partial: bool) -> str:
    text = strip_mlx_vlm_generation(text)
    if not text:
        return ""
    references = [prompt, default_mlx_vlm_prompt(partial), args.asr_initial_prompt]
    if any(is_prompt_echo_text(text, reference) for reference in references if reference):
        if args.debug or args.debug_dropped_asr:
            log_debug(f"dropped MLX-VLM prompt echo: {truncate_visible(text, 96)}")
        return ""
    return text


def mlx_vlm_result_text(result) -> str:
    if isinstance(result, str):
        return strip_mlx_vlm_generation(result)
    if isinstance(result, dict):
        for key in ("text", "output", "response"):
            if key in result:
                return strip_mlx_vlm_generation(str(result[key]))
    for attr in ("text", "output", "response"):
        if hasattr(result, attr):
            return strip_mlx_vlm_generation(str(getattr(result, attr)))
    return strip_mlx_vlm_generation(str(result))


def init_mlx_vlm_gemma_runtime(model_name: str | None = None) -> None:
    global mlx_vlm_model, mlx_vlm_processor, mlx_vlm_model_ref
    model_ref = (model_name or args.mlx_vlm_model or DEFAULT_MLX_VLM_AUDIO_MODEL).strip()
    if (
        getattr(mlx_vlm_thread_state, "model", None) is not None
        and getattr(mlx_vlm_thread_state, "processor", None) is not None
        and getattr(mlx_vlm_thread_state, "model_ref", "") == model_ref
    ):
        return
    ensure_mlx_vlm_loaded()
    if mlx_vlm_load is None:
        raise RuntimeError("mlx-vlm loader is unavailable")
    with mlx_vlm_lock:
        with mlx_thread_stream_context():
            with silent_external_output():
                model, processor = mlx_vlm_load(model_ref)
    mlx_vlm_thread_state.model = model
    mlx_vlm_thread_state.processor = processor
    mlx_vlm_thread_state.model_ref = model_ref
    # Keep the legacy globals populated for diagnostics and helper fallbacks.
    mlx_vlm_model = model
    mlx_vlm_processor = processor
    mlx_vlm_model_ref = model_ref


def apply_mlx_vlm_audio_template(prompt: str, num_audios: int, model=None, processor=None) -> str:
    if mlx_vlm_apply_chat_template is None:
        raise RuntimeError("mlx-vlm chat template helper is unavailable")
    active_model = model if model is not None else mlx_vlm_model
    active_processor = processor if processor is not None else mlx_vlm_processor
    config = getattr(active_model, "config", None)
    try:
        return mlx_vlm_apply_chat_template(
            active_processor,
            config,
            prompt,
            num_audios=num_audios,
            enable_thinking=False,
        )
    except TypeError:
        try:
            return mlx_vlm_apply_chat_template(active_processor, config, prompt, num_audios=num_audios)
        except TypeError:
            try:
                return mlx_vlm_apply_chat_template(
                    active_processor,
                    prompt,
                    num_audios=num_audios,
                    enable_thinking=False,
                )
            except TypeError:
                return mlx_vlm_apply_chat_template(active_processor, prompt, num_audios=num_audios)
    except ValueError:
        try:
            return mlx_vlm_apply_chat_template(
                active_processor, config, prompt, num_audios=num_audios
            )
        except TypeError:
            return mlx_vlm_apply_chat_template(active_processor, prompt, num_audios=num_audios)


def mlx_vlm_generation_kwargs(max_tokens: int | None = None, *, suppress_thinking: bool = True) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "max_tokens": max(1, int(max_tokens or args.mlx_vlm_max_tokens)),
        "temperature": float(args.mlx_vlm_temperature),
        "verbose": False,
        "skip_special_tokens": True,
    }
    if suppress_thinking:
        kwargs.update(
            {
                "enable_thinking": False,
                "thinking_budget": 0,
                "thinking_start_token": "<|channel>thought",
                "thinking_end_token": "<channel|>",
            }
        )
    return kwargs


def generate_mlx_vlm_safely(model, processor, formatted_prompt: str, *, audio=None, max_tokens: int | None = None):
    attempts = [
        mlx_vlm_generation_kwargs(max_tokens, suppress_thinking=True),
        mlx_vlm_generation_kwargs(max_tokens, suppress_thinking=False),
    ]
    last_exc: Exception | None = None
    for generate_kwargs in attempts:
        try:
            if audio is None:
                return mlx_vlm_generate(model, processor, formatted_prompt, **generate_kwargs)
            return mlx_vlm_generate(model, processor, formatted_prompt, audio=audio, **generate_kwargs)
        except TypeError as exc:
            last_exc = exc
            if audio is None:
                try:
                    return mlx_vlm_generate(model, processor, formatted_prompt, audio=[], **generate_kwargs)
                except TypeError as retry_exc:
                    last_exc = retry_exc
            continue
        except (IndexError, ValueError) as exc:
            # Older tokenizers may not know a thinking marker. Retry without the
            # budget knobs, then rely on the sanitizer as the hard output gate.
            last_exc = exc
            continue
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("mlx-vlm generator is unavailable")


def transcribe_audio_mlx_vlm_gemma(prepared_audio: np.ndarray, *, partial: bool = False) -> str:
    model_ref = effective_mlx_vlm_model(partial)
    init_mlx_vlm_gemma_runtime(model_ref)
    if mlx_vlm_generate is None:
        raise RuntimeError("mlx-vlm generator is unavailable")
    model = getattr(mlx_vlm_thread_state, "model", mlx_vlm_model)
    processor = getattr(mlx_vlm_thread_state, "processor", mlx_vlm_processor)
    if model is None or processor is None:
        raise RuntimeError("mlx-vlm Gemma model is not loaded for this ASR thread")
    tmpdir, wav_path = save_prepared_audio_wav(prepared_audio, "tts-transcribe-mlx-vlm-")
    with tmpdir:
        prompt = render_mlx_vlm_prompt(partial)
        formatted_prompt = apply_mlx_vlm_audio_template(prompt, 1, model=model, processor=processor)
        with mlx_vlm_lock:
            with mlx_thread_stream_context():
                with silent_external_output():
                    result = generate_mlx_vlm_safely(
                        model,
                        processor,
                        formatted_prompt,
                        audio=[wav_path],
                        max_tokens=max(1, int(args.mlx_vlm_max_tokens)),
                    )
    text = clean_mlx_vlm_transcript_output(mlx_vlm_result_text(result), prompt, partial=partial)
    last_decoder_quality.clear()
    last_decoder_quality.update({"raw_text": mlx_vlm_result_text(result), "metrics_available": False})
    return text if looks_like_real_text(text) else ""


def is_mlx_whisper_beam_unavailable(exc: Exception) -> bool:
    message = str(exc).casefold()
    return "beam search" in message and "not yet implemented" in message


def transcribe_audio_mlx_whisper(prepared_audio: np.ndarray, model_name: str | None = None, beam: int = 1) -> str:
    global mlx_whisper_beam_available, mlx_whisper_beam_fallback_reported
    model_ref = resolve_mlx_whisper_model_ref_for(model_name or args.whisper_model)
    mlx_module = None
    if MLX_WHISPER_INSTALLED:
        try:
            mlx_module = ensure_mlx_whisper_loaded()
        except Exception as exc:
            log_verbose(f"mlx-whisper Python backend unavailable; trying CLI fallback ({exc})")
    if mlx_module is not None:
        requested_beam = max(1, int(beam))
        decode_kwargs = {
            "path_or_hf_repo": model_ref,
            **asr_language_kwargs(),
            "task": "transcribe",
            "condition_on_previous_text": False,
            **asr_initial_prompt_kwargs(),
            "word_timestamps": False,
            "verbose": False,
            "temperature": 0.0,
            "compression_ratio_threshold": anti_repeat_thresholds()[2],
            "logprob_threshold": anti_repeat_thresholds()[1],
            "no_speech_threshold": anti_repeat_thresholds()[0],
        }
        if requested_beam > 1 and mlx_whisper_beam_available is not False:
            decode_kwargs["beam_size"] = requested_beam
        try:
            with silent_external_output():
                result = mlx_module.transcribe(prepared_audio, **decode_kwargs)
            if requested_beam > 1 and "beam_size" in decode_kwargs:
                mlx_whisper_beam_available = True
        except Exception as exc:
            if requested_beam <= 1 or not is_mlx_whisper_beam_unavailable(exc):
                raise
            mlx_whisper_beam_available = False
            if not mlx_whisper_beam_fallback_reported:
                log_verbose("mlx-whisper beam search is unavailable in this build; using cached greedy fallback")
                mlx_whisper_beam_fallback_reported = True
            decode_kwargs.pop("beam_size", None)
            with silent_external_output():
                result = mlx_module.transcribe(prepared_audio, **decode_kwargs)
        text = normalize_text(str(result.get("text", "")))
        last_decoder_quality.clear()
        last_decoder_quality.update(summarize_whisper_result(result))
        last_decoder_quality["requested_beam"] = requested_beam
        last_decoder_quality["effective_beam"] = requested_beam if "beam_size" in decode_kwargs else 1
        last_decoder_quality["beam_search_available"] = mlx_whisper_beam_available
        return text if looks_like_real_text(text) else ""

    cmd = resolve_stt_command("mlx_whisper", ("mlx-whisper",))
    tmpdir, wav_path = save_prepared_audio_wav(prepared_audio, "tts-transcribe-mlx-whisper-")
    with tmpdir:
        out_name = os.path.join(tmpdir.name, "result")
        proc = subprocess.run(
            [
                cmd,
                wav_path,
                "--model",
                model_ref,
                "--output-format",
                "txt",
                "--output-name",
                out_name,
                *asr_language_cli_args("--language"),
                "--task",
                "transcribe",
                *asr_initial_prompt_cli_args("--initial-prompt"),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        txt_path = f"{out_name}.txt"
        text = Path(txt_path).read_text(encoding="utf-8") if os.path.exists(txt_path) else proc.stdout
        text = normalize_text(text)
        return text if looks_like_real_text(text) else ""


def transcribe_audio_dictate_stt(prepared_audio: np.ndarray) -> str:
    cmd = resolve_stt_command("dictate-stt")
    tmpdir, wav_path = save_prepared_audio_wav(prepared_audio, "tts-transcribe-dictate-stt-")
    with tmpdir:
        proc = subprocess.run(
            [cmd, wav_path],
            capture_output=True,
            text=True,
            check=True,
        )
        raw = normalize_text(proc.stdout)
        if raw.startswith("{"):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    text = normalize_text(str(parsed.get("text", "")))
                    return text if looks_like_real_text(text) else ""
            except json.JSONDecodeError:
                pass
        return raw if looks_like_real_text(raw) else ""


def transcribe_audio_whisper_cli(prepared_audio: np.ndarray, model_path_override: str | None = None, beam_override: int | None = None) -> str:
    if not have("whisper-cli"):
        raise RuntimeError("whisper-cli is not installed")

    model_path = model_path_override or resolve_whisper_cli_model_path()
    with TemporaryDirectory(prefix="tts-transcribe-whisper-cli-") as tmpdir:
        wav_path = os.path.join(tmpdir, "chunk.wav")
        clipped = np.clip(prepared_audio, -1.0, 1.0)
        save_wav(wav_path, (clipped * 32767.0).astype(np.int16))

        cmd = [
            "whisper-cli",
            "-m",
            model_path,
            *asr_language_cli_args("-l"),
            "-f",
            wav_path,
            "-t",
            str(max(4, os.cpu_count() or 4)),
            "-bs",
            str(max(1, beam_override if beam_override is not None else args.asr_beam)),
            "-bo",
            "1",
            "-np",
            "-nt",
            *asr_initial_prompt_cli_args("--prompt"),
        ]
        if resolve_stt_device("whisper-cli") == "cpu":
            cmd.append("-ng")

        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        text = extract_whisper_cli_text("\n".join(part for part in [proc.stdout, proc.stderr] if part))
        return text if looks_like_real_text(text) else ""




def _shell_split_extra(extra: str) -> list[str]:
    if not extra.strip():
        return []
    try:
        return shlex.split(extra)
    except ValueError as exc:
        raise RuntimeError(f"Invalid --whispercpp-extra-args: {exc}") from exc


def transcribe_audio_whispercpp_coreml(prepared_audio: np.ndarray, *, beam: int) -> str:
    if not args.whispercpp_model:
        raise RuntimeError("--final-stt-backend whispercpp-coreml requires --whispercpp-model /path/to/ggml-model.bin")
    cmd_path = resolve_stt_command(args.whispercpp_command)
    model_path = str(Path(args.whispercpp_model).expanduser())
    if not Path(model_path).exists():
        raise RuntimeError(f"whisper.cpp model not found: {model_path}")
    tmpdir, wav_path = save_prepared_audio_wav(prepared_audio, "tts-transcribe-whispercpp-coreml-")
    with tmpdir:
        cmd = [
            cmd_path,
            "-m", model_path,
            *asr_language_cli_args("-l"),
            "-f", wav_path,
            "-t", str(max(4, os.cpu_count() or 4)),
            "-bs", str(max(1, beam)),
            "-bo", "1",
            "-np",
            "-nt",
            *asr_initial_prompt_cli_args("--prompt"),
            *_shell_split_extra(args.whispercpp_extra_args),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        text = extract_whisper_cli_text("\n".join(part for part in [proc.stdout, proc.stderr] if part))
        return text if looks_like_real_text(text) else ""


def transcribe_audio_apple_speech(prepared_audio: np.ndarray) -> str:
    if not args.apple_speech_command:
        raise RuntimeError("--partial-stt-backend apple-speech requires --apple-speech-command /path/to/apple_speech_transcriber")
    cmd_path = resolve_stt_command(args.apple_speech_command)
    tmpdir, wav_path = save_prepared_audio_wav(prepared_audio, "tts-transcribe-apple-speech-")
    with tmpdir:
        proc = subprocess.run([cmd_path, wav_path, source_language().apple_locale], capture_output=True, text=True, check=True)
        raw = normalize_text(proc.stdout)
        if raw.startswith("{"):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    raw = normalize_text(str(parsed.get("text", "")))
            except json.JSONDecodeError:
                pass
        return raw if looks_like_real_text(raw) else ""

def set_placeholder_language_ids() -> None:
    global ID_ENG, ID_TUR, ID_BY_LANG_CODE
    ordered_codes = list(dict.fromkeys([target_language_code(), source_language_code(), *LANGUAGE_SPECS.keys()]))
    ID_BY_LANG_CODE = {code: index + 1 for index, code in enumerate(ordered_codes)}
    ID_ENG = ID_BY_LANG_CODE.get("EN", 1)
    ID_TUR = ID_BY_LANG_CODE.get("TR", 2)


def init_parent_routing_runtime() -> None:
    global detected_translation_backend
    detected_translation_backend = resolve_translation_backend()
    args.translation_model = default_translation_model_for_backend(detected_translation_backend)
    # The translation process resolves real tokenizer IDs. The parent only needs stable direction markers.
    set_placeholder_language_ids()


def init_translation_runtime_only() -> None:
    global tokenizer, translator, use_mps, ID_ENG, ID_TUR, ID_BY_LANG_CODE
    global mlx_translation_model, mlx_translation_tokenizer, detected_translation_backend, mlx_translation_stop_ids

    detected_translation_backend = resolve_translation_backend()
    args.translation_model = default_translation_model_for_backend(detected_translation_backend)
    translation_ready = False
    if detected_translation_backend == "mlx-lm":
        translation_ready = mlx_translation_model is not None and mlx_translation_tokenizer is not None
    elif detected_translation_backend == "mlx-vlm-gemma":
        translation_ready = getattr(mlx_vlm_thread_state, "model", None) is not None
    else:
        translation_ready = tokenizer is not None and translator is not None

    if translation_ready:
        return

    eprint(f"🔥 Loading translation model ({detected_translation_backend}): {args.translation_model}")
    if detected_translation_backend == "mlx-vlm-gemma":
        try:
            args.translation_model = default_translation_model_for_backend(detected_translation_backend)
            init_mlx_vlm_gemma_runtime(args.translation_model)
            set_placeholder_language_ids()
        except Exception as exc:
            raise RuntimeError(f"Gemma translation failed for {args.translation_model}: {exc}") from exc

    if detected_translation_backend == "mlx-lm":
        try:
            ensure_mlx_lm_loaded()
            mlx_translation_model, mlx_translation_tokenizer = mlx_lm_load(args.translation_model)
            mlx_translation_stop_ids = configure_translation_stop_tokens(mlx_translation_tokenizer, args.translation_model)
            set_placeholder_language_ids()
        except Exception as exc:
            raise RuntimeError(f"MLX translation failed for {args.translation_model}: {exc}") from exc

    if detected_translation_backend == "nllb":
        if torch is None or AutoTokenizer is None or AutoModelForSeq2SeqLM is None:
            raise RuntimeError("Missing translation dependencies. Install torch and transformers for this script's Python environment.")
        tokenizer = load_pretrained_local_first(AutoTokenizer, args.translation_model)
        translator = load_pretrained_local_first(AutoModelForSeq2SeqLM, args.translation_model)
        translator.eval()
        if getattr(translator, "generation_config", None) is not None and getattr(translator.generation_config, "max_length", None) is not None:
            translator.generation_config.max_length = None
        use_mps = torch_mps_available()
        if use_mps:
            try:
                translator = translator.to(device="mps", dtype=torch.float16)
            except Exception:
                translator = translator.to("mps")

        ID_BY_LANG_CODE = {
            code: int(tokenizer.convert_tokens_to_ids(spec.nllb))
            for code, spec in LANGUAGE_SPECS.items()
        }
        ID_ENG = ID_BY_LANG_CODE.get("EN")
        ID_TUR = ID_BY_LANG_CODE.get("TR")


def init_models() -> None:
    global kokoro_tts, stt

    init_translation_runtime_only()

    if args.tts in {"auto", "kokoro"} and Kokoro is not None and kokoro_tts is None:
        model_path = Path(args.kokoro_model).expanduser()
        voices_path = Path(args.kokoro_voices).expanduser()
        if model_path.exists() and voices_path.exists():
            kokoro_tts = Kokoro(str(model_path), str(voices_path))

    if args.mode in {"audio", "both"} and stt is None:
        init_stt_runtime_only()


def init_stt_runtime_only() -> None:
    global stt, detected_stt_backend, detected_stt_device

    detected_stt_backend = resolve_stt_backend()
    detected_stt_device = resolve_stt_device(detected_stt_backend)
    args.whisper_model = default_whisper_model_for_backend(detected_stt_backend)
    validate_asr_model_language(args.whisper_model, detected_stt_backend)
    if detected_stt_backend == "whisper-cli":
        _ = resolve_whisper_cli_model_path()
        eprint(f"🔥 Using whisper-cli ({args.whisper_model}) on {detected_stt_device}...")
    elif detected_stt_backend == "mlx-vlm-gemma":
        if not MLX_VLM_INSTALLED:
            raise RuntimeError("Missing mlx-vlm dependency. Install it with: pip install -U mlx-vlm")
        model_ref = effective_mlx_vlm_model(partial=False)
        eprint(f"🔥 Using MLX-VLM Gemma audio STT ({model_ref}) on {detected_stt_device}...")
        if args.partial_mlx_vlm_model or args.final_mlx_vlm_model or args.partial_stt_backend != "same" or args.final_stt_backend != "same":
            eprint(f"🚦 ASR lanes: partial={args.partial_stt_backend}/{effective_mlx_vlm_model(partial=True)} | final={args.final_stt_backend}/{effective_mlx_vlm_model(partial=False)}")
    elif detected_stt_backend == "mlx-whisper":
        if not MLX_WHISPER_INSTALLED:
            _ = resolve_stt_command("mlx_whisper", ("mlx-whisper",))
        args.whisper_model = prepare_mlx_whisper_runtime_reference(args.whisper_model)
        for partial, attribute in ((True, "partial_whisper_model"), (False, "final_whisper_model")):
            if effective_lane_backend(partial) == "mlx-whisper" and getattr(args, attribute):
                setattr(args, attribute, prepare_mlx_whisper_runtime_reference(getattr(args, attribute)))
        model_ref = resolve_mlx_whisper_model_ref()
        eprint(f"🔥 Using mlx-whisper ({model_ref}) on {detected_stt_device}...")
        if args.partial_whisper_model or args.final_whisper_model or args.partial_stt_backend != "same" or args.final_stt_backend != "same":
            eprint(f"🚦 ASR lanes: partial={args.partial_stt_backend}/{args.partial_whisper_model or args.whisper_model}/beam{args.partial_asr_beam} | final={args.final_stt_backend}/{args.final_whisper_model or args.whisper_model}/beam{args.final_asr_beam or args.asr_beam}")
    elif detected_stt_backend == "dictate-stt":
        _ = resolve_stt_command("dictate-stt")
        eprint(f"🔥 Using dictate-stt on {detected_stt_device}...")
    else:
        if WhisperModel is None:
            raise RuntimeError("Missing faster-whisper dependency. Install faster-whisper or use --stt-backend whisper-cli/mlx-whisper.")
        eprint(f"🔥 Loading faster-whisper ({args.whisper_model}, {args.whisper_compute}) on CPU, language={asr_language_display()}...")
        stt = WhisperModel(
            args.whisper_model,
            device="cpu",
            compute_type=args.whisper_compute,
            num_workers=1,
        )

    if is_apple_silicon() and detected_stt_backend not in {"mlx-vlm-gemma", "mlx-whisper"}:
        log_verbose(f"Apple-native MLX STT unavailable; fell back to {detected_stt_backend} on {detected_stt_device}.")


def warmup_stt_runtime_only(report: bool = True, *, strict: bool = False) -> None:
    if not args.warmup or args.mode not in {"audio", "both"}:
        return
    try:
        silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
        if detected_stt_backend == "whisper-cli":
            _ = transcribe_audio_whisper_cli(silence)
        elif detected_stt_backend == "mlx-vlm-gemma":
            _ = transcribe_audio_mlx_vlm_gemma(silence, partial=False)
        elif detected_stt_backend == "mlx-whisper":
            _ = transcribe_audio_mlx_whisper(silence)
        elif detected_stt_backend == "dictate-stt":
            _ = transcribe_audio_dictate_stt(silence)
        elif stt is not None:
            list(
                stt.transcribe(
                    silence,
                    **asr_language_kwargs(),
                    beam_size=1,
                    best_of=1,
                    temperature=0.0,
                    condition_on_previous_text=False,
                    vad_filter=False,
                    word_timestamps=False,
                )[0]
            )
    except Exception as exc:
        if strict:
            raise RuntimeError(f"ASR initialization/warmup failed: {exc}") from exc
        if report:
            eprint(f"⚠️ ASR warmup skipped: {exc}")


def warmup_vad_runtime_only(report: bool = True) -> None:
    if not args.warmup or args.mode not in {"audio", "both"} or resolved_vad_backend != "silero":
        return
    try:
        ensure_silero_vad_loaded()
        _ = is_silero_speech_frame(np.zeros(FRAME_SAMPLES, dtype=np.int16))
        reset_silero_vad_state()
    except Exception as exc:
        if report:
            eprint(f"⚠️ Silero VAD warmup skipped: {exc}")


def warmup_translation_runtime_only(report: bool = True, *, strict: bool = False) -> None:
    if not args.warmup:
        return
    try:
        with translation_lock:
            _ = translate_chunk(
                f"Hello from {source_language().name}.",
                source_nllb_lang(),
                forced_bos_for_target(target_language_code()),
                max(1, args.mt_beams - 1),
            )
            _ = translate_chunk(
                f"Hello from {target_language().name}.",
                target_nllb_lang(),
                forced_bos_for_target(source_language_code()),
                max(1, args.mt_beams),
            )
    except Exception as exc:
        if strict:
            raise RuntimeError(f"Translation initialization/warmup failed: {exc}") from exc
        if report:
            eprint(f"⚠️ Translation warmup skipped: {exc}")


def warmup_models() -> None:
    if not args.warmup:
        return
    eprint("⚡ Warming up models...")
    init_models()
    warmup_vad_runtime_only(report=True)
    warmup_translation_runtime_only(report=True)
    warmup_stt_runtime_only(report=True)


def translation_target_lang(task_src_lang: str, forced_bos_id: int) -> str:
    reverse_ids = {value: code for code, value in ID_BY_LANG_CODE.items()}
    target_code = reverse_ids.get(int(forced_bos_id))
    if target_code:
        return language_spec(target_code).name
    src_code = cli_code_from_nllb(task_src_lang)
    return language_spec(reverse_language_code(src_code)).name


def forced_bos_for_target(target_lang: str) -> int:
    code = normalize_language_code(target_lang)
    if code not in ID_BY_LANG_CODE:
        if active_translation_backend() == "nllb" and tokenizer is not None:
            ID_BY_LANG_CODE[code] = int(tokenizer.convert_tokens_to_ids(language_spec(code).nllb))
        else:
            set_placeholder_language_ids()
    return int(ID_BY_LANG_CODE.get(code, 1))


def render_mt_prompt_template(template: str, text: str, src_lang: str, forced_bos_id: int) -> str:
    source_name = language_name_from_nllb(src_lang)
    target_name = translation_target_lang(src_lang, forced_bos_id)
    rendered = template.strip()
    replacements = {
        "{source_language}": source_name,
        "{target_language}": target_name,
        "{source_lang}": source_name,
        "{target_lang}": target_name,
        "{text}": text,
    }
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, value)
    return rendered


def mlx_translation_prompt(text: str, src_lang: str, forced_bos_id: int) -> str:
    if mlx_translation_tokenizer is None:
        raise RuntimeError("MLX tokenizer not initialized")
    source_name = language_name_from_nllb(src_lang)
    target_name = translation_target_lang(src_lang, forced_bos_id)
    custom_prompt = str(args.mt_prompt or "").strip()
    default_prompt = (
        f"Translate the following {source_name} text into natural {target_name}. "
        f"Return only {target_name} translation text. Do not answer in {source_name}.\n\n"
        f"{text}"
    )
    if custom_prompt and "{text}" in custom_prompt:
        user_prompt = render_mt_prompt_template(custom_prompt, text, src_lang, forced_bos_id)
    elif custom_prompt:
        rendered = render_mt_prompt_template(custom_prompt, text, src_lang, forced_bos_id)
        user_prompt = f"{rendered}\n\n{default_prompt}"
    else:
        user_prompt = default_prompt
    # TranslateGemma carries direction in structured content and accepts no system
    # role. Generic instruct models keep the already-rendered legacy prompt,
    # including custom --mt-prompt handling, without any text normalization here.
    messages = build_translation_messages(
        text,
        src_lang,
        target_name,
        default_translation_model_for_backend("mlx-lm"),
        generic_user_prompt=user_prompt,
    )
    try:
        return mlx_translation_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
    except TypeError:
        return mlx_translation_tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def mlx_vlm_translation_prompt(text: str, src_lang: str, forced_bos_id: int) -> str:
    source_name = language_name_from_nllb(src_lang)
    target_name = translation_target_lang(src_lang, forced_bos_id)
    custom_prompt = str(args.mt_prompt or "").strip()
    default_prompt = (
        f"Translate the following {source_name} text into natural {target_name}. "
        f"Return only {target_name} translation text. Do not include notes, labels, quotes, or the original text.\n\n"
        f"{text}"
    )
    if custom_prompt and "{text}" in custom_prompt:
        return render_mt_prompt_template(custom_prompt, text, src_lang, forced_bos_id)
    if custom_prompt:
        rendered = render_mt_prompt_template(custom_prompt, text, src_lang, forced_bos_id)
        return f"{rendered}\n\n{default_prompt}"
    return default_prompt


def generate_mlx_vlm_text(prompt: str, *, max_tokens: int | None = None) -> str:
    init_mlx_vlm_gemma_runtime(args.translation_model or args.mlx_vlm_model or DEFAULT_MLX_VLM_AUDIO_MODEL)
    if mlx_vlm_generate is None:
        raise RuntimeError("mlx-vlm generator is unavailable")
    model = getattr(mlx_vlm_thread_state, "model", mlx_vlm_model)
    processor = getattr(mlx_vlm_thread_state, "processor", mlx_vlm_processor)
    if model is None or processor is None:
        raise RuntimeError("mlx-vlm Gemma model is not loaded for this translation thread")
    formatted_prompt = apply_mlx_vlm_audio_template(prompt, 0, model=model, processor=processor)
    with mlx_vlm_lock:
        with mlx_thread_stream_context():
            with silent_external_output():
                result = generate_mlx_vlm_safely(model, processor, formatted_prompt, max_tokens=max_tokens)
    return strip_model_thinking(mlx_vlm_result_text(result))


def translate_chunk_mlx_vlm_gemma(chunk: str, src_lang: str, forced_bos_id: int, num_beams: int) -> str:
    del num_beams  # Gemma VLM text generation does not expose beam search here.
    prompt = mlx_vlm_translation_prompt(chunk, src_lang, forced_bos_id)
    text = generate_mlx_vlm_text(prompt)
    return normalize_text(text)


def translate_chunk_nllb(chunk: str, src_lang: str, forced_bos_id: int, num_beams: int) -> str:
    if tokenizer is None or translator is None:
        raise RuntimeError("Models not initialized")

    tokenizer.src_lang = src_lang
    inputs = tokenizer(chunk, return_tensors="pt", add_special_tokens=True)
    if use_mps:
        inputs = {k: v.to("mps") for k, v in inputs.items()}

    inference_ctx = torch.inference_mode() if torch is not None else nullcontext()
    with inference_ctx:
        generated = translator.generate(
            **inputs,
            forced_bos_token_id=forced_bos_id,
            max_new_tokens=NLLB_MAX_NEW_TOKENS,
            num_beams=num_beams,
            early_stopping=True,
            do_sample=False,
            length_penalty=1.0,
            repetition_penalty=1.08,
            no_repeat_ngram_size=3,
        )
    return normalize_text(tokenizer.decode(generated[0], skip_special_tokens=True))


def iter_translate_chunk_mlx_lm(chunk: str, src_lang: str, forced_bos_id: int, num_beams: int) -> Iterator[str]:
    ensure_mlx_lm_loaded()
    if mlx_translation_model is None or mlx_translation_tokenizer is None or mlx_lm_stream_generate is None:
        raise RuntimeError("MLX translation backend is not initialized")
    prompt = mlx_translation_prompt(chunk, src_lang, forced_bos_id)
    del num_beams  # MLX-LM streaming generation does not expose beam search here.
    for response in mlx_lm_stream_generate(mlx_translation_model, mlx_translation_tokenizer, prompt, max_tokens=256):
        piece = getattr(response, "text", "")
        if getattr(response, "token", None) in mlx_translation_stop_ids:
            # MLX-LM's final stop response can flush buffered answer characters.
            # Registered EOS itself was never added to the detokenizer.
            if getattr(response, "finish_reason", None) == "stop" and piece:
                yield piece  # Preserve buffered leading whitespace; sanitize the accumulated answer.
            break
        if piece:
            yield piece


def translate_chunk_mlx_lm(chunk: str, src_lang: str, forced_bos_id: int, num_beams: int) -> str:
    pieces: list[str] = []
    for piece in iter_translate_chunk_mlx_lm(chunk, src_lang, forced_bos_id, num_beams):
        pieces.append(piece)
    return strip_model_thinking("".join(pieces))


def translate_chunk(chunk: str, src_lang: str, forced_bos_id: int, num_beams: int) -> str:
    chunk = normalize_text(chunk)
    if not chunk:
        return ""
    if active_translation_backend() == "mlx-vlm-gemma":
        return translate_chunk_mlx_vlm_gemma(chunk, src_lang, forced_bos_id, num_beams)
    if active_translation_backend() == "mlx-lm":
        return translate_chunk_mlx_lm(chunk, src_lang, forced_bos_id, num_beams)
    return strip_model_thinking(translate_chunk_nllb(chunk, src_lang, forced_bos_id, num_beams))


def iter_translation_chunks(text: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    if active_translation_backend() in {"mlx-lm", "mlx-vlm-gemma"}:
        max_chars = int(args.mt_chunk_max_chars)
        if max_chars <= 0 or len(text) <= max_chars:
            return [text]
        return pack_by_chars(split_text_soft(text) or [text], max_chars)
    if tokenizer is None:
        raise RuntimeError("Tokenizer not initialized")
    return pack_by_tokens(split_text_soft(text) or [text], tokenizer, NLLB_SRC_TOKEN_BUDGET)


def translate_text(text: str, src_lang: str, forced_bos_id: int, num_beams: int) -> str:
    out_parts: list[str] = []
    for chunk in iter_translation_chunks(text):
        with translation_lock:
            decoded = translate_chunk(chunk, src_lang, forced_bos_id, num_beams)
        if decoded:
            out_parts.append(decoded)
    return normalize_text(" ".join(out_parts))


# ======================
# TTS
# ======================
def save_float_wav(path: str, audio_f32: np.ndarray, sample_rate: int) -> None:
    clipped = np.clip(audio_f32, -1.0, 1.0)
    save_wav(path, (clipped * 32767.0).astype(np.int16), sample_rate=sample_rate)


def kokoro_voice_lang(lang_hint: str) -> tuple[str, str]:
    code = active_tts_lang_code(lang_hint)
    spec = language_spec(code)
    if code == "EN":
        return args.kokoro_voice_en, args.kokoro_lang_en
    if code == "TR":
        voice = args.kokoro_voice_tr.strip() or args.kokoro_voice_en
        return voice, args.kokoro_lang_tr
    return args.kokoro_voice_en, spec.kokoro_lang


def tts_kokoro(text: str, wav_path: str, lang_hint: str) -> bool:
    global kokoro_tts
    if Kokoro is None:
        return False
    if kokoro_tts is None:
        model_path = Path(args.kokoro_model).expanduser()
        voices_path = Path(args.kokoro_voices).expanduser()
        if not (model_path.exists() and voices_path.exists()):
            return False
        try:
            kokoro_tts = Kokoro(str(model_path), str(voices_path))
        except Exception:
            return False

    voice, lang = kokoro_voice_lang(lang_hint)
    try:
        samples, sample_rate = kokoro_tts.create(text, voice=voice, speed=args.tts_speed, lang=lang)
        save_float_wav(wav_path, np.asarray(samples, dtype=np.float32), int(sample_rate))
        return True
    except Exception:
        return False


def tts_say(text: str, wav_path: str, lang_hint: str) -> bool:
    if not is_macos() or not have("say"):
        return False
    voice = pick_say_voice(lang_hint)
    with TemporaryDirectory(prefix="tts-transcribe-say-") as tmpdir:
        aiff = os.path.join(tmpdir, "out.aiff")
        source = Path(tmpdir) / "speech.txt"
        source.write_text(text, encoding="utf-8")
        try:
            subprocess.run(["say", "-v", voice, "-o", aiff, "-f", str(source)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=120)
            subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@22050", aiff, wav_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=30)
            with wave.open(wav_path, "rb") as rendered:
                if rendered.getnframes() <= 0:
                    raise RuntimeError(f"macOS voice {voice!r} produced no audio; check the installed voice and macOS speech service")
            return True
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"macOS speech synthesis timed out for {voice!r}") from exc
        except subprocess.CalledProcessError as exc:
            detail = normalize_text(exc.stderr or "")[:300]
            raise RuntimeError(f"macOS speech synthesis failed for {voice!r}" + (f": {detail}" if detail else "")) from exc


def tts_piper(text: str, wav_path: str, lang_hint: str) -> bool:
    if active_tts_lang_code(lang_hint) != "TR" or not have("piper") or not os.path.exists(VOICE_TR):
        return False
    try:
        subprocess.run(["piper", "-m", VOICE_TR, "-f", wav_path], input=text, text=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def edge_voice_for_lang(lang_hint: str) -> str:
    code = active_tts_lang_code(lang_hint)
    spec = language_spec(code)
    if args.edge_voice and (code == source_language_code() or not spec.edge_voice):
        return args.edge_voice
    return spec.edge_voice or args.edge_voice


def tts_edge(text: str, wav_path: str, lang_hint: str) -> bool:
    if not have("edge-tts"):
        return False
    with TemporaryDirectory(prefix="tts-transcribe-edge-") as tmpdir:
        mp3 = os.path.join(tmpdir, "out.mp3")
        pct = int((args.rate - 1.0) * 100)
        rate_str = f"{pct:+d}%"
        try:
            subprocess.run(
                ["edge-tts", "--voice", edge_voice_for_lang(lang_hint), "--rate", rate_str, "--text", text, "--write-media", mp3],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if have("ffmpeg"):
                subprocess.run(["ffmpeg", "-y", "-i", mp3, "-ar", "22050", "-ac", "1", wav_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            if is_macos() and have("afconvert"):
                subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@22050", mp3, wav_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
        except Exception:
            return False
    return False


def concat_wavs(parts: list[str], dest: str) -> None:
    if len(parts) == 1:
        shutil.copyfile(parts[0], dest)
        return
    with wave.open(parts[0], "rb") as first:
        params = first.getparams()
    with wave.open(dest, "wb") as out:
        out.setparams(params)
        for part in parts:
            with wave.open(part, "rb") as inp:
                out.writeframes(inp.readframes(inp.getnframes()))


def play_wav(path: str) -> None:
    if audio_bridge is not None:
        audio_bridge.play_wav(path)
        return
    if is_macos() and have("afplay"):
        subprocess.run(["afplay", path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elif have("ffplay"):
        subprocess.run(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path], check=True)
    else:
        raise RuntimeError("No audio playback command found (need afplay or ffplay)")


def tts_best_available(text: str, wav_out: str, lang_hint: str = "TR") -> str:
    text = normalize_text(text)
    if not text:
        raise RuntimeError("Empty text for TTS")

    if args.tts == "kokoro":
        order = [("kokoro", tts_kokoro)]
    elif args.tts == "say":
        order = [("say", tts_say)]
    elif args.tts == "piper":
        order = [("piper", tts_piper)]
    elif args.tts == "edge":
        order = [("edge", tts_edge)]
    elif active_tts_lang_code(lang_hint) == "EN":
        order = [("kokoro", tts_kokoro), ("say", tts_say), ("edge", tts_edge), ("piper", tts_piper)]
    elif active_tts_lang_code(lang_hint) == "TR":
        order = [("piper", tts_piper), ("say", tts_say), ("edge", tts_edge), ("kokoro", tts_kokoro)]
    else:
        order = [("say", tts_say), ("edge", tts_edge), ("kokoro", tts_kokoro), ("piper", tts_piper)]

    chunks = chunk_for_tts(text)
    with TemporaryDirectory(prefix="tts-transcribe-tts-") as tmpdir:
        part_paths: list[str] = []
        chosen_name = None
        chosen_fn = None
        failures: list[str] = []
        for name, fn in order:
            probe = os.path.join(tmpdir, "probe.wav")
            try:
                succeeded = fn(chunks[0], probe, lang_hint)
            except Exception as exc:
                failures.append(f"{name}: {exc}")
                continue
            if succeeded:
                chosen_name = name
                chosen_fn = fn
                part_paths.append(probe)
                break
            failures.append(f"{name}: unavailable or could not synthesize {active_tts_lang_code(lang_hint)}")
        if chosen_fn is None:
            raise RuntimeError("No TTS engine succeeded. " + "; ".join(failures))

        for idx, chunk in enumerate(chunks[1:], start=1):
            chunk_path = os.path.join(tmpdir, f"part_{idx:03d}.wav")
            if not chosen_fn(chunk, chunk_path, lang_hint):
                raise RuntimeError(f"TTS engine '{chosen_name}' failed while rendering chunk {idx + 1}.")
            part_paths.append(chunk_path)

        concat_wavs(part_paths, wav_out)
    return chosen_name or "unknown"



# ======================
# Timestamped ring streaming engine
# ======================
@dataclass(slots=True)
class TimestampedAudioFrame:
    pcm: np.ndarray
    timestamp_ns: int
    dbfs: float
    is_speech: bool
    speech_prob: float = 0.0


class TimestampedAppleVPIOBridge:
    """Reads timestamped Int16 mono frames emitted by apple_vpio_timestamp_bridge."""
    HEADER_SIZE = 20
    MAGIC = b"APCM"

    def __init__(self, path: str, mode: str = "vpio"):
        self.path = str(Path(path).expanduser())
        self.mode = mode
        self.proc: subprocess.Popen | None = None
        self.stderr_thread: threading.Thread | None = None
        self.stop_event = threading.Event()

    def __enter__(self):
        env = os.environ.copy()
        env["APPLE_VPIO_MODE"] = self.mode
        env["APPLE_VPIO_AGC"] = args.apple_vpio_agc
        env["APPLE_VPIO_BYPASS"] = args.apple_vpio_bypass_processing
        if args.verbose or args.debug:
            env["APPLE_VPIO_VERBOSE"] = "1"
        self.stop_event.clear()
        self.proc = subprocess.Popen([self.path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, bufsize=0, env=env)
        self.stderr_thread = threading.Thread(target=self._stderr_loop, daemon=True)
        self.stderr_thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop_event.set()
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=2)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        self.proc = None

    def _stderr_loop(self) -> None:
        if self.proc is None or self.proc.stderr is None:
            return
        for raw in self.proc.stderr:
            if self.stop_event.is_set():
                break
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            if "ready" in line:
                log_verbose(line)
            elif args.debug:
                log_debug(line)
            else:
                ui_update(status=f"⚠️ {line}")

    def _read_exact(self, n: int) -> bytes | None:
        assert self.proc is not None and self.proc.stdout is not None
        buf = bytearray()
        while len(buf) < n and not self.stop_event.is_set():
            chunk = self.proc.stdout.read(n - len(buf))
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)

    def frames(self) -> Iterator[tuple[int, np.ndarray]]:
        while not self.stop_event.is_set():
            header = self._read_exact(self.HEADER_SIZE)
            if not header:
                break
            if header[:4] != self.MAGIC:
                window = bytearray(header)
                while self.MAGIC not in window and not self.stop_event.is_set():
                    nxt = self._read_exact(1)
                    if not nxt:
                        return
                    window.extend(nxt)
                    if len(window) > 4096:
                        window = window[-4:]
                idx = window.find(self.MAGIC)
                rest = bytes(window[idx:])
                if len(rest) < self.HEADER_SIZE:
                    more = self._read_exact(self.HEADER_SIZE - len(rest))
                    if not more:
                        return
                    header = rest + more
                else:
                    header = rest[:self.HEADER_SIZE]
            timestamp_ns, sample_rate, samples = struct.unpack("<QII", header[4:20])
            payload = self._read_exact(samples * 2)
            if not payload:
                break
            if sample_rate != SAMPLE_RATE and args.debug:
                log_debug(f"timestamp bridge sample rate {sample_rate}; expected {SAMPLE_RATE}")
            yield int(timestamp_ns), np.frombuffer(payload, dtype=np.int16).copy()


class NativeTimestampRingBuffer:
    def __init__(self, seconds: float, sample_rate: int = SAMPLE_RATE):
        if not np.isfinite(sample_rate) or sample_rate <= 0 or not np.isfinite(seconds) or seconds <= 0:
            raise ValueError("ring duration and sample rate must be positive and finite")
        self.sample_rate = sample_rate
        self.max_samples = max(1, int(seconds * sample_rate))
        self.frames: deque[tuple[int, np.ndarray]] = deque()
        self.total_samples = 0
        self.lock = threading.Lock()

    def write(self, timestamp_ns: int, pcm: np.ndarray) -> None:
        if pcm.size == 0:
            return
        samples = np.asarray(pcm, dtype=np.int16).reshape(-1).copy()
        timestamp_ns = int(timestamp_ns)
        # A large callback still contributes its newest tail instead of evicting
        # itself entirely. Timestamps describe the retained first sample.
        if samples.size > self.max_samples:
            skipped = int(samples.size) - self.max_samples
            samples = samples[-self.max_samples:].copy()
            timestamp_ns += skipped * 1_000_000_000 // self.sample_rate
        with self.lock:
            self.frames.append((timestamp_ns, samples))
            self.total_samples += int(samples.size)
            overflow = self.total_samples - self.max_samples
            while self.frames and overflow > 0:
                old_timestamp, old = self.frames.popleft()
                trim = min(overflow, int(old.size))
                self.total_samples -= trim
                overflow -= trim
                if trim < old.size:
                    self.frames.appendleft((
                        old_timestamp + trim * 1_000_000_000 // self.sample_rate,
                        old[trim:].copy(),
                    ))

    def latest_ms(self, ms: int) -> np.ndarray:
        need = max(1, int(self.sample_rate * ms / 1000))
        chunks: list[np.ndarray] = []
        remaining = need
        with self.lock:
            for _ts, pcm in reversed(self.frames):
                if remaining <= 0:
                    break
                if pcm.size <= remaining:
                    chunks.append(pcm)
                    remaining -= int(pcm.size)
                else:
                    chunks.append(pcm[-remaining:])
                    remaining = 0
        if not chunks:
            return np.empty(0, dtype=np.int16)
        return np.concatenate(list(reversed(chunks))).astype(np.int16, copy=False)


class HybridVadEnergyGate:
    def __init__(self):
        self.audio_backend = resolve_audio_io_backend()
        self.hard_speech_floor_db = max(
            -78.0 if self.audio_backend == "apple-vpio" else -70.0,
            min_dbfs_utterance_threshold() - 12.0,
        )
        self.noise_floor_db = -70.0 if self.audio_backend == "apple-vpio" else -62.0
        self.vad_backend = resolved_vad_backend

    def detect(self, pcm: np.ndarray) -> tuple[bool, float, float]:
        db = rms_dbfs_i16(pcm)
        self.noise_floor_db = (1.0 - NOISE_EWMA_ALPHA) * self.noise_floor_db + NOISE_EWMA_ALPHA * min(db, self.noise_floor_db + 4.0)
        energy_ok = db >= max(min_dbfs_utterance_threshold(), self.noise_floor_db + args.noise_gate_db)
        vad_speech = False
        prob = 0.0
        if self.vad_backend == "webrtc" and vad is not None and pcm.size == FRAME_SAMPLES:
            try:
                vad_speech = bool(vad.is_speech(pcm.tobytes(), SAMPLE_RATE))
                prob = 1.0 if vad_speech else 0.0
            except Exception:
                vad_speech = energy_ok
        elif self.vad_backend == "silero":
            try:
                vad_speech = bool(is_silero_speech_frame(pcm))
                prob = 1.0 if vad_speech else 0.0
            except Exception:
                vad_speech = energy_ok
        else:
            vad_speech = energy_ok

        # VAD is allowed to carry quiet speech, but not near-digital silence.
        # This matches the legacy path's "VAD OR energy" behavior while keeping
        # a hard floor against open-mic noise and TTS bleed.
        speech = energy_ok or (vad_speech and db >= self.hard_speech_floor_db)
        return bool(speech), float(db), float(prob)


class PriorityAsrScheduler:
    def __init__(self):
        # Small queue by design: live speech wants freshness, not completeness of stale partials.
        self.q: PriorityQueue = PriorityQueue(maxsize=max(2, args.audio_queue_size * 3))
        self.seq = 0
        self.lock = threading.Lock()
        self.cancelled_partials: set[int] = set()
        self.final_submitted: set[int] = set()
        self.latest_partial_seq: dict[int, int] = {}
        self.partial_inflight_utterance: int | None = None

    def submit_with_provenance(self, utterance_id: int, pcm: np.ndarray, partial: bool, provenance: dict) -> bool:
        return self.submit(utterance_id, pcm, partial, provenance=provenance)

    def submit(self, utterance_id: int, pcm: np.ndarray, partial: bool, provenance: dict | None = None) -> bool:
        if pcm.size == 0:
            return False
        if asr_stage is not None:
            with self.lock:
                if partial and utterance_id in self.final_submitted:
                    return False
                if partial and args.partial_queue_latest_only:
                    snapshot = asr_stage.snapshot
                    if (snapshot.get("state") == "loading" or snapshot.get("committed_pending", 0)
                            or snapshot.get("inflight_partial") or snapshot.get("partials_pending", 0)):
                        # Let one preview finish rather than continually replacing
                        # its result when inference is slower than the cadence.
                        return False
                if not partial:
                    self.final_submitted.add(utterance_id)
                    self.cancelled_partials.add(utterance_id)
            accepted = submit_asr_stage(utterance_id, pcm, partial, provenance=provenance)
            if not accepted and not partial:
                with self.lock:
                    self.final_submitted.discard(utterance_id)
                    self.cancelled_partials.discard(utterance_id)
            return accepted
        priority = 10 if partial else 0
        marked_final_pending = False
        with self.lock:
            if partial:
                if utterance_id in self.final_submitted or utterance_id in self.cancelled_partials:
                    return False
                if args.partial_queue_latest_only and self.partial_inflight_utterance is not None:
                    if args.debug:
                        log_debug(
                            f"ring ASR skipped partial utterance={utterance_id}; partial utterance={self.partial_inflight_utterance} already in flight"
                        )
                    return False
            self.seq += 1
            seq = self.seq
            if partial and args.partial_queue_latest_only:
                self.latest_partial_seq[utterance_id] = seq
            if not partial:
                # Final ASR is authoritative: cancel queued speculative partials for this utterance.
                self.cancelled_partials.add(utterance_id)
                self.final_submitted.add(utterance_id)
                mark_final_asr_pending(utterance_id)
                marked_final_pending = True
                self.latest_partial_seq.pop(utterance_id, None)
                self._drop_queued_partials_for(utterance_id)
        item = (priority, seq, utterance_id, partial, pcm.astype(np.int16, copy=False).copy())
        while not shutdown_event.is_set():
            try:
                self.q.put_nowait(item)
                return True
            except Full:
                if self._drop_one_queued_partial():
                    continue
                if partial:
                    if args.debug:
                        log_debug(f"ring ASR dropped speculative partial utterance={utterance_id}; queue contains committed work")
                    return False
                try:
                    self.q.put(item, timeout=0.05)
                    return True
                except Full:
                    if args.debug:
                        log_debug(f"ring ASR waiting for final queue capacity utterance={utterance_id}")
                    continue
        if marked_final_pending:
            clear_final_asr_pending(utterance_id)
        return False

    def _drop_queued_partials_for(self, utterance_id: int) -> None:
        self._remove_queued_partials(utterance_id=utterance_id)

    def _drop_one_queued_partial(self) -> bool:
        return bool(self._remove_queued_partials(limit=1))

    def _remove_queued_partials(self, utterance_id: int | None = None, limit: int | None = None) -> int:
        # Draining and reinserting a live queue lets concurrent producers occupy
        # its slots and can lose/reorder committed finals. Change only selected
        # speculative entries while holding the queue's own synchronization lock.
        with self.q.mutex:
            candidates = sorted(
                (item for item in self.q.queue if item[3]
                 and (utterance_id is None or item[2] == utterance_id)),
                key=lambda item: item[1],
            )
            if limit is not None:
                candidates = candidates[:limit]
            removed_sequences = {item[1] for item in candidates}
            if not removed_sequences:
                return 0
            self.q.queue[:] = [item for item in self.q.queue if item[1] not in removed_sequences]
            heapify(self.q.queue)
            self.q.unfinished_tasks -= len(removed_sequences)
            if self.q.unfinished_tasks == 0:
                self.q.all_tasks_done.notify_all()
            self.q.not_full.notify_all()
            return len(removed_sequences)

    def _is_stale_partial(self, utterance_id: int, seq: int) -> bool:
        if utterance_id in self.cancelled_partials or utterance_id in self.final_submitted:
            return True
        if not args.partial_queue_latest_only:
            return False
        latest = self.latest_partial_seq.get(utterance_id)
        return latest is not None and seq != latest

    def worker(self) -> None:
        if detected_stt_backend == "mlx-vlm-gemma":
            try:
                warmup_stt_runtime_only(report=True)
            except Exception as exc:
                ui_update(status=f"❌ Ring ASR warmup error: {exc}")
        while not shutdown_event.is_set():
            try:
                _priority, seq, utterance_id, partial, pcm = self.q.get(timeout=0.1)
            except Empty:
                continue
            queued_result = False
            try:
                if partial and self._is_stale_partial(utterance_id, seq):
                    continue
                if partial:
                    with self.lock:
                        self.partial_inflight_utterance = utterance_id
                trace_update(utterance_id, **({"asr_partial_start_at": time.perf_counter()} if partial else {"asr_final_start_at": time.perf_counter()}))
                text = normalize_text(transcribe_pcm_i16(pcm, from_ring=True, partial=partial))
                if text or not partial:
                    queued_result = queue_put_asr_result(asr_results, AsrResult(utterance_id=utterance_id, text=text, partial=partial), "ring-asr-result")
            except Exception as exc:
                ui_update(status=f"❌ Ring ASR error: {exc}")
            finally:
                if not partial:
                    self.cancelled_partials.discard(utterance_id)
                    self.final_submitted.discard(utterance_id)
                    if not queued_result:
                        clear_final_asr_pending(utterance_id)
                else:
                    with self.lock:
                        if self.partial_inflight_utterance == utterance_id:
                            self.partial_inflight_utterance = None
                self.q.task_done()


class RingEndpointDetector:
    # Prefer a pause after the normal 18 s window. A 28 s hard limit leaves
    # headroom inside Whisper's 30 s context, including onset pre-roll.
    HARD_WINDOW_SECONDS = 28.0
    SOFT_BOUNDARY_SILENCE_MS = 240

    def __init__(self, ring: NativeTimestampRingBuffer, scheduler: PriorityAsrScheduler):
        self.ring = ring
        self.scheduler = scheduler
        self.vad_gate = HybridVadEnergyGate()
        self.lock = threading.RLock()
        self.reset_all()

    def reset_all(self) -> None:
        self.in_speech = False
        self.utterance_id = 0
        self.active: list[np.ndarray] = []
        self.pre_roll: list[np.ndarray] = []
        self.pre_roll_timestamps: list[int] = []
        self.active_timestamps: list[int] = []
        self.active_samples = 0
        self.capture_gap = False
        self.silence_ms = 0
        self.speech_ms = 0
        self.total_ms = 0
        self.speech_frames = 0
        self.total_frames = 0
        self.trailing_silence_frames = 0
        self.last_partial_ms = 0
        self.pending_final = False
        self.pending_final_started_sample: int | None = None
        self.pause_ewma_ms = float(args.silence_base_ms)
        self.adaptive_silence_ms = int(args.silence_base_ms)

    def _current_endpoint_threshold(self) -> int:
        # Apply the legacy adaptive/semantic endpointing logic to the ring path.
        text, stability, last_change_ms, locked_ratio = partial_endpoint_snapshot(self.utterance_id)
        base = compute_adaptive_threshold(self.pause_ewma_ms, self.speech_ms)
        base = semantic_endpoint_threshold(base, text)
        base = transcript_stability_bias(base, stability, last_change_ms)
        confidence = commit_confidence_score(text, stability, last_change_ms, self.silence_ms, locked_ratio)
        base = confidence_adjusted_endpoint_threshold(base, confidence)
        if should_semantic_endpoint(text, stability, last_change_ms, self.silence_ms, locked_ratio):
            base = min(base, max(args.semantic_min_silence_ms, args.silence_min_ms))
        self.adaptive_silence_ms = max(args.silence_min_ms, min(args.silence_max_ms, int(base)))
        if args.debug and self.silence_ms and self.silence_ms % 200 == 0:
            log_debug(f"ring endpoint silence_ms={self.silence_ms} threshold_ms={self.adaptive_silence_ms} text='{truncate_visible(text, 48)}'")
        return self.adaptive_silence_ms

    def _active_partial_window(self) -> np.ndarray:
        if not self.active:
            return np.empty(0, dtype=np.int16)
        max_frames = max(1, int(args.partial_asr_window_seconds * 1000 / FRAME_MS))
        frames = self.active[-max_frames:]
        if not frames:
            return np.empty(0, dtype=np.int16)
        return np.concatenate(frames).astype(np.int16, copy=False)

    def on_frame(self, timestamp_ns: int, pcm: np.ndarray) -> None:
        self.ring.write(timestamp_ns, pcm)
        # Explicit forced flushes can end on a sub-frame. VAD still receives its
        # required 20 ms input; ring history and ASR retain only captured samples.
        vad_pcm = np.pad(pcm, (0, FRAME_SAMPLES - pcm.size)) if pcm.size < FRAME_SAMPLES else pcm
        is_speech, db, prob = self.vad_gate.detect(vad_pcm)
        self.last_dbfs = db
        with self.lock:
            self._consume(TimestampedAudioFrame(pcm=pcm, timestamp_ns=timestamp_ns, dbfs=db, is_speech=is_speech, speech_prob=prob))

    def _window_provenance(self, pcm: np.ndarray, frame_count: int) -> dict:
        times = self.active_timestamps[-frame_count:]
        frames = self.active[-frame_count:]
        provenance = {"capture_gap": self.capture_gap, "frame_count": frame_count,
                      "timestamp_start_ns": times[0] if times else None,
                      "timestamp_end_ns": times[-1] + int(frames[-1].size) * 1_000_000_000 // SAMPLE_RATE if times and frames else None,
                      "sample_rate": SAMPLE_RATE, "channels": 1}
        if times and frames and len(times) == len(frames):
            duration_ns = provenance["timestamp_end_ns"] - provenance["timestamp_start_ns"]
            expected_ns = int(pcm.size) * 1_000_000_000 // SAMPLE_RATE
            provenance["source_span_minus_pcm_ns"] = duration_ns - expected_ns
            if abs(duration_ns - expected_ns) > 2 * 1_000_000_000 // SAMPLE_RATE:
                provenance["capture_gap"] = True
        return provenance

    def _submit_window(self, pcm: np.ndarray, partial: bool, frame_count: int,
                       *, boundary_reason: str | None = None, forced_speech_boundary: bool = False) -> bool:
        provenance = self._window_provenance(pcm, frame_count)
        if not partial:
            provenance.update(boundary_reason=boundary_reason or "endpoint",
                              forced_speech_boundary=forced_speech_boundary)
        method = getattr(self.scheduler, "submit_with_provenance", None)
        if method is not None:
            return method(self.utterance_id, pcm, partial, provenance)
        return self.scheduler.submit(self.utterance_id, pcm, partial=partial)

    def _consume(self, frame: TimestampedAudioFrame) -> None:
        if capture_paused_for_playback():
            return
        if not self.in_speech:
            self.pre_roll.append(frame.pcm)
            self.pre_roll_timestamps.append(frame.timestamp_ns)
            if len(self.pre_roll) > pre_roll_max:
                self.pre_roll = self.pre_roll[-pre_roll_max:]
                self.pre_roll_timestamps = self.pre_roll_timestamps[-pre_roll_max:]
            if frame.is_speech:
                self.utterance_id += 1
                self.in_speech = True
                reset_current_partial_state(next_utterance_id=self.utterance_id)
                trace_update(self.utterance_id, speech_started_at=time.perf_counter())
                # pre_roll already contains the triggering speech frame.
                self.active = list(self.pre_roll)
                self.active_timestamps = list(self.pre_roll_timestamps)
                self.active_samples = sum(int(pcm.size) for pcm in self.active)
                self.silence_ms = 0
                self.speech_ms = FRAME_MS
                self.total_ms = FRAME_MS
                self.speech_frames = 1
                self.total_frames = 1
                self.trailing_silence_frames = 0
                self.last_partial_ms = 0
                self.pause_ewma_ms = float(args.silence_base_ms)
                self.adaptive_silence_ms = int(args.silence_base_ms)
            return
        self.active.append(frame.pcm)
        self.active_timestamps.append(frame.timestamp_ns)
        self.active_samples += int(frame.pcm.size)
        self.total_ms += FRAME_MS
        self.total_frames += 1
        if frame.is_speech:
            self.speech_ms += FRAME_MS
            self.speech_frames += 1
            self.silence_ms = 0
            self.trailing_silence_frames = 0
            if self.pending_final:
                self.pending_final = False
                self.pending_final_started_sample = None
        else:
            self.silence_ms += FRAME_MS
            self.trailing_silence_frames += 1
            self.pause_ewma_ms = (1.0 - args.adapt_alpha) * self.pause_ewma_ms + args.adapt_alpha * float(self.silence_ms)
        if self.silence_ms == 0 and not self.pending_final and self.total_ms - self.last_partial_ms >= int(args.streaming_asr_chunk_seconds * 1000):
            self.last_partial_ms = self.total_ms
            partial_pcm = self._active_partial_window()
            if partial_pcm.size >= int(0.35 * SAMPLE_RATE):
                trace_update(self.utterance_id, partial_audio_ready_at=time.perf_counter())
                count = min(len(self.active), max(1, int(args.partial_asr_window_seconds * 1000 / FRAME_MS)))
                if self._submit_window(partial_pcm, True, count):
                    show_inflight_partial_transcript(self.utterance_id)
        if self.active_samples >= round(self.HARD_WINDOW_SECONDS * SAMPLE_RATE):
            self._finalize_locked(force=True, boundary_reason="hard_duration_limit", forced_speech_boundary=True)
            return
        if (self.active_samples >= round(MAX_UTTERANCE_SECONDS * SAMPLE_RATE)
                and self.silence_ms >= self.SOFT_BOUNDARY_SILENCE_MS):
            self._finalize_locked(boundary_reason="soft_duration_silence", keep_quiet_preroll=True)
            return
        endpoint_threshold = self._current_endpoint_threshold() if self.silence_ms else self.adaptive_silence_ms
        if self.silence_ms >= endpoint_threshold:
            if args.finalization_hold_ms <= 0:
                self._finalize_locked(force=False)
            elif not self.pending_final:
                self.pending_final = True
                self.pending_final_started_sample = self.active_samples
            elif ((self.active_samples - self.pending_final_started_sample) * 1000
                  >= args.finalization_hold_ms * SAMPLE_RATE):
                self._finalize_locked(force=False)

    def finalize(self, force: bool = False) -> None:
        with self.lock:
            self._finalize_locked(force=force, boundary_reason="forced_flush" if force else "endpoint")

    def _finalize_locked(self, force: bool = False, *, boundary_reason: str = "endpoint",
                         forced_speech_boundary: bool = False, keep_quiet_preroll: bool = False) -> None:
        if not self.in_speech or not self.active:
            self._reset_utterance()
            return
        # Endpoint waiting adds silence to the model input, not to the speech
        # span's quality denominator. Otherwise short valid words disappear
        # merely because the recognizer waited long enough for their ending.
        speech_span_frames = max(1, self.total_frames - self.trailing_silence_frames)
        speech_ratio = self.speech_frames / speech_span_frames
        duration_sec = self.active_samples / SAMPLE_RATE
        accepted = force or (duration_sec >= MIN_UTTERANCE_SECONDS
                             and self.speech_frames >= args.min_speech_frames
                             and speech_ratio >= args.min_speech_ratio)
        pcm = np.concatenate(self.active).astype(np.int16, copy=False)
        if accepted:
            trace_update(self.utterance_id, final_audio_ready_at=time.perf_counter())
            clear_inflight_partial_marker(self.utterance_id)
            self._submit_window(pcm, False, len(self.active), boundary_reason=boundary_reason,
                                forced_speech_boundary=forced_speech_boundary)
        if audio_evidence is not None and (not accepted or forced_speech_boundary):
            reason = "endpoint_speech_gate" if not accepted else "forced_speech_boundary"
            metadata = {"session_id": runtime_session_id, "source_language": source_language_code(),
                        "utterance_id": self.utterance_id, "boundary_reason": boundary_reason,
                        "forced_speech_boundary": forced_speech_boundary,
                        "speech_frames": self.speech_frames, "speech_span_frames": speech_span_frames,
                        "total_frames": self.total_frames, "trailing_silence_frames": self.trailing_silence_frames,
                        "speech_ratio": speech_ratio, "duration_seconds": duration_sec,
                        "min_speech_frames": args.min_speech_frames, "min_speech_ratio": args.min_speech_ratio,
                        **self._window_provenance(pcm, len(self.active))}
            audio_evidence.save_clip(pcm.astype("<i2", copy=False), metadata, [reason],
                                     label=f"u{self.utterance_id}-{reason}")
            audio_evidence.event({"type": reason, **metadata})
        quiet_count = min(pre_roll_max, self.trailing_silence_frames) if keep_quiet_preroll else 0
        quiet_frames = self.active[-quiet_count:] if quiet_count else []
        quiet_times = self.active_timestamps[-quiet_count:] if quiet_count else []
        self._reset_utterance()
        self.pre_roll = quiet_frames
        self.pre_roll_timestamps = quiet_times

    def _reset_utterance(self) -> None:
        self.in_speech = False
        self.active = []
        self.pre_roll = []
        self.pre_roll_timestamps = []
        self.active_timestamps = []
        self.active_samples = 0
        self.capture_gap = False
        self.silence_ms = 0
        self.speech_ms = 0
        self.total_ms = 0
        self.speech_frames = 0
        self.total_frames = 0
        self.trailing_silence_frames = 0
        self.last_partial_ms = 0
        self.pending_final = False
        self.pending_final_started_sample = None


ring_scheduler: PriorityAsrScheduler | None = None
ring_endpoint: RingEndpointDetector | None = None


@dataclass(slots=True)
class NativePCMFrame:
    sequence: int
    timestamp_ns: int
    pcm: np.ndarray
    discontinuity: bool = False
    valid_samples: int = FRAME_SAMPLES


class RingAudioFeeder:
    """Bounded single-source ingress for SoundDevice and the native Apple bridge.

    Callbacks only assemble/copy fixed 20 ms mono int16 frames. One worker feeds
    the endpoint in sequence order; model/VAD work never runs in the callback.
    SoundDevice ADC time is mapped to the monotonic clock once. Raw Apple PCM
    has no hardware timestamps, so its sample clock starts at first receipt.
    """

    def __init__(self, endpoint: RingEndpointDetector, queue_ms: int | None = None, *, clock_ns=None):
        self.endpoint = endpoint
        self.max_frames = max(1, int(args.capture_queue_ms if queue_ms is None else queue_ms) // FRAME_MS)
        self.clock_ns = clock_ns or time.monotonic_ns
        self.condition = threading.Condition()
        self.queue: deque[NativePCMFrame | None] = deque()  # None is an ordered forced-final boundary.
        self.queued_frames = 0
        self.pending = np.empty(0, dtype=np.int16)
        self.pending_timestamp_ns = 0
        self.next_timestamp_ns: int | None = None
        self.portaudio_clock_offset_ns: int | None = None
        self.next_sequence = 0
        self.discontinuity_pending = False
        self.dropped_samples = 0
        self.dropped_frames = 0
        self.status_events = 0
        self.input_overruns = 0
        self.callback_errors = 0
        self.captured_samples = 0
        self.last_input_at = 0.0
        self.last_capture_status = ""
        self.closed = False
        self.worker_thread: threading.Thread | None = None

    def __enter__(self):
        self.worker_thread = threading.Thread(target=self.worker, name="ring-capture", daemon=True)
        self.worker_thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _timestamp(self, time_info, samples: int) -> tuple[int, bool]:
        now_ns = self.clock_ns()
        try:
            def field_value(name):
                return float(time_info[name] if isinstance(time_info, dict) else getattr(time_info, name))
            adc, current = field_value("inputBufferAdcTime"), field_value("currentTime")
            if not np.isfinite(adc) or not np.isfinite(current) or adc <= 0 or current < adc:
                raise ValueError("no valid ADC clock")
            if self.portaudio_clock_offset_ns is None:
                self.portaudio_clock_offset_ns = now_ns - round(current * 1_000_000_000)
            timestamp_ns = self.portaudio_clock_offset_ns + round(adc * 1_000_000_000)
            gap = False
            if self.next_timestamp_ns is not None:
                # Sub-frame clock jitter must not fragment speech. A reported
                # device overflow always creates a boundary; an unreported ADC
                # jump of over half a frame also starts a separate utterance.
                if abs(timestamp_ns - self.next_timestamp_ns) > FRAME_MS * 1_000_000 // 2:
                    gap = True
                    self.status_events += 1
                    self.last_capture_status = "ADC timestamp gap/reset"
                    if timestamp_ns < self.next_timestamp_ns:
                        self.portaudio_clock_offset_ns += self.next_timestamp_ns - timestamp_ns
                    timestamp_ns = max(timestamp_ns, self.next_timestamp_ns)
                else:
                    timestamp_ns = self.next_timestamp_ns
            return timestamp_ns, gap
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
            if self.next_timestamp_ns is not None:
                return self.next_timestamp_ns, False
            return now_ns - samples * 1_000_000_000 // SAMPLE_RATE, False

    def _enqueue_frame(self, timestamp_ns: int, pcm: np.ndarray, valid_samples: int = FRAME_SAMPLES) -> None:
        while self.queued_frames >= self.max_frames:
            for index, old in enumerate(self.queue):
                if old is not None:
                    del self.queue[index]
                    self.queued_frames -= 1
                    self.dropped_frames += 1
                    self.dropped_samples += old.valid_samples
                    # Coalesce boundaries made adjacent by eviction, keeping
                    # control messages bounded along with the audio frames.
                    if 0 < index < len(self.queue) and self.queue[index - 1] is None and self.queue[index] is None:
                        del self.queue[index]
                    break
        self.queue.append(NativePCMFrame(
            self.next_sequence, int(timestamp_ns), pcm.copy(), self.discontinuity_pending, valid_samples,
        ))
        self.next_sequence += 1
        self.discontinuity_pending = False
        self.queued_frames += 1

    def callback(self, indata, frames: int, time_info, status) -> None:
        with self.condition:
            if self.closed or shutdown_event.is_set():
                return
            try:
                pcm = np.asarray(indata)
                if pcm.dtype != np.dtype(np.int16):
                    raise ValueError(f"capture dtype {pcm.dtype} is not Int16; refusing implicit PCM conversion")
                if pcm.ndim == 2 and pcm.shape[1] == 1:
                    pcm = pcm[:, 0]
                if pcm.ndim != 1 or int(frames) != pcm.size:
                    raise ValueError("capture requires 16 kHz mono int16 PCM with a matching frame count")
                source_start_sample = self.captured_samples
                self.captured_samples += int(pcm.size)
                self.last_input_at = time.monotonic()
                if status:
                    self.status_events += 1
                    self.input_overruns += int(bool(getattr(status, "input_overflow", False)))
                    self.last_capture_status = str(status)
                    self.discontinuity_pending = True
                if pcm.size == 0:
                    if status:
                        self.dropped_samples += int(self.pending.size)
                        self.pending = np.empty(0, dtype=np.int16)
                    self.condition.notify()
                    return
                timestamp_ns, clock_gap = self._timestamp(time_info, int(pcm.size))
                if audio_evidence is not None:
                    if audio_review_mode == "full":
                        audio_evidence.capture(pcm.astype("<i2", copy=False), start_sample=source_start_sample,
                                               timestamp_ns=timestamp_ns, discontinuity=bool(self.discontinuity_pending or status or clock_gap))
                    if status or clock_gap:
                        audio_evidence.event({"type": "capture_discontinuity", "source_start_sample": source_start_sample,
                                              "timestamp_ns": timestamp_ns, "input_status": str(status), "clock_gap": clock_gap})
                self.discontinuity_pending = self.discontinuity_pending or clock_gap
                if (status or clock_gap) and self.pending.size:
                    self.dropped_samples += int(self.pending.size)
                    self.pending = np.empty(0, dtype=np.int16)
                self.next_timestamp_ns = timestamp_ns + pcm.size * 1_000_000_000 // SAMPLE_RATE
                offset = 0
                if self.pending.size:
                    take = min(FRAME_SAMPLES - int(self.pending.size), int(pcm.size))
                    combined = np.concatenate((self.pending, pcm[:take]))
                    offset = take
                    if combined.size == FRAME_SAMPLES:
                        self._enqueue_frame(self.pending_timestamp_ns, combined)
                        self.pending = np.empty(0, dtype=np.int16)
                    else:
                        self.pending = combined
                while offset + FRAME_SAMPLES <= pcm.size:
                    self._enqueue_frame(timestamp_ns + offset * 1_000_000_000 // SAMPLE_RATE, pcm[offset:offset + FRAME_SAMPLES])
                    offset += FRAME_SAMPLES
                if offset < pcm.size:
                    self.pending = pcm[offset:].copy()
                    self.pending_timestamp_ns = timestamp_ns + offset * 1_000_000_000 // SAMPLE_RATE
            except Exception as exc:
                self.callback_errors += 1
                self.last_capture_status = str(exc)
                self.discontinuity_pending = True
                self.dropped_samples += int(self.pending.size)
                self.pending = np.empty(0, dtype=np.int16)
            self.condition.notify()

    def _flush_pending(self) -> None:
        if self.pending.size:
            pcm = np.pad(self.pending, (0, FRAME_SAMPLES - self.pending.size))
            self._enqueue_frame(self.pending_timestamp_ns, pcm, int(self.pending.size))
            self.pending = np.empty(0, dtype=np.int16)

    def request_finalize(self) -> None:
        with self.condition:
            if not self.closed:
                # A user flush owns all audio captured before this barrier,
                # including the tail awaiting the next full live callback.
                self._flush_pending()
                if not self.queue or self.queue[-1] is not None:
                    self.queue.append(None)
                self.condition.notify()

    def close(self) -> None:
        with self.condition:
            if not self.closed:
                self._flush_pending()
                if not self.queue or self.queue[-1] is not None:
                    self.queue.append(None)
                self.closed = True
                self.condition.notify_all()
        if self.worker_thread is not None and self.worker_thread is not threading.current_thread():
            self.worker_thread.join(timeout=2.0)

    def worker(self) -> None:
        expected_sequence = 0
        last_stats = (0, 0, 0, 0)
        needs_boundary = False
        while not shutdown_event.is_set():
            with self.condition:
                if not self.queue and not self.closed:
                    self.condition.wait(timeout=0.1)
                stats = (self.dropped_frames, self.dropped_samples, self.status_events, self.callback_errors)
                detail, overruns = self.last_capture_status, self.input_overruns
                has_item = bool(self.queue)
                done = self.closed and not has_item
                item = self.queue.popleft() if self.queue else None
                if item is not None:
                    self.queued_frames -= 1
            if stats != last_stats:
                ui_update(status=f"⚠️ Capture discontinuity | dropped={stats[0]} frames/{stats[1]} samples | input overruns={overruns} | status events={stats[2]} | callback errors={stats[3]} | {detail}")
                last_stats = stats
            if done:
                return
            try:
                if item is not None:
                    if needs_boundary or item.discontinuity or item.sequence != expected_sequence:
                        if hasattr(self.endpoint, "capture_gap"):
                            self.endpoint.capture_gap = True
                        self.endpoint.finalize(force=True)
                        if hasattr(self.endpoint, "capture_gap"):
                            self.endpoint.capture_gap = True
                    expected_sequence = item.sequence + 1
                    self.endpoint.on_frame(item.timestamp_ns, item.pcm[:item.valid_samples])
                    needs_boundary = False
                elif has_item:
                    self.endpoint.finalize(force=True)
            except Exception as exc:
                needs_boundary = True
                ui_update(status=f"❌ Ring capture worker error: {exc}")


def prepare_asr_audio(audio_i16: np.ndarray) -> np.ndarray:
    """Preserve captured PCM by default; the recognizer supplies its own Mel DSP."""
    global last_prepared_audio
    audio_f = np.asarray(audio_i16, dtype=np.int16).astype(np.float32) / 32768.0
    if args.asr_preprocessing == "legacy":
        filtered = np.empty_like(audio_f)
        for i in range(0, audio_f.size, FRAME_SAMPLES):
            filtered[i:i + FRAME_SAMPLES] = bandpass_voice(audio_f[i:i + FRAME_SAMPLES])
        prepared = rms_normalize(pre_emphasis(filtered))
    else:
        prepared = audio_f
    last_prepared_audio = np.ascontiguousarray(prepared, dtype=np.float32)
    last_asr_diagnostics["model_input"] = {
        "dtype": "float32", "byte_order": "little", "sample_rate": SAMPLE_RATE,
        "samples": int(prepared.size), "preprocessing": args.asr_preprocessing,
        "peak_abs": float(np.max(np.abs(prepared))) if prepared.size else 0.0,
        "out_of_range_fraction": float(np.mean(np.abs(prepared) > 1.0)) if prepared.size else 0.0,
        "sha256": hashlib.sha256(last_prepared_audio.astype("<f4", copy=False).tobytes()).hexdigest(),
    }
    return last_prepared_audio


def retain_asr_evidence(audio_i16: np.ndarray, text: str, error: str | None = None) -> None:
    raw_text = str(last_decoder_quality.get("raw_text", text))
    reasons = issue_reasons(last_asr_diagnostics["pcm"], text, raw_text=raw_text,
                            decoder_quality=last_decoder_quality,
                            expected_language=source_language().whisper, error=error,
                            gap=bool(asr_request_context.get("capture_gap")))
    if last_asr_diagnostics.get("model_input", {}).get("out_of_range_fraction", 0) > 0:
        reasons.append("model_input_out_of_range")
    if asr_request_context.get("forced_speech_boundary"):
        reasons.append("forced_speech_boundary")
    reasons.extend(last_asr_diagnostics.get("consistency_issues", []))
    last_asr_diagnostics.update({"decoder": dict(last_decoder_quality), "text": text,
                                 "issues": reasons, "error": error,
                                 "quality_flags_are_not_measured_accuracy": True})
    if audio_evidence is None or not audio_i16.size:
        return
    selected = reasons or (["debug"] if audio_review_mode == "full" else [])
    if not selected:
        return
    metadata = {"session_id": runtime_session_id, "source_language": source_language_code(),
                "backend": detected_stt_backend, "model": args.whisper_model,
                **last_asr_diagnostics}
    request_id = str(asr_request_context.get("request_id", "asr"))
    raw_saved = audio_evidence.save_clip(audio_i16.astype("<i2", copy=False).tobytes(), metadata, selected,
                                         label=f"{request_id}-raw")
    prepared_saved = True
    if last_prepared_audio is not None:
        prepared_saved = audio_evidence.save_blob(last_prepared_audio.astype("<f4", copy=False).tobytes(), metadata,
                                                  selected, label=f"{request_id}-model-input", suffix=".f32le")
    # This is the inference worker, not capture. Persist admitted evidence
    # before a fatal native error or worker exit can terminate its writer.
    audio_evidence.flush(timeout=3)
    last_asr_diagnostics["evidence"] = {"raw_accepted": raw_saved, "model_input_accepted": prepared_saved,
                                         "writer": audio_evidence.snapshot}


asr_preview_evidence: dict[int, str] = {}


def transcribe_pcm_i16(audio_i16: np.ndarray, *, from_ring: bool = False, partial: bool = False) -> str:
    global last_prepared_audio
    audio_i16 = np.ascontiguousarray(audio_i16, dtype=np.int16)
    last_prepared_audio = None
    last_decoder_quality.clear()
    last_asr_diagnostics.clear()
    last_asr_diagnostics.update({"request": dict(asr_request_context), "partial": partial,
                                 "pcm": pcm_quality(audio_i16), "decode_attempted": False})
    started = time.perf_counter()
    try:
        text = _transcribe_pcm_i16_impl(audio_i16, from_ring=from_ring, partial=partial)
    except Exception as exc:
        last_asr_diagnostics["processing_seconds"] = time.perf_counter() - started
        retain_asr_evidence(audio_i16, "", error=f"{type(exc).__name__}: {exc}")
        raise
    last_asr_diagnostics["processing_seconds"] = time.perf_counter() - started
    identity = asr_request_context.get("utterance_id")
    if identity is not None:
        if partial:
            if (text and not is_repetitive_text(text)
                    and (last_decoder_quality.get("max_compression_ratio") or 0) <= 2.9
                    and (last_decoder_quality.get("max_no_speech_prob") or 0) <= 0.6
                    and len(text) > len(asr_preview_evidence.get(identity, ""))):
                asr_preview_evidence[identity] = text[:2000]
                while len(asr_preview_evidence) > 32:
                    asr_preview_evidence.pop(next(iter(asr_preview_evidence)))
        else:
            preview = asr_preview_evidence.pop(identity, "")
            if final_preview_disagrees(text, preview, audio_i16.size / SAMPLE_RATE):
                last_asr_diagnostics["consistency_issues"] = ["final_preview_disagreement"]
                last_asr_diagnostics["preview_evidence"] = preview
    retain_asr_evidence(audio_i16, text)
    return text


def _transcribe_pcm_i16_impl(audio_i16: np.ndarray, *, from_ring: bool = False, partial: bool = False) -> str:
    if args.mode not in {"audio", "both"} or audio_i16.size == 0:
        return ""
    duration = audio_i16.size / SAMPLE_RATE
    if duration < 0.30:
        return ""
    if not should_send_to_asr(audio_i16, partial=partial):
        return ""
    backend = effective_lane_backend(partial)
    lane_model = effective_lane_model(partial)
    lane_beam = effective_lane_beam(partial)
    validate_asr_model_language(lane_model, backend)
    if backend == "faster-whisper" and stt is None:
        return ""
    if args.save_last_audio and not from_ring:
        save_wav(args.save_last_audio, audio_i16)
    if rms_dbfs_i16(audio_i16) < min_dbfs_utterance_threshold():
        return ""
    return decode_prepared_audio(prepare_asr_audio(audio_i16), partial=partial, from_ring=from_ring)


def decode_prepared_audio(prepared: np.ndarray, *, partial: bool = False, from_ring: bool = True) -> str:
    """Decode an exact recorded float input without applying capture DSP again."""
    prepared = np.ascontiguousarray(prepared, dtype=np.float32)
    if prepared.ndim != 1 or not np.all(np.isfinite(prepared)):
        raise ValueError("ASR model input must be finite mono float32 audio")
    backend, lane_model, lane_beam = effective_lane_backend(partial), effective_lane_model(partial), effective_lane_beam(partial)
    last_asr_diagnostics["decode_attempted"] = True
    actual_model = effective_mlx_vlm_model(partial) if backend == "mlx-vlm-gemma" else lane_model
    if backend == "mlx-whisper":
        actual_model = resolve_mlx_whisper_model_ref_for(lane_model)
    last_asr_diagnostics["decoder_config"] = {
        "backend": backend, "model": actual_model, "partial": partial, "requested_beam": lane_beam,
        "language": asr_language_display(), "task": "transcribe", "temperature": 0.0,
        "condition_on_previous_text": False, "initial_prompt": stt_initial_prompt(),
        "filter_thresholds": list(anti_repeat_thresholds()),
    }
    if backend == "whispercpp-coreml":
        return sanitize_asr_language_shape(sanitize_asr_text(transcribe_audio_whispercpp_coreml(prepared, beam=lane_beam), final=not partial), final=not partial)
    if backend == "apple-speech":
        return sanitize_asr_language_shape(sanitize_asr_text(transcribe_audio_apple_speech(prepared), final=not partial), final=not partial)
    if backend == "whisper-cli":
        model_override = str(Path(lane_model).expanduser()) if Path(lane_model).expanduser().exists() else None
        return sanitize_asr_language_shape(sanitize_asr_text(transcribe_audio_whisper_cli(prepared, model_path_override=model_override, beam_override=lane_beam), final=not partial), final=not partial)
    if backend == "mlx-vlm-gemma":
        return sanitize_asr_language_shape(sanitize_asr_text(transcribe_audio_mlx_vlm_gemma(prepared, partial=partial), final=not partial), final=not partial)
    if backend == "mlx-whisper":
        return sanitize_asr_language_shape(sanitize_asr_text(transcribe_audio_mlx_whisper(prepared, model_name=lane_model, beam=lane_beam), final=not partial), final=not partial)
    if backend == "dictate-stt":
        return sanitize_asr_language_shape(sanitize_asr_text(transcribe_audio_dictate_stt(prepared), final=not partial), final=not partial)
    segments, _ = stt.transcribe(
        prepared,
        **asr_language_kwargs(),
        task="transcribe",
        beam_size=lane_beam,
        best_of=1,
        temperature=0.0,
        condition_on_previous_text=False,
        **asr_initial_prompt_kwargs(),
        vad_filter=False if from_ring else True,
        vad_parameters={"min_silence_duration_ms": 200, "speech_pad_ms": 160},
        word_timestamps=False,
    )
    texts: list[str] = []
    no_speech_limit, min_logprob, max_compression = anti_repeat_thresholds()
    for seg in segments:
        if getattr(seg, "no_speech_prob", 0.0) > no_speech_limit:
            continue
        if getattr(seg, "avg_logprob", 0.0) < min_logprob:
            continue
        if getattr(seg, "compression_ratio", 0.0) > max_compression:
            continue
        piece = sanitize_asr_text(seg.text, final=not partial)
        if piece:
            texts.append(piece)
    text = normalize_text(" ".join(texts))
    return sanitize_asr_language_shape(sanitize_asr_text(text, final=not partial), final=not partial)

# ======================
# Audio state + queues
# ======================
audio_lock = threading.Lock()
capture_frames: Queue[bytes] = Queue(maxsize=max(2, args.capture_queue_ms // FRAME_MS))
audio_tasks: Queue[AudioTask] = Queue(maxsize=max(1, args.audio_queue_size))
asr_results: Queue[AsrResult] = Queue(maxsize=max(1, args.audio_queue_size * 2))
mt_tasks: PriorityQueue[PrioritizedTranslationTask] = PriorityQueue(maxsize=max(1, args.inference_queue_size))
tts_tasks: Queue[TtsTask] = Queue(maxsize=max(16, args.tts_queue_size))
shutdown_event = threading.Event()
tts_playing = threading.Event()
mt_revision_lock = threading.Lock()
latest_enqueued_audio_revision: dict[int, int] = {}
latency_lock = threading.Lock()
latency_traces: dict[int, dict[str, float]] = {}
audio_bridge: AppleVPIOBridge | None = None
asr_ipc_ctx: mp.context.BaseContext | None = None
asr_ipc_in = None
asr_ipc_out = None
asr_process: mp.Process | None = None
mt_ipc_ctx: mp.context.BaseContext | None = None
mt_ipc_in = None
mt_ipc_out = None
mt_process: mp.Process | None = None
asr_stage: ProcessStage | None = None
translation_stage: ProcessStage | None = None
translation_thread_error = ""
translation_stage_results: Queue[TranslationResult] = Queue(maxsize=128)
selection_tasks: dict[str, SelectionTask] = {}
selection_tasks_lock = threading.Lock()
stage_diagnostics_stream = None
pending_stage_asr: set[int] = set()
pending_stage_asr_lock = threading.Lock()
asr_slot_free: Queue[int] = Queue()
asr_slot_shms: dict[int, shared_memory.SharedMemory] = {}
asr_slot_size = int((MAX_UTTERANCE_SECONDS + 1.0) * SAMPLE_RATE * 2)
asr_slot_count = max(4, max(1, args.audio_queue_size) + max(1, args.audio_queue_size * 2) + 1)

resolved_vad_backend = resolve_vad_backend()
vad = webrtcvad.Vad(VAD_MODE) if resolved_vad_backend == "webrtc" and webrtcvad is not None else None
pre_roll = []
pre_roll_max = max(1, PRE_ROLL_MS // FRAME_MS)
active_frames: list[bytes] = []
utterance_frames: list[bytes] = []
max_frames = max(1, int(MAX_UTTERANCE_SECONDS * 1000 / FRAME_MS))
tail_frames = max(1, args.tail_pad_ms // FRAME_MS)
streaming_chunk_frames = max(1, int(args.streaming_asr_chunk_seconds * 1000 / FRAME_MS))
streaming_overlap_frames = max(1, args.streaming_asr_overlap_ms // FRAME_MS)
partial_asr_window_frames = max(streaming_chunk_frames, int(args.partial_asr_window_seconds * 1000 / FRAME_MS))

in_speech = False
silence_ms = 0
hangover_ms = 0
adaptive_silence_ms = args.silence_base_ms
pause_ewma_ms = float(args.silence_base_ms)
speech_run_ms = 0
speech_frames = 0
total_frames = 0
noise_floor_db = -62.0
playback_duck_ms = 0
playback_guard_until = 0.0
utterance_seq = 0
current_partial_text = ""
current_partial_display_text = ""
current_partial_display_locked_prefix = ""
current_partial_stability = 0
current_partial_last_change_ms = 0.0
current_partial_locked_prefix = ""
current_partial_locked_ratio = 0.0
current_partial_revision = 0
current_partial_utterance_id: int | None = None
current_partial_state_lock = threading.RLock()


def reset_current_partial_state(utterance_id: int | None = None, *, next_utterance_id: int | None = None) -> None:
    global current_partial_text, current_partial_display_text, current_partial_display_locked_prefix
    global current_partial_stability, current_partial_last_change_ms, current_partial_locked_prefix, current_partial_locked_ratio, current_partial_revision, current_partial_utterance_id
    with current_partial_state_lock:
        if utterance_id is not None and current_partial_utterance_id not in {None, utterance_id}:
            return
        current_partial_text = ""
        current_partial_display_text = ""
        current_partial_display_locked_prefix = ""
        current_partial_stability = 0
        current_partial_last_change_ms = 0.0
        current_partial_locked_prefix = ""
        current_partial_locked_ratio = 0.0
        current_partial_revision = 0
        current_partial_utterance_id = next_utterance_id


def partial_endpoint_snapshot(utterance_id: int) -> tuple[str, int, float, float]:
    """Take one coherent endpoint hint, excluding another utterance's results."""
    with current_partial_state_lock:
        if current_partial_utterance_id != utterance_id:
            return "", 0, 0.0, 0.0
        return (
            current_partial_text, current_partial_stability,
            current_partial_last_change_ms, current_partial_locked_ratio,
        )


def enqueue_translation_task(task: TranslationTask) -> bool:
    if translation_thread_error:
        ui_update(status=f"❌ Translation unavailable: {translation_thread_error}")
        return False
    if task.source == "audio":
        trace_update(
            task.utterance_id,
            **({"mt_partial_enqueued_at": time.perf_counter()} if task.partial else {"mt_final_enqueued_at": time.perf_counter()}),
        )
    if translation_stage is not None:
        accepted = translation_stage.submit(task, partial=task.partial,
                                            key=(task.source, task.utterance_id))
        if not accepted:
            if not task.partial:
                ui_update(status=f"❌ Translation not queued: {translation_stage.state}. Input is retained; check worker status or reduce the backlog.")
            return False
    elif mt_ipc_in is not None:
        accepted = queue_put_translation_task(mt_ipc_in, task, "mt")
    else:
        accepted = queue_put_translation_task(mt_tasks, task, "mt")
    if accepted and task.source == "audio" and task.utterance_id is not None:
        with mt_revision_lock:
            latest_enqueued_audio_revision[task.utterance_id] = max(
                latest_enqueued_audio_revision.get(task.utterance_id, 0),
                task.revision,
            )
    return accepted


def submit_typed_reply(text: str) -> bool:
    """A rejected submission keeps the editable input rather than clearing it."""
    global text_entry_seq
    normalized = normalize_text(text)
    if not normalized:
        return False
    text_entry_seq += 1
    entry_id = text_entry_seq
    with ui_lock:
        ui_set_panel_entry("right", entry_id, "src", f"⌨️ {target_label()}: {color_right_en(normalized)}", final=True)
    with session_log_lock:
        typed_input_texts[entry_id] = normalized
    accepted = enqueue_translation_task(TranslationTask(
        source="text", text=normalized, src_lang=target_nllb_lang(),
        beams=max(1, args.mt_beams), target_lang=source_language_code(),
        forced_bos_id=forced_bos_for_target(source_language_code()), utterance_id=entry_id,
    ))
    if not accepted:
        with session_log_lock:
            typed_input_texts.pop(entry_id, None)
        with ui_lock:
            ui_remove_panel_entry("right", entry_id)
    render_input("" if accepted else text)
    return accepted


def is_stale_audio_translation(task: TranslationTask) -> bool:
    if task.source != "audio" or task.utterance_id is None:
        return False
    # Partial MT is display-only. Let older completed partials keep/update the
    # live English draft instead of blanking the panel while newer MT is still
    # running; final UI updates still require an exact source revision match.
    if task.partial:
        return False
    with mt_revision_lock:
        return task.revision < latest_enqueued_audio_revision.get(task.utterance_id, task.revision)


def clear_audio_translation_revision(utterance_id: int, revision: int) -> None:
    with mt_revision_lock:
        if latest_enqueued_audio_revision.get(utterance_id) == revision:
            latest_enqueued_audio_revision.pop(utterance_id, None)


def trace_update(utterance_id: int | None, **values: float) -> None:
    if utterance_id is None or not args.latency_stats:
        return
    with latency_lock:
        trace = latency_traces.setdefault(int(utterance_id), {})
        trace.update(values)


def trace_get(utterance_id: int | None) -> dict[str, float]:
    if utterance_id is None or not args.latency_stats:
        return {}
    with latency_lock:
        return dict(latency_traces.get(int(utterance_id), {}))


def trace_drop(utterance_id: int | None) -> None:
    if utterance_id is None:
        return
    with latency_lock:
        latency_traces.pop(int(utterance_id), None)


def ms_between(trace: dict[str, float], start_key: str, end_key: str) -> int | None:
    start = trace.get(start_key)
    end = trace.get(end_key)
    if start is None or end is None:
        return None
    return max(0, int((end - start) * 1000))


def format_latency_status(utterance_id: int | None, *, include_tts: bool = False) -> str:
    trace = trace_get(utterance_id)
    if not trace:
        return ""
    parts: list[str] = []
    first_partial = ms_between(trace, "speech_started_at", "asr_first_partial_at")
    mt_first_delta = ms_between(trace, "speech_started_at", "mt_first_delta_at")
    asr_final = ms_between(trace, "speech_started_at", "asr_final_at")
    mt_final = ms_between(trace, "speech_started_at", "mt_final_at")
    tts_done = ms_between(trace, "speech_started_at", "tts_done_at")
    if first_partial is not None:
        parts.append(f"first partial {first_partial}ms")
    if mt_first_delta is not None:
        parts.append(f"MT first token {mt_first_delta}ms")
    if asr_final is not None:
        parts.append(f"ASR final {asr_final}ms")
    if mt_final is not None:
        parts.append(f"MT final {mt_final}ms")
    if include_tts and tts_done is not None:
        parts.append(f"TTS done {tts_done}ms")
    return " | ".join(parts)


def is_speculative_translation_task(item) -> bool:
    task = unwrap_translation_task(item)
    return isinstance(task, TranslationTask) and task.source == "audio" and task.partial


def queue_task_done_if_supported(q) -> None:
    if hasattr(q, "task_done"):
        q.task_done()


def _remove_optional_queue_item_locked(q: Queue, predicate) -> bool:
    """Remove one disposable item while the standard queue mutex is held."""
    for index, queued in enumerate(q.queue):
        if predicate(queued):
            del q.queue[index]
            if isinstance(q, PriorityQueue):
                heapify(q.queue)
            q.unfinished_tasks -= 1
            if q.unfinished_tasks == 0:
                q.all_tasks_done.notify_all()
            q.not_full.notify()
            return True
    return False


def _drop_optional_queue_item(q, predicate) -> bool:
    # Multiprocessing queues have no public atomic inspection operation. Refuse
    # eviction there rather than draining/reinserting someone else's committed
    # work, which can race with producers or stall indefinitely on a full queue.
    if not isinstance(q, Queue):
        return False
    with q.mutex:
        return _remove_optional_queue_item_locked(q, predicate)


def _put_without_waiting(q, item, optional_predicate) -> bool:
    """Atomically replace only optional work; never wait for queue capacity."""
    if shutdown_event.is_set():
        return False
    if isinstance(q, Queue):
        with q.mutex:
            if q.maxsize > 0 and q._qsize() >= q.maxsize:
                if not _remove_optional_queue_item_locked(q, optional_predicate):
                    return False
            q._put(item)
            q.unfinished_tasks += 1
            q.not_empty.notify()
            return True
    try:
        q.put_nowait(item)
        return True
    except Full:
        return False


def drop_one_speculative_translation_task(q, label: str) -> bool:
    dropped = _drop_optional_queue_item(q, is_speculative_translation_task)
    if dropped and args.debug:
        log_debug(f"dropped stale speculative {label} translation task to preserve committed work")
    return dropped


def queue_put_translation_task(q, task: TranslationTask, label: str) -> bool:
    item = wrap_translation_task(task) if isinstance(q, PriorityQueue) else task
    if _put_without_waiting(q, item, is_speculative_translation_task):
        return True
    if task.source == "text":
        ui_update(status="❌ Translation queue is full or stopping; this typed reply was not queued. Keep the text and retry after pending work finishes.")
    elif not task.partial:
        ui_update(status="❌ Translation queue is full or stopping; this final transcript was not queued for translation.")
    elif args.debug:
        log_debug(f"dropped speculative {label} translation task because higher-priority work is queued")
    return False


def is_stale_audio_tts_task(item) -> bool:
    return isinstance(item, TtsTask) and item.source == "audio"


def drop_one_stale_audio_tts_task(q: Queue, label: str) -> bool:
    dropped = _drop_optional_queue_item(q, is_stale_audio_tts_task)
    if dropped and args.debug:
        log_debug(f"dropped queued audio {label} task to preserve typed TTS")
    return dropped


def queue_put_tts_task(q: Queue, task: TtsTask, label: str) -> bool:
    """
    TTS is audible committed output, so do not use the generic drop-oldest queue.
    Text replies are user-authored and must be preserved. Optional spoken audio
    translations are allowed to coalesce behind newer audio, but never evict text.
    """
    if _put_without_waiting(q, task, is_stale_audio_tts_task):
        return True
    if task.source == "text":
        ui_update(status="❌ Speech queue is full or stopping; this reply was not queued for speech. Its translated text remains visible.")
    elif args.debug:
        log_debug(f"dropped incoming audio {label} task behind committed typed TTS")
    return False


def allocate_audio_slot() -> tuple[int | None, shared_memory.SharedMemory | None]:
    try:
        slot_id = asr_slot_free.get_nowait()
    except Empty:
        return None, None
    shm = asr_slot_shms.get(slot_id)
    if shm is None:
        return None, None
    return slot_id, shm


def make_audio_task(raw: bytes, utterance_id: int, partial: bool, use_shared_memory: bool) -> AudioTask:
    if use_shared_memory and raw:
        slot_id, shm = allocate_audio_slot()
        if shm is not None and len(raw) <= asr_slot_size:
            shm.buf[:len(raw)] = raw
            return AudioTask(utterance_id=utterance_id, partial=partial, shm_name=shm.name, slot_id=slot_id, num_bytes=len(raw))
        if slot_id is not None:
            release_audio_slot(slot_id)
        shm = shared_memory.SharedMemory(create=True, size=len(raw))
        shm.buf[:len(raw)] = raw
        shm.close()
        return AudioTask(utterance_id=utterance_id, partial=partial, shm_name=shm.name, num_bytes=len(raw))
    return AudioTask(utterance_id=utterance_id, partial=partial, raw=raw, num_bytes=len(raw))


def task_audio_bytes(task: AudioTask) -> bytes:
    if task.raw is not None:
        return task.raw
    if task.shm_name:
        shm = shared_memory.SharedMemory(name=task.shm_name)
        try:
            return bytes(shm.buf[:task.num_bytes])
        finally:
            shm.close()
    return b""


def flush_active_locked(force: bool = False, keep_overlap: bool = False) -> None:
    global in_speech, silence_ms, hangover_ms, speech_run_ms, speech_frames, total_frames, utterance_seq, current_partial_text, current_partial_revision

    if not active_frames or not utterance_frames:
        return

    salvage_partial = bool(speech_frames > 0 and total_frames > 0)
    if not force:
        if total_frames < 1 or speech_frames < args.min_speech_frames:
            if salvage_partial:
                force = True
            else:
                active_frames.clear()
                utterance_frames.clear()
                reset_current_partial_state()
                in_speech = False
                return
        if not force and (speech_frames / max(1, total_frames)) < args.min_speech_ratio:
            if salvage_partial:
                force = True
            else:
                active_frames.clear()
                utterance_frames.clear()
                reset_current_partial_state()
                in_speech = False
                return

    overlap = list(active_frames[-streaming_overlap_frames:]) if keep_overlap else []

    z = b"\x00" * (FRAME_SAMPLES * 2)
    source_frames = utterance_frames[-partial_asr_window_frames:] if keep_overlap else utterance_frames
    chunk_frames = list(source_frames)
    for _ in range(tail_frames):
        chunk_frames.append(z)

    raw = b"".join(chunk_frames)
    utterance_id = utterance_seq
    now = time.perf_counter()

    task = make_audio_task(raw, utterance_id=utterance_id, partial=keep_overlap, use_shared_memory=asr_process is not None)
    trace_update(utterance_id, **({"audio_partial_queued_at": now} if keep_overlap else {"audio_final_queued_at": now}))
    queued = True
    if args.prefer_latest_audio:
        queued = queue_put_audio_task(audio_tasks, task, "audio", has_task_done=True)
    else:
        audio_tasks.put(task)
    if keep_overlap and queued:
        show_inflight_partial_transcript(utterance_id)
    elif queued:
        clear_inflight_partial_marker(utterance_id)

    if keep_overlap:
        active_frames[:] = overlap
        silence_ms = 0
        hangover_ms = args.vad_hangover_ms
    else:
        active_frames.clear()
        utterance_frames.clear()
        reset_current_partial_state()
        in_speech = False
        silence_ms = 0
        hangover_ms = 0
        speech_run_ms = 0
        speech_frames = 0
        total_frames = 0
        utterance_seq += 1


def reset_active_capture_state() -> None:
    global in_speech, silence_ms, hangover_ms, speech_run_ms, speech_frames, total_frames, current_partial_text, current_partial_revision
    if active_frames and speech_frames > 0:
        flush_active_locked(force=True)
    active_frames.clear()
    utterance_frames.clear()
    reset_current_partial_state()
    in_speech = False
    silence_ms = 0
    hangover_ms = 0
    speech_run_ms = 0
    speech_frames = 0
    total_frames = 0
    if resolved_vad_backend == "silero":
        reset_silero_vad_state()


def force_flush_audio() -> None:
    with audio_lock:
        flush_active_locked(force=True)


def capture_paused_for_playback() -> bool:
    """Only explicit half-duplex capture pauses for playback or its tail guard."""
    return not args.duplex_capture and (tts_playing.is_set() or time.time() < playback_guard_until)


def process_capture_frame(pcm: bytes) -> None:
    global in_speech, silence_ms, hangover_ms, adaptive_silence_ms, pause_ewma_ms
    global speech_run_ms, speech_frames, total_frames, noise_floor_db, pre_roll, playback_duck_ms, playback_guard_until, current_partial_text, current_partial_stability, current_partial_last_change_ms, current_partial_locked_prefix, current_partial_locked_ratio, current_partial_revision

    if shutdown_event.is_set():
        return
    if capture_paused_for_playback():
        return
    pre_roll.append(pcm)
    if len(pre_roll) > pre_roll_max:
        del pre_roll[:-pre_roll_max]

    frame_i16 = np.frombuffer(pcm, dtype=np.int16)
    frame_db = rms_dbfs_i16(frame_i16)

    if not in_speech:
        noise_floor_db = (1.0 - NOISE_EWMA_ALPHA) * noise_floor_db + NOISE_EWMA_ALPHA * frame_db

    loud_enough = frame_db > noise_floor_db + args.noise_gate_db
    if resolved_vad_backend == "webrtc":
        speech_vad = bool(vad and vad.is_speech(pcm, SAMPLE_RATE))
    elif resolved_vad_backend == "silero":
        speech_vad = is_silero_speech_frame(frame_i16)
    else:
        speech_vad = False
    # Full-duplex listening must not classify all energy-VAD speech as playback
    # bleed while a reply is spoken. Echo suppression belongs in the selected
    # input route (for example VPIO); half-duplex is an explicit opt-out.
    playback_duck_ms = 0
    speech = bool(speech_vad or loud_enough)

    with audio_lock:
        if speech:
            if not in_speech:
                trace_update(utterance_seq, speech_started_at=time.perf_counter())
                active_frames.extend(pre_roll)
                utterance_frames.extend(pre_roll)
                in_speech = True
                silence_ms = 0
                hangover_ms = 0
                speech_run_ms = 0
                speech_frames = 0
                total_frames = 0
                reset_current_partial_state(next_utterance_id=utterance_seq)
            active_frames.append(pcm)
            utterance_frames.append(pcm)
            speech_run_ms += FRAME_MS
            hangover_ms = args.vad_hangover_ms
        elif in_speech:
            if hangover_ms > 0:
                active_frames.append(pcm)
                utterance_frames.append(pcm)
                hangover_ms -= FRAME_MS
            else:
                silence_ms += FRAME_MS
                pause_ewma_ms = (1.0 - args.adapt_alpha) * pause_ewma_ms + args.adapt_alpha * float(silence_ms)
                endpoint_text, endpoint_stability, endpoint_last_change, endpoint_locked_ratio = partial_endpoint_snapshot(utterance_seq)
                adaptive_silence_ms = compute_adaptive_threshold(pause_ewma_ms, speech_run_ms)
                adaptive_silence_ms = semantic_endpoint_threshold(adaptive_silence_ms, endpoint_text)
                adaptive_silence_ms = transcript_stability_bias(adaptive_silence_ms, endpoint_stability, endpoint_last_change)
                commit_confidence = commit_confidence_score(endpoint_text, endpoint_stability, endpoint_last_change, silence_ms, endpoint_locked_ratio)
                adaptive_silence_ms = confidence_adjusted_endpoint_threshold(adaptive_silence_ms, commit_confidence)
                if args.debug and endpoint_text:
                    score = sentence_completeness_score(endpoint_text)
                    log_debug(f"endpoint score={score} stability={endpoint_stability} locked_ratio={endpoint_locked_ratio:.2f} last_change_ms={endpoint_last_change:.0f} silence_ms={silence_ms} confidence={commit_confidence} threshold_ms={adaptive_silence_ms}")
                semantic_commit = should_semantic_endpoint(
                    endpoint_text,
                    endpoint_stability,
                    endpoint_last_change,
                    silence_ms,
                    endpoint_locked_ratio,
                )
                if semantic_commit or silence_ms >= adaptive_silence_ms:
                    if args.debug and semantic_commit:
                        log_debug("semantic endpoint committed stable complete partial")
                    flush_active_locked(force=False)

        if in_speech:
            total_frames += 1
            if speech:
                speech_frames += 1
            if len(active_frames) >= streaming_chunk_frames:
                flush_active_locked(force=True, keep_overlap=True)
            elif len(utterance_frames) > max_frames:
                flush_active_locked(force=True)


def audio_callback(indata, frames, time_info, status):
    if shutdown_event.is_set():
        return
    if audio_evidence is not None:
        timestamp_ns = time.monotonic_ns() - int(frames) * 1_000_000_000 // SAMPLE_RATE
        if audio_review_mode == "full":
            audio_evidence.capture(indata[:, 0], timestamp_ns=timestamp_ns, discontinuity=bool(status))
        if status:
            audio_evidence.save_clip(indata[:, 0], {"source_language": source_language_code(),
                                                    "capture_status": str(status), "streaming_engine": "legacy"},
                                     ["capture_status"], label="capture-issue", timestamp_ns=timestamp_ns)
    if status:
        return
    queue_put_latest_silent(capture_frames, indata[:, 0].tobytes())


# ======================
# Workers
# ======================
def audio_frame_worker() -> None:
    while not shutdown_event.is_set():
        try:
            pcm = capture_frames.get(timeout=0.1)
        except Empty:
            continue
        try:
            process_capture_frame(pcm)
        except Exception as exc:
            ui_update(status=f"❌ Capture pipeline error: {exc}")
        finally:
            capture_frames.task_done()


def transcribe_audio(raw: bytes) -> str:
    audio_i16 = np.frombuffer(raw, dtype=np.int16)
    return transcribe_pcm_i16(audio_i16, from_ring=False)


def asr_result_from_task(task: AudioTask) -> AsrResult:
    raw = task_audio_bytes(task)
    transcribed_text = normalize_text(transcribe_audio(raw))
    return AsrResult(
        utterance_id=task.utterance_id,
        text=transcribed_text,
        partial=task.partial,
        shm_name=task.shm_name,
        slot_id=task.slot_id,
    )


asr_decisions_lock = threading.Lock()


def record_asr_decision(result: AsrResult, text: str, decision: str, reason: str = "") -> None:
    """Keep source recognition independent of MT success and make losses auditable."""
    if not args.session_logs or runtime_session_dir is None:
        return
    diagnostics = result.diagnostics or {}
    decoder = diagnostics.get("decoder", {})
    raw_text = str(decoder.get("raw_text", result.text or ""))
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(), "session_id": runtime_session_id,
        "utterance_id": result.utterance_id, "request_id": result.request_id,
        "decision": decision, "reason": reason, "source_language": source_language_code(),
        "text": str(text)[:16000], "raw_text": raw_text[:16000],
        "text_truncated": len(str(text)) > 16000 or len(raw_text) > 16000,
        "request": diagnostics.get("request", {}), "issues": diagnostics.get("issues", []),
        "decoder_quality": {key: decoder.get(key) for key in (
            "mean_avg_logprob", "min_avg_logprob", "max_no_speech_prob", "max_compression_ratio",
            "requested_beam", "effective_beam", "detected_language")},
        "processing_seconds": diagnostics.get("processing_seconds"),
    }
    try:
        with asr_decisions_lock:
            with (runtime_session_dir / "asr-decisions.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
    except (OSError, ValueError) as exc:
        log_debug(f"could not write ASR decision journal: {exc}")
    if audio_evidence is not None:
        audio_evidence.event({"kind": "asr_decision", **payload})


def configure_inference_diagnostics(stage: str) -> None:
    """Keep child/library output out of the interactive terminal dashboard."""
    global stage_diagnostics_stream, runtime_session_id, runtime_session_dir, runtime_audio_dir, audio_review_mode, audio_evidence
    runtime_session_id = os.environ.get("TTS_TRANSLATE_SESSION_ID", runtime_session_id)
    directory = Path(os.environ.get("TTS_TRANSLATE_SESSION_DIR") or str(Path(__file__).resolve().parent / "runtime-reports"))
    runtime_session_dir = directory
    directory.mkdir(exist_ok=True)
    stage_diagnostics_stream = (directory / f"{stage}-worker.log").open("a", buffering=1, encoding="utf-8")
    sys.stdout = stage_diagnostics_stream
    sys.stderr = stage_diagnostics_stream
    eprint(f"\n--- {datetime.now(timezone.utc).isoformat()} {stage} starting ---")
    if stage == "asr" and os.environ.get("TTS_TRANSLATE_AUDIO_DIR"):
        runtime_audio_dir = Path(os.environ["TTS_TRANSLATE_AUDIO_DIR"])
        audio_review_mode = os.environ.get("TTS_TRANSLATE_AUDIO_MODE", "issues")
        if audio_review_mode != "off":
            audio_evidence = AudioEvidence(runtime_audio_dir, role="asr", full_stream=False,
                                           max_clip_bytes=args.issue_audio_budget_mb * 1024 * 1024)
            from multiprocessing.util import Finalize
            Finalize(audio_evidence, audio_evidence.close, exitpriority=10)


def initialize_asr_stage(configuration: dict | None = None) -> None:
    if configuration:
        vars(args).update(configuration)
    configure_inference_diagnostics("asr")
    init_stt_runtime_only()
    # Fail native initialization in this worker once, rather than repeatedly
    # retrying a poisoned extension from capture or UI threads.
    if detected_stt_backend == "mlx-whisper":
        ensure_mlx_whisper_loaded()
    elif detected_stt_backend == "mlx-vlm-gemma":
        init_mlx_vlm_gemma_runtime(effective_mlx_vlm_model(partial=False))
    warmup_stt_runtime_only(report=False, strict=True)


def handle_asr_stage_request(payload, emit) -> None:
    global asr_request_context
    utterance_id, partial, pcm = payload[:3]
    metadata = dict(payload[3]) if len(payload) > 3 else {}
    actual_hash = hashlib.sha256(np.ascontiguousarray(pcm, dtype="<i2").tobytes()).hexdigest()
    asr_request_context = {**metadata, "actual_pcm_sha256": actual_hash, "utterance_id": utterance_id}
    if metadata.get("pcm_sha256", actual_hash) != actual_hash:
        if audio_evidence is not None:
            audio_evidence.save_clip(pcm, asr_request_context, ["request_pcm_integrity_mismatch"], label="mismatched-input")
            audio_evidence.flush()
        raise RuntimeError("ASR request PCM integrity mismatch; refusing to decode another buffer")
    text = normalize_text(transcribe_pcm_i16(pcm, from_ring=True, partial=partial))
    emit(AsrResult(utterance_id=utterance_id, text=text, partial=partial,
                   request_id=metadata.get("request_id", ""), session_id=metadata.get("session_id", ""),
                   pcm_sha256=actual_hash, diagnostics=dict(last_asr_diagnostics)))


def initialize_translation_stage(configuration: dict | None = None) -> None:
    if configuration:
        vars(args).update(configuration)
    configure_inference_diagnostics("translation")
    init_translation_runtime_only()
    warmup_translation_runtime_only(report=False, strict=True)


def handle_translation_stage_request(task: TranslationTask, emit) -> None:
    if isinstance(task, SelectionTask):
        with translation_lock:
            translated = translate_chunk(task.text, language_spec(task.source_language).nllb,
                                         forced_bos_for_target(task.target_language), 1)
        translated = strip_model_thinking(translated)
        emit(SelectionResult(task, translated, "" if translated else "Selection translation returned no text"))
        return
    # The parent uses direction markers only; actual NLLB token IDs are owned
    # by the child tokenizer. MLX uses the structured language codes instead.
    task.forced_bos_id = forced_bos_for_target(task.target_lang)
    emit_translation_results(task, emit)


def brief_worker_detail(detail: str) -> str:
    if "No Metal device available" in detail:
        return "No Metal device available in this session"
    return detail.strip().splitlines()[-1] if detail.strip() else ""


def on_inference_event(event: StageEvent) -> None:
    if event.kind == "done" and runtime_session_dir is not None:
        try:
            with (runtime_session_dir / "inference-timing.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "stage": event.stage,
                    "job_id": event.job_id, "key": event.key, "partial": event.partial,
                    "background": event.background, "queue_wait_seconds": event.queue_wait_seconds,
                    "request_seconds": event.request_seconds}) + "\n")
        except OSError as exc:
            log_debug(f"could not write inference timing: {exc}")
    if event.stage == "translation":
        if event.kind == "failed":
            with selection_tasks_lock:
                identities = list(selection_tasks)
            for identity in identities:
                fail_selection_task(identity, "Translation worker failed; restart the session")
        elif event.kind in {"discarded", "rejected"} and isinstance(event.key, tuple) and event.key[:1] == ("lookup",):
            fail_selection_task(event.key[1], event.detail or "Live work superseded the selection; select it again")
    detail = brief_worker_detail(event.detail)
    if event.kind in {"loading", "ready", "busy", "done", "failed", "closed"}:
        update_stage_status(event.stage, event.state, detail)
    if event.kind in {"failed", "rejected", "callback_error"} and not (event.kind == "rejected" and event.background):
        directory = runtime_session_dir or Path(__file__).resolve().parent / "runtime-reports"
        directory.mkdir(exist_ok=True)
        with (directory / "pipeline-events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(),
                                     "stage": event.stage, "kind": event.kind,
                                     "detail": event.detail}, ensure_ascii=False) + "\n")
        if event.kind == "failed" and event.stage == "asr":
            with pending_stage_asr_lock:
                pending = tuple(pending_stage_asr)
                pending_stage_asr.clear()
                pending_clips = tuple(pending_audio_evidence.values())
                pending_audio_evidence.clear()
            if audio_evidence is not None:
                for pcm, metadata in pending_clips:
                    audio_evidence.save_clip(pcm, {**metadata, "worker_error": event.detail},
                                             ["worker_failure_pending_input"], label=metadata.get("request_id", "failed"))
            for utterance_id in pending:
                clear_final_asr_pending(utterance_id)
                clear_inflight_partial_marker(utterance_id)
                if ring_scheduler is not None:
                    with ring_scheduler.lock:
                        ring_scheduler.final_submitted.discard(utterance_id)
                        ring_scheduler.cancelled_partials.discard(utterance_id)
        if event.kind != "rejected" or not event.partial:
            ui_update(status=f"❌ {event.stage}: {detail or event.kind}. See {directory / 'pipeline-events.jsonl'}")


def receive_asr_stage_result(result: AsrResult) -> None:
    if result.session_id and result.session_id != runtime_session_id:
        ui_update(status="❌ Rejected an ASR result from another session")
        return
    with pending_stage_asr_lock:
        expected = pending_audio_evidence.get(result.utterance_id)
        if expected is not None and result.request_id:
            metadata = expected[1]
            if result.request_id != metadata.get("request_id"):
                return
            elif result.pcm_sha256 != metadata["pcm_sha256"]:
                ui_update(status="❌ ASR result audio hash mismatch; result quarantined")
                return
            else:
                pending_audio_evidence.pop(result.utterance_id, None)
    evidence = result.diagnostics.get("evidence", {})
    if evidence and (not evidence.get("raw_accepted", True) or not evidence.get("model_input_accepted", True)):
        ui_update(status="⚠️ Audio review storage limit reached or writer unavailable; check this session's recording summary")
    quality = result.diagnostics.get("decoder", {})
    if (result.partial and (quality.get("max_compression_ratio") or 0) > 2.9
            and is_repetitive_text(result.text)):
        # A pathological preview loop is not useful live text or MT input.
        # Retain its diagnostic audio and wait for authoritative final ASR.
        record_asr_decision(result, result.text, "partial_review_only", "Repetitive high-compression preview held for final decoding")
        clear_inflight_partial_marker(result.utterance_id)
        return
    if result.partial and ring_scheduler is not None:
        with ring_scheduler.lock:
            if result.utterance_id in ring_scheduler.final_submitted:
                return
    if not result.partial:
        with pending_stage_asr_lock:
            pending_stage_asr.discard(result.utterance_id)
        if ring_scheduler is not None:
            with ring_scheduler.lock:
                ring_scheduler.final_submitted.discard(result.utterance_id)
                ring_scheduler.cancelled_partials.discard(result.utterance_id)
    if not queue_put_asr_result(asr_results, result, "supervised-asr") and not result.partial:
        clear_final_asr_pending(result.utterance_id)


def receive_translation_stage_result(result: TranslationResult) -> None:
    if isinstance(result, SelectionResult):
        publish_selection_result(result)
        return
    optional = lambda queued: queued.stream_delta or queued.partial
    while not shutdown_event.is_set():
        if _put_without_waiting(translation_stage_results, result, optional):
            return
        if optional(result):
            return
        # Only this dedicated result-delivery thread may wait for committed
        # output capacity. Capture/UI submit and the model watchdog remain
        # independent; accepted final text must never be silently discarded.
        try:
            translation_stage_results.put(result, timeout=0.1)
            return
        except Full:
            continue


def start_asr_stage() -> None:
    global asr_stage
    if asr_stage is not None:
        return
    asr_stage = ProcessStage("asr", initialize_asr_stage, handle_asr_stage_request,
                             initargs=(vars(args).copy(),),
                             capacity=args.inference_queue_size,
                             startup_timeout=args.startup_timeout,
                             request_timeout=args.inference_timeout,
                             on_event=on_inference_event, on_result=receive_asr_stage_result)
    asr_stage.start()


def submit_asr_stage(utterance_id: int, pcm: np.ndarray, partial: bool, provenance: dict | None = None) -> bool:
    global asr_request_sequence
    if asr_stage is None:
        return False
    if not partial:
        mark_final_asr_pending(utterance_id)
        with pending_stage_asr_lock:
            pending_stage_asr.add(utterance_id)
    snapshot = np.ascontiguousarray(pcm, dtype=np.int16).copy()
    with pending_stage_asr_lock:
        asr_request_sequence += 1
        metadata = {"request_id": f"u{utterance_id}-{'p' if partial else 'f'}-{asr_request_sequence}",
                    "session_id": runtime_session_id, "samples": int(snapshot.size),
                    "source_language": source_language_code(), "partial": partial,
                    "pcm_sha256": hashlib.sha256(snapshot.astype("<i2", copy=False).tobytes()).hexdigest(),
                    **(provenance or {})}
        previous = pending_audio_evidence.get(utterance_id)
        pending_audio_evidence[utterance_id] = (snapshot, metadata)
    accepted = asr_stage.submit((utterance_id, partial, snapshot, metadata),
                                partial=partial, key=utterance_id)
    if accepted:
        with pending_stage_asr_lock:
            while len(pending_audio_evidence) > max(32, args.inference_queue_size * 2):
                pending_audio_evidence.pop(next(iter(pending_audio_evidence)))
    else:
        with pending_stage_asr_lock:
            current = pending_audio_evidence.get(utterance_id)
            if current is not None and current[1]["request_id"] == metadata["request_id"]:
                if previous is None:
                    pending_audio_evidence.pop(utterance_id, None)
                else:
                    pending_audio_evidence[utterance_id] = previous
    if not accepted and not partial:
        if audio_evidence is not None:
            audio_evidence.save_clip(snapshot, metadata, ["asr_request_rejected"], label=metadata["request_id"])
        with pending_stage_asr_lock:
            pending_stage_asr.discard(utterance_id)
        clear_final_asr_pending(utterance_id)
        ui_update(status=f"❌ ASR request not queued: {asr_stage.state}; capture continues. Check worker status/backlog.")
    return accepted


def update_capture_status(feeder: RingAudioFeeder) -> None:
    received = feeder.captured_samples / SAMPLE_RATE
    age = time.monotonic() - feeder.last_input_at if feeder.last_input_at else float("inf")
    if age > 2.0:
        state, detail = "no input", "No PCM received in 2s; check selected input/LitLink routing and microphone permission"
    else:
        level = getattr(feeder.endpoint, "last_dbfs", -100.0)
        speech = "speech" if feeder.endpoint.in_speech else "silence"
        state, detail = "listening", f"{received:.1f}s received | {level:.0f} dBFS | {speech} | dropped {feeder.dropped_samples} samples"
    if audio_evidence is not None:
        recording = audio_evidence.snapshot
        if recording["error"] or recording["rejected_capture_samples"] or recording["rejected_artifacts"]:
            detail += f" | REVIEW WARNING: dropped {recording['rejected_capture_samples']} recording samples; rejected {recording['rejected_artifacts']} clips"
    update_stage_status("capture", state, detail)
    for name, stage in (("asr", asr_stage), ("translation", translation_stage)):
        if stage is not None:
            snapshot = stage.snapshot
            error = brief_worker_detail(snapshot["error"])
            update_stage_status(name, snapshot["state"], error or f"queued {snapshot['pending']}; active {snapshot['inflight_key']}")


def asr_process_main(input_q, output_q) -> None:
    init_stt_runtime_only()
    warmup_stt_runtime_only(report=False)
    while True:
        task = input_q.get()
        if task is None:
            break
        try:
            result = asr_result_from_task(task)
            if result is not None:
                output_q.put(result)
            else:
                cleanup_audio_task_payload(task)
        finally:
            if task.slot_id is None:
                cleanup_audio_task_payload(task)


def start_asr_process() -> None:
    global asr_ipc_ctx, asr_ipc_in, asr_ipc_out, asr_process
    if asr_process is not None:
        return
    while True:
        try:
            _ = asr_slot_free.get_nowait()
        except Empty:
            break
    asr_slot_shms.clear()
    for slot_id in range(asr_slot_count):
        shm = shared_memory.SharedMemory(create=True, size=asr_slot_size)
        asr_slot_shms[slot_id] = shm
        asr_slot_free.put_nowait(slot_id)
    asr_ipc_ctx = mp.get_context("spawn")
    asr_ipc_in = asr_ipc_ctx.Queue(maxsize=max(1, args.audio_queue_size))
    asr_ipc_out = asr_ipc_ctx.Queue(maxsize=max(1, args.audio_queue_size * 2))
    asr_process = asr_ipc_ctx.Process(target=asr_process_main, args=(asr_ipc_in, asr_ipc_out), daemon=True)
    asr_process.start()


def close_mp_queue(queue_obj) -> None:
    if queue_obj is None:
        return
    try:
        queue_obj.close()
    except Exception:
        pass
    try:
        queue_obj.join_thread()
    except Exception:
        pass


def stop_asr_process() -> None:
    global asr_process, asr_ipc_in, asr_ipc_out, asr_ipc_ctx, asr_stage
    if asr_stage is not None:
        asr_stage.close(timeout=2)
        asr_stage = None
    if asr_ipc_in is not None:
        while True:
            try:
                leftover = asr_ipc_in.get_nowait()
            except Empty:
                break
            if leftover is not None:
                cleanup_audio_task_payload(leftover)
        try:
            asr_ipc_in.put_nowait(None)
        except Exception:
            pass
    if asr_process is not None:
        asr_process.join(timeout=2)
        if asr_process.is_alive():
            asr_process.terminate()
            asr_process.join(timeout=1)
    if asr_ipc_out is not None:
        while True:
            try:
                leftover = asr_ipc_out.get_nowait()
            except Empty:
                break
            if isinstance(leftover, AsrResult):
                cleanup_audio_result_payload(leftover)
    for shm in asr_slot_shms.values():
        try:
            shm.close()
        finally:
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
    asr_slot_shms.clear()
    while True:
        try:
            _ = asr_slot_free.get_nowait()
        except Empty:
            break
    close_mp_queue(asr_ipc_in)
    close_mp_queue(asr_ipc_out)
    asr_process = None
    asr_ipc_in = None
    asr_ipc_out = None
    asr_ipc_ctx = None


def asr_worker() -> None:
    while not shutdown_event.is_set():
        try:
            task = audio_tasks.get(timeout=0.1)
        except Empty:
            continue
        try:
            if args.debug:
                payload_bytes = task.num_bytes if task.num_bytes else (len(task.raw) if task.raw is not None else 0)
                log_debug(f"ASR task partial={task.partial} utterance_id={task.utterance_id} bytes={payload_bytes}")
            if asr_stage is not None:
                pcm = np.frombuffer(task_audio_bytes(task), dtype=np.int16).copy()
                submit_asr_stage(task.utterance_id, pcm, task.partial)
                cleanup_audio_task_payload(task)
            elif asr_ipc_in is not None:
                queue_put_audio_task(asr_ipc_in, task, "asr-audio", has_task_done=False)
            else:
                result = asr_result_from_task(task)
                if result is not None:
                    queue_put_asr_result(asr_results, result, "asr-result")
        except Exception as exc:
            ui_update(status=f"❌ ASR pipeline error: {exc}")
        finally:
            audio_tasks.task_done()


def asr_result_worker() -> None:
    if asr_ipc_out is None:
        return
    while not shutdown_event.is_set():
        try:
            result = asr_ipc_out.get(timeout=0.1)
        except Empty:
            continue
        try:
            cleanup_audio_result_payload(result)
            safe_result = AsrResult(
                utterance_id=result.utterance_id,
                text=result.text,
                partial=result.partial,
            )
            queue_put_asr_result(asr_results, safe_result, "asr-result")
        except Exception as exc:
            ui_update(status=f"❌ ASR result bridge error: {exc}")


def coordinator_worker() -> None:
    revision_cache: dict[int, int] = {}
    partial_cache: dict[int, str] = {}
    partial_source_spans: dict[int, tuple | None] = {}
    partial_display_cache: dict[int, str] = {}
    stable_translation_cache: dict[int, str] = {}
    partial_translation_candidates: dict[int, PartialTranslationCandidate] = {}
    stable_locks: dict[int, StableWordLock] = {}
    partial_mt_counters = {
        "queued": 0,
        "skipped_unstable": 0,
        "skipped_incomplete": 0,
        "skipped_unchanged": 0,
        "skipped_too_short": 0,
        "stale_candidate_removed": 0,
    }
    partial_mt_last_debug_at = 0.0

    def count_partial_mt(reason: str) -> None:
        partial_mt_counters[reason] = partial_mt_counters.get(reason, 0) + 1

    def maybe_log_partial_mt_counters(force: bool = False) -> None:
        nonlocal partial_mt_last_debug_at
        if not args.debug:
            return
        now = time.perf_counter()
        if not force and now - partial_mt_last_debug_at < 2.0:
            return
        partial_mt_last_debug_at = now
        summary = ", ".join(f"{key}={value}" for key, value in sorted(partial_mt_counters.items()))
        log_debug(f"partial MT counters: {summary}")

    # Watchdog state uses dedicated names so it cannot collide with partial
    # timestamps, trace values, revision caches, or other coordinator locals.
    watchdog_partial_first_seen_by_utt: dict[int, float] = {}
    watchdog_partial_last_seen_by_utt: dict[int, float] = {}
    watchdog_forced_final_by_utt: set[int] = set()
    watchdog_closed_by_utt: set[int] = set()
    finalized_utterances: set[int] = set()
    recent_final_transcripts: list[tuple[float, int, str, tuple | None]] = []

    global current_partial_text, current_partial_display_text, current_partial_display_locked_prefix
    global current_partial_stability, current_partial_last_change_ms, current_partial_locked_prefix, current_partial_locked_ratio, current_partial_revision, current_partial_utterance_id

    def watchdog_is_utterance_active(utterance_id: int) -> bool:
        endpoint = ring_endpoint
        if endpoint is None:
            return False
        try:
            return bool(endpoint.in_speech and endpoint.utterance_id == utterance_id)
        except Exception:
            return False

    def watchdog_clear_utterance(utterance_id: int) -> None:
        watchdog_partial_first_seen_by_utt.pop(utterance_id, None)
        watchdog_partial_last_seen_by_utt.pop(utterance_id, None)
        watchdog_forced_final_by_utt.discard(utterance_id)
        watchdog_closed_by_utt.discard(utterance_id)

    def watchdog_close_utterance(utterance_id: int, reason: str) -> None:
        partial_cache.pop(utterance_id, None)
        partial_source_spans.pop(utterance_id, None)
        partial_display_cache.pop(utterance_id, None)
        stable_translation_cache.pop(utterance_id, None)
        partial_translation_candidates.pop(utterance_id, None)
        stable_locks.pop(utterance_id, None)
        revision_cache.pop(utterance_id, None)
        watchdog_clear_utterance(utterance_id)
        watchdog_closed_by_utt.add(utterance_id)
        reset_current_partial_state(utterance_id)
        trace_drop(utterance_id)
        with ui_lock:
            ui_remove_panel_entry("left", utterance_id)
            render_dashboard()
        if args.debug:
            log_debug(f"watchdog closed utterance={utterance_id} reason={reason}")

    def selected_partial_translation_source(text: str, locked_prefix: str) -> str:
        if args.partial_translation_policy == "stable" or args.translation_stable_only:
            return normalize_text(locked_prefix)
        return normalize_text(text)

    def duplicate_recent_final_utterance_id(text: str, span: tuple | None, now: float) -> int | None:
        nonlocal recent_final_transcripts
        window_s = max(0.0, float(args.final_dedupe_window_ms) / 1000.0)
        if window_s <= 0:
            return None
        recent_final_transcripts = [
            (ts, utt, previous_text, previous_span)
            for ts, utt, previous_text, previous_span in recent_final_transcripts
            if now - ts <= window_s
        ]
        for _ts, utt, previous_text, previous_span in reversed(recent_final_transcripts):
            if (final_transcript_audio_spans_overlap(previous_span, span)
                    and final_transcripts_are_near_duplicates(previous_text, text)):
                return utt
        return None

    def note_recent_final_transcript(utterance_id: int, text: str, span: tuple | None, now: float) -> None:
        nonlocal recent_final_transcripts
        window_s = max(0.0, float(args.final_dedupe_window_ms) / 1000.0)
        if window_s <= 0:
            return
        recent_final_transcripts.append((now, utterance_id, text, span))
        recent_final_transcripts = [
            (ts, utt, previous_text, previous_span)
            for ts, utt, previous_text, previous_span in recent_final_transcripts
            if now - ts <= window_s
        ]

    def note_partial_translation_candidate(
        utterance_id: int,
        text: str,
        locked_prefix: str,
        revision: int,
        now: float,
    ) -> None:
        previous = partial_translation_candidates.get(utterance_id)
        stable_text = selected_partial_translation_source(text, locked_prefix)
        previous_stable_text = previous.stable_text if previous is not None else ""
        stable_changed_at = (
            previous.stable_changed_at
            if previous is not None and previous_stable_text == stable_text
            else now
        )
        last_sent_text = ""
        if previous is not None and stable_text.startswith(previous.last_sent_text):
            last_sent_text = previous.last_sent_text
        partial_translation_candidates[utterance_id] = PartialTranslationCandidate(
            text=text,
            locked_prefix=locked_prefix,
            revision=revision,
            stable_text=stable_text,
            stable_changed_at=stable_changed_at,
            last_sent_text=last_sent_text,
        )

    def choose_stable_partial_translation_text(candidate: PartialTranslationCandidate) -> str:
        return partial_translation_clump(candidate.stable_text)

    def sweep_partial_translations() -> None:
        if not args.stream_audio_translation or args.partial_translation_policy == "off":
            return
        quiet_seconds = max(0.0, float(args.partial_translation_stabilize_seconds))
        min_initial_words = max(1, int(args.stable_translation_min_words))
        min_delta_words = max(1, int(args.partial_translation_min_delta_words))
        now = time.perf_counter()
        for utterance_id, candidate in list(partial_translation_candidates.items()):
            if utterance_id not in partial_cache:
                partial_translation_candidates.pop(utterance_id, None)
                count_partial_mt("stale_candidate_removed")
                maybe_log_partial_mt_counters()
                continue
            translation_text = choose_stable_partial_translation_text(candidate)
            if not translation_text or translation_text == candidate.last_sent_text:
                count_partial_mt("skipped_unchanged" if translation_text else "skipped_incomplete")
                maybe_log_partial_mt_counters()
                continue
            if now - candidate.stable_changed_at < quiet_seconds:
                count_partial_mt("skipped_unstable")
                maybe_log_partial_mt_counters()
                continue
            previously_sent_words = candidate.last_sent_text.split()
            translation_words = translation_text.split()
            new_stable_words = max(0, len(translation_words) - len(previously_sent_words))
            if candidate.last_sent_text and translation_text.startswith(candidate.last_sent_text):
                if new_stable_words < min_delta_words:
                    count_partial_mt("skipped_too_short")
                    maybe_log_partial_mt_counters()
                    continue
            elif len(translation_words) < min_initial_words:
                count_partial_mt("skipped_too_short")
                maybe_log_partial_mt_counters()
                continue

            count_partial_mt("queued")
            maybe_log_partial_mt_counters(force=True)
            candidate.last_sent_text = translation_text
            stable_translation_cache[utterance_id] = translation_text
            enqueue_translation_task(
                TranslationTask(
                    source="audio",
                    text=translation_text,
                    src_lang=source_nllb_lang(),
                    beams=max(1, args.mt_beams - 1),
                    target_lang=target_language_code(),
                    forced_bos_id=forced_bos_for_target(target_language_code()),
                    utterance_id=utterance_id,
                    partial=True,
                    revision=candidate.revision,
                )
            )

    def sweep_stuck_partials() -> None:
        if not args.stuck_partial_watchdog:
            return
        timeout_s = max(0.5, args.stuck_partial_timeout_ms / 1000.0)
        now = time.perf_counter()
        for utterance_id, first_seen_at in list(watchdog_partial_first_seen_by_utt.items()):
            if utterance_id in watchdog_forced_final_by_utt or utterance_id in watchdog_closed_by_utt:
                continue
            partial_text = partial_cache.get(utterance_id, "")
            if not partial_text:
                watchdog_clear_utterance(utterance_id)
                continue
            if watchdog_is_utterance_active(utterance_id):
                continue
            if is_final_asr_pending(utterance_id):
                continue
            if now - first_seen_at < timeout_s:
                continue

            if args.stuck_partial_action == "close":
                watchdog_close_utterance(utterance_id, "stuck partial timeout")
                continue

            watchdog_forced_final_by_utt.add(utterance_id)
            fallback_text = sanitize_asr_text(partial_text, final=True)
            if fallback_text:
                if args.debug:
                    log_debug(f"watchdog force-finalizing utterance={utterance_id} from last partial")
                accepted = _put_without_waiting(
                    asr_results,
                    AsrResult(utterance_id=utterance_id, text=fallback_text, partial=False),
                    lambda queued: queued.partial,
                )
                if not accepted:
                    # This watchdog is also the queue consumer. Waiting for
                    # its own full queue would deadlock all ASR output. Keep
                    # the cached partial and retry after draining queued work.
                    watchdog_forced_final_by_utt.discard(utterance_id)
            else:
                watchdog_close_utterance(utterance_id, "stuck partial filtered empty")

    while not shutdown_event.is_set():
        try:
            result = asr_results.get(timeout=0.1)
        except Empty:
            sweep_stuck_partials()
            sweep_inflight_partial_markers(set(partial_cache.keys()))
            sweep_partial_translations()
            continue
        try:
            previous_text = partial_cache.get(result.utterance_id, "")
            clear_inflight_partial_marker(result.utterance_id)
            if result.partial:
                if result.utterance_id in finalized_utterances or result.utterance_id in watchdog_closed_by_utt:
                    if args.debug:
                        log_debug(f"dropped late partial ASR for closed utterance={result.utterance_id}")
                    continue
                result.text = sanitize_asr_text(result.text, final=False)
                if not result.text:
                    if not previous_text:
                        with ui_lock:
                            entry = ui_state["left_active"].get(result.utterance_id)
                            src = str(entry.get("src", "")) if entry else ""
                            if entry and "quality pending" in strip_ansi(src):
                                ui_remove_panel_entry("left", result.utterance_id)
                                render_dashboard()
                    continue
                partial_seen_time = time.perf_counter()
                watchdog_partial_first_seen_by_utt.setdefault(result.utterance_id, partial_seen_time)
                watchdog_partial_last_seen_by_utt[result.utterance_id] = partial_seen_time
                trace = trace_get(result.utterance_id)
                trace_update(
                    result.utterance_id,
                    asr_latest_partial_at=partial_seen_time,
                    **({} if "asr_first_partial_at" in trace else {"asr_first_partial_at": partial_seen_time}),
                )
                previous_span = partial_source_spans.get(result.utterance_id)
                current_span = final_transcript_audio_span(result)
                partial_text = merge_partial_transcript_with_provenance(previous_text, result.text,
                                                                          previous_span, current_span)
                revises_prefix = bool(
                    previous_text and previous_span is not None and current_span is not None
                    and previous_span[:3] == current_span[:3] and current_span[3] >= previous_span[3]
                    and not partial_text.startswith(previous_text)
                )
                if revises_prefix:
                    # A corrected full-window hypothesis must not remain hidden
                    # behind immutable words locked from the previous reading.
                    stable_locks.pop(result.utterance_id, None)
                    partial_display_cache.pop(result.utterance_id, None)
                    partial_translation_candidates.pop(result.utterance_id, None)
                    stable_translation_cache.pop(result.utterance_id, None)
                if current_span is None or previous_span is None or current_span[3] >= previous_span[3]:
                    partial_source_spans[result.utterance_id] = current_span
                prev_words = previous_text.split()
                cur_words = partial_text.split()
                prefix_words = longest_common_prefix_word_count(prev_words, cur_words) if previous_text else 0
                overlap_words = longest_suffix_prefix_word_overlap(prev_words, cur_words) if previous_text else 0
                similarity = word_sequence_similarity(prev_words, cur_words) if previous_text else 0.0
                changed_tail_words = max(0, len(cur_words) - prefix_words)
                materially_changed = (
                    not previous_text
                    or similarity < 0.72
                    or changed_tail_words >= 3
                    or abs(len(cur_words) - len(prev_words)) >= 3
                    or (partial_text != previous_text and prefix_words == 0 and overlap_words == 0)
                )
                if anti_repeat_enabled() or args.partial_translation_policy == "stable":
                    lock = stable_locks.setdefault(
                        result.utterance_id,
                        StableWordLock(args.stable_lock_confirmations, args.stable_mutable_tail_words),
                    )
                    locked_prefix, _mutable_tail, locked_ratio = lock.update(partial_text)
                else:
                    locked_prefix = common_prefix_text(previous_text, partial_text) if previous_text else ""
                    locked_prefix_words = locked_prefix.split() if locked_prefix else []
                    locked_ratio = (len(locked_prefix_words) / max(1, len(cur_words))) if cur_words else 0.0
                if previous_text and not materially_changed:
                    stability_boost = 2 if similarity >= 0.9 and changed_tail_words <= 1 else 1
                    _text, previous_stability, previous_last_change, _ratio = partial_endpoint_snapshot(result.utterance_id)
                    partial_stability = min(8, previous_stability + stability_boost)
                    partial_last_change_ms = previous_last_change + max(1, FRAME_MS * stability_boost)
                else:
                    partial_stability = 0 if not previous_text else 1
                    partial_last_change_ms = 0.0
                if args.debug and previous_text:
                    log_debug(
                        f"partial similarity={similarity:.2f} prefix_words={prefix_words} overlap_words={overlap_words} locked_ratio={locked_ratio:.2f} changed_tail={changed_tail_words} stability={partial_stability}"
                    )
                previous_display_text = partial_display_cache.get(result.utterance_id, "")
                display_text = partial_display_source_text(partial_text, locked_prefix)
                if previous_display_text and not display_text:
                    display_text = previous_display_text
                elif (
                    previous_display_text
                    and display_text
                    and not display_text.startswith(previous_display_text)
                    and args.partial_display_policy != "latest"
                ):
                    display_text = previous_display_text
                display_locked_prefix = locked_prefix if args.partial_display_policy == "latest" else display_text

                partial_cache[result.utterance_id] = partial_text
                partial_display_cache[result.utterance_id] = display_text
                previous_revision = revision_cache.get(result.utterance_id, 0)
                display_changed = revises_prefix or display_text != previous_display_text
                revision = previous_revision + 1 if display_changed else previous_revision
                revision_cache[result.utterance_id] = revision
                # Capture may start a newer utterance while this result is
                # sanitized/formatted. Recheck ownership at the actual commit.
                # The same lock makes capture resets and endpoint snapshots
                # atomic with this complete publication.
                with current_partial_state_lock:
                    if current_partial_utterance_id is None or result.utterance_id >= current_partial_utterance_id:
                        current_partial_text = partial_text
                        current_partial_display_text = display_text
                        current_partial_display_locked_prefix = display_locked_prefix
                        current_partial_stability = partial_stability
                        current_partial_last_change_ms = partial_last_change_ms
                        current_partial_locked_prefix = locked_prefix
                        current_partial_locked_ratio = locked_ratio
                        current_partial_utterance_id = result.utterance_id
                        current_partial_revision = revision
                display_is_inflight = not bool(display_text) or (args.partial_display_policy == "latest" and not display_locked_prefix)
                with ui_lock:
                    ui_set_panel_entry(
                        "left",
                        result.utterance_id,
                        "src",
                        format_partial_transcript_for_ui(display_text, display_locked_prefix, inflight=display_is_inflight),
                        final=False,
                        revision=revision,
                    )
                    render_dashboard()
                if args.stream_audio_translation and args.partial_translation_policy != "off":
                    if display_text:
                        note_partial_translation_candidate(
                            result.utterance_id,
                            display_text,
                            display_locked_prefix,
                            revision,
                            partial_seen_time,
                        )
                    else:
                        partial_translation_candidates.pop(result.utterance_id, None)
                    sweep_partial_translations()
                continue

            final_text = prepare_final_transcript_for_translation(result.text or "", previous_text)
            if result.utterance_id in finalized_utterances:
                if args.debug:
                    log_debug(f"dropped duplicate final ASR for already-finalized utterance={result.utterance_id}")
                continue
            if not final_text:
                record_asr_decision(result, "", "empty_final", "No final text remained after ASR filtering")
                if args.empty_final_action == "close":
                    watchdog_close_utterance(result.utterance_id, "empty final ASR")
                    continue
                if args.empty_final_action == "keep":
                    if args.debug:
                        log_debug("Coordinator received empty final text; keeping active partial")
                    continue
                watchdog_close_utterance(result.utterance_id, "empty final text after fallback")
                continue

            final_seen_at = time.perf_counter()
            final_audio_span = final_transcript_audio_span(result)
            duplicate_utterance_id = duplicate_recent_final_utterance_id(final_text, final_audio_span, final_seen_at)
            if duplicate_utterance_id is not None:
                record_asr_decision(result, final_text, "duplicate_audio_final",
                                    f"Repeated text over shared audio from utterance {duplicate_utterance_id}")
                finalized_utterances.add(result.utterance_id)
                watchdog_close_utterance(
                    result.utterance_id,
                    f"duplicate final ASR of utterance={duplicate_utterance_id}",
                )
                if args.debug:
                    log_debug(
                        f"dropped near-duplicate final utterance={result.utterance_id}; previous={duplicate_utterance_id}"
                    )
                continue

            revision = revision_cache.get(result.utterance_id, 0) + 1
            revision_cache[result.utterance_id] = revision
            trace_update(result.utterance_id, asr_final_at=final_seen_at)
            partial_cache.pop(result.utterance_id, None)
            partial_source_spans.pop(result.utterance_id, None)
            partial_display_cache.pop(result.utterance_id, None)
            stable_translation_cache.pop(result.utterance_id, None)
            partial_translation_candidates.pop(result.utterance_id, None)
            stable_locks.pop(result.utterance_id, None)
            watchdog_clear_utterance(result.utterance_id)
            finalized_utterances.add(result.utterance_id)
            note_recent_final_transcript(result.utterance_id, final_text, final_audio_span, final_seen_at)
            reset_current_partial_state(result.utterance_id)
            with ui_lock:
                ui_set_panel_entry("left", result.utterance_id, "src", f"🗣️ {source_label()}: {color_left_tr(final_text)}", final=True, revision=revision)
                render_dashboard()
            with session_log_lock:
                audio_final_transcripts[result.utterance_id] = final_text
            record_asr_decision(result, final_text, "accepted_final")
            translation_accepted = enqueue_translation_task(
                TranslationTask(
                    source="audio",
                    text=final_text,
                    src_lang=source_nllb_lang(),
                    beams=max(1, args.mt_beams),
                    target_lang=target_language_code(),
                    forced_bos_id=forced_bos_for_target(target_language_code()),
                    utterance_id=result.utterance_id,
                    partial=False,
                    revision=revision,
                )
            )
            if translation_accepted is False:
                record_asr_decision(result, final_text, "translation_rejected", "Final transcript retained; translation was not queued")
            revision_cache.pop(result.utterance_id, None)
        except Exception as exc:
            ui_update(status=f"❌ Coordinator error: {exc}")
        finally:
            if not result.partial:
                clear_final_asr_pending(result.utterance_id)
            asr_results.task_done()

def emit_best_effort_translation_result(emit, result: TranslationResult) -> None:
    """
    Streaming deltas are responsiveness hints, not authoritative output. Drop them
    instead of letting IPC back-pressure delay final MT/TTS results.
    """
    try:
        emit(result, block=False)
    except TypeError:
        emit(result)
    except Full:
        if args.debug:
            log_debug("dropped streaming MT delta because result queue is full")


def should_emit_mt_delta(accumulated: str, last_emitted: str, last_emit_at: float, *, force: bool = False) -> bool:
    current = strip_model_thinking(accumulated)
    if not current or current == last_emitted:
        return False
    if force or not last_emitted:
        return True

    changed_chars = max(0, len(current) - len(last_emitted))
    if changed_chars >= max(1, args.mt_delta_min_chars):
        return True
    if current.endswith((".", "?", "!", ",", ";", ":")) and changed_chars >= max(4, args.mt_delta_min_chars // 2):
        return True

    elapsed_ms = int((time.perf_counter() - last_emit_at) * 1000)
    return elapsed_ms >= max(1, args.mt_delta_min_ms) and changed_chars > 0


def emit_mt_delta(
    emit,
    task: TranslationTask,
    text: str,
    chunk_index: int,
    total_chunks: int,
    delta_index: int,
) -> None:
    emit_best_effort_translation_result(
        emit,
        TranslationResult(
            source=task.source,
            text=normalize_text(text),
            target_lang=task.target_lang,
            utterance_id=task.utterance_id,
            partial=task.partial,
            revision=task.revision,
            final=False,
            stream_delta=True,
            delta_index=delta_index,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
        ),
    )


def translate_chunk_with_optional_deltas(
    task: TranslationTask,
    chunk: str,
    chunk_index: int,
    total_chunks: int,
    forced_bos_id: int,
    emit,
) -> str:
    if active_translation_backend() != "mlx-lm" or not args.stream_mt_deltas:
        return translate_chunk(chunk, task.src_lang, forced_bos_id, task.beams)

    pieces: list[str] = []
    last_emitted = ""
    last_emit_at = 0.0
    delta_index = 0
    for piece in iter_translate_chunk_mlx_lm(chunk, task.src_lang, forced_bos_id, task.beams):
        pieces.append(piece)
        accumulated = strip_model_thinking("".join(pieces), final=False)
        if should_emit_mt_delta(accumulated, last_emitted, last_emit_at):
            last_emitted = accumulated
            last_emit_at = time.perf_counter()
            delta_index += 1
            emit_mt_delta(emit, task, last_emitted, chunk_index, total_chunks, delta_index)
    return strip_model_thinking("".join(pieces))


def emit_translation_results(task: TranslationTask, emit) -> None:
    if args.debug and mp.current_process().name == "MainProcess":
        log_debug(f"MT task source={task.source} partial={task.partial} rev={task.revision} chars={len(task.text)}")
    source_chunks = iter_translation_chunks(task.text)
    if not source_chunks:
        emit(
            TranslationResult(
                source=task.source,
                text="",
                target_lang=task.target_lang,
                utterance_id=task.utterance_id,
                partial=task.partial,
                revision=task.revision,
                error="Translation returned empty output",
            )
        )
        return

    out_parts: list[str] = []
    total_chunks = len(source_chunks)
    forced_bos_id = forced_bos_for_target(task.target_lang)
    for idx, chunk in enumerate(source_chunks, start=1):
        with translation_lock:
            translated_chunk = translate_chunk_with_optional_deltas(task, chunk, idx, total_chunks, forced_bos_id, emit)
        if not translated_chunk:
            continue
        out_parts.append(translated_chunk)

        if task.source == "text" and args.text_stream_tts:
            emit(
                TranslationResult(
                    source=task.source,
                    text=translated_chunk,
                    target_lang=task.target_lang,
                    utterance_id=task.utterance_id,
                    partial=task.partial,
                    revision=task.revision,
                    final=False,
                    chunk_index=idx,
                    total_chunks=total_chunks,
                )
            )

    if not out_parts:
        emit(
            TranslationResult(
                source=task.source,
                text="",
                target_lang=task.target_lang,
                utterance_id=task.utterance_id,
                partial=task.partial,
                revision=task.revision,
                error="Translation returned empty output",
            )
        )
        return

    emit(
        TranslationResult(
            source=task.source,
            text=normalize_text(" ".join(out_parts)),
            target_lang=task.target_lang,
            utterance_id=task.utterance_id,
            partial=task.partial,
            revision=task.revision,
            final=True,
            total_chunks=total_chunks,
        )
    )


def handle_translation_result(result: TranslationResult) -> None:
    if result.error:
        ui_update(status=f"❌ {result.error}")
        return
    result.text = strip_model_thinking(result.text)
    if not result.text:
        return
    if result.source == "audio":
        task_probe = TranslationTask(
            source="audio",
            text="",
            src_lang=source_nllb_lang(),
            beams=1,
            target_lang=result.target_lang,
            utterance_id=result.utterance_id,
            partial=result.partial,
            revision=result.revision,
        )
        if is_stale_audio_translation(task_probe):
            return
        if result.stream_delta:
            now = time.perf_counter()
            trace = trace_get(result.utterance_id)
            trace_update(
                result.utterance_id,
                mt_latest_partial_at=now,
                **({} if "mt_first_delta_at" in trace else {"mt_first_delta_at": now}),
            )
            with ui_lock:
                if int(result.utterance_id or 0) not in ui_state["left_active"]:
                    return
                updated = ui_set_panel_entry("left", int(result.utterance_id or 0), "dst", f"🌍 {target_label()}~: {color_left_en_partial(result.text)}", final=False, revision=result.revision)
                if updated:
                    render_dashboard()
            return
        if result.partial:
            trace_update(result.utterance_id, mt_latest_partial_at=time.perf_counter())
            text = f"🌍 {target_label()}~: {color_left_en_partial(result.text)}"
            final_flag = False
        else:
            trace_update(result.utterance_id, mt_final_at=time.perf_counter())
            text = f"🌍 {target_label()}: {color_left_en(result.text)}"
            final_flag = True
        with ui_lock:
            if result.partial and int(result.utterance_id or 0) not in ui_state["left_active"]:
                return
            updated = ui_set_panel_entry("left", int(result.utterance_id or 0), "dst", text, final=final_flag, revision=result.revision)
            if updated:
                render_dashboard()
        if not result.partial and result.utterance_id is not None:
            log_key = (int(result.utterance_id), int(result.revision))
            transcription_text = ""
            should_schedule_log = False
            with session_log_lock:
                if log_key not in logged_audio_translation_keys:
                    transcription_text = audio_final_transcripts.get(int(result.utterance_id), "")
                    logged_audio_translation_keys.add(log_key)
                    should_schedule_log = bool(transcription_text)
            if should_schedule_log:
                schedule_final_transcription_log(int(result.utterance_id), int(result.revision), transcription_text, result.text)
            if args.speak_audio_translation:
                trace_update(result.utterance_id, tts_enqueued_at=time.perf_counter())
                queue_put_tts_task(
                    tts_tasks,
                    TtsTask(text=result.text, wav_out=args.output_wav, lang_hint=target_language_code(), source="audio", utterance_id=result.utterance_id, revision=result.revision),
                    "tts",
                )
            elif args.latency_stats:
                status = format_latency_status(result.utterance_id)
                if status:
                    ui_update(status=f"⏱️ {status}")
                trace_drop(result.utterance_id)
            clear_audio_translation_revision(result.utterance_id, result.revision)
        return

    if result.stream_delta:
        with ui_lock:
            ui_set_panel_entry("right", int(result.utterance_id or 0), "dst", f"🌍 {source_label()}~[{result.chunk_index}/{result.total_chunks}]: {color_right_tr(result.text)}", final=False)
            render_dashboard()
        return

    if not result.final:
        with ui_lock:
            ui_set_panel_entry("right", int(result.utterance_id or 0), "dst", f"🌍 {source_label()}[{result.chunk_index}/{result.total_chunks}]: {color_right_tr(result.text)}", final=False)
            render_dashboard()
        if args.text_stream_tts:
            queue_put_tts_task(tts_tasks, TtsTask(text=result.text, wav_out=args.output_wav, lang_hint=result.target_lang, source="text", utterance_id=result.utterance_id, revision=result.revision), "tts")
        return

    if result.final and result.utterance_id is not None:
        log_key = (int(result.utterance_id), int(result.revision))
        input_text = ""
        with session_log_lock:
            if log_key not in logged_typed_translation_keys:
                input_text = typed_input_texts.get(int(result.utterance_id), "")
                logged_typed_translation_keys.add(log_key)
        if input_text:
            log_typed_translation(int(result.utterance_id), input_text, result.text)

    if not args.text_stream_tts:
        with ui_lock:
            ui_set_panel_entry("right", int(result.utterance_id or 0), "dst", f"🌍 {source_label()}: {color_right_tr(result.text)}", final=True)
            render_dashboard()
        queue_put_tts_task(tts_tasks, TtsTask(text=result.text, wav_out=args.output_wav, lang_hint=result.target_lang, source="text", utterance_id=result.utterance_id, revision=result.revision), "tts")
    else:
        with ui_lock:
            ui_set_panel_entry("right", int(result.utterance_id or 0), "dst", f"🌍 {source_label()}✓: {color_right_tr(result.text)}", final=True)
            render_dashboard()


def translation_worker() -> None:
    global translation_thread_error
    update_stage_status("translation", "loading", "thread mode; process isolation disabled")
    try:
        init_translation_runtime_only()
        warmup_translation_runtime_only(report=False, strict=True)
        update_stage_status("translation", "ready", "thread mode")
    except Exception as exc:
        translation_thread_error = str(exc)
        update_stage_status("translation", "failed", translation_thread_error)
        return
    while not shutdown_event.is_set():
        try:
            item = mt_tasks.get(timeout=0.1)
            task = unwrap_translation_task(item)
        except Empty:
            continue
        try:
            if is_stale_audio_translation(task):
                continue
            emit_translation_results(task, handle_translation_result)
        except Exception as exc:
            ui_update(status=f"❌ Translation pipeline error: {exc}")
        finally:
            mt_tasks.task_done()


def mt_process_main(input_q, output_q) -> None:
    try:
        init_translation_runtime_only()
        warmup_translation_runtime_only(report=False)
    except Exception as exc:
        output_q.put(
            TranslationResult(
                source="audio",
                text="",
                target_lang=target_language_code(),
                error=f"Translation process failed to start: {exc}",
            )
        )
        return
    while True:
        task = input_q.get()
        if task is None:
            break
        try:
            emit_translation_results(task, output_q.put)
        except Exception as exc:
            output_q.put(
                TranslationResult(
                    source=getattr(task, "source", "audio"),
                    text="",
                    target_lang=getattr(task, "target_lang", target_language_code()),
                    utterance_id=getattr(task, "utterance_id", None),
                    partial=getattr(task, "partial", False),
                    revision=getattr(task, "revision", 0),
                    error=f"Translation pipeline error: {exc}",
                )
            )


def start_mt_process() -> None:
    global translation_stage
    if translation_stage is not None:
        return
    translation_stage = ProcessStage("translation", initialize_translation_stage, handle_translation_stage_request,
                                     initargs=(vars(args).copy(),),
                                     capacity=args.inference_queue_size,
                                     startup_timeout=args.startup_timeout,
                                     request_timeout=args.inference_timeout,
                                     on_event=on_inference_event, on_result=receive_translation_stage_result)
    translation_stage.start()


def stop_mt_process() -> None:
    global mt_process, mt_ipc_in, mt_ipc_out, mt_ipc_ctx, translation_stage
    if translation_stage is not None:
        translation_stage.close(timeout=2)
        translation_stage = None
    if mt_ipc_in is not None:
        try:
            mt_ipc_in.put_nowait(None)
        except Exception:
            pass
    if mt_process is not None:
        mt_process.join(timeout=2)
        if mt_process.is_alive():
            mt_process.terminate()
            mt_process.join(timeout=1)
    close_mp_queue(mt_ipc_in)
    close_mp_queue(mt_ipc_out)
    mt_process = None
    mt_ipc_in = None
    mt_ipc_out = None
    mt_ipc_ctx = None


def translation_result_worker() -> None:
    while not shutdown_event.is_set():
        try:
            result = translation_stage_results.get(timeout=0.1)
        except Empty:
            continue
        try:
            handle_translation_result(result)
        except Exception as exc:
            ui_update(status=f"❌ Translation result error: {exc}")
        finally:
            translation_stage_results.task_done()


def tts_worker() -> None:
    global playback_guard_until
    while not shutdown_event.is_set():
        try:
            task = tts_tasks.get(timeout=0.1)
        except Empty:
            continue
        playback_started = False
        try:
            if task.source == "audio":
                trace_update(task.utterance_id, tts_start_at=time.perf_counter())
            if args.debug:
                log_debug(f"TTS task chars={len(task.text)} wav_out={task.wav_out}")
            engine = tts_best_available(task.text, task.wav_out, lang_hint=task.lang_hint)
            ui_update(status=f"🗣️ TTS engine: {engine} | WAV: {task.wav_out}")
            # Rendering can be slow. It is not playback, and must not mute or
            # reset active capture while the WAV is still being synthesized.
            if not args.duplex_capture:
                with audio_lock:
                    reset_active_capture_state()
            tts_playing.set()
            playback_started = True
            play_wav(task.wav_out)
            if task.source == "audio":
                trace_update(task.utterance_id, tts_done_at=time.perf_counter())
                if args.latency_stats:
                    status = format_latency_status(task.utterance_id, include_tts=True)
                    if status:
                        ui_update(status=f"⏱️ {status}")
                trace_drop(task.utterance_id)
        except Exception as exc:
            ui_update(status=f"❌ TTS pipeline error: {exc}")
        finally:
            playback_guard_until = (
                time.time() + (args.playback_guard_ms / 1000.0)
                if playback_started and not args.duplex_capture else 0.0
            )
            tts_playing.clear()
            tts_tasks.task_done()


def _assert_self_test(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_self_test() -> int:
    checks: list[tuple[str, object]] = []

    def check(name: str):
        def _wrap(fn):
            checks.append((name, fn))
            return fn
        return _wrap

    @check("MT queue preserves committed work")
    def _test_mt_queue() -> None:
        q: Queue[TranslationTask] = Queue(maxsize=1)
        q.put_nowait(
            TranslationTask(
                source="audio",
                text="partial",
                src_lang=source_nllb_lang(),
                beams=1,
                target_lang=target_language_code(),
                utterance_id=1,
                partial=True,
                revision=1,
            )
        )
        ok = queue_put_translation_task(
            q,
            TranslationTask(source="text", text="hello", src_lang=target_nllb_lang(), beams=1, target_lang=source_language_code(), utterance_id=2),
            "self-test-mt",
        )
        item = q.get_nowait()
        q.task_done()
        _assert_self_test(ok, "committed translation task was rejected")
        _assert_self_test(isinstance(item, TranslationTask) and item.source == "text", "speculative MT was not evicted")

    @check("ASR result queue preserves finals")
    def _test_asr_result_queue() -> None:
        q: Queue[AsrResult] = Queue(maxsize=1)
        q.put_nowait(AsrResult(utterance_id=1, text="partial", partial=True))
        ok = queue_put_asr_result(q, AsrResult(utterance_id=1, text="", partial=False), "self-test-asr")
        item = q.get_nowait()
        q.task_done()
        _assert_self_test(ok, "final ASR result was rejected")
        _assert_self_test(isinstance(item, AsrResult) and not item.partial, "partial ASR result was not evicted")

    @check("Final ASR pending gate blocks watchdog fallback")
    def _test_final_asr_pending_gate() -> None:
        scheduler = PriorityAsrScheduler()
        utterance_id = 4242
        pcm = np.ones(SAMPLE_RATE // 2, dtype=np.int16)
        try:
            ok = scheduler.submit(utterance_id, pcm, partial=False)
            _assert_self_test(ok, "final ASR task was not accepted")
            _assert_self_test(is_final_asr_pending(utterance_id), "final ASR was not marked pending")
            _priority, _seq, queued_utterance_id, partial, _queued_pcm = scheduler.q.get_nowait()
            scheduler.q.task_done()
            _assert_self_test(queued_utterance_id == utterance_id and not partial, "queued ASR task was not final")
        finally:
            clear_final_asr_pending(utterance_id)

    @check("Ring buffer retains exact newest samples across wraps and large callbacks")
    def _test_ring_sample_order() -> None:
        ring = NativeTimestampRingBuffer(0.003, sample_rate=8000)
        first = np.arange(17, dtype=np.int16)
        ring.write(0, first)
        first[:] = -1
        ring.write(2_125_000, np.arange(17, 40, dtype=np.int16))
        _assert_self_test(np.array_equal(ring.latest_ms(100), np.arange(16, 40)), "ring wrap lost or reordered retained samples")
        _assert_self_test(ring.total_samples == 24 and ring.frames[0][0] == 2_000_000, "partial eviction corrupted capacity or first timestamp")
        ring.write(5_000_000, np.arange(40, 90, dtype=np.int16))
        _assert_self_test(np.array_equal(ring.latest_ms(100), np.arange(66, 90)), "oversized capture discarded its newest tail")
        _assert_self_test(np.array_equal(ring.latest_ms(1), np.arange(82, 90)), "ring duration used the global rather than its own sample rate")
        _assert_self_test(ring.total_samples == 24 and ring.frames[0][0] == 8_250_000, "large capture retention timestamp is wrong")

    @check("Ring scheduler evicts only partials and preserves committed submission order")
    def _test_ring_scheduler_order() -> None:
        scheduler = PriorityAsrScheduler()
        scheduler.q = PriorityQueue(maxsize=4)
        pcm = np.ones(FRAME_SAMPLES, dtype=np.int16)
        ids = (101, 202, 303, 404)
        try:
            _assert_self_test(scheduler.submit(ids[0], pcm, partial=True), "first partial rejected")
            _assert_self_test(scheduler.submit(ids[1], pcm, partial=False), "first final rejected")
            _assert_self_test(scheduler.submit(ids[2], pcm, partial=True), "second partial rejected")
            _assert_self_test(scheduler.submit(ids[3], pcm, partial=False), "second final rejected")
            _assert_self_test(scheduler.submit(ids[0], pcm, partial=False), "promoted final rejected")
            _assert_self_test(scheduler.q.qsize() == 4, "promotion changed unrelated queue entries")
            _assert_self_test(scheduler._drop_one_queued_partial(), "remaining speculative task was not evicted")
            queued = []
            while not scheduler.q.empty():
                item = scheduler.q.get_nowait()
                queued.append((item[2], item[3]))
                scheduler.q.task_done()
            _assert_self_test(queued == [(202, False), (404, False), (101, False)], "committed finals were lost or reordered during partial eviction")
            _assert_self_test(scheduler.q.unfinished_tasks == 0, "eviction broke queue completion accounting")
        finally:
            for utterance_id in ids:
                clear_final_asr_pending(utterance_id)

    @check("Ring ingress assembles variable callbacks with ordered ADC timestamps")
    def _test_ring_ingress_callbacks() -> None:
        from types import SimpleNamespace
        from unittest.mock import patch

        class _Endpoint:
            def __init__(self):
                self.events = []
            def on_frame(self, timestamp_ns, pcm):
                self.events.append((timestamp_ns, pcm.copy()))
            def finalize(self, force=False):
                self.events.append("final")

        endpoint = _Endpoint()
        errors = []
        with patch.dict(globals(), {"shutdown_event": threading.Event(), "ui_update": lambda **kw: errors.append(kw)}):
            feeder = RingAudioFeeder(endpoint, queue_ms=200, clock_ns=lambda: 10_100_000_000)
            offset = 0
            for count in (113, 527, 800, 17, 143):
                pcm = np.arange(offset, offset + count, dtype=np.int16).reshape(-1, 1)
                jitter = 0.001 if offset else 0.0
                time_info = SimpleNamespace(inputBufferAdcTime=1.0 + offset / SAMPLE_RATE + jitter, currentTime=1.1 + offset / SAMPLE_RATE + jitter)
                feeder.callback(pcm, count, time_info, None)
                pcm[:] = -1  # capture buffers are reused as soon as callbacks return.
                offset += count
            packets = list(feeder.queue)
            _assert_self_test([packet.sequence for packet in packets] == list(range(5)), "callback splitting reordered frame sequence")
            _assert_self_test([packet.timestamp_ns for packet in packets] == [10_000_000_000 + n * 20_000_000 for n in range(5)], "ADC timestamps drifted across callback boundaries")
            feeder.request_finalize()
            feeder.close()
            feeder.worker()
            audio = [event[1] for event in endpoint.events if isinstance(event, tuple)]
            _assert_self_test(np.array_equal(np.concatenate(audio), np.arange(1600, dtype=np.int16)), "variable callbacks lost, duplicated, or retained mutable samples")
            _assert_self_test(endpoint.events[-1] == "final" and not errors, "ordered final boundary was lost or clean capture reported an error")

    @check("Forced ring flush owns its partial frame without leaking audio across utterances")
    def _test_ring_ingress_flush_partial_frame() -> None:
        from types import SimpleNamespace
        from unittest.mock import patch

        class _Scheduler:
            def __init__(self):
                self.finals = []
            def submit(self, utterance_id, pcm, partial):
                _assert_self_test(not partial, "short forced utterance unexpectedly submitted a partial")
                self.finals.append((utterance_id, pcm.copy()))
                return True

        class _Gate:
            def detect(self, pcm):
                _assert_self_test(pcm.size == FRAME_SAMPLES, "VAD received a truncated control frame")
                return True, -20.0, 1.0

        saved_partial = {key: value for key, value in globals().items() if key.startswith("current_partial_")}
        scheduler = _Scheduler()
        notices = []
        try:
            with patch.dict(globals(), {
                "shutdown_event": threading.Event(), "tts_playing": threading.Event(),
                "ui_update": lambda **kw: notices.append(kw), "latency_traces": {}, "inflight_partial_markers": {},
            }):
                endpoint = RingEndpointDetector(NativeTimestampRingBuffer(1), scheduler)
                endpoint.vad_gate = _Gate()
                feeder = RingAudioFeeder(endpoint, queue_ms=200, clock_ns=lambda: 10_100_000_000)
                feeder.callback(np.arange(100, dtype=np.int16), 100,
                                SimpleNamespace(inputBufferAdcTime=1.0, currentTime=1.1), None)
                feeder.request_finalize()
                feeder.callback(np.arange(100, 420, dtype=np.int16), 320,
                                SimpleNamespace(inputBufferAdcTime=1.00625, currentTime=1.10625), None)
                packets = [item for item in feeder.queue if item is not None]
                _assert_self_test([packet.valid_samples for packet in packets] == [100, 320], "forced flush left pre-barrier samples in live carry")
                _assert_self_test([packet.timestamp_ns for packet in packets] == [10_000_000_000, 10_006_250_000], "forced flush inserted synthetic time between captured samples")
                feeder.close()
                feeder.worker()
                _assert_self_test(len(scheduler.finals) == 2, "forced flush failed to split the two captured utterances")
                _assert_self_test(np.array_equal(scheduler.finals[0][1], np.arange(100, dtype=np.int16)), "pre-flush tail was lost, padded into ASR, or joined to later speech")
                _assert_self_test(np.array_equal(scheduler.finals[1][1], np.arange(100, 420, dtype=np.int16)), "post-flush speech inherited earlier samples")
                _assert_self_test(np.array_equal(endpoint.ring.latest_ms(1000), np.arange(420, dtype=np.int16)), "ring history contains synthetic flush padding or reordered samples")
                _assert_self_test(not notices, f"healthy forced flush produced capture errors: {notices}")
        finally:
            globals().update(saved_partial)

    @check("Ring ingress bounds large callbacks and reports sequence gaps before later audio")
    def _test_ring_ingress_drops() -> None:
        from unittest.mock import patch

        class _Endpoint:
            def __init__(self):
                self.events = []
            def on_frame(self, timestamp_ns, pcm):
                self.events.append((timestamp_ns, pcm.copy()))
            def finalize(self, force=False):
                self.events.append("final")

        endpoint = _Endpoint()
        notices = []
        with patch.dict(globals(), {"shutdown_event": threading.Event(), "ui_update": lambda **kw: notices.append(kw)}):
            feeder = RingAudioFeeder(endpoint, queue_ms=40, clock_ns=lambda: 1_000_000_000)
            pcm = np.arange(5 * FRAME_SAMPLES + 7, dtype=np.int16)
            feeder.callback(pcm, pcm.size, None, None)
            _assert_self_test(feeder.queued_frames == 2 and len(feeder.pending) == 7, "large callback exceeded the queue/sample carry budget")
            _assert_self_test([packet.sequence for packet in feeder.queue] == [3, 4], "large callback did not retain newest frames in order")
            feeder.close()  # Final seven real samples are retained; VAD-only padding never enters ASR/history.
            _assert_self_test(feeder.dropped_frames == 4 and feeder.dropped_samples == 4 * FRAME_SAMPLES, "queue eviction was not counted exactly")
            feeder.worker()
            audio = [event for event in endpoint.events if isinstance(event, tuple)]
            _assert_self_test(endpoint.events[0] == "final", "missing capture sequences were silently joined to the previous utterance")
            _assert_self_test(np.array_equal(np.concatenate([event[1] for event in audio]), pcm[1280:]), "large callback tail was lost, padded, or reordered")
            _assert_self_test(audio[-1][1].size == 7 and audio[1][0] - audio[0][0] == 20_000_000, "stopped sub-frame extent/timestamp is incorrect")
            _assert_self_test(len(notices) == 1 and "dropped=4 frames/1280 samples" in notices[0]["status"], "capture overflow was not reported")

    @check("Ring ingress separates device overruns and clock resets without repeated gaps")
    def _test_ring_ingress_overruns() -> None:
        from types import SimpleNamespace
        from unittest.mock import patch

        class _Endpoint:
            def __init__(self):
                self.events = []
            def on_frame(self, timestamp_ns, pcm):
                self.events.append((timestamp_ns, int(pcm[0])))
            def finalize(self, force=False):
                self.events.append("final")

        endpoint = _Endpoint()
        notices = []
        with patch.dict(globals(), {"shutdown_event": threading.Event(), "ui_update": lambda **kw: notices.append(kw)}):
            feeder = RingAudioFeeder(endpoint, queue_ms=200, clock_ns=lambda: 10_100_000_000)
            def send(count, value, adc, status=None):
                feeder.callback(np.full((count, 1), value, dtype=np.int16), count,
                                SimpleNamespace(inputBufferAdcTime=adc, currentTime=adc + 0.1), status)
            send(320, 1, 1.0)
            send(100, 2, 1.02)
            send(160, 3, 1.04, SimpleNamespace(input_overflow=True))
            send(160, 3, 1.05)
            send(320, 4, 0.5)  # PortAudio clock reset: re-anchor once and keep timestamps monotonic.
            send(320, 5, 0.52)
            packets = list(feeder.queue)
            _assert_self_test([packet.discontinuity for packet in packets] == [False, True, True, False], "overrun/reset boundary was lost or repeated on healthy callbacks")
            _assert_self_test([int(packet.pcm[0]) for packet in packets] == [1, 3, 4, 5], "incomplete audio crossed a capture discontinuity")
            _assert_self_test(feeder.dropped_samples == 100 and feeder.input_overruns == 1, "overrun carry loss was not counted")
            _assert_self_test([packet.timestamp_ns for packet in packets] == [10_000_000_000, 10_040_000_000, 10_060_000_000, 10_080_000_000], "clock reset caused overlapping or reversed capture timestamps")
            feeder.close()
            feeder.worker()
            _assert_self_test(endpoint.events == [(10_000_000_000, 1), "final", (10_040_000_000, 3), "final", (10_060_000_000, 4), (10_080_000_000, 5), "final"], "device gaps were not finalized before later audio")
            _assert_self_test(notices and "input overruns=1" in notices[0]["status"], "device overrun was not surfaced")

    @check("Ring ingress preserves flush barriers and bounded controls during repeated eviction")
    def _test_ring_ingress_flush_and_close() -> None:
        from unittest.mock import patch

        class _Endpoint:
            def __init__(self):
                self.events = []
            def on_frame(self, timestamp_ns, pcm):
                self.events.append(int(pcm[0]))
            def finalize(self, force=False):
                self.events.append("final")

        endpoint = _Endpoint()
        with patch.dict(globals(), {"shutdown_event": threading.Event(), "ui_update": lambda **kw: None}):
            feeder = RingAudioFeeder(endpoint, queue_ms=40, clock_ns=lambda: 1_000_000_000)
            for value in range(20):
                feeder.callback(np.full(FRAME_SAMPLES, value, dtype=np.int16), FRAME_SAMPLES, None, None)
                feeder.request_finalize()
                feeder.request_finalize()
            _assert_self_test(len(feeder.queue) <= 2 * feeder.max_frames + 1, "flush controls grew beyond the bounded audio queue")
            feeder.close()
            feeder.callback(np.full(FRAME_SAMPLES, 999, dtype=np.int16), FRAME_SAMPLES, None, None)
            feeder.worker()
            _assert_self_test([event for event in endpoint.events if event != "final"] == [18, 19], "close accepted later callbacks or reordered retained audio")
            _assert_self_test(endpoint.events[-4:] == [18, "final", 19, "final"], "manual finalization overtook an earlier frame")

    @check("Ring capture callback stays independent of a stalled worker and closes cleanly")
    def _test_ring_ingress_worker_lifecycle() -> None:
        from unittest.mock import patch
        entered, release = threading.Event(), threading.Event()

        class _Endpoint:
            def __init__(self):
                self.frames = []
                self.threads = []
            def on_frame(self, timestamp_ns, pcm):
                self.threads.append(threading.get_ident())
                self.frames.append(int(pcm[0]))
                if len(self.frames) == 1:
                    entered.set()
                    _assert_self_test(release.wait(timeout=2), "test worker release timed out")
            def finalize(self, force=False):
                pass

        endpoint = _Endpoint()
        notices = []
        with patch.dict(globals(), {"shutdown_event": threading.Event(), "ui_update": lambda **kw: notices.append(kw)}):
            feeder = RingAudioFeeder(endpoint, queue_ms=40, clock_ns=lambda: 1_000_000_000)
            try:
                with feeder:
                    feeder.callback(np.zeros(FRAME_SAMPLES, dtype=np.int16), FRAME_SAMPLES, None, None)
                    _assert_self_test(entered.wait(timeout=1), "capture worker did not receive a frame")
                    for value in range(1, 7):
                        feeder.callback(np.full(FRAME_SAMPLES, value, dtype=np.int16), FRAME_SAMPLES, None, None)
                    _assert_self_test(feeder.dropped_frames == 4, "stalled worker did not enforce the capture budget")
                    release.set()
            finally:
                release.set()
                feeder.close()
            _assert_self_test(feeder.worker_thread is not None and not feeder.worker_thread.is_alive(), "capture worker failed to exit after source shutdown")
            _assert_self_test(endpoint.frames == [0, 5, 6], "stalled worker reordered or duplicated retained audio")
            _assert_self_test(all(ident != threading.get_ident() for ident in endpoint.threads), "endpoint work ran on the capture callback thread")
            _assert_self_test(notices and "dropped=4 frames/1280 samples" in notices[0]["status"], "live worker failed to report its overrun")

    @check("Selected SoundDevice inputs resolve numeric indexes and bypass native auto capture")
    def _test_sounddevice_input_selection() -> None:
        from unittest.mock import patch
        with patch.object(args, "audio_io", "auto"), patch.object(args, "input_device", "17"):
            _assert_self_test(resolve_audio_io_backend() == "sounddevice" and sounddevice_input_device() == 17, "numeric device selection was ignored or treated as a name")
        with patch.object(args, "input_device", "LitLink Audio"):
            _assert_self_test(sounddevice_input_device() == "LitLink Audio", "named virtual audio source was changed")

    @check("Ring endpoint includes each onset frame exactly once")
    def _test_ring_onset_sequence() -> None:
        from unittest.mock import patch
        saved_partial = {key: value for key, value in globals().items() if key.startswith("current_partial_")}
        try:
            with patch.dict(globals(), {"tts_playing": threading.Event()}):
                endpoint = RingEndpointDetector(NativeTimestampRingBuffer(1), PriorityAsrScheduler())
                before = np.full(FRAME_SAMPLES, 11, dtype=np.int16)
                onset = np.full(FRAME_SAMPLES, 22, dtype=np.int16)
                endpoint._consume(TimestampedAudioFrame(before, 0, -60.0, False))
                endpoint._consume(TimestampedAudioFrame(onset, 20_000_000, -20.0, True))
                _assert_self_test(np.array_equal(np.concatenate(endpoint.active), np.concatenate([before, onset])), "speech-onset frame was duplicated or reordered")
                _assert_self_test(current_partial_utterance_id == endpoint.utterance_id, "new capture did not own endpoint partial context")
        finally:
            globals().update(saved_partial)

    @check("Coordinator ignores closed partials without erasing newer utterance context")
    def _test_coordinator_interleaving() -> None:
        from copy import deepcopy
        from unittest.mock import patch
        stop = threading.Event()

        class _FiniteResults(Queue):
            def task_done(self):
                super().task_done()
                if self.unfinished_tasks == 0:
                    stop.set()

        results = _FiniteResults()
        for utterance_id, text, partial in [
            (10, "final ten", False), (20, "future twenty", True),
            (10, "late closed ten", True), (5, "earlier final five", False),
            (30, "newest thirty", True), (20, "older twenty revision", True),
            (20, "final twenty", False), (20, "late closed twenty", True),
            (40, "", False), (40, "late empty-final forty", True),
            (50, "racing fifty", True),
        ]:
            results.put_nowait(AsrResult(utterance_id, text, partial))
        translated: list[TranslationTask] = []
        errors: list[dict] = []
        state = deepcopy(ui_state)
        state.update(left_active={}, left_order=[], left_done=[])

        def sanitize_with_capture_interleaving(text: str, final: bool = False) -> str:
            if text == "racing fifty":
                _assert_self_test(
                    current_partial_utterance_id == 30 and current_partial_text == "newest thirty",
                    "older results erased or replaced future partial context",
                )
                # A newer capture can start while the coordinator prepares an
                # older ASR result; admission must be rechecked at publication.
                reset_current_partial_state(next_utterance_id=60)
            return text

        replacement = {
            "asr_results": results, "shutdown_event": stop, "ui_state": state,
            "render_dashboard": lambda: None, "ui_update": lambda **kw: errors.append(kw),
            "enqueue_translation_task": translated.append,
            "sanitize_asr_text": sanitize_with_capture_interleaving,
            "prepare_final_transcript_for_translation": lambda text, previous: text,
            "forced_bos_for_target": lambda target: 1,
            "audio_final_transcripts": {}, "latency_traces": {}, "inflight_partial_markers": {},
            "ring_endpoint": None, "current_partial_utterance_id": None,
        }
        saved_partial = {key: value for key, value in globals().items() if key.startswith("current_partial_")}
        try:
            with ExitStack() as stack:
                stack.enter_context(patch.dict(globals(), replacement))
                stack.enter_context(patch.object(args, "stream_audio_translation", False))
                stack.enter_context(patch.object(args, "partial_display_policy", "latest"))
                stack.enter_context(patch.object(args, "final_dedupe_window_ms", 0))
                stack.enter_context(patch.object(args, "empty_final_action", "close"))
                coordinator_worker()
                _assert_self_test(not errors, f"coordinator raised errors: {errors}")
                _assert_self_test([task.utterance_id for task in translated] == [10, 5, 20], "committed earlier/future IDs were dropped or reordered")
                _assert_self_test(state["left_active"][10]["src_final"] and "late" not in str(state["left_active"][10]["src"]), "late partial reopened finalized ten")
                _assert_self_test(state["left_active"][20]["src_final"] and "late" not in str(state["left_active"][20]["src"]), "late partial reopened finalized twenty")
                _assert_self_test(40 not in state["left_active"], "late partial reopened a closed empty final")
                _assert_self_test(30 in state["left_active"], "newer partial UI entry was lost")
                _assert_self_test(current_partial_utterance_id == 60 and not current_partial_text, "older result overwrote capture that started during result preparation")
                _assert_self_test(results.unfinished_tasks == 0, "coordinator failed to acknowledge an ignored result")
        finally:
            globals().update(saved_partial)

    @check("TTS queue preserves typed speech")
    def _test_tts_queue() -> None:
        q: Queue[TtsTask] = Queue(maxsize=1)
        q.put_nowait(TtsTask(text="old audio", wav_out="out.wav", lang_hint="en", source="audio", utterance_id=1))
        ok = queue_put_tts_task(
            q,
            TtsTask(text="typed reply", wav_out="out.wav", lang_hint="tr", source="text", utterance_id=2),
            "self-test-tts",
        )
        item = q.get_nowait()
        q.task_done()
        _assert_self_test(ok, "typed TTS task was rejected")
        _assert_self_test(isinstance(item, TtsTask) and item.source == "text", "audio TTS evicted a typed reply")

        q2: Queue[TtsTask] = Queue(maxsize=1)
        q2.put_nowait(TtsTask(text="typed reply", wav_out="out.wav", lang_hint="tr", source="text", utterance_id=3))
        ok = queue_put_tts_task(
            q2,
            TtsTask(text="audio reply", wav_out="out.wav", lang_hint="en", source="audio", utterance_id=4),
            "self-test-tts",
        )
        item = q2.get_nowait()
        q2.task_done()
        _assert_self_test(not ok, "audio TTS should not displace typed TTS")
        _assert_self_test(isinstance(item, TtsTask) and item.source == "text", "typed TTS was displaced by audio")

    @check("MLX Whisper beam fallback is cached")
    def _test_mlx_beam_fallback() -> None:
        global MLX_WHISPER_INSTALLED, mlx_whisper, mlx_whisper_beam_available, mlx_whisper_beam_fallback_reported

        class _FakeMlxWhisper:
            def __init__(self):
                self.calls: list[dict] = []

            def transcribe(self, _audio, **kwargs):
                self.calls.append(dict(kwargs))
                if "beam_size" in kwargs:
                    raise RuntimeError("Beam search decoder is not yet implemented")
                return {"text": "merhaba dunya"}

        old_installed = MLX_WHISPER_INSTALLED
        old_module = mlx_whisper
        old_available = mlx_whisper_beam_available
        old_reported = mlx_whisper_beam_fallback_reported
        fake = _FakeMlxWhisper()
        try:
            MLX_WHISPER_INSTALLED = True
            mlx_whisper = fake
            mlx_whisper_beam_available = None
            mlx_whisper_beam_fallback_reported = False
            audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
            text = transcribe_audio_mlx_whisper(audio, model_name="tiny", beam=3)
            text_again = transcribe_audio_mlx_whisper(audio, model_name="tiny", beam=3)
            beam_attempts = sum(1 for call in fake.calls if "beam_size" in call)
            _assert_self_test(text == "merhaba dunya", "beam fallback did not return greedy text")
            _assert_self_test(text_again == "merhaba dunya", "cached greedy fallback failed")
            _assert_self_test(beam_attempts == 1, "beam fallback was not cached")
            _assert_self_test(mlx_whisper_beam_available is False, "beam availability was not cached as false")
        finally:
            MLX_WHISPER_INSTALLED = old_installed
            mlx_whisper = old_module
            mlx_whisper_beam_available = old_available
            mlx_whisper_beam_fallback_reported = old_reported

    @check("ASR language is forced into backend calls")
    def _test_asr_language_forcing() -> None:
        old_force = args.force_asr_language
        old_src = args.source_language
        old_prompt = args.asr_initial_prompt
        try:
            args.force_asr_language = True
            args.source_language = "AR"
            args.asr_initial_prompt = ""
            _assert_self_test(asr_language_kwargs() == {"language": "ar"}, "ASR kwargs did not force Arabic")
            _assert_self_test(asr_language_cli_args("-l") == ["-l", "ar"], "ASR CLI args did not force Arabic")
            _assert_self_test(stt_initial_prompt() is None, "ASR initial prompt should default empty to avoid prompt-language leakage")
        finally:
            args.force_asr_language = old_force
            args.source_language = old_src
            args.asr_initial_prompt = old_prompt

    @check("MLX-VLM Gemma prompt stays source-language transcription")
    def _test_mlx_vlm_prompt_defaults() -> None:
        old_prompt = args.mlx_vlm_prompt
        old_partial_prompt = args.mlx_vlm_partial_prompt
        old_final_prompt = args.mlx_vlm_final_prompt
        try:
            args.mlx_vlm_prompt = ""
            args.mlx_vlm_partial_prompt = ""
            args.mlx_vlm_final_prompt = ""
            prompt = render_mlx_vlm_prompt(partial=False)
            _assert_self_test("Do not translate" in prompt, "default Gemma prompt should not collapse ASR and MT")
            _assert_self_test(source_language().name in prompt, "default Gemma prompt omitted source language")
            args.mlx_vlm_prompt = "Listen to {source_language}; target={target_language}; {task_hint}"
            rendered = render_mlx_vlm_prompt(partial=True)
            _assert_self_test(source_language().name in rendered and target_language().name in rendered, "Gemma prompt placeholders did not render")
        finally:
            args.mlx_vlm_prompt = old_prompt
            args.mlx_vlm_partial_prompt = old_partial_prompt
            args.mlx_vlm_final_prompt = old_final_prompt

    @check("MLX-VLM output cleanup removes wrappers")
    def _test_mlx_vlm_output_cleanup() -> None:
        cleaned = strip_mlx_vlm_generation("<think>hidden</think> Transcript: Merhaba dünya")
        _assert_self_test(cleaned == "Merhaba dünya", "Gemma output cleanup left wrapper text")
        leaked = "<|channel>thought Thinking Process: 1. **Analyze the Request:** explain internals"
        _assert_self_test(strip_mlx_vlm_generation(leaked) == "", "unclosed Gemma thought channel was not suppressed")
        final_after_thought = "<|channel>thought hidden<channel|><|channel>final Transcript: Merhaba dünya"
        _assert_self_test(strip_mlx_vlm_generation(final_after_thought) == "Merhaba dünya", "Gemma final channel was not recovered after thought stripping")
        _assert_self_test(sanitize_asr_text(leaked, final=False) == "", "ASR sanitizer allowed Gemma thinking text")

    @check("MLX-VLM prompt echoes are rejected")
    def _test_mlx_vlm_prompt_echo_rejection() -> None:
        prompt = "Bu ses Türkçe konuşmadır. Türkçe konuşmayı Türkçe olarak eksiksiz yazıya dök. Açıklama ekleme."
        leaked = "Bu ses Türkçe konuşmadır. Türkçe konuşmayı Türkçe olarak eksiksiz yazıya dök. Açıklama ekleme."
        _assert_self_test(clean_mlx_vlm_transcript_output(leaked, prompt, partial=True) == "", "Gemma prompt echo was not rejected")
        _assert_self_test(clean_mlx_vlm_transcript_output("EMPTY", prompt, partial=True) == "", "Gemma EMPTY sentinel was not rejected")
        _assert_self_test(clean_mlx_vlm_transcript_output("Merhaba nasılsınız", prompt, partial=True) == "Merhaba nasılsınız", "real transcript was rejected as prompt echo")

    @check("Known hallucination filter removes caption credits")
    def _test_hallucination_filter() -> None:
        old_filter = args.known_hallucination_filter
        try:
            args.known_hallucination_filter = "strict"
            cleaned = sanitize_asr_text("Altyazı M. K.", final=True)
            _assert_self_test(cleaned == "", "caption-credit hallucination was not filtered")
        finally:
            args.known_hallucination_filter = old_filter

    @check("Overlapping ASR repetitions are collapsed")
    def _test_overlapping_asr_repetition_cleanup() -> None:
        noisy = "Diş hekimliği 30 Nisan 2026 tarihinde olan diş hekimliği 30 Nisan 2026 tarihinde olan randevunuzu hatırlatırız"
        cleaned = collapse_overlapping_transcript_repetitions(noisy)
        _assert_self_test(cleaned.casefold().count("diş hekimliği") == 1, "overlapping ASR repeat was not collapsed")
        _assert_self_test("randevunuzu hatırlatırız" in cleaned, "repeat cleanup dropped the useful suffix")

    @check("Final dedupe preserves corrections and requires shared audio")
    def _test_near_duplicate_final_detection() -> None:
        old_ratio = args.final_dedupe_contained_ratio
        try:
            args.final_dedupe_contained_ratio = 0.78
            first = "لا إله إلا الله وحده لا شريك له حمده على كل شيء"
            second = "لا إله إلا إنه الله وحده لا شريك له وله له المذكور الحمد يحيى يوم يعد بيد الخير وهو على كل شيء لشيء قدير"
            _assert_self_test(not final_transcripts_are_near_duplicates(first, second), "different decoded words were treated as duplicate speech")
            _assert_self_test(final_transcripts_are_near_duplicates(first, first), "exact repeated transcript was not recognized")
            _assert_self_test(not final_transcript_audio_spans_overlap(None, None), "missing audio provenance suppressed a repeated phrase")
            _assert_self_test(not final_transcript_audio_spans_overlap(("session", "AR", 0, 100), ("session", "AR", 200, 300)), "disjoint source intervals were treated as duplicates")
            _assert_self_test(
                not final_transcripts_are_near_duplicates("مرحبا كيف حالك اليوم", "أريد حجز موعد جديد"),
                "unrelated Arabic finals were treated as duplicates",
            )
        finally:
            args.final_dedupe_contained_ratio = old_ratio

    @check("Partial MT waits for complete clumps")
    def _test_partial_mt_clump_readiness() -> None:
        old_score = args.partial_translation_min_completeness_score
        old_force = args.partial_translation_force_words
        try:
            args.partial_translation_min_completeness_score = 2
            args.partial_translation_force_words = 10
            _assert_self_test(partial_translation_clump("I would like to") == "", "incomplete partial was translated too early")
            ready = partial_translation_clump("Randevu almak istiyorum")
            _assert_self_test(ready == "Randevu almak istiyorum", "complete stable clump was not accepted")
        finally:
            args.partial_translation_min_completeness_score = old_score
            args.partial_translation_force_words = old_force

    @check("Partial UI shows stable and inflight text separately")
    def _test_partial_ui_inflight_format() -> None:
        old_confidence_ui = args.confidence_ui
        try:
            args.confidence_ui = True
            rendered = strip_ansi(format_partial_transcript_for_ui("merhaba dunya", "merhaba", inflight=True))
            _assert_self_test("TR✓:" in rendered, "stable partial marker was not shown")
            _assert_self_test("dunya" in rendered, "unverified mutable tail was not shown")
            _assert_self_test("quality pending" in rendered, "inflight quality-pending marker was not shown")
        finally:
            args.confidence_ui = old_confidence_ui

    @check("Arabic terminal text is not mutated")
    def _test_arabic_terminal_safe_text() -> None:
        text = "لا لا"
        rendered = terminal_safe_text(text)
        _assert_self_test(rendered == text, "terminal-safe Arabic must preserve lam-alef text exactly")
        _assert_self_test(display_width(rendered) == 5, "Arabic lam-alef should keep canonical terminal width")
        _assert_self_test(count_lam_alef_clusters(rendered) == 2, "lam-alef compensation count failed")

    @check("Arabic presentation forms normalize before ASR/MT/logging")
    def _test_arabic_presentation_form_normalization() -> None:
        text = "ﻻ ﺈﻠﻫ ﺇﻻ ﺎﻠﻠﻫ"
        normalized = normalize_text(text)
        _assert_self_test("ﻻ" not in normalized, "presentation lam-alef was not normalized")
        _assert_self_test("لا" in normalized, "canonical lam-alef was not restored")

    @check("In-flight partial markers expire without ASR text")
    def _test_inflight_partial_marker_expiry() -> None:
        saved_state = {
            key: value.copy() if isinstance(value, (dict, list)) else value
            for key, value in ui_state.items()
        }
        old_timeout = args.inflight_partial_timeout_ms
        try:
            ui_state["left_done"] = []
            ui_state["right_done"] = []
            ui_state["left_active"] = {}
            ui_state["right_active"] = {}
            ui_state["left_order"] = []
            ui_state["right_order"] = []
            args.inflight_partial_timeout_ms = 1
            show_inflight_partial_transcript(999, revision=1)
            inflight_partial_markers[999] = time.perf_counter() - 1.0
            sweep_inflight_partial_markers(set())
            _assert_self_test(999 not in ui_state["left_active"], "expired in-flight marker was not removed")
        finally:
            args.inflight_partial_timeout_ms = old_timeout
            inflight_partial_markers.pop(999, None)
            ui_state.clear()
            ui_state.update(saved_state)

    @check("Partial translation draft lingers across source revisions")
    def _test_partial_translation_ui_lingers() -> None:
        saved_state = {
            key: value.copy() if isinstance(value, (dict, list)) else value
            for key, value in ui_state.items()
        }
        try:
            ui_state["left_done"] = []
            ui_state["right_done"] = []
            ui_state["left_active"] = {}
            ui_state["right_active"] = {}
            ui_state["left_order"] = []
            ui_state["right_order"] = []
            ui_state["input"] = ""
            ui_state["status"] = ""

            _assert_self_test(
                ui_set_panel_entry("left", 77, "src", "TR partial 1", final=False, revision=1),
                "initial partial source was rejected",
            )
            _assert_self_test(
                ui_set_panel_entry("left", 77, "dst", "EN draft 1", final=False, revision=1),
                "initial partial translation was rejected",
            )
            _assert_self_test(
                ui_set_panel_entry("left", 77, "src", "TR partial 2", final=False, revision=2),
                "newer partial source was rejected",
            )
            _assert_self_test(
                ui_state["left_active"][77]["dst"] == "EN draft 1",
                "partial translation disappeared when source revision changed",
            )
            _assert_self_test(
                not ui_set_panel_entry("left", 77, "dst", "EN draft 0", final=False, revision=0),
                "older partial translation overwrote a newer visible draft",
            )
            _assert_self_test(
                ui_set_panel_entry("left", 77, "dst", "EN draft 2", final=False, revision=2),
                "newer partial translation did not update the visible draft",
            )
            _assert_self_test(
                not ui_set_panel_entry("left", 77, "dst", "EN final stale", final=True, revision=1),
                "stale final translation was accepted",
            )
            ui_remove_panel_entry("left", 77)
            _assert_self_test(
                not ui_set_panel_entry("left", 77, "dst", "EN ghost", final=True, revision=3),
                "missing final translation target was accepted",
            )
            _assert_self_test(77 not in ui_state["left_active"] and 77 not in ui_state["left_order"], "stale final created a ghost UI entry")
        finally:
            ui_state.clear()
            ui_state.update(saved_state)

    @check("Partial display uses stable text by default")
    def _test_partial_display_source_policy() -> None:
        old_policy = args.partial_display_policy
        old_min_words = args.partial_display_min_words
        try:
            args.partial_display_policy = "stable"
            args.partial_display_min_words = 2
            _assert_self_test(
                partial_display_source_text("merhaba dunya belki", "merhaba dunya") == "merhaba dunya",
                "stable display did not select the locked prefix",
            )
            _assert_self_test(
                partial_display_source_text("merhaba dunya belki", "merhaba") == "",
                "stable display exposed too-short locked text",
            )
            args.partial_display_policy = "latest"
            _assert_self_test(
                partial_display_source_text("merhaba dunya belki", "merhaba") == "merhaba dunya belki",
                "latest display policy did not expose the full partial",
            )
        finally:
            args.partial_display_policy = old_policy
            args.partial_display_min_words = old_min_words

    @check("MLX MT keeps short utterances in one context")
    def _test_mlx_mt_chunking_keeps_context() -> None:
        old_backend = detected_translation_backend
        old_max_chars = args.mt_chunk_max_chars
        globals()["detected_translation_backend"] = "mlx-lm"
        try:
            args.mt_chunk_max_chars = 1800
            chunks = iter_translation_chunks("First sentence. Second sentence. Third sentence.")
            _assert_self_test(chunks == ["First sentence. Second sentence. Third sentence."], "MLX MT split a short utterance into fragments")
        finally:
            globals()["detected_translation_backend"] = old_backend
            args.mt_chunk_max_chars = old_max_chars

    @check("MT prompt template injects source text")
    def _test_mt_prompt_template() -> None:
        old_prompt = args.mt_prompt
        try:
            args.mt_prompt = "Translate {source_language} to {target_language}: {text}"
            rendered = render_mt_prompt_template(args.mt_prompt, "Merhaba", source_nllb_lang(), 1)
            _assert_self_test("Merhaba" in rendered and target_language().name in rendered, "MT prompt template did not render placeholders")
        finally:
            args.mt_prompt = old_prompt

    @check("MLX prompt selects TranslateGemma direction schema and preserves generic templates")
    def _test_mlx_translation_prompt_model_contract() -> None:
        class _FakeTokenizer:
            def __init__(self):
                self.messages = None

            # This deliberately omits enable_thinking to exercise the existing
            # compatibility retry used with tokenizers that reject that option.
            def apply_chat_template(self, messages, *, add_generation_prompt, tokenize):
                _assert_self_test(add_generation_prompt and not tokenize, "incorrect prompt serialization options")
                self.messages = messages
                return "formatted-prompt"

        saved = {name: globals()[name] for name in (
            "mlx_translation_tokenizer", "USER_SET_TRANSLATION_MODEL", "ID_BY_LANG_CODE",
        )}
        old_model, old_prompt = args.translation_model, args.mt_prompt
        fake = _FakeTokenizer()
        globals()["mlx_translation_tokenizer"] = fake
        globals()["USER_SET_TRANSLATION_MODEL"] = True
        globals()["ID_BY_LANG_CODE"] = {code: index + 11 for index, code in enumerate(LANGUAGE_SPECS)}
        try:
            args.translation_model = "/models/translategemma-12b-it-4bit"
            args.mt_prompt = "Generic wrapper must not enter the specialized translation content: {text}"
            text = "  שלום 123، مرحباً\nCafé, İstanbul, Москва  "
            for source_code in ("AR", "HE", "DE", "TR", "FR", "RU"):
                for target_code in ("AR", "HE", "DE", "TR", "FR", "RU"):
                    if source_code == target_code:
                        continue
                    formatted = mlx_translation_prompt(
                        text, language_spec(source_code).nllb, ID_BY_LANG_CODE[target_code],
                    )
                    expected = [{"role": "user", "content": [{
                        "type": "text", "source_lang_code": language_spec(source_code).whisper,
                        "target_lang_code": language_spec(target_code).whisper, "text": text,
                    }]}]
                    _assert_self_test(formatted == "formatted-prompt" and fake.messages == expected,
                                      f"TranslateGemma direction/text corrupted for {source_code}->{target_code}")

            args.translation_model = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
            args.mt_prompt = "Use formal {target_language}: {text}"
            mlx_translation_prompt(text, language_spec("AR").nllb, ID_BY_LANG_CODE["HE"])
            _assert_self_test(fake.messages == [
                {"role": "system", "content": "You are a strict real-time translator. Output only the requested target-language translation with no commentary, no quotes, and no extra notes."},
                {"role": "user", "content": f"Use formal Hebrew: {text}"},
            ], "generic custom template changed during TranslateGemma integration")
        finally:
            globals().update(saved)
            args.translation_model, args.mt_prompt = old_model, old_prompt

    @check("Final ASR merge preserves partial details")
    def _test_final_merge_preserves_partial_details() -> None:
        old_merge = args.final_merge_last_partial
        old_mode = args.final_translation_completeness_mode
        old_min_words = args.final_merge_min_final_words
        try:
            args.final_merge_last_partial = True
            args.final_translation_completeness_mode = "conservative"
            args.final_merge_min_final_words = 4
            merged = prepare_final_transcript_for_translation(
                "مرحبا أريد حجز موعد",
                "مرحبا أريد حجز موعد غدا صباحا مع الطبيب",
            )
            _assert_self_test("غدا" in merged and "صباحا" in merged, "final merge dropped details from last partial")
        finally:
            args.final_merge_last_partial = old_merge
            args.final_translation_completeness_mode = old_mode
            args.final_merge_min_final_words = old_min_words

    @check("Terminal Arabic lam-alef width is compensated without mutation")
    def _test_terminal_arabic_lam_alef_width() -> None:
        sample = "لا لا لا"
        _assert_self_test(terminal_safe_text(sample) == sample, "terminal rendering mutated Arabic lam-alef text")
        _assert_self_test(display_width(sample) == 8, "canonical lam-alef width should remain two cells")
        _assert_self_test(count_lam_alef_clusters(sample) == 3, "lam-alef divider compensation count is incorrect")

    failures: list[str] = []
    for name, fn in checks:
        try:
            fn()
            print(f"self-test OK: {name}")
        except Exception as exc:
            failures.append(f"{name}: {exc}")
            print(f"self-test FAIL: {name}: {exc}", file=sys.stderr)

    if failures:
        print("\nSelf-test failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print("self-test OK: all fast pipeline invariants passed")
    return 0


# ======================
# Main loop
# ======================
def web_state_snapshot() -> dict:
    from tts_runtime.web_transport import build_web_snapshot
    return build_web_snapshot(ui_state, {
        "source_language": source_language_code(), "target_language": target_language_code(),
        "input_device": args.input_device, "debug": args.debug, "mode": args.mode,
        "audio_review": args.audio_review,
    }, session_id=runtime_session_id, running=not shutdown_event.is_set(),
        audio_review_mode=audio_review_mode, recording_directory=str(runtime_audio_dir or ""))


def mark_audio_for_review(reason: str = "user_flagged") -> bool:
    if audio_evidence is None or ring_endpoint is None:
        return False
    ring = ring_endpoint.ring
    remaining = SAMPLE_RATE * 15
    chunks, start_ns, end_ns = [], None, None
    with ring.lock:
        for timestamp_ns, samples in reversed(ring.frames):
            frame_end = timestamp_ns + int(samples.size) * 1_000_000_000 // SAMPLE_RATE
            if start_ns is not None and abs(frame_end - start_ns) > 2 * 1_000_000_000 // SAMPLE_RATE:
                break  # Never join two disjoint recording intervals for review.
            take = min(remaining, int(samples.size))
            chunks.append(samples[-take:].copy())
            start_ns = timestamp_ns + (int(samples.size) - take) * 1_000_000_000 // SAMPLE_RATE
            end_ns = frame_end if end_ns is None else end_ns
            remaining -= take
            if remaining <= 0:
                break
    if not chunks:
        return False
    pcm = np.concatenate(list(reversed(chunks)))
    return audio_evidence.save_clip(pcm, {"session_id": runtime_session_id,
                                          "source_language": source_language_code(),
                                          "timestamp_start_ns": start_ns, "timestamp_end_ns": end_ns,
                                          "selection": "newest contiguous interval, up to 15 seconds"},
                                    [reason], label="manual-review", timestamp_ns=start_ns)


def publish_selection_result(result: SelectionResult) -> None:
    with selection_tasks_lock:
        expected = selection_tasks.get(result.task.lookup_id)
        if expected != result.task or result.task.session_id != runtime_session_id:
            return
        selection_tasks.pop(result.task.lookup_id, None)
    if headless_transport is not None:
        headless_transport.send_packet({"type": "lookup_result", "lookup_id": result.task.lookup_id,
            "selection_id": result.task.selection_id, "source_language": result.task.source_language,
            "target_language": result.task.target_language,
            "translation": clean_model_output(result.translation), "error": str(result.error)[:1000]})


def fail_selection_task(identity: str, error: str) -> None:
    with selection_tasks_lock:
        task = selection_tasks.get(identity)
    if task is not None:
        publish_selection_result(SelectionResult(task, error=error))


def submit_selection_task(command: dict) -> dict:
    if translation_stage is None or translation_stage.state not in {"loading", "ready", "busy"}:
        return {"accepted": False, "message": "Start a session with an available translation worker first"}
    task = SelectionTask.from_command(command, runtime_session_id)
    with selection_tasks_lock:
        if len(selection_tasks) >= 4 or task.lookup_id in selection_tasks:
            return {"accepted": False, "message": "Selection translation is busy; wait or select again shortly"}
        selection_tasks[task.lookup_id] = task
    accepted = translation_stage.submit(task, background=True, key=("lookup", task.lookup_id))
    if not accepted:
        fail_selection_task(task.lookup_id, "Live work takes priority; select the phrase again shortly")
    return {"accepted": accepted, "message": "Selection queued" if accepted else "Translation is busy"}


def process_headless_action(command: dict) -> dict:
    action = command.get("action")
    if action == "lookup":
        return submit_selection_task(command)
    if action == "reply":
        text = command.get("text", "")
        if not isinstance(text, str) or not text.strip() or len(text) > 10000:
            return {"accepted": False, "message": "Enter a reply of 1–10000 characters"}
        accepted = submit_typed_reply(text)
        return {"accepted": accepted, "message": "Queued" if accepted else "Translation unavailable or queue full; draft retained"}
    if action == "flush":
        if active_ring_feeder is not None:
            active_ring_feeder.request_finalize()
        else:
            force_flush_audio()
        return {"accepted": True, "message": "Audio flush requested"}
    if action == "mark":
        accepted = mark_audio_for_review(str(command.get("reason") or "user_flagged"))
        return {"accepted": accepted, "message": "Review clip queued" if accepted else "No captured audio available or audio review is disabled/full"}
    if action == "stop":
        return {"accepted": True, "message": "Stopping", "stop": True}
    return {"accepted": False, "message": "Unknown command"}


def read_headless_command(timeout: float) -> str | None:
    if headless_transport is not None and headless_transport.poll(timeout):
        if headless_transport.snapshot.get("output_failed"):
            raise RuntimeError("Web output channel stalled, overflowed or failed; restart the session after reconnecting")
        shutdown_event.set()
        return "\x1b"
    return None


def run_web_interface() -> int:
    from tts_runtime.web_controller import WebController
    from tts_runtime.web_server import WebInterface
    excluded = {"--ui", "--web-host", "--web-port", "--source-language", "--target-language", "--input-device", "--audio-review"}
    forwarded = []
    tokens = iter(sys.argv[1:])
    for token in tokens:
        flag = token.split("=", 1)[0]
        if flag == "--debug":
            continue
        if flag in excluded:
            if "=" not in token:
                next(tokens, None)
            continue
        forwarded.append(token)
    controller = WebController(core_path=Path(__file__).resolve(), python_executable=sys.executable,
                               core_arguments=forwarded,
                               initial_config={"source_language": source_language_code(),
                                               "target_language": target_language_code(),
                                               "input_device": args.input_device, "debug": args.debug,
                                               "audio_review": args.audio_review})
    server = WebInterface(controller, host=args.web_host, port=args.web_port)
    try:
        url = server.start()
        eprint(f"Live TTS web UI: {url}")
        eprint("Keep this process running; open the URL and choose Start. Ctrl-C stops the local server.")
        while True:
            time.sleep(0.25)
    except KeyboardInterrupt:
        return 0
    finally:
        controller.close()
        server.close()


def main() -> int:
    global audio_bridge, text_entry_seq, detected_stt_backend, detected_stt_device, active_ring_feeder
    if args.self_test:
        return run_self_test()
    if args.list_devices:
        list_audio_devices()
        return 0

    if args.dump_config:
        eprint(json.dumps(effective_config(), indent=2, sort_keys=True))
        return 0

    if args.ui == "web":
        return run_web_interface()

    validate_runtime_configuration()
    initialize_session_runtime()

    if args.mode in {"audio", "both"}:
        ensure_audio_runtime()

    streaming_engine = resolve_streaming_engine() if args.mode in {"audio", "both"} else "legacy"
    # Resolve metadata only. Every heavy model import/load/warmup belongs to
    # its worker, after terminal setup; capture and keyboard never wait for it.
    init_parent_routing_runtime()
    if args.mode in {"audio", "both"}:
        detected_stt_backend = resolve_stt_backend()
        detected_stt_device = resolve_stt_device(detected_stt_backend)
        args.whisper_model = default_whisper_model_for_backend(detected_stt_backend)
    setup_terminal()

    workers: list[threading.Thread] = []
    global ring_scheduler, ring_endpoint
    if args.mode in {"audio", "both"}:
        if streaming_engine == "ring":
            ring_scheduler = PriorityAsrScheduler()
            ring_endpoint = RingEndpointDetector(NativeTimestampRingBuffer(args.ring_buffer_seconds), ring_scheduler)
            workers.append(threading.Thread(target=coordinator_worker, daemon=True))
        else:
            workers.append(threading.Thread(target=audio_frame_worker, daemon=True))
            workers.append(threading.Thread(target=asr_worker, daemon=True))
            workers.append(threading.Thread(target=coordinator_worker, daemon=True))
    if args.mt_process:
        workers.append(threading.Thread(target=translation_result_worker, daemon=True))
    else:
        workers.append(threading.Thread(target=translation_worker, daemon=True))
    workers.append(threading.Thread(target=tts_worker, daemon=True))
    for worker in workers:
        worker.start()

    mode_lines = [f"🚀 Mode: {args.mode.upper()} | profile={args.profile}"]
    if args.mode == "audio":
        mode_lines.append(f"🎙️ Play {source_language().name} phone-call audio near the mic. Silence auto-translates to {target_language().name} text; ENTER forces flush; ESC exits.")
    elif args.mode == "text":
        mode_lines.append(f"⌨️ Type {target_language().name} and press ENTER for spoken {source_language().name}. ESC exits.")
    else:
        mode_lines.append(f"🎙️ Incoming {source_language().name} call audio -> {target_language().name} text")
        mode_lines.append(f"⌨️ Type {target_language().name} anytime -> spoken {source_language().name} reply")
        mode_lines.append("ENTER sends typed text; ENTER on empty line forces audio flush; ESC exits.")
    if args.session_logs:
        mode_lines.append(f"📝 Session logs ON: {args.transcription_log}, {args.tts_log} (--no-session-logs disables)")
    init_dashboard(mode_lines)
    ui_update(status=f"Audio review: {audio_review_mode} | {runtime_audio_dir or 'disabled'} | Ctrl-R saves a review clip")

    text_buffer = ""
    ring_feeder: RingAudioFeeder | None = None
    last_capture_status_at = 0.0

    try:
        if args.mode in {"audio", "both"}:
            start_asr_stage()
        else:
            update_stage_status("capture", "off", "typed replies only")
            update_stage_status("asr", "off")
        if args.mt_process:
            start_mt_process()
        if args.mode in {"audio", "both"}:
            audio_backend = resolve_audio_io_backend()
            log_verbose(f"audio backend: {audio_backend} | streaming engine: {streaming_engine}")

            def handle_interactive_key(timeout: float = 0.0) -> bool:
                nonlocal text_buffer, last_capture_status_at
                global text_entry_seq
                now = time.monotonic()
                if ring_feeder is not None and now - last_capture_status_at >= 0.5:
                    last_capture_status_at = now
                    update_capture_status(ring_feeder)
                key = read_key(timeout)
                if key is None:
                    return True
                if key == "\x12":
                    saved = mark_audio_for_review()
                    ui_update(status="Review clip queued" if saved else "No audio available for review or recording disabled/full")
                    return True
                if key == "\x1b":
                    if text_buffer:
                        text_buffer = ""
                        render_input(text_buffer, active=False)
                        ui_update(status="↩️ Text input cancelled.")
                        return True
                    return False
                if key in ("\n", "\r"):
                    if text_buffer.strip():
                        if submit_typed_reply(text_buffer):
                            text_buffer = ""
                    else:
                        if streaming_engine == "ring" and ring_feeder is not None:
                            ring_feeder.request_finalize()
                        else:
                            force_flush_audio()
                        ui_update(status="🎙️ Forced audio flush.")
                    return True
                if key == "\x7f":
                    if text_buffer:
                        text_buffer = text_buffer[:-1]
                        render_input(text_buffer)
                    return True
                if args.mode != "audio" and key.isprintable():
                    text_buffer += key
                    render_input(text_buffer)
                return True

            if streaming_engine == "ring":
                if ring_endpoint is None:
                    raise RuntimeError("ring endpoint was not initialized")
                with RingAudioFeeder(ring_endpoint) as feeder:
                    ring_feeder = feeder
                    active_ring_feeder = feeder
                    if audio_backend in {"apple-raw", "apple-vpio"}:
                        preferred_modes = ("raw",) if audio_backend == "apple-raw" else ("vpio", "raw")
                        audio_ctx = AppleVPIOBridge(feeder.callback, preferred_modes=preferred_modes)
                    else:
                        audio_ctx = sd.InputStream(
                            samplerate=SAMPLE_RATE,
                            channels=CHANNELS,
                            dtype=DTYPE,
                            blocksize=FRAME_SAMPLES,
                            device=sounddevice_input_device(),
                            callback=feeder.callback,
                        )
                    with audio_ctx as ctx:
                        if audio_backend in {"apple-raw", "apple-vpio"}:
                            audio_bridge = ctx  # same bridge handles playback via stdin
                            capture_label = f"apple_vpio_bridge mode={ctx.mode}"
                        else:
                            capture_label = f"SoundDevice input={sounddevice_input_device()!r}"
                        ui_update(status=f"🎙️ {capture_label} | ordered ring scheduler ON")
                        update_stage_status("capture", "listening", capture_label)
                        while True:
                            if shutdown_event.is_set():
                                break
                            if not handle_interactive_key(0.05):
                                break
                ring_feeder = None
                active_ring_feeder = None
            else:
                if audio_backend in {"apple-raw", "apple-vpio"}:
                    preferred_modes = ("raw",) if audio_backend == "apple-raw" else ("vpio", "raw")
                    audio_ctx = AppleVPIOBridge(audio_callback, preferred_modes=preferred_modes)
                else:
                    audio_ctx = sd.InputStream(
                        samplerate=SAMPLE_RATE,
                        channels=CHANNELS,
                        dtype=DTYPE,
                        blocksize=FRAME_SAMPLES,
                        device=sounddevice_input_device(),
                        callback=audio_callback,
                    )
                with audio_ctx as ctx:
                    if audio_backend in {"apple-raw", "apple-vpio"}:
                        audio_bridge = ctx
                    update_stage_status("capture", "listening", f"{audio_backend}; legacy endpointing")
                    while True:
                        if not handle_interactive_key(0.05):
                            break
        else:
            while True:
                key = read_key(0.05)
                if key is None:
                    continue
                if key == "\x1b":
                    break
                if key in ("\n", "\r"):
                    if text_buffer.strip():
                        if submit_typed_reply(text_buffer):
                            text_buffer = ""
                    continue
                if key == "\x7f":
                    if text_buffer:
                        text_buffer = text_buffer[:-1]
                        render_input(text_buffer)
                    continue
                if key.isprintable():
                    text_buffer += key
                    render_input(text_buffer)
    finally:
        shutdown_event.set()
        stop_asr_process()
        stop_mt_process()
        audio_bridge = None
        active_ring_feeder = None
        if audio_evidence is not None:
            audio_evidence.close(timeout=5)
        if headless_transport is not None and sys.exc_info()[0] is None:
            headless_transport.close(timeout=1)
        restore_terminal()
        if args.ui != "headless":
            print("\x1b[H\x1b[2JExited.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        if args.ui == "headless":
            update_stage_status("capture", "failed", str(exc))
            ui_update(status=f"Unable to run: {exc}")
        eprint(f"❌ Unable to run: {exc}")
        error_directory = runtime_session_dir or Path(__file__).resolve().parent / "runtime-reports"
        eprint(f"Check the input device/LitLink route and the ASR/TRANSLATION status. Full worker errors: {error_directory / 'pipeline-events.jsonl'}")
        if args.debug:
            import traceback
            traceback.print_exc()
        raise SystemExit(1)
    finally:
        if headless_transport is not None:
            headless_transport.close(timeout=1)
