"""Observable ASR review signals; these are not measured recognition accuracy.

No metric here is WER or CER. Decoder probabilities and audio characteristics
only decide which clips deserve human review; they never reject a transcript.
"""
from __future__ import annotations

from collections import Counter
import math
import re
from typing import Any

import numpy as np
from .text_output import WHISPER_CONTROL_TOKEN


DEFAULT_THRESHOLDS = {
    "quiet_dbfs": -55.0,
    "clipped_fraction": 0.001,
    "dc_offset_fraction": 0.05,
    "min_avg_logprob": -1.0,
    "max_no_speech_prob": 0.6,
    "max_compression_ratio": 2.4,
}

CONTROL_TOKEN = re.compile(
    r"<end_of_turn>|<start_of_turn>|</?think\b[^>]*>|"
    r"<\|(?:im_start|im_end|eot_id|endoftext|start_header_id|end_header_id|channel|end)\|?>|"
    r"<(?:bos|eos|pad)>", re.IGNORECASE,
)


def pcm_quality(audio, sample_rate: int = 16000) -> dict[str, Any]:
    """Describe mono signed 16-bit PCM without altering any sample."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if isinstance(audio, (bytes, bytearray, memoryview)):
        if len(audio) % 2:
            raise ValueError("Int16 PCM must contain complete two-byte samples")
        samples = np.frombuffer(audio, dtype="<i2")
    else:
        samples = np.asarray(audio)
        if samples.ndim != 1 or samples.dtype.kind != "i" or samples.dtype.itemsize != 2:
            raise ValueError("Expected mono Int16 PCM")
    count = int(samples.size)
    if not count:
        return {
            "samples": 0, "sample_rate": sample_rate, "duration_seconds": 0.0,
            "peak_abs": 0, "peak_fraction": 0.0, "rms_dbfs": -120.0,
            "dc_offset_fraction": 0.0, "clipped_fraction": 0.0,
            "zero_fraction": 1.0, "near_silence": True,
            "accuracy_measured": False,
        }
    # int16 abs(-32768) overflows; widen before measuring peaks/clipping.
    wide = samples.astype(np.int32)
    floating = wide.astype(np.float64) / 32768.0
    peak = int(np.max(np.abs(wide)))
    rms = float(np.sqrt(np.mean(floating * floating)))
    dbfs = max(-120.0, 20.0 * math.log10(max(rms, 1e-12)))
    return {
        "samples": count, "sample_rate": sample_rate,
        "duration_seconds": count / sample_rate,
        "peak_abs": peak, "peak_fraction": peak / 32768.0,
        "rms_dbfs": dbfs,
        "dc_offset_fraction": float(np.mean(floating)),
        "clipped_fraction": float(np.mean(np.abs(wide) >= 32760)),
        "zero_fraction": float(np.mean(wide == 0)),
        "near_silence": dbfs < DEFAULT_THRESHOLDS["quiet_dbfs"],
        "accuracy_measured": False,
    }


def _finite(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def final_preview_disagrees(final_text: str, preview_text: str, audio_seconds: float) -> bool:
    """Flag a collapsed final despite substantial preview speech; never replace it."""
    words = lambda value: re.findall(r"\w+", str(value).casefold(), re.UNICODE)
    final, preview = words(final_text), words(preview_text)
    if audio_seconds < 6 or len(preview) < 8 or len(final) > max(5, len(preview) // 2):
        return False
    shared = len(set(final) & set(preview)) / max(1, len(set(preview)))
    return shared < 0.25


def summarize_whisper_result(result: dict[str, Any]) -> dict[str, Any]:
    """Retain Whisper's segment diagnostics instead of discarding them."""
    if not isinstance(result, dict):
        raise ValueError("Whisper result must be a dictionary")
    segments = []
    for segment in result.get("segments") or []:
        if not isinstance(segment, dict):
            continue
        segments.append({
            "id": segment.get("id"), "text": str(segment.get("text", "")),
            **{key: _finite(segment.get(key)) for key in (
                "start", "end", "temperature", "avg_logprob",
                "no_speech_prob", "compression_ratio",
            )},
        })
    logprobs = [s["avg_logprob"] for s in segments if s["avg_logprob"] is not None]
    speech = [s["no_speech_prob"] for s in segments if s["no_speech_prob"] is not None]
    compression = [s["compression_ratio"] for s in segments if s["compression_ratio"] is not None]
    return {
        "raw_text": str(result.get("text", "")),
        "detected_language": result.get("language"),
        "segment_count": len(segments), "segments": segments,
        "min_avg_logprob": min(logprobs) if logprobs else None,
        "mean_avg_logprob": sum(logprobs) / len(logprobs) if logprobs else None,
        "max_no_speech_prob": max(speech) if speech else None,
        "max_compression_ratio": max(compression) if compression else None,
        "confidence_proxies_only": True, "accuracy_measured": False,
    }


def issue_reasons(
    pcm_metrics: dict[str, Any], decoded_text: str, *,
    raw_text: str | None = None, decoder_quality: dict[str, Any] | None = None,
    expected_language: str | None = None, error: str | Exception | None = None,
    gap: bool = False, thresholds: dict[str, float] | None = None,
) -> list[str]:
    """Return review reasons, never an assertion that a transcript is wrong."""
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    reasons = []
    if not pcm_metrics.get("samples", 0):
        reasons.append("empty_audio")
    elif pcm_metrics.get("rms_dbfs", -120.0) < limits["quiet_dbfs"]:
        reasons.append("quiet_audio")
    if pcm_metrics.get("clipped_fraction", 0.0) >= limits["clipped_fraction"]:
        reasons.append("clipped_audio")
    if abs(pcm_metrics.get("dc_offset_fraction", 0.0)) >= limits["dc_offset_fraction"]:
        reasons.append("large_dc_offset")
    if not str(decoded_text or "").strip():
        reasons.append("empty_transcript")
        if raw_text and raw_text.strip():
            reasons.append("filtered_transcript")
    if CONTROL_TOKEN.search(raw_text or decoded_text or "") or WHISPER_CONTROL_TOKEN.search(raw_text or decoded_text or ""):
        reasons.append("model_control_tokens")
    words = str(decoded_text or "").casefold().split()
    if len(words) >= 8 and (max(Counter(words).values()) / len(words) > 0.4 or len(set(words)) / len(words) < 0.35):
        reasons.append("repetitive_transcript")
    quality = decoder_quality or {}
    for field, comparison, reason in (
        ("min_avg_logprob", lambda value: value < limits["min_avg_logprob"], "low_decoder_logprob"),
        ("max_no_speech_prob", lambda value: value > limits["max_no_speech_prob"], "high_no_speech_probability"),
        ("max_compression_ratio", lambda value: value > limits["max_compression_ratio"], "high_compression_ratio"),
    ):
        value = _finite(quality.get(field))
        if value is not None and comparison(value):
            reasons.append(reason)
    detected = quality.get("detected_language")
    if detected and expected_language and str(detected).casefold() != str(expected_language).casefold():
        reasons.append("language_mismatch")
    if error:
        reasons.append("decoder_error")
    if gap:
        reasons.append("capture_discontinuity")
    return reasons
