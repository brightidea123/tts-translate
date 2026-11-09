"""Bounded acoustic re-decode plans and timestamp-owned transcript stitching.

This module never promotes a previous text hypothesis into final recognition.
Recovery words must come from a new decode of the same captured samples.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import unicodedata


CORE_SECONDS = 8
CONTEXT_SECONDS = 0.5
MAX_CORES = 4
TIMESTAMP_TOLERANCE_SECONDS = 0.08


@dataclass(frozen=True, slots=True)
class RecoveryWindow:
    start_sample: int
    end_sample: int
    core_start_sample: int
    core_end_sample: int


@dataclass(frozen=True, slots=True)
class RecoveryTranscript:
    text: str = ""
    error: str = ""

    @property
    def valid(self) -> bool:
        return bool(self.text) and not self.error


def recovery_windows(count: int, sample_rate: int = 16000) -> tuple[RecoveryWindow, ...]:
    """Partition capture into at most four balanced cores of at most 8 s each.

    Cores cover every captured sample exactly once. Context is at most 0.5 s
    on each side and is clipped to the original recording's bounds. Balancing
    prevents a tiny trailing remainder from becoming a mostly silent core.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("captured sample count must be a positive integer")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("sample rate must be a positive integer")
    core_samples = CORE_SECONDS * sample_rate
    if count > MAX_CORES * core_samples:
        raise ValueError("recovery audio exceeds the four-core limit")
    context_samples = round(CONTEXT_SECONDS * sample_rate)
    num_cores = (count + core_samples - 1) // core_samples
    boundaries = [round(index * count / num_cores) for index in range(num_cores + 1)]
    return tuple(
        RecoveryWindow(
            max(0, start - context_samples),
            min(count, end + context_samples),
            start,
            end,
        )
        for start, end in zip(boundaries, boundaries[1:])
    )


def _timestamp(value) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _join_word_tokens(tokens: list[str]) -> str:
    # Whisper tokens normally carry their own leading spaces. Preserve those;
    # supply a space for adapters that instead return bare word strings.
    closing = ".,!?;:%،؛؟…)]}»”’"
    opening = "([{«“‘"
    pieces: list[str] = []
    previous = ""
    for token in tokens:
        if previous and not (previous[-1].isspace() or token[0].isspace()
                             or token[0] in closing or previous[-1] in opening):
            pieces.append(" ")
        pieces.append(token)
        previous = token
    return "".join(pieces).strip()


def _lexical_content(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(char for char in normalized if char.isalnum())


def stitch_recovery(
    windows: Sequence[RecoveryWindow],
    summaries: Sequence[Mapping],
    sample_rate: int = 16000,
) -> RecoveryTranscript:
    """Select freshly decoded words by acoustic midpoint, never by text dedupe.

    Each summary needs nonempty ``raw_text`` (or ``text``) and segments with
    ``words`` dictionaries containing ``word``, ``start`` and ``end``. Word
    times are seconds relative to that window, before adding its capture offset.
    Timestamped tokens must cover all decoded lexical content in order. Any
    incomplete or invalid window invalidates the whole candidate. A valid
    timestamp plan is not a measurement of recognition accuracy.
    """
    def invalid(reason: str) -> RecoveryTranscript:
        return RecoveryTranscript(error=reason)

    if not windows or summaries is None or len(windows) != len(summaries):
        return invalid("recovery needs one decoded summary per window")
    if any(not isinstance(window, RecoveryWindow) for window in windows):
        return invalid("invalid recovery window")
    try:
        expected = recovery_windows(windows[-1].core_end_sample, sample_rate)
    except ValueError as exc:
        return invalid(str(exc))
    if tuple(windows) != expected:
        return invalid("recovery windows do not match a complete bounded capture plan")

    tokens: list[str] = []
    for index, (window, summary) in enumerate(zip(windows, summaries)):
        label = f"window {index + 1}"
        if not isinstance(summary, Mapping):
            return invalid(f"{label}: missing decoder summary")
        text = summary.get("raw_text", summary.get("text", ""))
        if not isinstance(text, str) or not text.strip():
            return invalid(f"{label}: empty decoded text")
        if summary.get("error"):
            return invalid(f"{label}: decoder error")
        segments = summary.get("segments")
        if not isinstance(segments, (list, tuple)) or not segments:
            return invalid(f"{label}: missing word timestamps")
        duration = (window.end_sample - window.start_sample) / sample_rate
        previous_start = previous_end = -math.inf
        owned: list[str] = []
        timestamped_tokens: list[str] = []
        usable_words = 0
        for segment in segments:
            if not isinstance(segment, Mapping):
                return invalid(f"{label}: invalid segment")
            segment_text = segment.get("text", "")
            if not isinstance(segment_text, str):
                return invalid(f"{label}: invalid segment text")
            words = segment.get("words")
            if not isinstance(words, (list, tuple)) or (segment_text.strip() and not words):
                return invalid(f"{label}: missing word timestamps")
            for word in words:
                if not isinstance(word, Mapping):
                    return invalid(f"{label}: invalid word")
                token = word.get("word")
                if not isinstance(token, str) or not token.strip():
                    return invalid(f"{label}: empty word token")
                start, end = _timestamp(word.get("start")), _timestamp(word.get("end"))
                if start is None or end is None:
                    return invalid(f"{label}: nonfinite or missing word timestamp")
                if start > end or start < previous_start or end < previous_end:
                    return invalid(f"{label}: unordered word timestamps")
                if start < -TIMESTAMP_TOLERANCE_SECONDS or end > duration + TIMESTAMP_TOLERANCE_SECONDS:
                    return invalid(f"{label}: word timestamps exceed captured audio")
                previous_start, previous_end = start, end
                usable_words += 1
                timestamped_tokens.append(token)
                # Permit rounding around the recording edges, but never give
                # a word ownership outside this exact captured sample span.
                start = max(0.0, min(duration, start))
                end = max(0.0, min(duration, end))
                midpoint = window.start_sample + round((start + end) * 0.5 * sample_rate)
                owned_here = window.core_start_sample <= midpoint < window.core_end_sample
                if index == len(windows) - 1 and midpoint == window.core_end_sample:
                    owned_here = True
                if owned_here:
                    owned.append(token)
        lexical_text = _lexical_content(text)
        if not lexical_text or lexical_text != _lexical_content("".join(timestamped_tokens)):
            return invalid(f"{label}: word alignment does not cover decoded text")
        if not usable_words or not owned:
            return invalid(f"{label}: no timestamped words owned by its core")
        tokens.extend(owned)
    text = _join_word_tokens(tokens)
    return RecoveryTranscript(text=text) if text else invalid("recovery produced no owned text")
