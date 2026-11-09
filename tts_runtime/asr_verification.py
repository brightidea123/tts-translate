"""Conservative agreement checks for a fresh acoustic ASR verification pass.

This module does not decode audio or claim measured recognition accuracy. It
requires explicit language-detection provenance, usable decoder diagnostics and
compatible source text before a caller can accept a verification result.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from difflib import SequenceMatcher
import math
import re
import unicodedata


_SCRIPTS = {
    "AR": "ARABIC", "HE": "HEBREW", "DE": "LATIN", "TR": "LATIN",
    "FR": "LATIN", "RU": "CYRILLIC", "EN": "LATIN", "ES": "LATIN",
}
_SHORT_RESPONSES = {
    "AR": {"نعم", "لا", "شكرا"}, "HE": {"כן", "לא", "תודה"},
    "DE": {"ja", "nein", "ok"}, "TR": {"evet", "hayır", "yok", "var"},
    "FR": {"oui", "non", "merci"}, "RU": {"да", "нет", "ага", "спасибо"},
    "EN": {"yes", "no", "ok", "okay", "thanks"}, "ES": {"sí", "si", "no", "gracias"},
}


@dataclass(frozen=True, slots=True)
class VerificationAssessment:
    verified: bool = False
    text: str = ""
    language: str = ""
    reason: str = ""


def lexical_words(text: str) -> tuple[str, ...]:
    if not isinstance(text, str):
        return ()
    normalized = unicodedata.normalize("NFKC", text).casefold()
    # Formatting direction marks and combining accents are not lexical words.
    normalized = "".join(char for char in normalized
                         if unicodedata.category(char) not in {"Cf", "Mn", "Mc", "Me"})
    normalized = "".join(str(unicodedata.decimal(char)) if char.isdecimal() else char
                         for char in normalized)
    return tuple(re.findall(r"[^\W_]+", normalized, re.UNICODE))


def _numbers(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = "".join(str(unicodedata.decimal(char)) if char.isdecimal() else char
                         for char in normalized)
    return tuple(re.findall(r"[+−-]?\d+(?:[.,]\d+)*(?:%|٪)?", normalized))


def equivalent_text(left: str, right: str) -> bool:
    """Exact lexical equality while tolerating Unicode/spacing/punctuation."""
    if not isinstance(left, str) or not isinstance(right, str):
        return False
    return bool(lexical_words(left)) and lexical_words(left) == lexical_words(right) and _numbers(left) == _numbers(right)


def _language(value) -> str:
    return value.strip().upper() if isinstance(value, str) else ""


def _finite(value) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def assess_verification(
    candidate_text: str,
    verification_summary: Mapping,
    *,
    expected_language: str | None = None,
    require_agreement: bool = True,
    require_auto: bool = True,
) -> VerificationAssessment:
    """Assess a newly decoded hypothesis, never merely a forced language label.

    ``require_auto=False`` permits the caller's subsequent forced-language
    confirmation, normally with ``expected_language`` from a prior accepted
    automatic detection. Ordinary short thanks are not blacklisted.
    """
    def reject(reason: str) -> VerificationAssessment:
        return VerificationAssessment(reason=reason)

    if not isinstance(verification_summary, Mapping):
        return reject("missing verification summary")
    if require_auto and verification_summary.get("language_auto_detected") is not True:
        return reject("verification language was not automatically detected")
    language = _language(verification_summary.get("detected_language"))
    if language not in _SCRIPTS:
        return reject("unsupported or missing verification language")
    if expected_language is not None and _language(expected_language) != language:
        return reject("verification language disagrees with expected language")
    raw_text = verification_summary.get("raw_text")
    if not isinstance(raw_text, str) or not raw_text.strip():
        return reject("empty verification transcript")
    text = " ".join(raw_text.split())
    mean = _finite(verification_summary.get("mean_avg_logprob"))
    minimum = _finite(verification_summary.get("min_avg_logprob"))
    no_speech = _finite(verification_summary.get("max_no_speech_prob"))
    compression = _finite(verification_summary.get("max_compression_ratio"))
    if any(value is None for value in (mean, minimum, no_speech, compression)):
        return reject("missing or nonfinite verification diagnostics")
    if mean < -.8 or not 0 <= no_speech <= .5 or not 0 < compression <= 2.4:
        return reject("verification decoder quality is insufficient")
    letters = [char for char in unicodedata.normalize("NFKC", text) if char.isalpha()]
    words = lexical_words(text)
    short_response = len(words) == 1 and words[0] in _SHORT_RESPONSES[language]
    if len(letters) < (2 if short_response else 4):
        return reject("insufficient verification script evidence")
    expected_script = _SCRIPTS[language]
    script_count = sum(expected_script in unicodedata.name(char, "") for char in letters)
    if script_count / len(letters) < .5:
        return reject("verification text does not fit the detected language script")
    if require_agreement:
        if not isinstance(candidate_text, str):
            return reject("missing candidate text for agreement")
        candidate = lexical_words(candidate_text)
        if not candidate or not words:
            return reject("missing candidate text for agreement")
        if _numbers(candidate_text) != _numbers(text):
            return reject("verification changed numbers")
        length_ratio = min(len(candidate), len(words)) / max(len(candidate), len(words))
        agreement = SequenceMatcher(None, candidate, words, autojunk=False).ratio()
        if length_ratio < .65 or agreement < .72:
            return reject("verification text does not sufficiently agree with candidate")
    return VerificationAssessment(True, text, language, "fresh acoustic verification agreed")
