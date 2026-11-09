"""Narrow consistency checks that request fresh acoustic verification.

Ordinary thanks are valid speech. These checks require a substantial conflicting
preview or strong decoder-loop evidence; they never delete repeated words.
"""
from __future__ import annotations

import math
import re

from .asr_verification import lexical_words


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold(), re.UNICODE)


_SHORT_THANKS = frozenset(
    tuple(_words(text)) for text in (
        "thank you", "thanks", "תודה", "תודה רבה", "شكرا", "شكرا لكم", "شكراً جزيلاً",
        "danke", "danke schön", "merci", "merci beaucoup", "teşekkürler", "teşekkür ederim", "спасибо",
    )
)
_VIDEO_OUTRO = re.compile(
    r"thanks? (?:you )?for watching|thank you for watching|"
    r"תודה (?:רבה )?(?:על הצפייה|שצפיתם)|כתוביות (?:ותרגום|מאת)|"
    r"شكرا(?:ً)? (?:لكم )?على (?:المشاهدة|المتابعة)|"
    r"danke fürs zuschauen|merci (?:d'avoir regardé|de votre attention)|"
    r"спасибо за просмотр|подписывайтесь на канал|"
    r"izlediğiniz için teşekkür|abone olmayı unutmayın",
    re.IGNORECASE,
)


def is_outro_candidate(text: str) -> bool:
    words = _words(text)
    return bool(words and len(words) <= 14 and (tuple(words) in _SHORT_THANKS or _VIDEO_OUTRO.search(text)))


def outro_conflicts_with_preview(text: str, preview: str) -> bool:
    current, previous = _words(text), _words(preview)
    if len(previous) < 8 or not is_outro_candidate(text):
        return False
    overlap = len(set(current) & set(previous)) / max(1, len(set(current)))
    return overlap < 0.5


def decoder_repetition_loop(text: str, compression_ratio: float | None) -> bool:
    """Identify dominant tandem loops only with high decoder compression.

    This is a request to re-decode audio, not permission to deduplicate text.
    Two spoken copies and brief emphasis remain valid. A one-word loop needs
    six copies; longer cycles need at least three copies and nine words. Its
    contiguous periodic span must occupy half the output, or at least 24 words
    and one quarter of the output. Partial cycles count toward span coverage,
    allowing a slightly changed prefix without rewriting any lexical token.
    """
    if isinstance(compression_ratio, bool):
        return False
    try:
        compression = float(compression_ratio)
    except (TypeError, ValueError, OverflowError):
        return False
    if not math.isfinite(compression) or compression <= 2.9:
        return False
    words = lexical_words(text)
    count = len(words)
    for period in range(1, min(64, count // 3) + 1):
        run = period
        for index in range(period, count):
            run = run + 1 if words[index] == words[index - period] else period
            if period == 1:
                substantial = run >= 6
            else:
                substantial = run >= 3 * period and run >= 9
            if substantial and (run * 2 >= count or (run >= 24 and run * 4 >= count)):
                return True
    return False
