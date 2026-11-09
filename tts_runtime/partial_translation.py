"""Bounded batching of confirmed source prefixes for live translation."""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil


def _extends(text: str, prefix: str) -> bool:
    return not prefix or text == prefix or text.startswith(prefix + " ")


@dataclass(slots=True)
class PartialTranslationCandidate:
    text: str
    locked_prefix: str
    revision: int
    stable_text: str
    stable_changed_at: float
    last_sent_text: str = ""
    last_admitted_at: float | None = None

    def ready(self, now: float, interval: float) -> bool:
        return (now - self.stable_changed_at >= interval
                and (self.last_admitted_at is None or now - self.last_admitted_at >= interval))

    def needs_refresh(self, text: str, now: float, *, min_delta_words: int, interval: float) -> bool:
        """Batch small additions to long prefixes without starving live MT.

        Each request retranslates its entire source prefix to retain context.
        Require proportionate growth, or at most six seconds between eligible
        refreshes, so two new words cannot repeatedly retransmit a long phrase.
        The first preview and corrected prefixes keep their existing deadline.
        """
        if not text or text == self.last_sent_text or not self.ready(now, interval):
            return False
        if not self.last_sent_text or not _extends(text, self.last_sent_text):
            return True
        previous_words = len(self.last_sent_text.split())
        added_words = len(text.split()) - previous_words
        minimum = max(1, min_delta_words)
        if added_words < minimum:
            return False
        if added_words >= max(minimum, ceil(previous_words * 0.25)):
            return True
        return (self.last_admitted_at is not None
                and now - self.last_admitted_at >= max(6.0, interval))

    def admitted(self, text: str, now: float) -> None:
        self.last_sent_text = text
        self.last_admitted_at = now


def update_candidate(previous: PartialTranslationCandidate | None, *, text: str,
                     locked_prefix: str, revision: int, stable_text: str,
                     now: float, batch_stable_prefix: bool) -> PartialTranslationCandidate:
    changed_at = now
    last_sent = ""
    last_admitted = None
    if previous is not None:
        if stable_text == previous.stable_text:
            changed_at = previous.stable_changed_at
        elif (batch_stable_prefix and previous.stable_text
              and previous.stable_text != previous.last_sent_text
              and _extends(stable_text, previous.stable_text)):
            # Already-confirmed, unsent words keep their original batching
            # deadline while speech extends them. Waiting for the complete
            # growing prefix to become quiet starves live MT until final ASR.
            changed_at = previous.stable_changed_at
        if _extends(stable_text, previous.last_sent_text):
            last_sent = previous.last_sent_text
        last_admitted = previous.last_admitted_at
    return PartialTranslationCandidate(text, locked_prefix, revision, stable_text,
                                       changed_at, last_sent, last_admitted)
