"""Retain preceding words while aligning overlapping live ASR windows.

This is a display/partial-MT hypothesis, never a replacement for final ASR.
Only a mutable tail is revised after words have been retained as stable.
"""
from __future__ import annotations

from dataclasses import dataclass
from difflib import Match, SequenceMatcher
import unicodedata


def word_key(word: str) -> str:
    # Compare lexical tokens across punctuation/casing changes without
    # equating different numbers, negation, or inflected words.
    key = unicodedata.normalize("NFKC", word).casefold()
    meaningful = {"-", "+", "−", "%", "٪"}
    while key and unicodedata.category(key[0]).startswith("P") and key[0] not in meaningful:
        key = key[1:]
    while key and unicodedata.category(key[-1]).startswith("P") and key[-1] not in meaningful:
        key = key[:-1]
    return key or word


@dataclass(frozen=True)
class PreviewMerge:
    text: str
    accepted: bool = True
    reason: str = ""


def merge_preview(previous: str, current: str, previous_span: tuple | None,
                  current_span: tuple | None, stable_prefix: str = "") -> PreviewMerge:
    previous, current = " ".join(previous.split()), " ".join(current.split())
    if not previous:
        return PreviewMerge(current)
    if not current:
        return PreviewMerge(previous, False, "empty preview")
    before, after = previous.split(), current.split()
    old_keys, new_keys = list(map(word_key, before)), list(map(word_key, after))
    stable = stable_prefix.split()
    if before[:len(stable)] != stable:
        stable = []
    fixed = len(stable)
    same_start = False
    if previous_span is not None and current_span is not None:
        if previous_span[:2] != current_span[:2]:
            return PreviewMerge(previous, False, "different audio source")
        if current_span[2] < previous_span[2] or current_span[3] < previous_span[3]:
            return PreviewMerge(previous, False, "older audio window")
        same_start = current_span[2] == previous_span[2]
        if current_span[2] >= previous_span[3]:
            return PreviewMerge(" ".join([*before, *after]))

    # A temporarily shorter read, including just an inner slice of the rolling
    # window, must not blank words already visible. Meaningful corrections can
    # still revise the provisional tail; final ASR can revise the whole phrase.
    if len(after) < len(before) and any(old_keys[index:index + len(after)] == new_keys
                                        for index in range(len(before) - len(after) + 1)):
        return PreviewMerge(previous, False, "temporarily shortened preview")
    if old_keys == new_keys:
        return PreviewMerge(previous)
    if same_start:
        if not fixed:
            return PreviewMerge(current)
        if new_keys[:fixed] == old_keys[:fixed]:
            return PreviewMerge(" ".join([*stable, *after[fixed:]]))

    # A multiword lexical anchor survives punctuation changes at a shifted
    # window boundary. Preserve everything before it, rather than replacing
    # the accumulated sentence with the most recent six seconds.
    blocks = [block for block in SequenceMatcher(None, old_keys, new_keys, autojunk=False).get_matching_blocks()
              if block.size >= 2 and block.a + block.size >= fixed]
    for overlap in range(min(len(before), len(after)), 1, -1):
        if old_keys[-overlap:] == new_keys[:overlap]:
            # Repeated spoken phrases must attach to the most recent matching
            # suffix, rather than an earlier occurrence in retained history.
            blocks.append(Match(len(before) - overlap, 0, overlap))
            break
    if blocks:
        # Start with the earliest reliable anchor in the new reading so a
        # correction in its tail is applied rather than frozen behind a later
        # unchanged phrase. Resolve repeated equal anchors toward recent audio.
        anchor = max(blocks, key=lambda block: (-block.b, block.size, block.a + block.size))
        if anchor.a <= fixed <= anchor.a + anchor.size:
            cut = anchor.b + fixed - anchor.a
            words = [*stable, *after[cut:]]
        else:
            words = [*before[:anchor.a], *after[anchor.b:]]
        return PreviewMerge(" ".join(words))

    if not fixed and (previous_span is None or current_span is None) and old_keys[0] == new_keys[0]:
        return PreviewMerge(current)
    # Unanchored overlapping audio is ambiguous. Keep visible words until a
    # better preview/final instead of deleting history or duplicating a window.
    return PreviewMerge(previous, False, "no reliable overlap anchor")
