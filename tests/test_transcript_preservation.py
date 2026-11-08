"""Preserve decoded speech and suppress only proven duplicate source audio."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import importlib.util
import io
from pathlib import Path
from queue import Queue
import sys
import threading
import unittest
from unittest.mock import patch
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]


def load_core():
    name = "transcript_core_" + uuid4().hex
    spec = importlib.util.spec_from_file_location(name, ROOT / "tts-transcribe.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    with patch.object(sys, "argv", [str(spec.origin), "--source-language", "HE", "--target-language", "AR"]), \
            redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        spec.loader.exec_module(module)
    return module


class TranscriptPreservationTests(unittest.TestCase):
    def setUp(self):
        self.core = load_core()

    def result(self, utterance_id, text, start=0, duration=1, **overrides):
        core = self.core
        request = {"session_id": "same-capture-session", "source_language": "HE",
                   "timestamp_start_ns": int(start * 1e9), "timestamp_end_ns": int((start + duration) * 1e9),
                   "samples": round(duration * core.SAMPLE_RATE), "sample_rate": core.SAMPLE_RATE,
                   "channels": 1, "capture_gap": False, **overrides}
        return core.AsrResult(utterance_id, text, False, session_id="same-capture-session",
                              diagnostics={"request": request})

    def test_brief_multilingual_speech_is_not_discarded_by_an_extra_hidden_threshold(self):
        for text in ["שלום.", "כן", "לא", "Ja", "Nein", "Oui", "Evet", "Нет.", "نعم", "لا"]:
            with self.subTest(text=text):
                self.assertTrue(self.core.looks_like_real_text(text))
                self.assertEqual(self.core.sanitize_asr_text(text, final=True), text)
        with patch.object(self.core.args, "min_text_words", 2):
            self.assertFalse(self.core.looks_like_real_text("שלום."))
        with patch.object(self.core.args, "min_text_chars", 8):
            self.assertFalse(self.core.looks_like_real_text("שלום."))

    def test_final_sanitizer_preserves_similar_clauses_corrections_and_emphasis(self):
        for text in [
            "I want to go to London today and I want to go to Paris tomorrow.",
            "We need five blue chairs and we need six red chairs.",
            "Please send the package to Berlin. Please send the invoice to Paris.",
            "My account number is one two three four and your account number is five six seven eight.",
            "מאוד מאוד קשה", "no no please stop", "yes yes yes yes",
        ]:
            with self.subTest(text=text):
                self.assertEqual(self.core.sanitize_asr_text(text, final=True), text)

    def test_dedupe_does_not_equate_negation_numbers_or_different_words(self):
        for left, right in [
            ("Please send the package today", "Please do not send the package today"),
            ("I want five red chairs", "I want six red chairs"),
            ("My appointment is at nine today", "My appointment is at ten today"),
        ]:
            with self.subTest(left=left, right=right):
                self.assertFalse(self.core.final_transcripts_are_near_duplicates(left, right))

    def test_growing_window_hypothesis_replaces_corrected_words_instead_of_appending(self):
        merge = self.core.merge_partial_transcript_with_provenance
        first = ("session", "HE", 0, 1_000_000_000)
        grown = ("session", "HE", 0, 2_000_000_000)
        shifted = ("session", "HE", 1_000_000_000, 3_000_000_000)
        self.assertEqual(merge("we need five chairs", "we need six chairs", first, grown), "we need six chairs")
        self.assertEqual(merge("please send the package", "please do not send the package", first, grown), "please do not send the package")
        self.assertEqual(merge("we need six chairs", "we need five chairs", grown, first), "we need six chairs")
        self.assertEqual(merge("we need six chairs", "six chairs tomorrow", grown, shifted), "we need six chairs tomorrow")

    def test_audio_dedupe_requires_valid_same_source_overlapping_intervals(self):
        core = self.core
        original = core.final_transcript_audio_span(self.result(1, "repeat", start=5))
        overlap = core.final_transcript_audio_span(self.result(2, "repeat", start=5.1))
        self.assertTrue(core.final_transcript_audio_spans_overlap(original, overlap))
        for result in [
            self.result(2, "repeat", start=7),
            self.result(2, "repeat", start=5.5),
            self.result(2, "repeat", start=5, capture_gap=True),
            self.result(2, "repeat", start=5, source_language="AR"),
            self.result(2, "repeat", start=5, session_id="another-session"),
            self.result(2, "repeat", start=5, samples=1),
            self.result(2, "repeat", start=5, sample_rate=48000),
            self.result(2, "repeat", start=5, channels=2),
            core.AsrResult(2, "repeat", False),
        ]:
            with self.subTest(result=result):
                span = core.final_transcript_audio_span(result)
                self.assertFalse(core.final_transcript_audio_spans_overlap(original, span))

    def test_coordinator_keeps_disjoint_repeats_and_logs_final_before_translation(self):
        core = self.core
        stop = threading.Event()

        class FiniteQueue(Queue):
            def task_done(self):
                super().task_done()
                if self.unfinished_tasks == 0:
                    stop.set()

        results = FiniteQueue()
        for result in [
            self.result(1, "מה זה אומר?"),
            self.result(2, "מה זה אומר?", start=10),
            core.AsrResult(3, "מה זה אומר?", False),
            self.result(4, "מה זה אומר?", start=10.05),
            self.result(5, "מה זה לא אומר?", start=10),
            self.result(6, "שלום.", start=20),
            self.result(7, "", start=25),
            self.result(8, "transcription with unavailable translation", start=30),
        ]:
            results.put_nowait(result)
        state = deepcopy(core.ui_state)
        state.update(left_active={}, left_order=[], left_done=[])
        translated, decisions, errors = [], [], []

        def record(result, text, decision, reason=""):
            decisions.append((result.utterance_id, text, decision))

        def translate(task):
            self.assertIn((task.utterance_id, task.text, "accepted_final"), decisions)
            translated.append(task)
            return task.utterance_id != 8

        with patch.dict(core.__dict__, {
            "asr_results": results, "shutdown_event": stop, "ui_state": state,
            "render_dashboard": lambda: None, "ui_update": lambda **kw: errors.append(kw),
            "enqueue_translation_task": translate, "record_asr_decision": record,
            "forced_bos_for_target": lambda _target: None, "ring_endpoint": None,
            "audio_final_transcripts": {}, "latency_traces": {}, "inflight_partial_markers": {},
        }), patch.object(core.args, "stream_audio_translation", False):
            core.coordinator_worker()
        self.assertEqual(errors, [])
        self.assertEqual([task.utterance_id for task in translated], [1, 2, 3, 5, 6, 8])
        self.assertIn((4, "מה זה אומר?", "duplicate_audio_final"), decisions)
        self.assertIn((7, "", "empty_final"), decisions)
        self.assertIn((8, "transcription with unavailable translation", "translation_rejected"), decisions)

    def test_coordinator_replaces_same_start_partial_hypothesis(self):
        core = self.core
        stop = threading.Event()

        class FiniteQueue(Queue):
            def task_done(self):
                super().task_done()
                if self.unfinished_tasks == 0:
                    stop.set()

        results = FiniteQueue()
        first = self.result(1, "we need five chairs", duration=1)
        first.partial = True
        second = self.result(1, "we need six chairs", duration=2)
        second.partial = True
        results.put_nowait(first)
        results.put_nowait(second)
        state = deepcopy(core.ui_state)
        state.update(left_active={}, left_order=[], left_done=[])
        errors = []
        with patch.dict(core.__dict__, {
            "asr_results": results, "shutdown_event": stop, "ui_state": state,
            "render_dashboard": lambda: None, "ui_update": lambda **kw: errors.append(kw),
            "ring_endpoint": None, "latency_traces": {}, "inflight_partial_markers": {},
            "current_partial_utterance_id": None,
        }), patch.object(core.args, "stream_audio_translation", False), \
                patch.object(core.args, "partial_display_policy", "latest"):
            core.coordinator_worker()
            self.assertEqual(core.current_partial_text, "we need six chairs")
            self.assertIn("six chairs", core.strip_ansi(state["left_active"][1]["src"]))
            self.assertNotIn("five", core.strip_ansi(state["left_active"][1]["src"]))
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
