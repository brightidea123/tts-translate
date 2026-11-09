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


    def run_preview_results(self, results_to_deliver, timing_observations=None, *, decisions=None,
                            translations=None, status_updates=None, initial_entries=None):
        core = self.core
        stop = threading.Event()
        snapshots, errors = [], []
        state = deepcopy(core.ui_state)
        state.update(left_active={}, left_order=[], left_done=[])
        if initial_entries:
            state["left_active"] = deepcopy(initial_entries)
            state["left_order"] = list(initial_entries)

        class FiniteQueue(Queue):
            def task_done(self):
                super().task_done()
                snapshots.append(deepcopy(state))
                if timing_observations is not None:
                    uid = core.current_partial_utterance_id
                    timing_observations.append(core.partial_endpoint_observation(uid))
                if self.unfinished_tasks == 0:
                    stop.set()

        results = FiniteQueue()
        for result in results_to_deliver:
            results.put_nowait(result)
        with patch.dict(core.__dict__, {
            "asr_results": results, "shutdown_event": stop, "ui_state": state,
            "render_dashboard": lambda: None, "ui_update": lambda **kw: errors.append(kw),
            "ring_endpoint": None, "latency_traces": {}, "inflight_partial_markers": {},
            "current_partial_utterance_id": None, "audio_final_transcripts": {},
            "record_asr_decision": lambda *args: decisions.append(args) if decisions is not None else None,
            "enqueue_translation_task": lambda task: (translations.append(task) if translations is not None else None) or True,
            "forced_bos_for_target": lambda _target: None,
        }), patch.object(core.args, "stream_audio_translation", False):
            core.coordinator_worker()
        if status_updates is None:
            self.assertEqual(errors, [])
        else:
            status_updates.extend(errors)
        return snapshots

    def test_unresolved_final_cannot_commit_fallback_or_erase_visible_words_and_late_work_stays_closed(self):
        core = self.core
        preview = "אנחנו צריכים להזמין חמש כיסאות כחולים ושישה כיסאות אדומים לפגישה מחר"
        first = self.result(7, preview, duration=6)
        first.partial = True
        unresolved = self.result(7, "", duration=24.64)
        unresolved.diagnostics.update(final_recovery={"status": "unresolved"},
                                      preview_evidence="a different worker preview must not replace visible words")
        late = self.result(7, "תודה רבה.", duration=25)
        late.partial = True
        decisions, translations, updates = [], [], []
        with patch.object(core.args, "empty_final_action", "fallback"):
            snapshots = self.run_preview_results([first, unresolved, late], decisions=decisions,
                                                 translations=translations, status_updates=updates)
        self.assertEqual(translations, [])
        self.assertEqual([row[2] for row in decisions], ["unresolved_final"])
        self.assertEqual(snapshots[1]["left_active"], {})
        self.assertEqual(snapshots[1]["left_order"], [])
        archived = snapshots[1]["left_done"][0]
        self.assertEqual(archived["entry_id"], 7)
        self.assertEqual(archived["source_preview"]["text"], preview)
        self.assertFalse(archived["source_preview"]["inflight"])
        self.assertTrue(archived["review_required"])
        self.assertFalse(archived["src_final"])
        self.assertFalse(archived["dst_final"])
        self.assertEqual(snapshots[2]["left_done"], snapshots[1]["left_done"])
        self.assertEqual(snapshots[2]["left_active"], {})
        self.assertEqual(len(updates), 1)
        self.assertIn("recognition needs review", updates[0]["status"])
        state = snapshots[-1]
        before = deepcopy(state)
        with patch.dict(core.__dict__, {"ui_state": state, "render_dashboard": lambda: None}), \
                patch.object(core.args, "speak_audio_translation", False):
            for partial, stream_delta in [(True, False), (True, True), (False, False)]:
                core.handle_translation_result(core.TranslationResult(source="audio", text="late translation",
                    target_lang="AR", utterance_id=7, partial=partial, stream_delta=stream_delta,
                    revision=archived["src_revision"]))
                self.assertEqual(state, before)

    def test_unresolved_final_without_a_displayed_preview_retains_worker_evidence_as_unverified(self):
        core = self.core
        preview = "שלום חברים אנחנו צריכים לקבוע פגישה חדשה למחר בשעה תשע בבוקר"
        unresolved = self.result(2, "", duration=20.08)
        unresolved.diagnostics.update(final_recovery={"status": "unresolved"}, preview_evidence=preview)
        decisions, translations, updates = [], [], []
        with patch.object(core.args, "empty_final_action", "fallback"):
            snapshots = self.run_preview_results([unresolved], decisions=decisions,
                                                 translations=translations, status_updates=updates)
        archived = snapshots[0]["left_done"][0]
        self.assertIn(preview, core.strip_ansi(archived["src"]))
        self.assertEqual(archived["source_preview"], {"text": preview, "stable": "", "inflight": False})
        self.assertTrue(archived["review_required"])
        self.assertFalse(archived["src_final"])
        self.assertFalse(archived["dst_final"])
        self.assertEqual(snapshots[0]["left_active"], {})
        self.assertEqual(translations, [])
        self.assertEqual([row[2] for row in decisions], ["unresolved_final"])

    def test_unresolved_final_replaces_only_a_blank_placeholder_with_unverified_worker_evidence(self):
        core = self.core
        preview = "אנחנו צריכים לקבוע פגישה חדשה למחר בשעה תשע בבוקר ליד התחנה"
        entry = {**core.make_panel_entry(), "src": "HE~: ..",
            "source_preview": {"text": "", "stable": "", "inflight": True}}
        unresolved = self.result(2, "", duration=20.08)
        unresolved.diagnostics.update(final_recovery={"status": "unresolved"}, preview_evidence=preview)
        archived = self.run_preview_results([unresolved], status_updates=[], initial_entries={2: entry})[0]["left_done"][0]
        self.assertEqual(archived["source_preview"]["text"], preview)
        self.assertIn(preview, core.strip_ansi(archived["src"]))
        self.assertEqual(archived["audio_duration_seconds"], 20.08)
        self.assertTrue(archived["review_required"])
        self.assertFalse(archived["src_final"])

    def test_first_word_is_visible_by_default_and_can_be_revised_then_finalized(self):
        core = self.core
        self.assertEqual(core.args.partial_display_policy, "latest")
        first = self.result(1, "שלום", duration=1)
        first.partial = True
        corrected = self.result(1, "שלומך", duration=2)
        corrected.partial = True
        final = self.result(1, "שלום לך!", duration=3)
        late = self.result(1, "late preview", duration=2)
        late.partial = True
        snapshots = self.run_preview_results([first, corrected, final, late])
        initial = snapshots[0]["left_active"][1]
        self.assertEqual(initial["source_preview"], {"text": "שלום", "stable": "", "inflight": False})
        self.assertFalse(initial["src_final"])
        self.assertIn(core.ANSI_PROVISIONAL, initial["src"])
        self.assertEqual(snapshots[1]["left_active"][1]["source_preview"]["text"], "שלומך")
        final_entry = snapshots[2]["left_active"][1]
        self.assertTrue(final_entry["src_final"])
        self.assertNotIn("source_preview", final_entry)
        self.assertEqual(snapshots[3]["left_active"][1], final_entry)

    def test_coordinator_stability_age_uses_audio_seconds_instead_of_result_count(self):
        results = [self.result(1, "שלום", duration=1), self.result(1, "שלום", duration=2),
                   self.result(1, "שלום עולם", duration=3), self.result(1, "שלום", duration=2.5)]
        for result in results:
            result.partial = True
        observations = []
        self.run_preview_results(results, observations)
        self.assertEqual([item[2] for item in observations], [0, 1000, 0, 0])
        self.assertEqual([item[4] for item in observations], [1_000_000_000, 2_000_000_000, 3_000_000_000, 3_000_000_000])

    def test_empty_later_preview_keeps_the_last_visible_words(self):
        first = self.result(1, "שלום", duration=1)
        first.partial = True
        empty = self.result(1, "", duration=2)
        empty.partial = True
        snapshots = self.run_preview_results([first, empty])
        self.assertEqual(snapshots[0]["left_active"][1]["src"], snapshots[1]["left_active"][1]["src"])
        self.assertEqual(snapshots[1]["left_active"][1]["source_preview"]["text"], "שלום")

    def test_rolling_preview_retains_stable_prefix_and_revises_only_new_tail_until_final(self):
        first = self.result(1, "please keep these words while we buy five blue chairs", duration=5)
        second = self.result(1, "please keep these words while we buy five blue chairs today", duration=6)
        shifted = self.result(1, "THESE words while we buy six red chairs today and tomorrow", start=1, duration=6)
        shortened = self.result(1, "these words while we", start=1, duration=7)
        for result in [first, second, shifted, shortened]:
            result.partial = True
        final = self.result(1, "actually replace the entire phrase with the authoritative final", duration=9)
        snapshots = self.run_preview_results([first, second, shifted, shortened, final])
        steady = snapshots[1]["left_active"][1]["source_preview"]["stable"]
        self.assertEqual(steady, "please keep these words while we")
        live = snapshots[2]["left_active"][1]["source_preview"]
        self.assertEqual(live["stable"], steady)
        self.assertEqual(live["text"], "please keep these words while we buy six red chairs today and tomorrow")
        self.assertEqual(snapshots[3]["left_active"][1]["source_preview"], live)
        final_entry = snapshots[4]["left_active"][1]
        self.assertNotIn("source_preview", final_entry)
        self.assertIn("authoritative final", self.core.strip_ansi(final_entry["src"]))
        self.assertNotIn("please keep", self.core.strip_ansi(final_entry["src"]))

    def test_busy_marker_is_two_dots_and_never_blinks_over_visible_words(self):
        core = self.core
        core.reset_current_partial_state(next_utterance_id=1)
        with patch.object(core, "render_dashboard"):
            core.show_inflight_partial_transcript(1)
            entry = core.ui_state["left_active"][1]
            self.assertEqual(core.strip_ansi(entry["src"]), "🗣️ HE~: ..")
            core.ui_set_panel_entry("left", 1, "src", core.format_partial_transcript_for_ui("שלום", ""), False,
                                    revision=2, preview=core.source_preview("שלום", "", inflight=False))
            visible = entry["src"]
            core.show_inflight_partial_transcript(1)
            self.assertEqual(entry["src"], visible)
            self.assertEqual(entry["src_revision"], 2)
            self.assertTrue(entry["source_preview"]["inflight"])
            core.clear_inflight_partial_marker(1)
            self.assertFalse(entry["source_preview"]["inflight"])
            self.assertEqual(entry["src"], visible)
            core.show_inflight_partial_transcript(1)
            core.inflight_partial_markers[1] = core.time.perf_counter() - 30
            core.sweep_inflight_partial_markers()
            self.assertEqual(core.ui_state["left_active"][1]["src"], visible)

    def test_pending_hint_cannot_copy_another_utterance_or_resurrect_final_text(self):
        core = self.core
        core.reset_current_partial_state(next_utterance_id=2)
        core.current_partial_display_text = "another utterance"
        core.current_partial_display_locked_prefix = "another"
        with patch.object(core, "render_dashboard"):
            core.show_inflight_partial_transcript(3)
            self.assertEqual(core.ui_state["left_active"][3]["source_preview"]["text"], "")
            core.show_inflight_partial_transcript(1)
            self.assertNotIn(1, core.ui_state["left_active"])
            core.ui_set_panel_entry("left", 3, "src", "HE: finalized", True, revision=4)
            before = deepcopy(core.ui_state["left_active"][3])
            core.show_inflight_partial_transcript(3)
            self.assertEqual(core.ui_state["left_active"][3], before)
            core.ui_set_panel_entry("left", 3, "dst", "AR: final translation", True, revision=4)
            self.assertNotIn(3, core.ui_state["left_active"])
            core.show_inflight_partial_transcript(3)
            self.assertNotIn(3, core.ui_state["left_active"])

    def test_placeholder_waits_for_active_capture_then_expires_without_guessing_from_text(self):
        core = self.core
        core.reset_current_partial_state(next_utterance_id=1)
        with patch.object(core, "render_dashboard"), patch.object(core, "capture_utterance_active", return_value=True):
            core.show_inflight_partial_transcript(1)
            core.inflight_partial_markers[1] = core.time.perf_counter() - 30
            core.sweep_inflight_partial_markers()
            self.assertIn(1, core.ui_state["left_active"])
        with patch.object(core, "render_dashboard"), patch.object(core, "capture_utterance_active", return_value=False):
            core.sweep_inflight_partial_markers()
            self.assertNotIn(1, core.ui_state["left_active"])
        self.assertFalse(core.placeholder_only_entry({"src": "HE~: ..", "src_final": False}))

    def test_empty_result_does_not_orphan_card_when_request_timer_is_cleared(self):
        core = self.core
        core.reset_current_partial_state(next_utterance_id=1)
        with patch.object(core, "render_dashboard"), patch.object(core, "capture_utterance_active", return_value=True):
            core.show_inflight_partial_transcript(1)
            core.clear_inflight_partial_marker(1)
            core.ui_state["left_active"][1]["source_placeholder_since"] -= 30
            self.assertNotIn(1, core.inflight_partial_markers)
            core.sweep_inflight_partial_markers()
            self.assertIn(1, core.ui_state["left_active"])
        with patch.object(core, "render_dashboard") as render, \
                patch.object(core, "capture_utterance_active", return_value=False):
            core.sweep_inflight_partial_markers()
            self.assertNotIn(1, core.ui_state["left_active"])
            self.assertNotIn(1, core.ui_state["left_order"])
            render.assert_called_once()

    def test_final_pending_protects_empty_card_until_result_or_failure(self):
        core = self.core
        core.reset_current_partial_state(next_utterance_id=1)
        with patch.object(core, "render_dashboard"), patch.object(core, "capture_utterance_active", return_value=False):
            core.show_inflight_partial_transcript(1)
            core.clear_inflight_partial_marker(1)
            core.ui_state["left_active"][1]["source_placeholder_since"] -= 30
            core.mark_final_asr_pending(1)
            core.sweep_inflight_partial_markers()
            self.assertIn(1, core.ui_state["left_active"])
            core.clear_final_asr_pending(1)
            core.sweep_inflight_partial_markers()
            self.assertNotIn(1, core.ui_state["left_active"])

    def test_empty_card_cleanup_preserves_actual_source_translation_and_final_text(self):
        core = self.core
        with patch.object(core, "render_dashboard"), patch.object(core, "capture_utterance_active", return_value=False):
            for uid in [1, 2, 3, 4]:
                core.show_inflight_partial_transcript(uid)
                core.clear_inflight_partial_marker(uid)
                core.ui_state["left_active"][uid]["source_placeholder_since"] -= 30
            core.ui_set_panel_entry("left", 2, "src", core.format_partial_transcript_for_ui("שלום", ""), False,
                                    preview=core.source_preview("שלום", "", inflight=False))
            core.ui_set_panel_entry("left", 3, "dst", "AR~: ترجمة", False)
            core.ui_set_panel_entry("left", 4, "src", "HE: מקור סופי", True)
            self.assertNotIn("source_placeholder_since", core.ui_state["left_active"][2])
            core.sweep_inflight_partial_markers()
            self.assertNotIn(1, core.ui_state["left_active"])
            self.assertEqual(set(core.ui_state["left_active"]), {2, 3, 4})
            core.prune_empty_source_placeholders(force=True)
            self.assertEqual(set(core.ui_state["left_active"]), {2, 3, 4})

    def test_placeholder_housekeeping_is_independent_of_optional_watchdog_and_stop_clears_fresh_blanks(self):
        core = self.core
        with patch.object(core, "render_dashboard"), patch.object(core, "capture_utterance_active", return_value=False), \
                patch.object(core.args, "stuck_partial_watchdog", False):
            core.show_inflight_partial_transcript(1)
            core.clear_inflight_partial_marker(1)
            core.ui_state["left_active"][1]["source_placeholder_since"] -= 30
            core.sweep_inflight_partial_markers()
            self.assertNotIn(1, core.ui_state["left_active"])
            core.show_inflight_partial_transcript(2)
            core.prune_empty_source_placeholders()
            self.assertIn(2, core.ui_state["left_active"])
            core.prune_empty_source_placeholders(force=True)
            self.assertNotIn(2, core.ui_state["left_active"])


if __name__ == "__main__":
    unittest.main()
