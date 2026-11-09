"""Verified code-switched audio retains its own MT/UI/log source language."""
from __future__ import annotations

from copy import deepcopy
from queue import Queue
import threading
from unittest import TestCase
from unittest.mock import patch

from test_accuracy_integration import load_core
from tts_runtime.web_transport import build_web_snapshot


class VerifiedLanguageRoutingTests(TestCase):
    def setUp(self):
        self.core = load_core()  # Configured incoming HE → AR; typed AR → HE.

    def result(self, uid, text, *, partial=False, verification=None, actual_source=None):
        core = self.core
        request = {"session_id": "same-session", "source_language": "HE", "sample_rate": 16000,
                   "channels": 1, "samples": 96000, "timestamp_start_ns": uid * 10_000_000_000,
                   "timestamp_end_ns": uid * 10_000_000_000 + 6_000_000_000, "capture_gap": False}
        diagnostics = {"request": request}
        if verification is not None:
            diagnostics["final_verification"] = {"status": verification}
        if actual_source is not None:
            diagnostics["verified_source_language"] = actual_source
        return core.AsrResult(uid, text, partial, session_id="same-session", diagnostics=diagnostics)

    def deliver(self, results):
        core = self.core
        stop = threading.Event()
        state = deepcopy(core.ui_state)
        state.update(left_active={}, left_order=[], left_done=[])
        tasks, decisions, updates, languages, texts = [], [], [], {}, {}

        class FiniteQueue(Queue):
            def task_done(self):
                super().task_done()
                if self.unfinished_tasks == 0:
                    stop.set()

        queue = FiniteQueue()
        for result in results:
            queue.put_nowait(result)
        with patch.dict(core.__dict__, {
            "asr_results": queue, "shutdown_event": stop, "ui_state": state,
            "render_dashboard": lambda: None, "ui_update": lambda **kw: updates.append(kw),
            "ring_endpoint": None, "latency_traces": {}, "inflight_partial_markers": {},
            "current_partial_utterance_id": None, "audio_final_transcripts": texts,
            "audio_final_source_languages": languages,
            "record_asr_decision": lambda *args: decisions.append(args),
            "enqueue_translation_task": lambda task: tasks.append(task) or True,
            "forced_bos_for_target": lambda _target: None,
        }), patch.object(core.args, "stream_audio_translation", False):
            core.coordinator_worker()
        return state, tasks, decisions, updates, languages, texts

    def test_verified_english_routes_to_arabic_without_changing_neighboring_hebrew_or_typed_direction(self):
        core = self.core
        english = self.result(11, "The meeting begins tomorrow morning at nine.",
                              verification="verified", actual_source="EN")
        hebrew = self.result(12, "אנחנו צריכים לקבוע פגישה חדשה למחר בבוקר")
        state, tasks, decisions, updates, languages, texts = self.deliver([english, hebrew])
        self.assertEqual(updates, [])
        self.assertEqual([row[2] for row in decisions], ["accepted_final", "accepted_final"])
        self.assertEqual([(task.src_lang, task.target_lang) for task in tasks],
                         [("eng_Latn", "AR"), ("heb_Hebr", "AR")])
        self.assertEqual([state["left_active"][uid]["source_language"] for uid in [11, 12]], ["EN", "HE"])
        self.assertIn("EN:", core.strip_ansi(state["left_active"][11]["src"]))
        self.assertEqual(languages, {11: "EN", 12: "HE"})
        self.assertEqual(english.diagnostics["request"]["source_language"], "HE")

        typed, records = [], []
        with patch.dict(core.__dict__, {"ui_state": state, "audio_final_transcripts": texts,
                "audio_final_source_languages": languages, "render_dashboard": lambda: None,
                "ui_update": lambda **kw: None,
                "schedule_final_transcription_log": lambda *args: None,
                "enqueue_translation_task": lambda task: typed.append(task) or True,
                "forced_bos_for_target": lambda _target: 0,
                "append_session_log": lambda path, record: records.append(record)}), \
                patch.object(core.args, "speak_audio_translation", False), \
                patch.object(core.args, "latency_stats", False):
            for task in tasks:
                core.handle_translation_result(core.TranslationResult("audio", "الترجمة العربية", "AR",
                    utterance_id=task.utterance_id, revision=task.revision, partial=False))
                core.log_final_transcription(task.utterance_id, task.text, "الترجمة العربية")
            self.assertTrue(core.submit_typed_reply("يرجى تأكيد الموعد"))
        self.assertEqual(state["left_active"], {})
        self.assertEqual([entry["source_language"] for entry in state["left_done"]], ["EN", "HE"])
        self.assertEqual([record["transcription_lang"] for record in records], ["EN", "HE"])
        self.assertEqual([(task.src_lang, task.target_lang) for task in typed], [("arb_Arab", "HE")])
        self.assertEqual(core.source_language_code(), "HE")
        self.assertEqual(core.target_language_code(), "AR")
        snapshot = build_web_snapshot(state, {"source_language": "HE", "target_language": "AR"},
                                      "same-session", True, "full", None)
        self.assertEqual([entry["source_language"] for entry in snapshot["transcript"]], ["EN", "HE"])
        self.assertEqual(snapshot["replies"][0]["source_language"], "AR")
        self.assertEqual(snapshot["replies"][0]["target_language"], "HE")

    def test_unverified_or_unsupported_source_hint_cannot_override_configured_language(self):
        cases = [
            self.result(1, "text", actual_source="EN"),
            self.result(2, "text", verification="unresolved", actual_source="EN"),
            self.result(3, "text", verification="pending", actual_source="EN"),
            self.result(4, "text", verification="verified", actual_source="XX"),
            self.result(5, "text", verification="verified"),
        ]
        cases[-1].diagnostics["verified_source_language"] = None
        cases.append(self.result(6, "text", verification="verified", actual_source="en"))
        for result in cases:
            with self.subTest(diagnostics=result.diagnostics):
                self.assertEqual(self.core.result_source_language(result), "HE")
        verified = self.result(7, "text", verification="verified", actual_source="EN")
        self.assertEqual(self.core.result_source_language(verified), "EN")

    def test_unresolved_verification_cannot_use_fallback_or_translate_provisional_words(self):
        core = self.core
        text = "אנחנו צריכים לקבוע פגישה חדשה למחר בבוקר ליד התחנה"
        partial = self.result(11, text, partial=True)
        unresolved = self.result(11, "", verification="unresolved", actual_source="EN")
        unresolved.diagnostics["preview_evidence"] = "worker evidence must not overwrite the displayed words"
        with patch.object(core.args, "empty_final_action", "fallback"):
            state, tasks, decisions, updates, languages, texts = self.deliver([partial, unresolved])
        self.assertEqual(tasks, [])
        self.assertEqual([row[2] for row in decisions], ["unresolved_final"])
        self.assertEqual(languages, {})
        self.assertEqual(texts, {})
        self.assertEqual(state["left_active"], {})
        entry = state["left_done"][0]
        self.assertEqual(entry["source_preview"]["text"], text)
        self.assertTrue(entry["review_required"])
        self.assertFalse(entry["src_final"])
        self.assertFalse(entry["dst_final"])
        self.assertIn("recognition needs review", updates[0]["status"])
