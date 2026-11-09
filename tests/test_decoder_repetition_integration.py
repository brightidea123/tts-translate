"""Looping ASR finals require fresh audio recovery before committed translation."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from queue import Queue
import threading
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from test_accuracy_integration import load_core
from tts_runtime.asr_quality import summarize_whisper_result
from tts_runtime.asr_recovery import recovery_windows


LOOP = "אנחנו צריכים לקבוע פגישה. " * 8
PREVIEW = "אנחנו צריכים להזמין חמש כיסאות כחולים ושישה כיסאות אדומים לפגישה מחר בבוקר"
FRESH = ("אנחנו צריכים להזמין חמש כיסאות כחולים ושישה כיסאות אדומים לקראת הפגישה מחר בבוקר "
         "ליד תחנת הרכבת ולבדוק את הכתובת החדשה עם המנהל עוד היום")


class DecoderRepetitionIntegrationTests(TestCase):
    def setUp(self):
        self.core = load_core()
        self.core.asr_request_context = {"utterance_id": 7, "request_id": "same-captured-audio"}
        self.audio = ((np.arange(363520, dtype=np.int32) % 12000) - 6000).astype(np.int16)
        self.windows = recovery_windows(int(self.audio.size))

    def summary(self, text, *, compression=5.2, words=None):
        segment = {"text": text, "start": 0, "end": 6, "temperature": 0,
                   "avg_logprob": -.2, "no_speech_prob": .1, "compression_ratio": compression}
        if words is not None:
            segment["words"] = words
        return summarize_whisper_result({"text": text, "language": "he", "segments": [segment]})

    def primary(self, pcm, **_kwargs):
        self.core.prepare_asr_audio(pcm)
        self.core.last_decoder_quality.update(self.summary(LOOP))
        return LOOP.strip()

    def decode_final(self, retry):
        with patch.object(self.core, "_transcribe_pcm_i16_impl", side_effect=self.primary), \
                patch.object(self.core, "effective_lane_backend", return_value="mlx-whisper"), \
                patch.object(self.core, "transcribe_audio_mlx_whisper", side_effect=retry) as decode:
            text = self.core.transcribe_pcm_i16(self.audio, partial=False)
        result = self.core.AsrResult(7, text, False, diagnostics=deepcopy(self.core.last_asr_diagnostics))
        return result, decode

    def deliver(self, results):
        core = self.core
        stop = threading.Event()
        state = deepcopy(core.ui_state)
        state.update(left_active={}, left_order=[], left_done=[])
        tasks, decisions, updates = [], [], []

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
            "current_partial_utterance_id": None, "audio_final_transcripts": {}, "audio_final_source_languages": {},
            "record_asr_decision": lambda *args: decisions.append(args),
            "enqueue_translation_task": lambda task: tasks.append(task) or True,
            "forced_bos_for_target": lambda _target: None,
        }), patch.object(core.args, "stream_audio_translation", False):
            core.coordinator_worker()
        return state, tasks, decisions, updates

    def test_looping_final_is_redecoded_from_exact_audio_and_only_fresh_words_reach_mt(self):
        calls, expected = [], []
        source_words = FRESH.split()
        def retry(audio, **kwargs):
            window = self.windows[len(calls)]
            np.testing.assert_array_equal(audio, self.audio[window.start_sample:window.end_sample].astype(np.float32) / 32768)
            self.assertTrue(kwargs["word_timestamps"])
            self.assertNotIn("language_override", kwargs)
            tokens = source_words[len(calls) * 8:(len(calls) + 1) * 8]
            expected.extend(tokens)
            offset = (window.core_start_sample - window.start_sample) / 16000
            words = [{"word": " " + token, "start": offset + .1 + index * .3,
                      "end": offset + .3 + index * .3} for index, token in enumerate(tokens)]
            text = " ".join(tokens)
            calls.append(window)
            self.core.last_decoder_quality.clear()
            self.core.last_decoder_quality.update(self.summary(text, compression=1.2, words=words))
            return text
        result, decode = self.decode_final(retry)
        self.assertEqual(result.text, " ".join(expected))
        self.assertEqual(decode.call_count, 3)
        recovery = result.diagnostics["final_recovery"]
        self.assertEqual((recovery["status"], recovery["reason"]), ("recovered", "decoder_repetition_loop"))
        self.assertIn("decoder_repetition_loop", result.diagnostics["issues"])
        self.assertEqual(result.diagnostics["primary_decoder"]["raw_text"], LOOP)
        prepared = (self.audio.astype(np.float32) / 32768).astype("<f4").tobytes()
        self.assertEqual(result.diagnostics["model_input"]["sha256"], hashlib.sha256(prepared).hexdigest())
        self.assertEqual(result.diagnostics["request"]["request_id"], "same-captured-audio")
        self.assertEqual(sum(w.core_end_sample - w.core_start_sample for w in calls), self.audio.size)
        _state, tasks, decisions, updates = self.deliver([result])
        self.assertEqual(updates, [])
        self.assertEqual([row[2] for row in decisions], ["accepted_final"])
        self.assertEqual([task.text for task in tasks], [result.text])
        self.assertNotEqual(tasks[0].text, LOOP.strip())

    def test_recovery_window_that_still_loops_stops_without_committing_or_recursing(self):
        def retry(_audio, **_kwargs):
            tokens = LOOP.split()
            words = [{"word": " " + token, "start": index * .1, "end": index * .1 + .08}
                     for index, token in enumerate(tokens)]
            self.core.last_decoder_quality.clear()
            self.core.last_decoder_quality.update(self.summary(LOOP, words=words))
            return LOOP
        result, decode = self.decode_final(retry)
        self.assertEqual(result.text, "")
        self.assertEqual(decode.call_count, 1)
        self.assertEqual(result.diagnostics["final_recovery"]["status"], "unresolved")
        self.assertIn("final_recovery_unresolved", result.diagnostics["issues"])
        self.assertEqual(self.core.last_decoder_quality["raw_text"], LOOP)
        _state, tasks, decisions, updates = self.deliver([result])
        self.assertEqual(tasks, [])
        self.assertEqual([row[2] for row in decisions], ["unresolved_final"])
        self.assertIn("needs review", updates[0]["status"])

    def test_recovery_decoder_exception_does_not_publish_the_original_loop(self):
        def retry(_audio, **_kwargs):
            raise RuntimeError("decoder unavailable")
        result, decode = self.decode_final(retry)
        self.assertEqual(result.text, "")
        self.assertEqual(decode.call_count, 1)
        self.assertEqual(result.diagnostics["final_recovery"]["status"], "unresolved")
        self.assertEqual(result.diagnostics["final_recovery"]["reason"], "decoder_repetition_loop")

    def test_two_spoken_sentence_copies_do_not_add_recovery_or_remove_words(self):
        text = "אנחנו צריכים לקבוע פגישה למחר בבוקר. " * 2
        def primary(pcm, **_kwargs):
            self.core.prepare_asr_audio(pcm)
            self.core.last_decoder_quality.update(self.summary(text, compression=2.0))
            return text.strip()
        with patch.object(self.core, "_transcribe_pcm_i16_impl", side_effect=primary), \
                patch.object(self.core, "recover_inconsistent_final") as recovery, \
                patch.object(self.core, "transcribe_audio_mlx_whisper") as decode:
            self.assertEqual(self.core.transcribe_pcm_i16(self.audio, partial=False), text.strip())
        recovery.assert_not_called()
        decode.assert_not_called()

    def test_preview_loop_is_review_only_but_ordinary_repeated_emphasis_remains_visible(self):
        core = self.core
        loop = core.AsrResult(7, LOOP, True, diagnostics={"decoder": {"max_compression_ratio": 5.2}})
        emphasis = core.AsrResult(7, "לא לא לא בבקשה תפסיק", True,
                                  diagnostics={"decoder": {"max_compression_ratio": 1.2}})
        decisions = []
        with patch.object(core, "record_asr_decision", side_effect=lambda *args: decisions.append(args)), \
                patch.object(core, "clear_inflight_partial_marker"):
            core.receive_asr_stage_result(loop)
            self.assertTrue(core.asr_results.empty())
            core.receive_asr_stage_result(emphasis)
            self.assertIs(core.asr_results.get_nowait(), emphasis)
            core.asr_results.task_done()
        self.assertEqual([row[2] for row in decisions], ["partial_review_only"])

    def test_publication_guard_blocks_unrecovered_legacy_loop_and_partial_fallback(self):
        core = self.core
        preview = core.AsrResult(7, PREVIEW, True)
        final = core.AsrResult(7, LOOP, False, diagnostics={"decoder": {"max_compression_ratio": 5.2}})
        late = core.AsrResult(7, LOOP, True)
        with patch.object(core.args, "empty_final_action", "fallback"):
            state, tasks, decisions, updates = self.deliver([preview, final, late])
        self.assertEqual(tasks, [])
        self.assertEqual([row[2] for row in decisions], ["unresolved_final"])
        self.assertEqual(final.diagnostics["final_recovery"]["status"], "unresolved")
        self.assertEqual(state["left_active"], {})
        self.assertEqual(len(state["left_done"]), 1)
        entry = state["left_done"][0]
        self.assertTrue(entry["review_required"])
        self.assertFalse(entry["src_final"])
        self.assertEqual(entry["source_preview"]["text"], PREVIEW)
        self.assertNotIn(LOOP.strip(), core.strip_ansi(entry["src"]))
        self.assertEqual(len(updates), 1)
