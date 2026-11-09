"""Source-clock endpointing and word-boundary preservation without inference."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from uuid import uuid4

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_core():
    name = "endpoint_core_" + uuid4().hex
    spec = importlib.util.spec_from_file_location(name, ROOT / "tts-transcribe.py")
    core = importlib.util.module_from_spec(spec)
    sys.modules[name] = core
    with patch.object(sys, "argv", [str(spec.origin), "--ui", "headless", "--audio-review", "off"]):
        spec.loader.exec_module(core)
    return core


class Scheduler:
    def __init__(self):
        self.requests = []

    def submit_with_provenance(self, utterance_id, pcm, partial, provenance):
        self.requests.append((utterance_id, pcm.copy(), partial, dict(provenance)))
        return True

    @property
    def finals(self):
        return [request for request in self.requests if not request[2]]


class Evidence:
    def __init__(self):
        self.clips, self.events = [], []

    def save_clip(self, pcm, metadata, reasons, **kwargs):
        self.clips.append((np.array(pcm, copy=True), dict(metadata), list(reasons), kwargs))
        return True

    def event(self, metadata):
        self.events.append(dict(metadata))


class EndpointBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.core = load_core()
        self.core.show_inflight_partial_transcript = lambda *args, **kwargs: None
        self.core.clear_inflight_partial_marker = lambda *args: None
        self.scheduler = Scheduler()
        self.endpoint = self.core.RingEndpointDetector(self.core.NativeTimestampRingBuffer(1), self.scheduler)
        self.frame_number = 0

    def feed(self, count, speech, *, value=None, endpoint=None):
        endpoint = endpoint or self.endpoint
        value = (4500 if speech else 0) if value is None else value
        for _ in range(count):
            pcm = np.full(self.core.FRAME_SAMPLES, value, dtype=np.int16)
            frame = self.core.TimestampedAudioFrame(pcm, self.frame_number * 20_000_000,
                                                   -20.0 if speech else -120.0, speech)
            endpoint._consume(frame)
            self.frame_number += 1

    def test_short_valid_words_survive_trailing_endpoint_wait(self):
        for speech_frames in (5, 7, 9):
            with self.subTest(speech_ms=speech_frames * 20):
                scheduler = Scheduler()
                endpoint = self.core.RingEndpointDetector(self.core.NativeTimestampRingBuffer(1), scheduler)
                self.feed(speech_frames, True, endpoint=endpoint)
                self.feed(100, False, endpoint=endpoint)
                self.assertEqual(len(scheduler.finals), 1)
                _, pcm, _, provenance = scheduler.finals[0]
                np.testing.assert_array_equal(pcm[:speech_frames * 320], np.full(speech_frames * 320, 4500))
                self.assertFalse(np.any(pcm[speech_frames * 320:]))
                self.assertEqual(provenance["boundary_reason"], "endpoint")
                self.assertFalse(provenance["forced_speech_boundary"])

    def test_rejected_short_noise_has_review_clip_and_reason(self):
        evidence = self.core.audio_evidence = Evidence()
        self.feed(4, True)
        self.feed(100, False)
        self.assertEqual(self.scheduler.finals, [])
        self.assertEqual(len(evidence.clips), 1)
        pcm, metadata, reasons, _ = evidence.clips[0]
        self.assertEqual(reasons, ["endpoint_speech_gate"])
        self.assertEqual(metadata["speech_frames"], 4)
        self.assertEqual(metadata["speech_span_frames"], 4)
        self.assertEqual(metadata["speech_ratio"], 1.0)
        np.testing.assert_array_equal(pcm[:4 * 320], np.full(4 * 320, 4500))
        self.assertEqual(evidence.events[0]["type"], "endpoint_speech_gate")

    def test_rejected_sparse_speech_retires_blank_card_after_request_timer_is_cleared(self):
        # Exercise real UI lifecycle helpers; the ordinary endpoint fixtures
        # replace those helpers because most tests only inspect captured PCM.
        core = self.core = load_core()
        self.scheduler = Scheduler()
        self.endpoint = core.RingEndpointDetector(core.NativeTimestampRingBuffer(1), self.scheduler)
        core.ring_endpoint = self.endpoint
        evidence = core.audio_evidence = Evidence()
        self.endpoint._current_endpoint_threshold = lambda: 5000
        clock = [100.0]
        with patch.object(core, "render_dashboard"), patch.object(core.time, "perf_counter", side_effect=lambda: clock[0]):
            # Enough source-clock duration for a preview, but fewer voiced
            # frames than the endpoint speech gate accepts for a final.
            self.feed(1, True)
            self.feed(55, False)
            self.feed(3, True)
            utterance_id = self.endpoint.utterance_id
            self.assertTrue(any(request[2] for request in self.scheduler.requests))
            self.assertTrue(core.placeholder_only_entry(core.ui_state["left_active"][utterance_id]))
            core.clear_inflight_partial_marker(utterance_id)
            self.assertNotIn(utterance_id, core.inflight_partial_markers)
            self.endpoint.finalize()
            self.assertEqual(self.scheduler.finals, [])
            self.assertEqual(evidence.events[0]["type"], "endpoint_speech_gate")
            self.assertFalse(self.endpoint.in_speech)
            clock[0] += core.args.inflight_partial_timeout_ms / 1000.0 + 1
            core.prune_empty_source_placeholders()
            self.assertNotIn(utterance_id, core.ui_state["left_active"])
            self.assertNotIn(utterance_id, core.ui_state["left_order"])

    def test_speech_gate_keeps_interior_pauses_in_its_denominator(self):
        evidence = self.core.audio_evidence = Evidence()
        self.core.args.min_speech_ratio = 0.9
        self.endpoint._current_endpoint_threshold = lambda: 5000
        self.feed(5, True)
        self.feed(20, False)
        self.feed(5, True)
        self.feed(60, False)
        self.endpoint.finalize()
        self.assertEqual(self.scheduler.finals, [])
        metadata = evidence.clips[0][1]
        self.assertEqual(metadata["speech_span_frames"], 30)
        self.assertEqual(metadata["trailing_silence_frames"], 60)
        self.assertAlmostEqual(metadata["speech_ratio"], 1 / 3)

    def test_paced_and_backlogged_audio_produce_identical_endpoints(self):
        finals = []
        for paced in (True, False):
            scheduler = Scheduler()
            endpoint = self.core.RingEndpointDetector(self.core.NativeTimestampRingBuffer(1), scheduler)
            clock = [0.0]
            self.frame_number = 0
            with patch.object(self.core.time, "perf_counter", side_effect=lambda: clock[0]), \
                    patch.object(self.core.time, "time_ns", return_value=1_000_000_000_000_000_000), \
                    patch.object(self.core.time, "monotonic_ns", return_value=0):
                for index in range(125):
                    clock[0] = index * 0.02 if paced else 0
                    self.feed(1, index < 25 or index >= 100, endpoint=endpoint)
            self.assertEqual(len(scheduler.finals), 1)
            self.assertEqual(endpoint.utterance_id, 2)
            finals.append(scheduler.finals[0])
        np.testing.assert_array_equal(finals[0][1], finals[1][1])
        self.assertEqual(finals[0][3], finals[1][3])

    def test_resumed_speech_cancels_pending_source_clock_hold(self):
        self.endpoint._current_endpoint_threshold = lambda: 100
        self.feed(10, True)
        self.feed(5, False)
        self.assertTrue(self.endpoint.pending_final)
        self.feed(10, True)
        self.assertFalse(self.endpoint.pending_final)
        self.assertIsNone(self.endpoint.pending_final_started_sample)
        self.feed(30, False)
        self.assertEqual(len(self.scheduler.finals), 1)
        pcm = self.scheduler.finals[0][1]
        self.assertTrue(np.all(pcm[15 * 320:25 * 320] == 4500))

    def test_only_completed_word_pauses_train_the_estimate_and_new_phrases_keep_it(self):
        self.endpoint._current_endpoint_threshold = lambda: 5000
        self.feed(20, True)
        initial = self.endpoint.pause_ewma_ms
        self.feed(10, False)
        self.assertEqual(self.endpoint.pause_ewma_ms, initial)
        self.assertEqual(self.endpoint.completed_pause_count, 0)
        self.feed(1, True)
        self.assertEqual(self.endpoint.pause_ewma_ms, initial)
        self.assertEqual(self.endpoint.completed_pause_count, 1)
        self.feed(35, False)
        self.assertEqual(self.endpoint.pause_ewma_ms, initial)
        self.feed(1, True)
        self.assertAlmostEqual(self.endpoint.pause_ewma_ms, 468.8)
        self.assertEqual(self.endpoint.completed_pause_count, 2)
        self.endpoint.finalize(force=True)
        self.feed(1, True)
        self.assertAlmostEqual(self.endpoint.pause_ewma_ms, 468.8)
        self.assertEqual(self.endpoint.completed_pause_count, 0)

    def test_stale_complete_text_cannot_accelerate_or_delay_a_new_word_gap(self):
        core = self.core
        self.feed(50, True)
        core.current_partial_text = "The earlier sentence is complete."
        core.current_partial_stability = 5
        core.current_partial_locked_ratio = 1.0
        core.current_partial_timing.update(core.current_partial_text, 0, 0)
        core.current_partial_timing.update(core.current_partial_text, 500_000_000, 1)
        self.feed(10, False)
        self.assertFalse(self.endpoint.endpoint_text_fresh)
        expected = core.compute_adaptive_threshold(self.endpoint.pause_ewma_ms, self.endpoint.speech_ms)
        self.assertEqual(self.endpoint.adaptive_silence_ms, expected)
        core.current_partial_timing.update(core.current_partial_text, self.endpoint.last_speech_end_ns, 2)
        threshold = self.endpoint._current_endpoint_threshold()
        self.assertTrue(self.endpoint.endpoint_text_fresh)
        self.assertLess(threshold, expected)

    def test_repeated_ellipsis_or_continuation_does_not_become_an_early_sentence_end(self):
        core = self.core
        self.feed(50, True)
        for text in ["I was going to...", "This is an unfinished thought…", "we were talking about the"]:
            with self.subTest(text=text):
                core.current_partial_text = text
                core.current_partial_stability = 8
                core.current_partial_locked_ratio = 1.0
                core.current_partial_timing = core.PartialTextTiming()
                core.current_partial_timing.update(text, 0, 0)
                core.current_partial_timing.update(text, self.endpoint.last_speech_end_ns, 2)
                threshold = self.endpoint._current_endpoint_threshold()
                acoustic = core.compute_adaptive_threshold(self.endpoint.pause_ewma_ms, self.endpoint.speech_ms)
                self.assertGreaterEqual(threshold, acoustic)

    def test_semantic_sentence_hints_recognize_arabic_questions_and_unfinished_hebrew(self):
        core = self.core
        with patch.object(core.args, "source_language", "AR"):
            self.assertGreater(core.sentence_completeness_score("هل وصل القطار؟"), core.sentence_completeness_score("هل وصل القطار..."))
            self.assertLessEqual(core.sentence_completeness_score("نحن نتحدث عن..."), 0)
        with patch.object(core.args, "source_language", "HE"):
            self.assertLessEqual(core.sentence_completeness_score("אנחנו מדברים על..."), 0)

    def test_legacy_word_gap_counter_resets_when_speech_resumes(self):
        core = self.core
        voice = np.full(320, 4500, dtype=np.int16).tobytes()
        quiet = np.zeros(320, dtype=np.int16).tobytes()
        with patch.object(core, "resolved_vad_backend", "energy"):
            for _ in range(10): core.process_capture_frame(voice)
            for _ in range(10): core.process_capture_frame(quiet)
            self.assertGreater(core.silence_ms, 0)
            core.process_capture_frame(voice)
            self.assertEqual(core.silence_ms, 0)

    def test_soft_limit_does_not_cut_voiced_audio(self):
        self.feed(950, True)
        self.assertEqual(self.scheduler.finals, [])
        self.assertTrue(self.endpoint.in_speech)
        self.assertEqual(self.endpoint.active_samples, 19 * 16000)

    def test_extended_pause_deadline_restarts_the_hold_instead_of_spending_it_early(self):
        threshold = [300]
        self.endpoint._current_endpoint_threshold = lambda: threshold[0]
        self.feed(100, True)
        self.feed(15, False)
        self.assertTrue(self.endpoint.pending_final)
        threshold[0] = 600
        self.feed(1, False)
        self.assertFalse(self.endpoint.pending_final)
        self.feed(14, False)
        self.assertTrue(self.endpoint.pending_final)
        self.assertEqual(self.scheduler.finals, [])
        self.feed(6, False)
        self.assertEqual(self.scheduler.finals, [])
        self.feed(1, False)
        self.assertEqual(len(self.scheduler.finals), 1)
        self.assertEqual(self.scheduler.finals[0][3]["trailing_silence_ms"], 740)

    def test_soft_window_does_not_use_a_short_word_pause_as_a_paragraph_boundary(self):
        self.feed(950, True)
        self.feed(12, False)
        self.assertEqual(self.scheduler.finals, [])
        self.feed(10, True, value=7000)
        self.feed(32, False)
        self.assertEqual(len(self.scheduler.finals), 1)
        pcm = self.scheduler.finals[0][1]
        self.assertTrue(np.all(pcm[962 * 320:972 * 320] == 7000))
        provenance = self.scheduler.finals[0][3]
        self.assertEqual(provenance["boundary_reason"], "soft_duration_silence")
        self.assertGreaterEqual(provenance["trailing_silence_ms"], provenance["endpoint_threshold_ms"] + self.core.args.finalization_hold_ms)

    def test_later_text_cannot_spend_the_hold_before_its_longer_pause_deadline(self):
        threshold = [300]
        self.endpoint._current_endpoint_threshold = lambda: threshold[0]
        self.feed(100, True)
        self.feed(20, False)
        self.assertTrue(self.endpoint.pending_final)
        # The new 400 ms deadline has already elapsed when this hint arrives;
        # its hold still cannot be borrowed from the old 300 ms deadline.
        threshold[0] = 400
        self.feed(6, False)
        self.assertEqual(self.scheduler.finals, [])
        self.feed(1, False)
        self.assertEqual(len(self.scheduler.finals), 1)
        self.assertEqual(self.scheduler.finals[0][3]["trailing_silence_ms"], 540)

    def test_soft_limit_ends_on_quiet_tail_and_keeps_only_quiet_preroll(self):
        self.feed(950, True)
        self.feed(11, False)
        self.assertEqual(self.scheduler.finals, [])
        self.feed(1, False)
        self.assertEqual(self.scheduler.finals, [])
        self.feed(20, False)
        self.assertEqual(len(self.scheduler.finals), 1)
        _, pcm, _, provenance = self.scheduler.finals[0]
        self.assertEqual(provenance["boundary_reason"], "soft_duration_silence")
        self.assertFalse(provenance["forced_speech_boundary"])
        self.assertEqual(pcm.size, (950 + 32) * 320)
        self.assertTrue(np.all(pcm[:950 * 320] == 4500))
        self.assertFalse(np.any(pcm[-32 * 320:]))
        self.feed(1, True, value=7000)
        self.assertEqual(self.endpoint.utterance_id, 2)
        onset = np.concatenate(self.endpoint.active)
        self.assertFalse(np.any(onset[:-320]))
        self.assertTrue(np.all(onset[-320:] == 7000))
        self.assertEqual(onset.size, self.core.pre_roll_max * 320)

    def test_hard_limit_is_bounded_flagged_and_saved_for_review(self):
        evidence = self.core.audio_evidence = Evidence()
        self.feed(1399, True)
        self.assertEqual(self.scheduler.finals, [])
        self.feed(1, True)
        self.assertEqual(len(self.scheduler.finals), 1)
        _, pcm, _, provenance = self.scheduler.finals[0]
        self.assertEqual(pcm.size, 28 * 16000)
        self.assertTrue(np.all(pcm == 4500))
        self.assertEqual(provenance["boundary_reason"], "hard_duration_limit")
        self.assertTrue(provenance["forced_speech_boundary"])
        self.assertEqual(evidence.clips[0][2], ["forced_speech_boundary"])
        np.testing.assert_array_equal(evidence.clips[0][0], pcm)
        self.assertEqual(evidence.events[0]["type"], "forced_speech_boundary")
        self.feed(1, True, value=7000)
        self.assertEqual(self.endpoint.utterance_id, 2)
        np.testing.assert_array_equal(np.concatenate(self.endpoint.active), np.full(320, 7000))

    def test_forced_subframe_flush_preserves_exact_captured_pcm(self):
        pcm = np.arange(137, dtype=np.int16)
        self.endpoint._consume(self.core.TimestampedAudioFrame(pcm, 0, -20.0, True))
        self.endpoint.finalize(force=True)
        self.assertEqual(len(self.scheduler.finals), 1)
        np.testing.assert_array_equal(self.scheduler.finals[0][1], pcm)
        self.assertEqual(self.scheduler.finals[0][3]["timestamp_end_ns"], pcm.size * 1_000_000_000 // 16000)


if __name__ == "__main__":
    unittest.main()
