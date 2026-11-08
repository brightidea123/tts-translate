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
        self.core.show_inflight_partial_transcript = lambda *args: None
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
            with patch.object(self.core.time, "perf_counter", side_effect=lambda: clock[0]):
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

    def test_soft_limit_does_not_cut_voiced_audio(self):
        self.feed(950, True)
        self.assertEqual(self.scheduler.finals, [])
        self.assertTrue(self.endpoint.in_speech)
        self.assertEqual(self.endpoint.active_samples, 19 * 16000)

    def test_soft_limit_ends_on_quiet_tail_and_keeps_only_quiet_preroll(self):
        self.feed(950, True)
        self.feed(11, False)
        self.assertEqual(self.scheduler.finals, [])
        self.feed(1, False)
        self.assertEqual(len(self.scheduler.finals), 1)
        _, pcm, _, provenance = self.scheduler.finals[0]
        self.assertEqual(provenance["boundary_reason"], "soft_duration_silence")
        self.assertFalse(provenance["forced_speech_boundary"])
        self.assertEqual(pcm.size, (950 + 12) * 320)
        self.assertTrue(np.all(pcm[:950 * 320] == 4500))
        self.assertFalse(np.any(pcm[-12 * 320:]))
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
