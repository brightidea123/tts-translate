"""Captured-clip timing stays independent of rolling previews and inference."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import importlib.util
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from uuid import uuid4

import numpy as np

from tts_runtime.clip_timing import capture_clip_timing
from tts_runtime.web_transport import build_web_snapshot


ROOT = Path(__file__).resolve().parents[1]
SECOND = 1_000_000_000
OFFSET = int(datetime(2026, 11, 8, tzinfo=timezone.utc).timestamp()) * SECOND - 5 * SECOND
START = "2026-11-08T00:00:00.000Z"


def load_core():
    name = "clip_timing_core_" + uuid4().hex
    spec = importlib.util.spec_from_file_location(name, ROOT / "tts-transcribe.py")
    core = importlib.util.module_from_spec(spec)
    sys.modules[name] = core
    with patch.object(sys, "argv", [str(spec.origin), "--source-language", "HE", "--target-language", "AR"]), \
            redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        spec.loader.exec_module(core)
    return core


class CaptureClipTimingTests(unittest.TestCase):
    def test_monotonic_capture_time_converts_to_utc_without_decode_time(self):
        result = capture_clip_timing({"timestamp_start_ns": 5 * SECOND + 125_000_000,
                                      "samples": 24000, "sample_rate": 16000}, OFFSET)
        self.assertEqual(result, {"audio_started_at": "2026-11-08T00:00:00.125Z",
                                  "audio_duration_seconds": 1.5})
        captured_before_sleep = capture_clip_timing({"timestamp_start_ns": 5 * SECOND,
                                                     "capture_wall_clock_offset_ns": OFFSET}, OFFSET + 60 * SECOND)
        self.assertEqual(captured_before_sleep, {"audio_started_at": START})

    def test_full_utterance_timing_wins_over_six_second_preview_window(self):
        result = capture_clip_timing({"timestamp_start_ns": 8 * SECOND,
                                      "samples": 6 * 16000, "sample_rate": 16000,
                                      "utterance_start_ns": 5 * SECOND, "utterance_samples": 9 * 16000}, OFFSET)
        self.assertEqual(result, {"audio_started_at": START, "audio_duration_seconds": 9.0})

    def test_missing_or_invalid_provenance_does_not_invent_timing(self):
        for request in [{}, {"timestamp_start_ns": -1, "samples": -1, "sample_rate": 16000},
                        {"timestamp_start_ns": True, "samples": True, "sample_rate": 16000},
                        {"timestamp_start_ns": "5", "samples": "16000", "sample_rate": 16000},
                        {"samples": 16000, "sample_rate": 0}]:
            with self.subTest(request=request):
                self.assertEqual(capture_clip_timing(request, OFFSET), {})
        self.assertEqual(capture_clip_timing({"samples": 0, "sample_rate": 16000}, OFFSET),
                         {"audio_duration_seconds": 0.0})

    def test_ring_provenance_keeps_full_clip_origin_when_model_window_shifts(self):
        core = load_core()
        endpoint = core.RingEndpointDetector.__new__(core.RingEndpointDetector)
        endpoint.active = [np.zeros(core.FRAME_SAMPLES, dtype=np.int16) for _ in range(450)]
        endpoint.active_timestamps = [5 * SECOND + index * 20_000_000 for index in range(450)]
        endpoint.active_samples = 9 * core.SAMPLE_RATE
        endpoint.capture_gap = False
        endpoint.audio_wall_clock_offset_ns = OFFSET
        window = np.concatenate(endpoint.active[-300:])
        request = endpoint._window_provenance(window, 300)
        self.assertEqual(request["timestamp_start_ns"], 8 * SECOND)
        self.assertEqual(request["timestamp_end_ns"], 14 * SECOND)
        self.assertEqual(capture_clip_timing(request, OFFSET),
                         {"audio_started_at": START, "audio_duration_seconds": 9.0})

    def test_delayed_partial_cannot_shrink_live_duration_and_final_is_authoritative(self):
        core = load_core()
        first = {"audio_started_at": START, "audio_duration_seconds": 9.0}
        older = {"audio_started_at": "2026-11-08T00:00:03.000Z", "audio_duration_seconds": 6.0}
        core.ui_set_panel_entry("left", 1, "src", "HE~: preview", False, revision=1, audio_timing=first)
        core.ui_set_panel_entry("left", 1, "src", "HE~: refined preview", False, revision=2, audio_timing=older)
        entry = core.ui_state["left_active"][1]
        self.assertEqual(entry["audio_started_at"], START)
        self.assertEqual(entry["audio_duration_seconds"], 9.0)
        final = {"audio_started_at": START, "audio_duration_seconds": 8.95}
        core.ui_set_panel_entry("left", 1, "src", "HE: final", True, revision=3, audio_timing=final)
        self.assertEqual(entry["audio_duration_seconds"], 8.95)

    def test_live_timing_update_does_not_replace_existing_source_or_revision(self):
        core = load_core()
        core.reset_current_partial_state(next_utterance_id=1)
        core.ui_set_panel_entry("left", 1, "src", "HE~: visible words", False, revision=3,
                                preview=core.source_preview("visible words", "visible", inflight=False),
                                audio_timing={"audio_started_at": START, "audio_duration_seconds": 5.0})
        entry = core.ui_state["left_active"][1]
        with patch.object(core, "render_dashboard") as render:
            core.show_inflight_partial_transcript(1, audio_timing={"audio_started_at": START, "audio_duration_seconds": 7.0})
            self.assertEqual(entry["src"], "HE~: visible words")
            self.assertEqual(entry["src_revision"], 3)
            self.assertEqual(entry["audio_duration_seconds"], 7.0)
            render.assert_called_once()
            core.show_inflight_partial_transcript(1, audio_timing={"audio_started_at": START, "audio_duration_seconds": 6.0})
            render.assert_called_once()

    def test_committed_phrase_retains_timing_in_web_history_but_replies_do_not(self):
        core = load_core()
        timing = {"audio_started_at": START, "audio_duration_seconds": 3.4}
        core.ui_set_panel_entry("left", 4, "src", "HE: שלום", True, revision=2, audio_timing=timing)
        core.ui_set_panel_entry("left", 4, "dst", "AR: مرحباً", True, revision=2)
        self.assertNotIn(4, core.ui_state["left_active"])
        self.assertEqual(core.ui_state["left_done"][0]["audio_started_at"], START)
        core.ui_state["right_done"] = [{"entry_id": 1, "src": "AR: reply", "dst": "HE: תשובה", **timing}]
        snapshot = build_web_snapshot(core.ui_state, {"source_language": "HE", "target_language": "AR"}, "s", True, "off", None)
        phrase = snapshot["transcript"][0]
        self.assertEqual(phrase["audio_started_at"], START)
        self.assertEqual(phrase["audio_duration_seconds"], 3.4)
        self.assertNotIn("audio_started_at", snapshot["replies"][0])
        self.assertNotIn("audio_duration_seconds", snapshot["replies"][0])
