"""Capture remains live through blocked synthesis/playback and selects language voices."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from queue import Queue
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

import numpy as np


def load_core():
    path = Path(__file__).resolve().parents[1] / "tts-transcribe.py"
    spec = importlib.util.spec_from_file_location("duplex_test_core", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    with patch.object(sys, "argv", [str(path), "--source-language", "AR", "--target-language", "HE"]):
        spec.loader.exec_module(module)
    return module


class DuplexPlaybackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.core = load_core()

    def test_playback_guard_only_pauses_explicit_half_duplex(self):
        core = self.core
        playing = threading.Event()
        playing.set()
        with patch.dict(core.__dict__, {"tts_playing": playing, "playback_guard_until": float("inf")}):
            with patch.object(core.args, "duplex_capture", True):
                self.assertFalse(core.capture_paused_for_playback())
                endpoint = core.RingEndpointDetector(core.NativeTimestampRingBuffer(1), SimpleNamespace())
                endpoint.in_speech = True
                endpoint.utterance_id = 987
                endpoint._current_endpoint_threshold = lambda: 10_000
                frame = core.TimestampedAudioFrame(np.ones(core.FRAME_SAMPLES, dtype=np.int16), 0, -60.0, False)
                endpoint._consume(frame)
                self.assertEqual(len(endpoint.active), 1)
            with patch.object(core.args, "duplex_capture", False):
                self.assertTrue(core.capture_paused_for_playback())
                endpoint._consume(frame)
                self.assertEqual(len(endpoint.active), 1)
            playing.clear()
            with patch.object(core.args, "duplex_capture", False):
                self.assertTrue(core.capture_paused_for_playback(), "half-duplex tail guard must remain active")

    def test_blocked_tts_render_and_playback_do_not_reset_or_pause_full_duplex(self):
        core = self.core
        tasks = Queue()
        tasks.put(SimpleNamespace(text="שלום", wav_out="unused.wav", lang_hint="HE", source="text"))
        stop = threading.Event()
        playing = threading.Event()
        rendering = threading.Event()
        release_render = threading.Event()
        playback = threading.Event()
        release_playback = threading.Event()

        def render(*_args, **_kwargs):
            rendering.set()
            if not release_render.wait(timeout=3):
                raise RuntimeError("test render release timed out")
            return "fake"

        def play(_path):
            playback.set()
            if not release_playback.wait(timeout=3):
                raise RuntimeError("test playback release timed out")
            stop.set()

        with patch.dict(core.__dict__, {"tts_tasks": tasks, "shutdown_event": stop, "tts_playing": playing,
                                       "playback_guard_until": 0.0, "tts_best_available": render,
                                       "play_wav": play, "ui_update": lambda **_kw: None}), \
                patch.object(core.args, "duplex_capture", True), \
                patch.object(core, "reset_active_capture_state") as reset:
            worker = threading.Thread(target=core.tts_worker, daemon=True)
            worker.start()
            try:
                self.assertTrue(rendering.wait(timeout=2))
                self.assertFalse(playing.is_set(), "synthesis is not playback")
                self.assertFalse(core.capture_paused_for_playback())
                reset.assert_not_called()
                release_render.set()
                self.assertTrue(playback.wait(timeout=2))
                self.assertTrue(playing.is_set())
                self.assertFalse(core.capture_paused_for_playback(), "full duplex must remain live during actual playback")
                reset.assert_not_called()
            finally:
                release_render.set()
                release_playback.set()
                stop.set()
                worker.join(timeout=3)
            self.assertFalse(worker.is_alive())
            self.assertFalse(playing.is_set())
            self.assertEqual(core.playback_guard_until, 0.0)
            self.assertEqual(tasks.unfinished_tasks, 0)

    def test_legacy_capture_passes_frames_during_full_duplex_playback(self):
        core = self.core
        playing = threading.Event()
        playing.set()
        roll = []
        with patch.dict(core.__dict__, {
            "shutdown_event": threading.Event(), "tts_playing": playing,
            "playback_guard_until": float("inf"), "pre_roll": roll,
            "in_speech": False, "noise_floor_db": -60.0,
            "resolved_vad_backend": "energy", "playback_duck_ms": 999,
            "rms_dbfs_i16": lambda _pcm: -60.0,
        }), patch.object(core.args, "duplex_capture", True):
            core.process_capture_frame(np.zeros(core.FRAME_SAMPLES, dtype=np.int16).tobytes())
            self.assertEqual(len(roll), 1)
            self.assertEqual(core.playback_duck_ms, 0)

    def test_half_duplex_pauses_only_for_playback_and_its_tail(self):
        core = self.core
        tasks = Queue()
        tasks.put(SimpleNamespace(text="reply", wav_out="unused.wav", lang_hint="TR", source="text"))
        stop = threading.Event()
        playing = threading.Event()
        phases = []

        def render(*_args, **_kwargs):
            phases.append("render")
            self.assertFalse(playing.is_set())
            self.assertFalse(core.capture_paused_for_playback())
            return "fake"

        def play(_path):
            phases.append("play")
            self.assertTrue(playing.is_set())
            self.assertTrue(core.capture_paused_for_playback())
            stop.set()

        with patch.dict(core.__dict__, {"tts_tasks": tasks, "shutdown_event": stop, "tts_playing": playing,
                                       "playback_guard_until": 0.0, "tts_best_available": render,
                                       "play_wav": play, "ui_update": lambda **_kw: None}), \
                patch.object(core.args, "duplex_capture", False), \
                patch.object(core.args, "playback_guard_ms", 10_000), \
                patch.object(core, "reset_active_capture_state") as reset:
            core.tts_worker()
            self.assertEqual(phases, ["render", "play"])
            reset.assert_called_once()
            self.assertFalse(playing.is_set())
            self.assertTrue(core.capture_paused_for_playback())
            self.assertEqual(tasks.unfinished_tasks, 0)


class LanguageVoiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.core = DuplexPlaybackTests.core if hasattr(DuplexPlaybackTests, "core") else load_core()

    def test_voice_inventory_preserves_names_and_uses_each_language(self):
        core = self.core
        rows = [("Maged", "ar_SA"), ("Carmit", "he_IL"), ("Anna", "de_DE"),
                ("Yelda", "tr_TR"), ("Marie Claire", "fr_FR"), ("Milena", "ru_RU"),
                ("Samantha", "en_US"), ("Mónica", "es_ES")]
        inventory = "\n".join(f"{name:35} {locale} # Sample" for name, locale in rows)
        core.available_say_voice_records.cache_clear()
        try:
            with patch.object(core, "is_macos", return_value=True), \
                    patch.object(core, "have", return_value=True), \
                    patch.object(core.subprocess, "run", return_value=SimpleNamespace(stdout=inventory)), \
                    patch.object(core.args, "say_voice", ""):
                self.assertEqual(core.available_say_voice_records(), rows)
                for name, locale in rows:
                    self.assertEqual(core.pick_say_voice(locale[:2].upper()), name)
                self.assertEqual(core.pick_say_voice("FR"), "Marie Claire")
                with patch.object(core.args, "edge_voice", ""):
                    for code, spec in core.LANGUAGE_SPECS.items():
                        self.assertEqual(core.edge_voice_for_lang(code), spec.edge_voice)
            with patch.object(core, "available_say_voice_records", return_value=rows[:1]), \
                    patch.object(core.args, "say_voice", ""):
                with self.assertRaisesRegex(RuntimeError, "No installed macOS voice matches Hebrew"):
                    core.pick_say_voice("HE")
        finally:
            core.available_say_voice_records.cache_clear()

    def test_empty_macos_speech_is_rejected_with_actionable_error(self):
        core = self.core

        def run(command, **_kwargs):
            if command[0] == "say":
                self.assertIn("-f", command)
                self.assertEqual(Path(command[command.index("-f") + 1]).read_text(encoding="utf-8"), "שלום")
            if command[0] == "afconvert":
                with wave.open(command[-1], "wb") as output:
                    output.setnchannels(1)
                    output.setsampwidth(2)
                    output.setframerate(22050)
                    output.writeframes(b"")
            return SimpleNamespace(returncode=0)

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(core, "is_macos", return_value=True), \
                patch.object(core, "have", return_value=True), \
                patch.object(core, "pick_say_voice", return_value="Carmit"), \
                patch.object(core.subprocess, "run", side_effect=run), \
                patch.object(core.args, "tts", "say"):
            with self.assertRaisesRegex(RuntimeError, "Carmit.*produced no audio"):
                core.tts_best_available("שלום", str(Path(directory) / "reply.wav"), "HE")


if __name__ == "__main__":
    unittest.main()
