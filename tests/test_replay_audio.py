from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

import numpy as np

from scripts import replay_audio


class ReplayAudioTests(unittest.TestCase):
    @staticmethod
    def wav(path):
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(np.array([0, 5000, -5000, 1000], dtype="<i2").tobytes())

    def test_default_inspection_preserves_audio_and_never_loads_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "clip.wav"
            self.wav(path)
            path.with_suffix(".json").write_text('{"source_language":"HE","text":"שלום"}', encoding="utf-8")
            before = path.read_bytes()
            with patch.object(replay_audio, "load_core", side_effect=AssertionError("Model import during inspection")), redirect_stdout(io.StringIO()) as output:
                self.assertEqual(replay_audio.main([str(path)]), 0)
            report = json.loads(output.getvalue())
            self.assertFalse(report["model_inference_run"])
            self.assertFalse(report["downloads_allowed"])
            self.assertEqual(report["metadata"]["text"], "שלום")
            self.assertEqual(report["audio_metrics"]["samples"], 4)
            self.assertEqual(path.read_bytes(), before)

    def test_prepared_input_is_replayed_exactly_without_int16_roundtrip(self):
        original = np.array([0.00001234, -0.00001234, 1.2, -1.1], dtype="<f4")
        observed = {}

        def decode(samples, **kwargs):
            observed["samples"] = samples.copy()
            return "مرحباً"

        core = SimpleNamespace(init_stt_runtime_only=lambda: None, decode_prepared_audio=decode,
                               transcribe_pcm_i16=lambda *a, **k: self.fail("Prepared replay used PCM/DSP path"),
                               source_language_code=lambda: "AR", detected_stt_backend="mlx-whisper",
                               last_asr_diagnostics={"accuracy_measured": False})
        with tempfile.TemporaryDirectory() as temporary, patch.object(replay_audio, "load_core", return_value=core), patch.dict(replay_audio.os.environ):
            path = Path(temporary) / "model-input.f32le"
            path.write_bytes(original.tobytes())
            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(replay_audio.main(["--prepared-f32", str(path), "--run", "--source", "AR"]), 0)
            report = json.loads(output.getvalue())
            self.assertTrue(report["model_inference_run"])
            self.assertEqual(report["audio_metrics"]["out_of_range_fraction"], 0.5)
            self.assertEqual(replay_audio.os.environ["HF_HUB_OFFLINE"], "1")
            np.testing.assert_array_equal(observed["samples"], original)

    def test_nonfinite_or_incomplete_float_input_is_rejected_before_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.f32le"
            for payload in (b"\x00", np.array([np.nan], dtype="<f4").tobytes()):
                path.write_bytes(payload)
                with patch.object(replay_audio, "load_core", side_effect=AssertionError("Bad audio reached model")), redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(replay_audio.main(["--prepared-f32", str(path), "--run"]), 1)
                self.assertFalse(json.loads(output.getvalue())["model_inference_run"])

    def test_gated_audio_does_not_claim_decoder_inference(self):
        core = SimpleNamespace(init_stt_runtime_only=lambda: None,
                               transcribe_pcm_i16=lambda *args, **kwargs: "",
                               source_language_code=lambda: "HE", detected_stt_backend="mlx-whisper",
                               last_asr_diagnostics={"decode_attempted": False})
        with patch.object(replay_audio, "load_core", return_value=core):
            result = replay_audio.replay(np.zeros(16000, dtype=np.int16), prepared=False, source="HE",
                                         backend="mlx-whisper", model=None, preprocessing="raw", allow_download=False)
        self.assertFalse(result["model_inference_run"])
        self.assertIn("gated", result["note"])

    def test_model_initialization_error_does_not_claim_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "clip.wav"
            self.wav(path)
            with patch.object(replay_audio, "load_core", side_effect=ImportError("No Metal device")), redirect_stdout(io.StringIO()) as output:
                self.assertEqual(replay_audio.main([str(path), "--run", "--source", "HE"]), 1)
            report = json.loads(output.getvalue())
            self.assertFalse(report["model_inference_run"])
            self.assertIn("No Metal", report["errors"][0])


if __name__ == "__main__":
    unittest.main()
