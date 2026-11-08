from __future__ import annotations

import json
import unittest

import numpy as np

from tts_runtime.asr_quality import issue_reasons, pcm_quality, summarize_whisper_result, final_preview_disagrees


class AudioQualityTests(unittest.TestCase):
    def test_collapsed_final_is_a_review_signal_even_with_confident_decoder(self):
        preview = "Önce kendimiz yaşayıp hem de ekranlarınızın başında sizler çocuklarınız aileleriniz"
        self.assertTrue(final_preview_disagrees("İzlediğiniz için teşekkür ederim.", preview, 18.68))
        self.assertFalse(final_preview_disagrees(preview + " birlikte", preview, 18.68))
        self.assertFalse(final_preview_disagrees("שלום", "שלום חבר", 1.5))
        self.assertFalse(final_preview_disagrees("one two three four", "one two three four five six seven eight", 8))

    def test_signed_minimum_clipping_is_measured_without_int16_overflow(self):
        audio = np.array([-32768, 32767, 0, 0], dtype=np.int16)
        metrics = pcm_quality(audio)
        self.assertEqual(metrics["peak_abs"], 32768)
        self.assertEqual(metrics["clipped_fraction"], 0.5)
        self.assertEqual(metrics["duration_seconds"], 4 / 16000)
        self.assertIn("clipped_audio", issue_reasons(metrics, "שלום"))
        json.dumps(metrics, allow_nan=False)

    def test_clean_sine_is_not_flagged_and_pcm_bytes_match_array(self):
        audio = (np.sin(np.arange(16000) * 2 * np.pi * 440 / 16000) * 6000).astype(np.int16)
        before = audio.copy()
        metrics = pcm_quality(audio)
        self.assertEqual(pcm_quality(audio.astype("<i2").tobytes()), metrics)
        self.assertEqual(issue_reasons(metrics, "Bonjour, rendez-vous demain matin."), [])
        np.testing.assert_array_equal(audio, before)

    def test_empty_and_malformed_audio_are_explicit(self):
        metrics = pcm_quality(np.empty(0, dtype=np.int16))
        self.assertEqual(issue_reasons(metrics, ""), ["empty_audio", "empty_transcript"])
        with self.assertRaises(ValueError):
            pcm_quality(b"\x00")
        with self.assertRaises(ValueError):
            pcm_quality(np.zeros((2, 2), dtype=np.int16))
        with self.assertRaises(ValueError):
            pcm_quality(np.zeros(2, dtype=np.float32))

    def test_decoder_diagnostics_are_finite_and_independent_of_wer(self):
        summary = summarize_whisper_result({"text": "שלום", "language": "he", "segments": [
            {"id": 0, "text": "שלום", "start": 0, "end": 1, "avg_logprob": -1.5,
             "no_speech_prob": 0.8, "compression_ratio": 3.2},
            {"id": 1, "avg_logprob": float("nan"), "no_speech_prob": float("inf")},
        ]})
        json.dumps(summary, allow_nan=False)
        self.assertEqual(summary["min_avg_logprob"], -1.5)
        self.assertIsNone(summary["segments"][1]["avg_logprob"])
        self.assertFalse(summary["accuracy_measured"])
        reasons = issue_reasons(pcm_quality(np.array([5000, -5000], dtype=np.int16)), "שלום",
                                decoder_quality=summary, expected_language="ar")
        self.assertEqual(reasons, ["low_decoder_logprob", "high_no_speech_probability", "high_compression_ratio", "language_mismatch"])

    def test_filtered_controls_repetition_and_capture_errors_are_review_signals(self):
        metrics = pcm_quality(np.array([5000, -5000], dtype=np.int16))
        reasons = issue_reasons(metrics, "", raw_text="<end_of_turn>", error=RuntimeError("decode failed"), gap=True)
        self.assertEqual(reasons, ["empty_transcript", "filtered_transcript", "model_control_tokens", "decoder_error", "capture_discontinuity"])
        self.assertIn("repetitive_transcript", issue_reasons(metrics, "yes " * 10))
        self.assertNotIn("model_control_tokens", issue_reasons(metrics, "x < y and z > 3"))
        self.assertIn("model_control_tokens", issue_reasons(metrics, "האזרח הזה", raw_text="האזרח הזה<|lt|>"))


if __name__ == "__main__":
    unittest.main()
