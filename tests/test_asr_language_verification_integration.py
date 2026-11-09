"""Acoustic verification is separate from forced source-language decoding."""
from __future__ import annotations

import hashlib
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from test_accuracy_integration import load_core


ENGLISH = "The appointment is tomorrow morning at the new office near the station."
HEBREW = "אנחנו צריכים לקבוע פגישה למחר בבוקר ליד התחנה החדשה בעיר הגדולה"


def quality(text, language, *, automatic=False, no_speech=.1):
    return {"raw_text": text, "detected_language": language, "language_auto_detected": automatic,
            "mean_avg_logprob": -.2, "min_avg_logprob": -.3, "max_no_speech_prob": no_speech,
            "max_compression_ratio": 1.2, "segments": []}


class AcousticLanguageVerificationTests(TestCase):
    def setUp(self):
        self.core = load_core()
        self.core.asr_request_context = {"utterance_id": 11, "request_id": "original-audio"}
        self.audio = ((np.arange(137600, dtype=np.int32) % 12000) - 6000).astype(np.int16)

    def run_final(self, primary_text, summaries, *, seconds=8.6):
        core, calls = self.core, []
        audio = self.audio[:round(seconds * 16000)]
        def primary(pcm, **_kwargs):
            core.prepare_asr_audio(pcm)
            core.last_decoder_quality.update(quality(primary_text, "he"))
            return core.sanitize_asr_language_shape(primary_text, final=True)
        def verify(prepared, **kwargs):
            np.testing.assert_array_equal(prepared, audio.astype(np.float32) / 32768)
            calls.append(kwargs)
            core.last_decoder_quality.clear()
            core.last_decoder_quality.update(summaries[len(calls) - 1])
            return core.last_decoder_quality["raw_text"]
        with patch.object(core, "_transcribe_pcm_i16_impl", side_effect=primary), \
                patch.object(core, "effective_lane_backend", return_value="mlx-whisper"), \
                patch.object(core, "transcribe_audio_mlx_whisper", side_effect=verify), \
                patch.object(core, "recover_inconsistent_final") as recovery:
            text = core.transcribe_pcm_i16(audio, partial=False)
        recovery.assert_not_called()
        return text, calls

    def test_verified_english_insert_survives_hebrew_script_filter_without_using_bad_hebrew_preview(self):
        self.core.asr_preview_evidence[11] = HEBREW
        text, calls = self.run_final(ENGLISH, [quality(ENGLISH, "en", automatic=True)])
        self.assertEqual(text, ENGLISH)
        self.assertEqual([c["language_override"] for c in calls], ["auto"])
        self.assertEqual(self.core.last_asr_diagnostics["verified_source_language"], "EN")
        self.assertEqual(self.core.last_asr_diagnostics["final_verification"]["status"], "verified")
        self.assertIn("verified_language_switch", self.core.last_asr_diagnostics["issues"])
        self.assertNotIn("language_mismatch", self.core.last_asr_diagnostics["issues"])
        prepared = (self.audio.astype(np.float32) / 32768).astype("<f4").tobytes()
        self.assertEqual(self.core.last_asr_diagnostics["model_input"]["sha256"], hashlib.sha256(prepared).hexdigest())
        self.assertEqual(self.core.last_asr_diagnostics["request"]["request_id"], "original-audio")

    def test_forced_language_is_not_independent_verification(self):
        text, calls = self.run_final(ENGLISH, [quality(ENGLISH, "he", automatic=False)])
        self.assertEqual(text, "")
        self.assertEqual(len(calls), 1)
        self.assertIn("final_verification_unresolved", self.core.last_asr_diagnostics["issues"])

    def test_genuine_short_thanks_can_pass_automatic_acoustic_verification(self):
        text, calls = self.run_final("תודה רבה.", [quality("תודה רבה.", "he", automatic=True)], seconds=3)
        self.assertEqual(text, "תודה רבה.")
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.core.last_asr_diagnostics["verified_source_language"], "HE")

    def test_short_outro_replaced_by_foreign_words_requires_a_second_confirming_decode(self):
        summaries = [quality(ENGLISH, "en", automatic=True), quality(ENGLISH, "en")]
        text, calls = self.run_final("תודה רבה.", summaries, seconds=3)
        self.assertEqual(text, ENGLISH)
        self.assertEqual([c["language_override"] for c in calls], ["auto", "EN"])
        self.assertEqual(self.core.last_asr_diagnostics["verified_source_language"], "EN")

    def test_short_outro_with_high_no_speech_probability_remains_unverified(self):
        text, calls = self.run_final("תודה.", [quality("תודה.", "he", automatic=True, no_speech=.64)], seconds=1.46)
        self.assertEqual(text, "")
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.core.last_asr_diagnostics["final_verification"]["status"], "unresolved")
        self.assertEqual(self.core.last_decoder_quality["raw_text"], "תודה.")

    def test_changed_number_or_unrelated_verification_cannot_release_foreign_text(self):
        for candidate, verified in [("We need 15 blue chairs tomorrow", "We need 50 blue chairs tomorrow"),
                                    (ENGLISH, "The weather is cold and windy today.")]:
            with self.subTest(candidate=candidate):
                text, _ = self.run_final(candidate, [quality(verified, "en", automatic=True)])
                self.assertEqual(text, "")

    def test_changed_alternative_requires_agreement_not_just_good_decoder_metrics(self):
        text, calls = self.run_final("תודה רבה.", [quality(ENGLISH, "en", automatic=True),
            quality("Please call the office on Monday instead.", "en")], seconds=3)
        self.assertEqual(text, "")
        self.assertEqual(len(calls), 2)

    def test_native_non_outro_text_does_not_add_a_verification_decode(self):
        text, calls = self.run_final(HEBREW, [])
        self.assertEqual(text, HEBREW)
        self.assertEqual(calls, [])

    def test_verification_decoder_error_is_bounded_and_retains_original_diagnostics(self):
        core = self.core
        core.prepare_asr_audio(self.audio)
        core.last_asr_diagnostics["consistency_issues"] = []
        core.last_decoder_quality.update(quality(ENGLISH, "he"))
        with patch.object(core, "transcribe_audio_mlx_whisper", side_effect=RuntimeError("decoder unavailable")) as decode:
            self.assertEqual(core.verify_final_audio(ENGLISH, reason="foreign_script"), "")
        self.assertEqual(decode.call_count, 1)
        self.assertEqual(core.last_decoder_quality["raw_text"], ENGLISH)
        self.assertEqual(core.last_asr_diagnostics["final_verification"]["status"], "unresolved")
        self.assertIn("final_verification_unresolved", core.last_asr_diagnostics["consistency_issues"])

    def test_real_wrapper_uses_automatic_language_instead_of_the_forced_session_language(self):
        core, calls = self.core, []
        def decode(_audio, **kwargs):
            calls.append(kwargs)
            return {"text": ENGLISH, "language": "en", "segments": []}
        with patch.object(core, "MLX_WHISPER_INSTALLED", True), \
                patch.object(core, "ensure_mlx_whisper_loaded", return_value=SimpleNamespace(transcribe=decode)), \
                patch.object(core, "resolve_mlx_whisper_model_ref_for", return_value="local-model"):
            core.transcribe_audio_mlx_whisper(np.zeros(16000, dtype=np.float32), language_override="auto")
        self.assertIsNone(calls[0]["language"])
        self.assertEqual(calls[0]["task"], "transcribe")
        self.assertFalse(calls[0]["condition_on_previous_text"])
        self.assertTrue(core.last_decoder_quality["language_auto_detected"])
