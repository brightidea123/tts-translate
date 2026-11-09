"""Verify collapsed finals from exact source audio without running MLX."""
from __future__ import annotations

import hashlib
from unittest import TestCase
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from test_accuracy_integration import load_core
from tts_runtime.asr_consistency import outro_conflicts_with_preview
from tts_runtime.asr_quality import summarize_whisper_result
from tts_runtime.asr_recovery import recovery_windows


PREVIEW = "אנחנו צריכים לשלוח את החבילה מחר בבוקר לכתובת החדשה ולבדוק שוב את מספר הבית לפני הנסיעה אל העיר הגדולה"


class FinalRecoveryIntegrationTests(TestCase):
    def setUp(self):
        self.core = load_core()
        self.core.asr_request_context = {"utterance_id": 4, "request_id": "same-capture", "pcm_sha256": "original"}
        self.audio = ((np.arange(394240, dtype=np.int32) % 16000) - 8000).astype(np.int16)
        self.windows = recovery_windows(self.audio.size)

    def primary(self, audio, *, partial=False, **_kwargs):
        self.core.prepare_asr_audio(audio)
        text = PREVIEW if partial else "תודה רבה."
        self.core.last_decoder_quality.update(raw_text=text, detected_language="he", max_no_speech_prob=.1,
                                             max_compression_ratio=1.0, segments=[])
        return text

    def run_recovery(self, decode):
        core = self.core
        with patch.object(core, "_transcribe_pcm_i16_impl", side_effect=self.primary), \
                patch.object(core, "effective_lane_backend", return_value="mlx-whisper"), \
                patch.object(core, "transcribe_audio_mlx_whisper", side_effect=decode) as retry:
            core.transcribe_pcm_i16(self.audio[:96000], partial=True)
            text = core.transcribe_pcm_i16(self.audio, partial=False)
        return text, retry

    def summary(self, tokens, window):
        offset = (window.core_start_sample - window.start_sample) / 16000
        words = [{"word": " " + token, "start": offset + .2 + i * .3,
                  "end": offset + .4 + i * .3} for i, token in enumerate(tokens)]
        return summarize_whisper_result({"text": " ".join(tokens), "language": "he", "segments": [
            {"text": " ".join(tokens), "start": words[0]["start"], "end": words[-1]["end"],
             "avg_logprob": -.2, "no_speech_prob": .05, "compression_ratio": 1.1, "words": words}]})

    def test_shorter_acoustic_windows_recover_in_order_without_reprocessing_or_promoting_preview(self):
        calls = []
        words = (PREVIEW + " בשעה שתים עשרה בדיוק").split()
        expected = []
        def decode(audio, **kwargs):
            window = self.windows[len(calls)]
            np.testing.assert_array_equal(audio, self.audio[window.start_sample:window.end_sample].astype(np.float32) / 32768)
            self.assertTrue(kwargs["word_timestamps"])
            tokens = words[len(calls) * 6:(len(calls) + 1) * 6]
            expected.extend(tokens)
            calls.append(window)
            self.core.last_decoder_quality.clear()
            self.core.last_decoder_quality.update(self.summary(tokens, window))
            return " ".join(tokens)
        text, retry = self.run_recovery(decode)
        self.assertEqual(text, " ".join(expected))
        self.assertEqual(retry.call_count, 4)
        self.assertEqual(self.core.last_asr_diagnostics["final_recovery"]["status"], "recovered")
        self.assertEqual(self.core.last_asr_diagnostics["primary_decoder"]["raw_text"], "תודה רבה.")
        self.assertEqual(self.core.last_decoder_quality["raw_text"], text)
        raw = (self.audio.astype(np.float32) / 32768).astype("<f4").tobytes()
        self.assertEqual(self.core.last_asr_diagnostics["model_input"]["sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(self.core.last_asr_diagnostics["request"]["request_id"], "same-capture")
        self.assertEqual(sum(w.core_end_sample - w.core_start_sample for w in calls), self.audio.size)

    def test_failed_window_does_not_commit_a_partial_recovery_or_retry_forever(self):
        def fail(_audio, **_kwargs):
            raise RuntimeError("decoder failed")
        text, retry = self.run_recovery(fail)
        self.assertEqual(text, "")
        self.assertEqual(retry.call_count, 1)
        self.assertEqual(self.core.last_asr_diagnostics["final_recovery"]["status"], "unresolved")
        self.assertIn("final_recovery_unresolved", self.core.last_asr_diagnostics["issues"])
        self.assertEqual(self.core.last_decoder_quality["raw_text"], "תודה רבה.")

    def test_repeated_outros_in_recovery_cannot_pass_as_restored_speech(self):
        calls = []
        def decode(_audio, **_kwargs):
            window = self.windows[len(calls)]
            calls.append(window)
            self.core.last_decoder_quality.clear()
            self.core.last_decoder_quality.update(self.summary(["תודה", "רבה"], window))
            return "תודה רבה"
        text, retry = self.run_recovery(decode)
        self.assertEqual(text, "")
        self.assertEqual(retry.call_count, 4)
        self.assertEqual(self.core.last_asr_diagnostics["final_recovery"]["status"], "unresolved")

    def test_python_backend_requests_and_retains_real_word_alignment_for_recovery(self):
        core = self.core
        calls = []
        def decode(audio, **kwargs):
            calls.append(kwargs)
            return {"text": "שלום עולם", "language": "he", "segments": [
                {"text": "שלום עולם", "words": [{"word": " שלום", "start": .1, "end": .5},
                                                 {"word": " עולם", "start": .6, "end": 1.0}]}]}
        with patch.object(core, "MLX_WHISPER_INSTALLED", True), \
                patch.object(core, "ensure_mlx_whisper_loaded", return_value=SimpleNamespace(transcribe=decode)), \
                patch.object(core, "resolve_mlx_whisper_model_ref_for", return_value="local-model"):
            self.assertEqual(core.transcribe_audio_mlx_whisper(np.zeros(32000, dtype=np.float32), word_timestamps=True), "שלום עולם")
        self.assertTrue(calls[0]["word_timestamps"])
        self.assertFalse(calls[0]["condition_on_previous_text"])
        self.assertEqual(calls[0]["task"], "transcribe")
        self.assertEqual(core.last_decoder_quality["segments"][0]["words"][1]["word"], " עולם")

    def test_timestamp_recovery_does_not_fall_back_to_unaligned_cli_output(self):
        with patch.object(self.core, "MLX_WHISPER_INSTALLED", False), \
                patch.object(self.core, "resolve_mlx_whisper_model_ref_for", return_value="local-model"), \
                patch.object(self.core, "resolve_stt_command") as command, \
                self.assertRaisesRegex(RuntimeError, "Python backend"):
            self.core.transcribe_audio_mlx_whisper(np.zeros(32000, dtype=np.float32), word_timestamps=True)
        command.assert_not_called()

    def test_missing_word_alignment_cannot_silently_claim_recovered_coverage(self):
        def decode(_audio, **_kwargs):
            self.core.last_decoder_quality.clear()
            self.core.last_decoder_quality.update(raw_text=PREVIEW, segments=[])
            return PREVIEW
        text, retry = self.run_recovery(decode)
        self.assertEqual(text, "")
        self.assertEqual(retry.call_count, 1)

    def test_short_thanks_without_verification_input_are_not_silently_committed(self):
        with patch.object(self.core, "_transcribe_pcm_i16_impl", return_value="תודה רבה."), \
                patch.object(self.core, "transcribe_audio_mlx_whisper") as retry:
            text = self.core.transcribe_pcm_i16(self.audio[:24000], partial=False)
        self.assertEqual(text, "")
        self.assertEqual(self.core.last_asr_diagnostics["final_verification"]["status"], "unresolved")
        retry.assert_not_called()

    def test_conflicting_outro_preview_does_not_replace_substantial_existing_speech(self):
        core = self.core
        texts = iter([PREVIEW, "תודה רבה."])
        def decode(_audio, **_kwargs):
            text = next(texts)
            core.last_decoder_quality.update(raw_text=text, max_no_speech_prob=.1, max_compression_ratio=1)
            return text
        with patch.object(core, "_transcribe_pcm_i16_impl", side_effect=decode):
            self.assertEqual(core.transcribe_pcm_i16(self.audio[:96000], partial=True), PREVIEW)
            self.assertEqual(core.transcribe_pcm_i16(self.audio[:96000], partial=True), "")
        self.assertIn("partial_outro_conflict", core.last_asr_diagnostics["issues"])
        self.assertEqual(core.asr_preview_evidence[4], PREVIEW)


class OutroConsistencyTests(TestCase):
    def test_conflicting_signoffs_are_reviewed_across_supported_languages(self):
        for text in ["תודה רבה.", "תודה על הצפייה!", "כתוביות ותרגום: צוות", "Thank you for watching.",
                     "شكرا لكم على المشاهدة", "Merci beaucoup", "Danke", "Спасибо за просмотр"]:
            with self.subTest(text=text):
                self.assertTrue(outro_conflicts_with_preview(text, PREVIEW))

    def test_ordinary_thanks_and_discussion_of_video_platforms_are_not_blacklisted(self):
        self.assertFalse(outro_conflicts_with_preview("תודה רבה.", "תודה רבה"))
        self.assertFalse(outro_conflicts_with_preview("We are comparing YouTube channels today", PREVIEW))
        self.assertFalse(outro_conflicts_with_preview("תודה רבה.", PREVIEW + " תודה רבה"))
