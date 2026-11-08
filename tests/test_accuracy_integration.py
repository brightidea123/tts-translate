"""Accuracy-path integrity and audio evidence, without models or devices."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from queue import Empty
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4
import wave

import numpy as np

from tts_runtime.audio_evidence import AudioEvidence
from tts_runtime.asr_quality import pcm_quality


ROOT = Path(__file__).resolve().parents[1]


def load_core():
    name = "accuracy_core_" + uuid4().hex
    spec = importlib.util.spec_from_file_location(name, ROOT / "tts-transcribe.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    with patch.object(sys, "argv", [str(spec.origin), "--source-language", "HE", "--target-language", "AR"]):
        spec.loader.exec_module(module)
    return module


class GemmaTokenizer:
    eos_token_id, unk_token_id = 1, 0

    def __init__(self):
        self.eos_token_ids = {1}

    def convert_tokens_to_ids(self, token):
        return 106 if token == "<end_of_turn>" else 0

    def convert_ids_to_tokens(self, token):
        return "<end_of_turn>" if token == 106 else "<unk>"

    def encode(self, text, add_special_tokens=False):
        return [self.convert_tokens_to_ids(text)]

    def add_eos_token(self, marker):
        self.eos_token_ids.add(self.convert_tokens_to_ids(marker))


class Mailbox:
    state = "ready"

    def __init__(self):
        self.payloads = []

    def submit(self, payload, **kwargs):
        self.payloads.append(payload)
        return True


class AccuracyIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.core = load_core()

    def test_core_registers_end_turn_and_keeps_the_last_detokenized_word(self):
        core, tokenizer = self.core, GemmaTokenizer()
        core.ensure_mlx_lm_loaded = lambda: None
        core.mlx_lm_load = lambda reference: (object(), tokenizer)
        core.resolve_translation_backend = lambda: "mlx-lm"
        core.args.translation_model = "models/translategemma-12b-it-4bit"
        core.init_translation_runtime_only()
        self.assertEqual(core.mlx_translation_stop_ids, frozenset({1, 106}))
        self.assertEqual(tokenizer.eos_token_ids, {1, 106})
        core.mlx_translation_prompt = lambda *args: "structured prompt"
        observed = []

        def generate(*args, **kwargs):
            for token, text, reason in [(10, "שלום", None), (106, " עולם!", "stop"), (12, "MUST NOT APPEAR", None)]:
                observed.append(token)
                yield SimpleNamespace(token=token, text=text, finish_reason=reason)

        core.mlx_lm_stream_generate = generate
        pieces = list(core.iter_translate_chunk_mlx_lm("source", "heb_Hebr", 1, 1))
        self.assertEqual("".join(pieces), "שלום עולם!")
        self.assertEqual(observed, [10, 106])

    def test_partial_translation_snapshots_never_show_model_tag_fragments(self):
        core = self.core
        core.active_translation_backend = lambda: "mlx-lm"
        core.args.stream_mt_deltas = True
        core.args.mt_delta_min_chars = 1
        core.args.mt_delta_min_ms = 0
        raw = "<start_of_turn>model\n<think>PRIVATE</think>مرحباً بالعالم<end_of_turn>"
        core.iter_translate_chunk_mlx_lm = lambda *args: iter(raw)
        task = core.TranslationTask("audio", "שלום", "heb_Hebr", 1, "AR", utterance_id=1, partial=True)
        previews = []
        answer = core.translate_chunk_with_optional_deltas(task, task.text, 1, 1, 1, previews.append)
        self.assertEqual(answer, "مرحباً بالعالم")
        self.assertTrue(previews)
        for preview in previews:
            self.assertNotIn("<", preview.text)
            self.assertNotIn("PRIVATE", preview.text)
            self.assertTrue(answer.startswith(preview.text))

    def test_raw_model_input_preserves_samples_while_legacy_dsp_can_overshoot(self):
        core = self.core
        source = np.array([-32768, -23000, -1, 0, 1, 23000, 32767], dtype=np.int16)
        core.args.asr_preprocessing = "raw"
        prepared = core.prepare_asr_audio(source)
        np.testing.assert_array_equal(prepared, source.astype(np.float32) / 32768.0)
        self.assertTrue(prepared.flags.c_contiguous)
        self.assertLessEqual(float(np.max(np.abs(prepared))), 1)
        self.assertEqual(core.last_asr_diagnostics["model_input"]["sha256"], hashlib.sha256(prepared.astype("<f4").tobytes()).hexdigest())
        impulse = np.zeros(16000, dtype=np.int16)
        impulse[1600] = 32767
        core.args.asr_preprocessing = "legacy"
        legacy = core.prepare_asr_audio(impulse)
        self.assertGreater(float(np.max(np.abs(legacy))), 1)
        self.assertGreater(core.last_asr_diagnostics["model_input"]["out_of_range_fraction"], 0)

    def test_whisper_language_marker_cannot_leak_into_final_source_text(self):
        core = self.core
        self.assertEqual(core.sanitize_asr_text("האזרח הזה<|lt|>", final=True), "האזרח הזה")
        self.assertEqual(core.sanitize_asr_text("<|startoftranscript|><|he|>שלום.<|endoftext|>", final=True), "שלום.")

    def test_source_decision_journal_does_not_depend_on_translation_output(self):
        core = self.core
        with tempfile.TemporaryDirectory() as temporary:
            core.runtime_session_dir = Path(temporary)
            core.runtime_session_id = "journal-session"
            result = core.AsrResult(8, "שלום.", False, request_id="u8-f-9", diagnostics={
                "decoder": {"raw_text": " שלום.", "mean_avg_logprob": -0.3},
                "request": {"timestamp_start_ns": 100, "timestamp_end_ns": 200}, "issues": []})
            core.record_asr_decision(result, result.text, "accepted_final")
            core.record_asr_decision(result, result.text, "translation_rejected", "Translation queue rejected final work")
            rows = [json.loads(line) for line in (Path(temporary) / "asr-decisions.jsonl").read_text().splitlines()]
            self.assertEqual([row["decision"] for row in rows], ["accepted_final", "translation_rejected"])
            self.assertTrue(all(row["text"] == "שלום." for row in rows))
            self.assertFalse((Path(temporary) / "transcription.log").exists())
            core.args.session_logs = False
            core.record_asr_decision(result, result.text, "must-not-log")
            self.assertEqual(len((Path(temporary) / "asr-decisions.jsonl").read_text().splitlines()), 2)

    def test_stage_submission_owns_immutable_pcm_and_rejects_tampering_before_decode(self):
        core = self.core
        core.runtime_session_id = "session-under-test"
        core.asr_stage = Mailbox()
        source = np.arange(320, dtype=np.int16) - 160
        original = source.copy()
        self.assertTrue(core.submit_asr_stage(7, source, False, {"source_start_sample": 640, "source_end_sample": 960}))
        source[:] = -12345
        payload = core.asr_stage.payloads[0]
        self.assertEqual(len(payload), 4)
        np.testing.assert_array_equal(payload[2], original)
        self.assertEqual(payload[3]["session_id"], "session-under-test")
        self.assertEqual(payload[3]["samples"], 320)
        self.assertEqual(payload[3]["pcm_sha256"], hashlib.sha256(original.astype("<i2").tobytes()).hexdigest())
        decode = Mock(return_value="שלום")
        core.transcribe_pcm_i16 = decode
        emitted = []
        core.handle_asr_stage_request(payload, emitted.append)
        decode.assert_called_once()
        self.assertEqual(emitted[0].request_id, payload[3]["request_id"])
        self.assertEqual(emitted[0].pcm_sha256, payload[3]["pcm_sha256"])
        tampered = payload[2].copy()
        tampered[0] += 1
        decode.reset_mock()
        with self.assertRaisesRegex(RuntimeError, "integrity mismatch"):
            core.handle_asr_stage_request((*payload[:2], tampered, payload[3]), emitted.append)
        decode.assert_not_called()

    def test_parent_quarantines_foreign_session_old_request_and_wrong_pcm_results(self):
        core = self.core
        core.runtime_session_id = "current-session"
        core.asr_stage = Mailbox()
        self.assertTrue(core.submit_asr_stage(9, np.arange(640, dtype=np.int16), False))
        metadata = core.asr_stage.payloads[-1][3]
        for override in [{"session_id": "another-session"}, {"request_id": "old-request"}, {"pcm_sha256": "wrong-hash"}]:
            with self.subTest(override=override):
                values = {"utterance_id": 9, "text": "quarantined", "partial": False,
                          "session_id": metadata["session_id"], "request_id": metadata["request_id"],
                          "pcm_sha256": metadata["pcm_sha256"], **override}
                core.receive_asr_stage_result(core.AsrResult(**values))
                with self.assertRaises(Empty):
                    core.asr_results.get_nowait()
                self.assertIn(9, core.pending_audio_evidence)
        result = core.AsrResult(9, "valid answer", False, session_id=metadata["session_id"],
                               request_id=metadata["request_id"], pcm_sha256=metadata["pcm_sha256"])
        core.receive_asr_stage_result(result)
        self.assertIs(core.asr_results.get_nowait(), result)
        core.asr_results.task_done()
        self.assertNotIn(9, core.pending_audio_evidence)

    def test_full_debug_records_exact_input_before_endpoint_eviction(self):
        core = self.core
        with tempfile.TemporaryDirectory() as temporary:
            core.args.debug = True
            core.args.audio_review = "auto"
            core.audio_review_mode = core.resolved_audio_review_mode()
            self.assertEqual(core.audio_review_mode, "full")
            evidence = AudioEvidence(temporary, role="capture", full_stream=True)
            core.audio_evidence = evidence
            feeder = core.RingAudioFeeder(SimpleNamespace(), queue_ms=20)
            source = np.arange(320 * 5 + 17, dtype=np.int16) - 1000
            original = source.copy()
            try:
                feeder.callback(source[:, None], source.size, None, None)
                source[:] = -999
                self.assertEqual(feeder.callback_errors, 0, feeder.last_capture_status)
                self.assertGreaterEqual(feeder.dropped_frames, 4)
            finally:
                evidence.close()
            recorded = []
            for path in sorted(Path(temporary).glob("*.wav")):
                with wave.open(str(path), "rb") as stream:
                    recorded.extend(np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").tolist())
            np.testing.assert_array_equal(recorded, original)
            self.assertEqual(evidence.snapshot["written_capture_samples"], original.size)

    def test_issue_mode_saves_only_flagged_raw_and_exact_float_input(self):
        core = self.core
        with tempfile.TemporaryDirectory() as temporary:
            core.audio_review_mode = "issues"
            core.audio_evidence = AudioEvidence(temporary, role="asr", full_stream=False)
            core.runtime_session_id = "issue-session"
            core.args.asr_preprocessing = "raw"
            good = (6000 * np.sin(2 * np.pi * np.arange(16000) * 440 / 16000)).astype(np.int16)
            try:
                core.last_asr_diagnostics.update(pcm=pcm_quality(good))
                core.last_decoder_quality.update(raw_text="שלום עולם", detected_language="he", min_avg_logprob=-0.2)
                core.prepare_asr_audio(good)
                core.retain_asr_evidence(good, "שלום עולם")
                self.assertEqual(core.audio_evidence.snapshot["clip_admitted_bytes"], 0)
                clipped = np.tile(np.array([32767, -32768], dtype=np.int16), 160)
                core.last_asr_diagnostics.clear()
                core.last_asr_diagnostics["pcm"] = pcm_quality(clipped)
                core.last_decoder_quality.clear()
                core.asr_request_context = {"request_id": "flagged-input", "session_id": "issue-session"}
                exact_input = core.prepare_asr_audio(clipped).astype("<f4").tobytes()
                core.retain_asr_evidence(clipped, "")
            finally:
                core.audio_evidence.close()
            wavs, floats = list(Path(temporary).glob("*.wav")), list(Path(temporary).glob("*.f32le"))
            self.assertEqual((len(wavs), len(floats)), (1, 1))
            with wave.open(str(wavs[0]), "rb") as stream:
                self.assertEqual(stream.readframes(stream.getnframes()), clipped.astype("<i2").tobytes())
            self.assertEqual(floats[0].read_bytes(), exact_input)
            metadata = json.loads(Path(str(wavs[0]) + ".json").read_text())
            self.assertIn("clipped_audio", metadata["reasons"])
            self.assertIn("empty_transcript", metadata["reasons"])
            self.assertFalse(metadata["metadata"]["pcm"]["accuracy_measured"])

    def test_two_sessions_never_share_reply_log_or_audio_paths(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict("os.environ"):
            cores = [load_core(), load_core()]
            try:
                for core in cores:
                    core.__file__ = str(Path(temporary) / "app.py")
                    core.args.recordings_dir = str(Path(temporary) / "recordings")
                    core.args.debug = True
                    core.args.audio_review = "auto"
                    core.initialize_session_runtime()
                self.assertNotEqual(cores[0].runtime_session_id, cores[1].runtime_session_id)
                for attribute in ("output_wav", "transcription_log", "tts_log"):
                    self.assertNotEqual(getattr(cores[0].args, attribute), getattr(cores[1].args, attribute))
                    self.assertTrue(Path(getattr(cores[0].args, attribute)).resolve().is_relative_to(Path(temporary).resolve()))
                self.assertNotEqual(cores[0].runtime_audio_dir, cores[1].runtime_audio_dir)
                self.assertEqual(cores[0].audio_review_mode, "full")
                self.assertEqual(cores[1].audio_review_mode, "full")
            finally:
                for core in cores:
                    if core.audio_evidence is not None:
                        core.audio_evidence.close()


if __name__ == "__main__":
    unittest.main()
