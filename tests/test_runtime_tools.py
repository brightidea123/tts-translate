"""Offline tests for runtime tools; no ML backend, microphone or network needed."""

from __future__ import annotations

from contextlib import redirect_stdout
import importlib.metadata
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import wave

from scripts import doctor, run_profile, smoke_models

_REAL_IMPORT = __import__


def _import_without_whisper(name, *args, **kwargs):
    if name == "mlx_whisper":
        raise ImportError("test backend unavailable")
    return _REAL_IMPORT(name, *args, **kwargs)


class ProfileLaunchTests(unittest.TestCase):
    def test_explicit_baseline_backends_and_languages(self):
        with tempfile.TemporaryDirectory() as folder:
            command = run_profile.build_command("baseline", "Arabic", "Hebrew", models_dir=Path(folder), python="/test/python")
        self.assertEqual(command[0], "/test/python")
        for flag, value in {
            "--source-language": "AR", "--target-language": "HE", "--stt-backend": "mlx-whisper",
            "--translation-backend": "mlx-lm", "--whisper-model": "mlx-community/whisper-large-v3-mlx",
            "--translation-model": "mlx-community/translategemma-12b-it-4bit",
            "--audio-io": "sounddevice", "--streaming-engine": "ring", "--vad-backend": "energy",
        }.items():
            self.assertEqual(command[command.index(flag) + 1], value)

    def test_local_models_and_passthrough_keep_audio_device_as_one_argument(self):
        with tempfile.TemporaryDirectory() as folder:
            models = Path(folder)
            (models / "whisper-large-v3-mlx").mkdir()
            (models / "translategemma-12b-it-4bit").mkdir()
            command = run_profile.build_command("baseline", "TR", "DE", models_dir=models,
                input_device="LitLink Audio Bridge", extra_args=["--no-warmup"])
            self.assertEqual(command[command.index("--whisper-model") + 1], str((models / "whisper-large-v3-mlx").resolve()))
            self.assertEqual(command[command.index("--translation-model") + 1], str((models / "translategemma-12b-it-4bit").resolve()))
        self.assertEqual(command[command.index("--input-device") + 1], "LitLink Audio Bridge")
        self.assertEqual(command[-1], "--no-warmup")

    def test_optional_gemma_profile_uses_audio_model_flag(self):
        command = run_profile.build_command("gemma4_audio_optional", "HE", "FR")
        self.assertIn("--mlx-vlm-model", command)
        self.assertNotIn("--whisper-model", command)
        self.assertEqual(command[command.index("--translation-backend") + 1], "mlx-lm")

    def test_explicit_audio_overrides_are_preserved(self):
        command = run_profile.build_command("baseline", "HE", "FR", audio_io="apple-raw", streaming_engine="legacy", vad_backend="webrtc")
        for flag, value in {"--audio-io": "apple-raw", "--streaming-engine": "legacy", "--vad-backend": "webrtc"}.items():
            self.assertEqual(command[command.index(flag) + 1], value)

    def test_invalid_profile_and_language_fail_before_launch(self):
        for args in [("not-a-profile", "TR", "HE"), ("baseline", "TR", "TR"), ("baseline", "ja", "HE")]:
            with self.assertRaises(ValueError):
                run_profile.build_command(*args)

    def test_launch_is_offline_by_default_and_download_requires_flag(self):
        class Completed:
            returncode = 0
        with patch.object(run_profile.subprocess, "run", return_value=Completed()) as run:
            self.assertEqual(run_profile.main(["baseline", "--source", "AR", "--target", "HE", "--dump-config"]), 0)
            self.assertEqual(run.call_args.kwargs["env"]["HF_HUB_OFFLINE"], "1")
            self.assertIn("--dump-config", run.call_args.args[0])
            run_profile.main(["baseline", "--allow-download", "--", "--no-warmup"])
            self.assertEqual(run.call_args.kwargs["env"]["HF_HUB_OFFLINE"], "0")
            self.assertEqual(run.call_args.args[0][-1], "--no-warmup")


class DoctorTests(unittest.TestCase):
    def test_missing_dependencies_are_reported_without_import_or_device_query(self):
        with patch.object(doctor, "package_version", return_value=None), patch.object(doctor, "isolated_probe") as probes:
            result = doctor.inspect_runtime(probe_metal=False, import_backends=False)
        self.assertEqual(result["missing_baseline_packages"], list(doctor.BASELINE_PACKAGES))
        self.assertFalse(result["ready_for_model_smoke"])
        self.assertFalse(result["model_inference"]["run"])
        self.assertFalse(result["audio"]["queried"])
        probes.assert_not_called()

    def test_model_presence_is_not_claimed_as_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder)
            (model / "config.json").write_text("{}")
            self.assertFalse(doctor.model_status(model)["appears_complete"])
            (model / "model.safetensors").write_bytes(b"fixture")
            result = doctor.model_status(model)
        self.assertTrue(result["appears_complete"])
        self.assertIn("not tested", result["validation"])

    def test_metal_import_failure_is_diagnostic(self):
        failure = {"tested": True, "available": False, "compute_ok": False, "error": "ImportError: missing mlx", "returncode": 1}
        with patch.object(doctor, "isolated_probe", return_value=failure):
            result = doctor.metal_probe()
        self.assertFalse(result["compute_ok"])
        self.assertIn("ImportError", result["error"])

    def test_installed_metadata_does_not_hide_broken_imports(self):
        with patch.object(doctor, "package_version", return_value="1.0"), patch.object(doctor, "isolated_probe", return_value={"ok": False, "error": "broken backend", "returncode": -6}):
            result = doctor.inspect_runtime(probe_metal=False, import_backends=True)
        self.assertEqual(result["missing_baseline_packages"], [])
        self.assertFalse(result["ready_for_model_smoke"])
        self.assertTrue(all(not item["ok"] and "broken backend" in item["error"] for item in result["backend_imports"].values()))

    def test_failed_import_is_reported_from_real_child_without_loading_parent(self):
        result = doctor.isolated_probe("import", "_tts_doctor_missing_backend_fixture")
        self.assertFalse(result["ok"])
        self.assertIn("ModuleNotFoundError", result["error"])
        self.assertEqual(result["returncode"], 1)

    def test_native_abort_stays_in_child_and_preserves_error(self):
        aborted = subprocess.CompletedProcess([], -6, stdout="", stderr="nanobind: duplicate DeviceType")
        with patch.object(doctor.subprocess, "run", return_value=aborted) as run:
            result = doctor.isolated_probe("metal")
        self.assertFalse(result["compute_ok"])
        self.assertEqual(result["returncode"], -6)
        self.assertIn("signal 6", result["error"])
        self.assertIn("duplicate DeviceType", result["stderr"])
        self.assertEqual(run.call_args.kwargs["env"]["HF_HUB_OFFLINE"], "1")

    def test_hung_native_probe_has_bounded_timeout(self):
        with patch.object(doctor.subprocess, "run", side_effect=subprocess.TimeoutExpired("probe", 0.1)):
            result = doctor.isolated_probe("import", "mlx_lm", timeout=0.1)
        self.assertFalse(result["ok"])
        self.assertTrue(result["timed_out"])
        self.assertIsNone(result["returncode"])

    def test_skipping_native_checks_cannot_claim_smoke_readiness(self):
        with patch.object(doctor, "package_version", return_value="1.0"):
            result = doctor.inspect_runtime(probe_metal=False, import_backends=False)
        self.assertFalse(result["ready_for_model_smoke"])


class SmokeToolTests(unittest.TestCase):
    def test_all_directed_pairs_are_unique_and_preserve_unicode(self):
        cases = smoke_models.translation_cases("TR", "HE", None, True)
        self.assertEqual(len(cases), 30)
        self.assertEqual(len({(src, dst) for src, dst, _ in cases}), 30)
        self.assertTrue(all(src != dst for src, dst, _ in cases))
        self.assertEqual(next(text for src, dst, text in cases if src == "he" and dst == "ar"), smoke_models.SPEECH_FIXTURES["he"])

    def test_custom_text_whitespace_and_direction_are_preserved(self):
        text = "  שלום 123، مرحباً\nFR: rendez-vous?  "
        self.assertEqual(smoke_models.translation_cases("HE", "AR", text, False), [("he", "ar", text)])

    def test_default_is_explicit_no_inference_and_no_model_import(self):
        with patch.object(smoke_models, "asr_smoke") as asr, patch.object(smoke_models, "translation_smoke") as translate, io.StringIO() as capture, redirect_stdout(capture):
            status = smoke_models.main([])
            result = json.loads(capture.getvalue())
        self.assertEqual(status, 0)
        self.assertFalse(result["model_inference_requested"])
        self.assertFalse(result["model_inference_attempted"])
        self.assertFalse(result["model_inference_run"])
        self.assertFalse(result["model_inference_completed"])
        self.assertFalse(result["downloads_allowed"])
        asr.assert_not_called()
        translate.assert_not_called()

    def test_requested_backend_import_failure_does_not_claim_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fixture.wav"
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(1)
                handle.setframerate(16000)
                handle.setsampwidth(2)
                handle.writeframes(b"\x00\x00" * 160)
            with patch("builtins.__import__", side_effect=_import_without_whisper), io.StringIO() as capture, redirect_stdout(capture):
                status = smoke_models.main(["--asr-model", "/local/whisper", "--audio", str(path)])
                result = json.loads(capture.getvalue())
        self.assertEqual(status, 1)
        self.assertTrue(result["model_inference_requested"])
        self.assertTrue(result["model_inference_attempted"])
        self.assertFalse(result["model_inference_run"])
        self.assertFalse(result["model_inference_completed"])
        self.assertFalse(result["stages"]["asr"]["completed"])
        self.assertIn("ImportError: test backend unavailable", result["stages"]["asr"]["error"])

    def test_missing_audio_is_rejected_before_backend_import(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "missing.wav"
            with patch("builtins.__import__", side_effect=_import_without_whisper), io.StringIO() as capture, redirect_stdout(capture):
                status = smoke_models.main(["--asr-model", "/local/whisper", "--audio", str(path)])
                result = json.loads(capture.getvalue())
        self.assertEqual(status, 1)
        self.assertFalse(result["model_inference_run"])
        self.assertIn("FileNotFoundError", result["stages"]["asr"]["error"])

    def test_one_failed_stage_does_not_hide_other_completed_inference(self):
        with patch.object(smoke_models, "asr_smoke", return_value={"ok": True, "text": "recognized"}), \
                patch.object(smoke_models, "translation_smoke", side_effect=ImportError("missing translation backend")), \
                io.StringIO() as capture, redirect_stdout(capture):
            status = smoke_models.main(["--asr-model", "/local/whisper", "--audio", "fixture.wav", "--translation-model", "/local/translator"])
            result = json.loads(capture.getvalue())
        self.assertEqual(status, 1)
        self.assertTrue(result["model_inference_run"])
        self.assertFalse(result["model_inference_completed"])
        self.assertTrue(result["stages"]["asr"]["completed"])
        self.assertFalse(result["stages"]["translation"]["completed"])
        self.assertEqual(set(result["results"]), {"asr"})

    def test_failed_first_stage_skips_native_backend_retry(self):
        with patch.object(smoke_models, "asr_smoke", side_effect=ImportError("Metal unavailable")), \
                patch.object(smoke_models, "translation_smoke") as translate, \
                io.StringIO() as capture, redirect_stdout(capture):
            status = smoke_models.main(["--asr-model", "/local/whisper", "--audio", "fixture.wav", "--translation-model", "/local/translator"])
            result = json.loads(capture.getvalue())
        self.assertEqual(status, 1)
        self.assertTrue(result["model_inference_requested"])
        self.assertTrue(result["model_inference_attempted"])
        self.assertFalse(result["model_inference_run"])
        self.assertFalse(result["model_inference_completed"])
        self.assertTrue(result["stages"]["asr"]["attempted"])
        self.assertFalse(result["stages"]["translation"]["attempted"])
        self.assertTrue(result["stages"]["translation"]["skipped"])
        self.assertIn("not retried", result["stages"]["translation"]["skip_reason"])
        translate.assert_not_called()

    def test_pcm_fixture_uses_known_sample_values_and_rejects_wrong_format(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fixture.wav"
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(1)
                handle.setframerate(16000)
                handle.setsampwidth(2)
                handle.writeframes(bytes([0, 128, 0, 0, 0, 64]))
            samples = smoke_models.load_pcm_fixture(path)
            self.assertEqual(samples.tolist(), [-1.0, 0.0, 0.5])
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(2)
                handle.setframerate(48000)
                handle.setsampwidth(2)
                handle.writeframes(bytes([0, 0, 0, 0]))
            with self.assertRaisesRegex(ValueError, "16 kHz mono"):
                smoke_models.load_pcm_fixture(path)


if __name__ == "__main__":
    unittest.main()
