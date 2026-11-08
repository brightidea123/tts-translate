"""MLX model selection resolves the full-v3 alias without silently using turbo."""
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tts_runtime.model_assets import require_cached_mlx_whisper


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "tts-transcribe.py"
LARGE_V3 = "mlx-community/whisper-large-v3-mlx"


class MlxWhisperModelResolutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.object(sys, "argv", [str(CORE), "--dump-config"]), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            scope = runpy.run_path(str(CORE), run_name="model_resolution_test")
        cls.resolve = staticmethod(scope["resolve_mlx_whisper_model_ref_for"])
        cls.namespace = cls.resolve.__globals__

    def test_full_v3_names_resolve_to_canonical_mlx_repository(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch.dict(self.namespace, __file__=str(Path(temporary) / "tts-transcribe.py")):
            for model in ["large-v3", "mlx-community/whisper-large-v3", "mlx-community:whisper-large-v3", LARGE_V3]:
                with self.subTest(model=model):
                    self.assertEqual(self.resolve(model), LARGE_V3)

    def test_turbo_and_custom_repositories_are_not_replaced(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch.dict(self.namespace, __file__=str(Path(temporary) / "tts-transcribe.py")):
            for supplied, expected in [
                ("large-v3-turbo", "mlx-community/whisper-large-v3-turbo"),
                ("custom/recognizer", "custom/recognizer"),
                ("medium", "mlx-community/whisper-medium"),
            ]:
                with self.subTest(model=supplied):
                    self.assertEqual(self.resolve(supplied), expected)

    def test_existing_explicit_local_path_stays_authoritative(self):
        with tempfile.TemporaryDirectory() as temporary:
            local = Path(temporary) / "my-whisper"
            local.mkdir()
            (local / "config.json").write_text("{}", encoding="utf-8")
            (local / "weights.npz").write_bytes(b"test weights")
            with patch.dict(self.namespace, prefer_project_model=lambda model: (_ for _ in ()).throw(AssertionError("local selection must not consult Hub alias"))):
                self.assertEqual(self.resolve(str(local)), str(local.resolve()))

    def test_incomplete_project_folder_does_not_hide_the_hub_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            incomplete = Path(temporary) / "models/whisper-large-v3-mlx"
            incomplete.mkdir(parents=True)
            (incomplete / "config.json").write_text("{}", encoding="utf-8")
            (incomplete / "weights.npz").touch()
            with patch.dict(self.namespace, __file__=str(Path(temporary) / "tts-transcribe.py")):
                self.assertEqual(self.resolve("large-v3"), LARGE_V3)

    def test_complete_project_copy_is_used_for_alias_and_canonical_name(self):
        with tempfile.TemporaryDirectory() as temporary:
            complete = Path(temporary) / "models/whisper-large-v3-mlx"
            complete.mkdir(parents=True)
            (complete / "config.json").write_text("{}", encoding="utf-8")
            (complete / "weights.safetensors").write_bytes(b"test weights")
            with patch.dict(self.namespace, __file__=str(Path(temporary) / "tts-transcribe.py")):
                self.assertEqual(self.resolve("large-v3"), str(complete.resolve()))
                self.assertEqual(self.resolve(LARGE_V3), str(complete.resolve()))

    def test_complete_legacy_full_v3_folder_is_reused_without_turbo_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            models = Path(temporary) / "models"
            turbo = models / "whisper-large-v3-turbo"
            turbo.mkdir(parents=True)
            (turbo / "config.json").write_text("{}", encoding="utf-8")
            (turbo / "weights.npz").write_bytes(b"turbo weights")
            with patch.dict(self.namespace, __file__=str(Path(temporary) / "tts-transcribe.py")):
                self.assertEqual(self.resolve("large-v3"), LARGE_V3)
                legacy = models / "whisper-large-v3"
                legacy.mkdir()
                (legacy / "config.json").write_text("{}", encoding="utf-8")
                (legacy / "weights.npz").write_bytes(b"full weights")
                self.assertEqual(self.resolve("large-v3"), str(legacy.resolve()))

    def test_missing_cache_fails_initialization_before_mlx_import_even_without_warmup(self):
        # Exercise real init_stt_runtime_only: skipping warmup must not merely
        # postpone missing-asset failure until the first captured utterance.
        arguments = SimpleNamespace(**vars(self.namespace["args"]))
        arguments.mode = "audio"
        arguments.warmup = False
        arguments.stt_backend = "mlx-whisper"
        arguments.whisper_model = "large-v3"
        repair = "Missing full-v3 cache. Run: python scripts/prepare_models.py --download --models whisper"
        preflight = Mock(side_effect=RuntimeError(repair))
        native_import, warmup, diagnostics = Mock(), Mock(), Mock()
        with patch.dict(self.namespace, args=arguments, MLX_WHISPER_INSTALLED=True,
                        prepare_mlx_whisper_runtime_reference=preflight,
                        ensure_mlx_whisper_loaded=native_import,
                        warmup_stt_runtime_only=warmup,
                        configure_inference_diagnostics=diagnostics):
            with self.assertRaisesRegex(RuntimeError, "Missing full-v3 cache.*prepare_models.py"):
                self.namespace["initialize_asr_stage"]()
        diagnostics.assert_called_once_with("asr")
        preflight.assert_called_once()
        native_import.assert_not_called()
        warmup.assert_not_called()


class MlxWhisperCachePreflightTests(unittest.TestCase):
    command = "python scripts/prepare_models.py --download --models whisper"

    def require(self, reference, *, offline=True):
        return require_cached_mlx_whisper(reference, offline=offline, preparation_command=self.command)

    @staticmethod
    def complete(directory):
        (directory / "config.json").write_text("{}", encoding="utf-8")
        (directory / "weights.npz").write_bytes(b"complete test weights")

    def test_explicit_complete_model_never_looks_up_the_hub(self):
        download = Mock(side_effect=AssertionError("local assets must not access Hub"))
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.complete(directory)
            with patch.dict(sys.modules, huggingface_hub=SimpleNamespace(snapshot_download=download)):
                self.assertEqual(self.require(str(directory)), str(directory.resolve()))
        download.assert_not_called()

    def test_explicit_incomplete_model_reports_repair_without_hub_fallback(self):
        download = Mock(side_effect=AssertionError("explicit local selection must not fall back to Hub"))
        with tempfile.TemporaryDirectory() as temporary, \
                patch.dict(sys.modules, huggingface_hub=SimpleNamespace(snapshot_download=download)):
            with self.assertRaises(RuntimeError) as failure:
                self.require(temporary)
        self.assertIn("Incomplete MLX Whisper model", str(failure.exception))
        self.assertIn(self.command, str(failure.exception))
        download.assert_not_called()

    def test_missing_hub_snapshot_is_local_only_and_reports_the_repair_command(self):
        download = Mock(side_effect=OSError("not cached"))
        with patch.dict(sys.modules, huggingface_hub=SimpleNamespace(snapshot_download=download)):
            with self.assertRaises(RuntimeError) as failure:
                self.require(LARGE_V3)
        download.assert_called_once_with(repo_id=LARGE_V3, local_files_only=True)
        self.assertIn("Offline mode is enabled", str(failure.exception))
        self.assertIn(LARGE_V3, str(failure.exception))
        self.assertIn(self.command, str(failure.exception))

    def test_incomplete_cached_snapshot_does_not_reach_model_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "config.json").write_text("{}", encoding="utf-8")
            (directory / "weights.npz").touch()
            download = Mock(return_value=temporary)
            with patch.dict(sys.modules, huggingface_hub=SimpleNamespace(snapshot_download=download)):
                with self.assertRaises(RuntimeError) as failure:
                    self.require(LARGE_V3)
        download.assert_called_once_with(repo_id=LARGE_V3, local_files_only=True)
        self.assertIn("incomplete", str(failure.exception))
        self.assertIn(self.command, str(failure.exception))

    def test_complete_hub_snapshot_becomes_a_local_model_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.complete(directory)
            download = Mock(return_value=temporary)
            with patch.dict(sys.modules, huggingface_hub=SimpleNamespace(snapshot_download=download)):
                self.assertEqual(self.require(LARGE_V3), str(directory.resolve()))
        download.assert_called_once_with(repo_id=LARGE_V3, local_files_only=True)

    def test_online_mode_defers_the_selected_repo_without_preflight_download(self):
        download = Mock(side_effect=AssertionError("preflight must not download"))
        with patch.dict(sys.modules, huggingface_hub=SimpleNamespace(snapshot_download=download)):
            self.assertEqual(self.require(LARGE_V3, offline=False), LARGE_V3)
        download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
