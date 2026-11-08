"""Startup keeps model libraries dormant and honors the selected full model."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from optional_dependencies import LazyDependency, optional_dependency


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "tts-transcribe.py"


class LightweightStartupTests(unittest.TestCase):
    def test_direct_configuration_does_not_import_any_model_runtime(self):
        code = """
import json, runpy, sys
sys.argv = ['tts-transcribe.py', '--dump-config']
scope = runpy.run_path('tts-transcribe.py', run_name='configuration_only')
config = scope['effective_config']()
print(json.dumps({'config': config, 'loaded': [name for name in
    ['torch', 'transformers', 'faster_whisper', 'kokoro_onnx', 'mlx.core', 'mlx_lm', 'mlx_whisper']
    if name in sys.modules]}))
"""
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["loaded"], [])
        config = report["config"]
        self.assertEqual(config["audio_io"], "sounddevice")
        self.assertEqual(config["streaming_engine"], "ring")
        if sys.platform == "darwin" and os.uname().machine == "arm64":
            self.assertEqual(config["resolved_stt_backend"], "mlx-whisper")
            self.assertTrue(config["whisper_model"].endswith(("whisper-large-v3-mlx", "whisper-large-v3")))
            self.assertEqual(config["resolved_translation_backend"], "mlx-lm")
            self.assertTrue(config["translation_model"].endswith("translategemma-12b-it-4bit"))
        if (ROOT / "models/whisper-large-v3-mlx/config.json").is_file() and any(
                (ROOT / "models/whisper-large-v3-mlx" / name).is_file()
                for name in ("weights.safetensors", "weights.npz")):
            self.assertEqual(config["whisper_model"], str(ROOT / "models/whisper-large-v3-mlx"))
        if (ROOT / "models/translategemma-12b-it-4bit").is_dir():
            self.assertEqual(config["translation_model"], str(ROOT / "models/translategemma-12b-it-4bit"))

    def test_explicit_backend_models_and_capture_options_remain_authoritative(self):
        result = subprocess.run([
            sys.executable, str(CORE), "--dump-config",
            "--stt-backend", "mlx-vlm-gemma", "--translation-backend", "nllb",
            "--translation-model", "custom/translation-model",
            "--whisper-model", "custom/recognition-model",
            "--audio-io", "apple-raw", "--streaming-engine", "legacy",
        ], cwd=ROOT, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        config = json.loads(result.stderr)
        self.assertEqual(config["resolved_stt_backend"], "mlx-vlm-gemma")
        self.assertEqual(config["resolved_translation_backend"], "nllb")
        self.assertEqual(config["translation_model"], "custom/translation-model")
        self.assertEqual(config["whisper_model"], "custom/recognition-model")
        self.assertEqual(config["audio_io"], "apple-raw")
        self.assertEqual(config["streaming_engine"], "legacy")

    def test_missing_local_pretrained_asset_never_retries_remote_implicitly(self):
        with patch.object(sys, "argv", [str(CORE), "--dump-config"]), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            scope = runpy.run_path(str(CORE), run_name="local_loader_test")
        calls = []

        class MissingLoader:
            @classmethod
            def from_pretrained(cls, model, **kwargs):
                calls.append((model, kwargs))
                raise OSError("weights not present")

        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}):
            with self.assertRaisesRegex(RuntimeError, "Cannot load local model"):
                scope["load_pretrained_local_first"](MissingLoader, "custom/missing")
        self.assertEqual(calls, [("custom/missing", {"local_files_only": True})])


class LazyDependencyTests(unittest.TestCase):
    def test_backend_is_imported_once_when_first_used_across_threads(self):
        imported = []
        configured = []
        fake = SimpleNamespace(tensor=lambda value: value * 2)
        dependency = LazyDependency("test_backend", configure=configured.append)

        def load(name):
            imported.append(name)
            return fake

        self.assertFalse(dependency.loaded)
        self.assertIn("loaded=False", repr(dependency))
        results = []
        with patch("optional_dependencies.importlib.import_module", side_effect=load):
            threads = [threading.Thread(target=lambda: results.append(dependency.tensor(3))) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=2)
                self.assertFalse(thread.is_alive())
        self.assertEqual(results, [6] * 8)
        self.assertEqual(imported, ["test_backend"])
        self.assertEqual(configured, [fake])

    def test_failed_native_initialization_is_not_retried(self):
        dependency = LazyDependency("failed_native_runtime")
        with patch("optional_dependencies.importlib.import_module", side_effect=ImportError("no native device")) as loader:
            for _ in range(2):
                with self.assertRaisesRegex(ImportError, "no native device"):
                    dependency.tensor(1)
        self.assertEqual(loader.call_count, 1)

    def test_missing_optional_package_keeps_none_without_importing(self):
        with patch("optional_dependencies.importlib.util.find_spec", return_value=None), \
                patch("optional_dependencies.importlib.import_module") as loader:
            self.assertIsNone(optional_dependency("not_installed", "Model"))
        loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
