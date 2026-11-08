from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

from scripts import generate_fixtures, prepare_models


class AssetPreparationTests(unittest.TestCase):
    def test_default_status_has_no_network_or_generated_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "not-created"
            for module, argv, boundary in [
                (prepare_models, ["--models-dir", str(missing)], "fetch_model"),
                (generate_fixtures, ["--output-dir", str(missing)], "generate_fixture"),
            ]:
                with patch.object(module, boundary, side_effect=AssertionError("unexpected side effect")), redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(module.main(argv), 0)
                report = json.loads(output.getvalue())
                self.assertFalse(report.get("downloads_run", report.get("generation_run")))
                self.assertFalse(missing.exists())

    def test_metadata_fetch_preserves_existing_weights_and_records_honest_checksums(self):
        with tempfile.TemporaryDirectory() as temporary:
            models_dir = Path(temporary)
            directory = prepare_models.model_directory(models_dir, prepare_models.BASELINE_MODELS["whisper"])
            directory.mkdir()
            weights = b"existing browser-downloaded weights"
            (directory / "weights.npz").write_bytes(weights)
            calls = []

            def snapshot_download(**kwargs):
                calls.append(kwargs)
                (Path(kwargs["local_dir"]) / "config.json").write_text('{"model_type":"whisper"}', encoding="utf-8")
                cache = directory / ".cache" / "huggingface"
                cache.mkdir(parents=True)
                (cache / "download.lock").write_text("not part of the model")
                return str(directory)

            with patch.dict(prepare_models.os.environ), \
                    patch.object(prepare_models.importlib, "import_module", return_value=SimpleNamespace(snapshot_download=snapshot_download)):
                record = prepare_models.fetch_model("whisper", models_dir, metadata_only=True, max_workers=2)
            manifest_path = prepare_models.write_manifest(models_dir, {"whisper": record})
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            model = manifest["models"]["whisper"]
            self.assertEqual(calls[0]["revision"], prepare_models.LARGE_V3_REVISION)
            self.assertIn("*.npz", calls[0]["ignore_patterns"])
            self.assertIn("*.safetensors", calls[0]["ignore_patterns"])
            self.assertEqual((directory / "weights.npz").read_bytes(), weights)
            self.assertEqual(model["files"]["weights.npz"]["sha256"], hashlib.sha256(weights).hexdigest())
            self.assertFalse(model["files"]["weights.npz"]["included_in_snapshot_request"])
            self.assertFalse(model["complete_snapshot_requested"])
            self.assertIn("has not been verified", model["integrity_note"])
            self.assertEqual(set(model["files"]), {"weights.npz", "config.json"})

    def test_missing_matching_voice_fails_before_synthesis(self):
        inventory = "Samantha            en_US    # Hello\nCarmit             he_IL    # שלום\n"
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(generate_fixtures.platform, "system", return_value="Darwin"), \
                    patch.object(generate_fixtures.shutil, "which", side_effect=lambda name: "/usr/bin/" + name), \
                    patch.object(generate_fixtures.subprocess, "run", return_value=SimpleNamespace(stdout=inventory)) as run, \
                    patch.object(generate_fixtures, "generate_fixture", side_effect=AssertionError("unexpected synthesis")), \
                    redirect_stdout(io.StringIO()) as output:
                code = generate_fixtures.main(["--generate", "--output-dir", str(Path(temporary) / "new"), "--languages", "he", "ar"])
            self.assertEqual(code, 1)
            self.assertIn("No installed AR voice", json.loads(output.getvalue())["error"])
            self.assertEqual(run.call_count, 1)
            self.assertFalse((Path(temporary) / "new").exists())

    def test_full_whisper_download_verifies_pinned_weight_bytes(self):
        weights = b"downloaded model fixture"
        expected = hashlib.sha256(weights).hexdigest()
        for pinned_hash in (expected, "0" * 64):
            with self.subTest(pinned_hash=pinned_hash), tempfile.TemporaryDirectory() as temporary:
                models = Path(temporary)
                fixture_model = {**prepare_models.BASELINE_MODELS["whisper"],
                                 "expected_sha256": {"weights.npz": pinned_hash}}

                def download(**kwargs):
                    directory = Path(kwargs["local_dir"])
                    directory.mkdir(parents=True)
                    (directory / "config.json").write_text("{}", encoding="utf-8")
                    (directory / "weights.npz").write_bytes(weights)

                with patch.dict(prepare_models.os.environ), \
                        patch.dict(prepare_models.BASELINE_MODELS, whisper=fixture_model), \
                        patch.object(prepare_models.importlib, "import_module", return_value=SimpleNamespace(snapshot_download=download)):
                    if pinned_hash == expected:
                        result = prepare_models.fetch_model("whisper", models, metadata_only=False, max_workers=1)
                        self.assertTrue(result["complete_snapshot_requested"])
                        self.assertEqual(result["files"]["weights.npz"]["sha256"], expected)
                    else:
                        with self.assertRaisesRegex(RuntimeError, "failed the pinned SHA-256 check"):
                            prepare_models.fetch_model("whisper", models, metadata_only=False, max_workers=1)

    def test_synthesis_preserves_hebrew_source_and_produces_pcm_without_playback(self):
        calls = []
        phrase = generate_fixtures.SPEECH_FIXTURES["he"]

        def run(command, **kwargs):
            calls.append(command)
            if command[0] == "/usr/bin/say":
                self.assertEqual(Path(command[command.index("-f") + 1]).read_text(encoding="utf-8"), phrase + "\n")
                Path(command[command.index("-o") + 1]).write_bytes(b"temporary aiff")
            else:
                self.assertEqual(command[0], "/usr/bin/afconvert")
                with wave.open(command[-1], "wb") as handle:
                    handle.setnchannels(1)
                    handle.setsampwidth(2)
                    handle.setframerate(16000)
                    handle.writeframes(b"\x01\x00" * 160)
            return SimpleNamespace(returncode=0)

        with tempfile.TemporaryDirectory() as temporary, patch.object(generate_fixtures.subprocess, "run", side_effect=run):
            output_dir = Path(temporary)
            result = generate_fixtures.generate_fixture("he", {"name": "Carmit", "locale": "he_IL"}, output_dir, "/usr/bin/say", "/usr/bin/afconvert")
            self.assertEqual(result["text"], phrase)
            self.assertEqual(result["sample_rate"], 16000)
            self.assertEqual(result["duration_seconds"], 0.01)
            self.assertEqual(result["sha256"], hashlib.sha256((output_dir / "he.wav").read_bytes()).hexdigest())
            self.assertEqual({command[0] for command in calls}, {"/usr/bin/say", "/usr/bin/afconvert"})
            self.assertEqual([path.name for path in output_dir.iterdir()], ["he.wav"])


if __name__ == "__main__":
    unittest.main()
