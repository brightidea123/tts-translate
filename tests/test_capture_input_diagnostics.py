"""Capture preflight persists its resolved input without hardware in metadata paths."""
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from test_accuracy_integration import load_core
from tts_runtime.audio_devices import describe_input_device


class StopAfterPreflight(Exception):
    pass


class CaptureInputDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.core = load_core()
        self.info = {"index": 7, "name": "LitLink Audio Bridge", "hostapi": 0,
                     "max_input_channels": 2, "default_samplerate": 48000.0}

    def fake_sounddevice(self):
        return SimpleNamespace(
            query_devices=Mock(return_value=dict(self.info)),
            check_input_settings=Mock(),
            default=SimpleNamespace(device=[7, 0]),
            InputStream=Mock(side_effect=AssertionError("Test must not open an audio stream")),
        )

    def test_preflight_resolution_is_persisted_before_models_or_capture_and_reaches_web_config(self):
        core, sd = self.core, self.fake_sounddevice()
        core.args.input_device = None
        core.args.audio_review = "off"
        core.args.ui = "headless"
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(core.os.environ, {}, clear=False), \
                patch.object(core, "sd", sd), \
                patch.object(core, "validate_runtime_configuration"), \
                patch.object(core, "ensure_audio_runtime"), \
                patch.object(core, "resolve_audio_io_backend", return_value="sounddevice"), \
                patch.object(core, "create_session_directory", return_value=Path(directory)), \
                patch.object(core, "init_parent_routing_runtime", side_effect=StopAfterPreflight), \
                patch.object(core, "ProcessStage") as stage, \
                patch.object(core, "setup_terminal") as terminal:
            with self.assertRaises(StopAfterPreflight):
                core.main()
            metadata = json.loads((Path(directory) / "session.json").read_text())
            resolved = metadata["config"]["resolved_input_device"]
            self.assertEqual(resolved["name"], "LitLink Audio Bridge")
            self.assertEqual(resolved["index"], 7)
            self.assertTrue(resolved["requested_default"])
            self.assertTrue(resolved["is_default"])
            self.assertEqual(resolved["sample_rate"], 16000)
            self.assertEqual(resolved["channels"], 1)
            self.assertEqual(resolved["dtype"], "int16")
            self.assertIsNone(metadata["config"]["input_device"])
            self.assertEqual(core.web_state_snapshot()["config"]["resolved_input_device"], resolved)
            self.assertIsNone(core.web_state_snapshot()["config"]["input_device"])
            sd.query_devices.assert_called_once_with(None, kind="input")
            sd.check_input_settings.assert_called_once_with(device=7, samplerate=16000, channels=1, dtype="int16")
            sd.InputStream.assert_not_called()
            stage.assert_not_called()
            terminal.assert_not_called()

    def test_periodic_listening_and_no_input_status_retain_the_actual_resolved_name(self):
        core, sd = self.core, self.fake_sounddevice()
        core.args.input_device = None
        core.resolved_capture_input = describe_input_device(self.info, None, samplerate=16000,
            channels=1, dtype="int16", default_index=7)
        feeder = SimpleNamespace(captured_samples=32000, last_input_at=99.5, dropped_samples=0,
                                 endpoint=SimpleNamespace(last_dbfs=-22, in_speech=True))
        worker = SimpleNamespace(snapshot={"state": "ready", "error": "", "pending": 0, "inflight_key": None})
        with patch.object(core, "sd", sd), patch.object(core, "asr_stage", worker), \
                patch.object(core, "translation_stage", worker), patch.object(core, "render_dashboard"), \
                patch.object(core.time, "monotonic", return_value=100):
            states = []
            for input_at in [99.5, 0, 99.5]:
                feeder.last_input_at = input_at
                # A later default-route change cannot rewrite this capture's identity.
                sd.default.device = [2, 0]
                core.update_capture_status(feeder)
                states.append(core.web_state_snapshot()["stages"]["capture"])
            self.assertTrue(states[0].startswith("listening |"))
            self.assertTrue(states[1].startswith("no input |"))
            self.assertTrue(states[2].startswith("listening |"))
            for status in states:
                self.assertIn("'LitLink Audio Bridge' (index 7)", status)
            self.assertEqual(core.web_state_snapshot()["config"]["resolved_input_device"]["index"], 7)
            self.assertIsNone(core.args.input_device)
            sd.query_devices.assert_not_called()
            sd.check_input_settings.assert_not_called()
            sd.InputStream.assert_not_called()

    def test_metadata_and_web_snapshot_leave_unresolved_default_unknown_without_querying(self):
        core, sd = self.core, self.fake_sounddevice()
        core.args.input_device = None
        core.resolved_capture_input = None
        with patch.object(core, "sd", sd), \
                patch.object(core, "resolve_audio_io_backend", return_value="sounddevice"):
            self.assertIsNone(core.effective_config()["resolved_input_device"])
            self.assertIsNone(core.web_state_snapshot()["config"]["resolved_input_device"])
            sd.query_devices.assert_not_called()
            sd.check_input_settings.assert_not_called()
            sd.InputStream.assert_not_called()

    def test_dump_config_does_not_resolve_a_device_or_initialize_audio(self):
        core, sd = self.core, self.fake_sounddevice()
        core.args.dump_config = True
        core.args.input_device = None
        core.resolved_capture_input = None
        output = io.StringIO()
        with patch.object(core, "sd", sd), redirect_stderr(output), \
                patch.object(core, "resolve_audio_io_backend", return_value="sounddevice"), \
                patch.object(core, "ensure_audio_runtime") as audio, \
                patch.object(core, "initialize_session_runtime") as session, \
                patch.object(core, "ProcessStage") as stage:
            self.assertEqual(core.main(), 0)
        config = json.loads(output.getvalue())
        self.assertIsNone(config["input_device"])
        self.assertIsNone(config["resolved_input_device"])
        audio.assert_not_called()
        session.assert_not_called()
        stage.assert_not_called()
        sd.query_devices.assert_not_called()
        sd.check_input_settings.assert_not_called()
        sd.InputStream.assert_not_called()
