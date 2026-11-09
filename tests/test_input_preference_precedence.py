"""Saved named inputs cannot override an explicit current routing choice."""
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from uuid import uuid4

from tts_runtime import local_settings


ROOT = Path(__file__).resolve().parents[1]


def load_with(arguments, preference="LitLink Audio Bridge", error=None):
    name = "input_preference_core_" + uuid4().hex
    spec = importlib.util.spec_from_file_location(name, ROOT / "tts-transcribe.py")
    core = importlib.util.module_from_spec(spec)
    sys.modules[name] = core
    with patch.object(sys, "argv", [str(spec.origin), *arguments]), \
            patch.object(local_settings, "load_input_preference", return_value=preference, side_effect=error) as read, \
            redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        spec.loader.exec_module(core)
    return core, read


class InputPreferencePrecedenceTests(unittest.TestCase):
    def test_saved_named_input_is_used_without_hardware_initialization(self):
        core, read = load_with([])
        self.assertEqual(core.args.input_device, "LitLink Audio Bridge")
        self.assertEqual(core.effective_config()["input_device"], "LitLink Audio Bridge")
        self.assertIsNone(core.effective_config()["resolved_input_device"])
        read.assert_called_once_with(ROOT / "settings.local.json")

    def test_explicit_named_selection_overrides_saved_preference(self):
        core, read = load_with(["--input-device", "Other Input"])
        self.assertEqual(core.sounddevice_input_device(), "Other Input")
        read.assert_not_called()

    def test_explicit_default_does_not_inherit_saved_litlink(self):
        core, read = load_with(["--input-device=default"])
        self.assertIsNone(core.args.input_device)
        self.assertIsNone(core.sounddevice_input_device())
        read.assert_not_called()

    def test_explicit_native_capture_does_not_inherit_sounddevice_preference(self):
        for backend in ["apple-vpio", "apple-raw"]:
            with self.subTest(backend=backend):
                core, read = load_with(["--audio-io", backend])
                self.assertIsNone(core.args.input_device)
                read.assert_not_called()

    def test_explicit_numeric_input_remains_available_without_saving_unstable_index(self):
        core, read = load_with(["--input-device=17"])
        self.assertEqual(core.sounddevice_input_device(), 17)
        read.assert_not_called()

    def test_invalid_saved_settings_stop_startup_instead_of_capturing_default(self):
        with self.assertRaises(SystemExit) as raised:
            load_with([], error=ValueError("Invalid local settings"))
        self.assertEqual(raised.exception.code, 2)
