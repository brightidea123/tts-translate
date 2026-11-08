"""Device routing failures without opening hardware or importing model engines."""
import json
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tts_runtime.audio_devices import checked_input_device, fresh_input_devices, input_devices


def device(name, index, channels):
    return {"name": name, "index": index, "max_input_channels": channels, "default_samplerate": 48000}


class AudioDeviceTests(unittest.TestCase):
    def sounddevice(self):
        devices = [device("Speakers", 0, 0), device("LitLink Audio Bridge", 1, 2), device("Mic", 2, 1)]

        def query(requested=None, kind=None):
            if kind is None:
                return devices
            selected = 1 if requested is None else requested
            if isinstance(selected, str):
                matches = [info for info in devices if info["name"] == selected and info["max_input_channels"] > 0]
                if len(matches) != 1:
                    raise ValueError("Missing or ambiguous input device")
                return matches[0]
            info = devices[selected]
            if info["max_input_channels"] < 1:
                raise ValueError(f"Not an input device: {info['name']!r}")
            return info

        return SimpleNamespace(query_devices=query, default=SimpleNamespace(device=[1, 0]),
                               check_input_settings=Mock()), devices

    def check(self, sd, requested):
        return checked_input_device(sd, requested, samplerate=16000, channels=1, dtype="int16")

    def test_inventory_uses_names_and_filters_outputs(self):
        sd, _ = self.sounddevice()
        result = input_devices(sd)
        self.assertEqual([info["id"] for info in result], ["LitLink Audio Bridge", "Mic"])
        self.assertEqual(result[0]["index"], 1)
        self.assertTrue(result[0]["default"])

    def test_name_resolves_current_index_after_device_reordering(self):
        sd, devices = self.sounddevice()
        self.assertEqual(self.check(sd, "LitLink Audio Bridge")["index"], 1)
        devices[0], devices[1] = devices[1], devices[0]
        for index, info in enumerate(devices):
            info["index"] = index
        self.assertEqual(self.check(sd, "LitLink Audio Bridge")["index"], 0)
        sd.check_input_settings.assert_called_with(device=0, samplerate=16000, channels=1, dtype="int16")

    def test_output_only_index_fails_with_available_inputs_without_fallback(self):
        sd, _ = self.sounddevice()
        with self.assertRaisesRegex(RuntimeError, "Not an input device.*Available inputs: 1: LitLink Audio Bridge"):
            self.check(sd, 0)
        sd.check_input_settings.assert_not_called()

    def test_default_is_resolved_and_missing_selection_is_never_substituted(self):
        sd, _ = self.sounddevice()
        self.assertEqual(self.check(sd, None)["name"], "LitLink Audio Bridge")
        with self.assertRaisesRegex(RuntimeError, "Missing or ambiguous input device"):
            self.check(sd, "Disconnected LitLink")

    def test_unavailable_default_is_rejected_without_choosing_another_input(self):
        sd, _ = self.sounddevice()
        query = sd.query_devices
        def missing_default(requested=None, kind=None):
            if requested is None and kind == "input":
                raise ValueError("No default input device")
            return query(requested, kind)
        sd.query_devices = missing_default
        with self.assertRaisesRegex(RuntimeError, "system default.*No default input device"):
            self.check(sd, None)
        sd.check_input_settings.assert_not_called()

    def test_unsupported_format_has_actionable_error(self):
        sd, _ = self.sounddevice()
        sd.check_input_settings.side_effect = RuntimeError("Invalid sample rate")
        with self.assertRaisesRegex(RuntimeError, "Invalid sample rate.*--list-devices"):
            self.check(sd, 1)

    def test_duplicate_names_are_identified_and_rejected(self):
        sd, devices = self.sounddevice()
        devices.append(device("LitLink Audio Bridge", 3, 2))
        inputs = input_devices(sd)
        self.assertTrue(inputs[0]["ambiguous"])
        self.assertTrue(inputs[-1]["ambiguous"])
        with self.assertRaisesRegex(RuntimeError, "ambiguous"):
            self.check(sd, "LitLink Audio Bridge")

    def test_refresh_uses_new_interpreter_with_timeout_and_reports_errors(self):
        with patch("tts_runtime.audio_devices.subprocess.run") as run:
            run.return_value = SimpleNamespace(stdout=json.dumps({"devices": [{"id": "LitLink"}]}))
            self.assertEqual(fresh_input_devices("/venv/python"), [{"id": "LitLink"}])
            self.assertEqual(run.call_args.args[0], ["/venv/python", "-m", "tts_runtime.audio_devices"])
            self.assertEqual(run.call_args.kwargs["timeout"], 8)
            run.return_value.stdout = json.dumps({"devices": [], "error": "CoreAudio unavailable"})
            with self.assertRaisesRegex(RuntimeError, "CoreAudio unavailable"):
                fresh_input_devices("/venv/python")
            run.side_effect = subprocess.TimeoutExpired("inventory", 8)
            with self.assertRaises(subprocess.TimeoutExpired):
                fresh_input_devices("/venv/python")
