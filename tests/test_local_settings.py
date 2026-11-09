"""Stable local input preferences without hardware or model initialization."""
from __future__ import annotations

import builtins
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tts_runtime.local_settings import load_input_preference


class LocalSettingsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "settings.local.json"

    def write(self, value):
        self.path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def test_absent_file_has_no_remembered_override(self):
        self.assertIsNone(load_input_preference(self.path))

    def test_named_preference_preserves_unicode_and_internal_spaces(self):
        for name in ["LitLink Audio Bridge", "  קלט שמע  ", "Input: Room 2"]:
            with self.subTest(name=name):
                self.write({"input_device": name})
                self.assertEqual(load_input_preference(self.path), name.strip())

    def test_null_or_empty_object_selects_system_default(self):
        for settings in [{"input_device": None}, {}]:
            self.write(settings)
            self.assertIsNone(load_input_preference(self.path))

    def test_malformed_json_and_non_objects_do_not_silently_fall_back(self):
        self.path.write_text("{broken", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Invalid JSON.*settings.local.json"):
            load_input_preference(self.path)
        for settings in [None, [], "LitLink", True, 7]:
            with self.subTest(settings=settings):
                self.write(settings)
                with self.assertRaisesRegex(ValueError, "must be an object"):
                    load_input_preference(self.path)

    def test_unknown_keys_are_explicit_errors(self):
        self.write({"input_device": "LitLink", "input_devce": "other"})
        with self.assertRaisesRegex(ValueError, "only input_device"):
            load_input_preference(self.path)

    def test_invalid_device_values_or_control_characters_are_rejected(self):
        for value in [True, False, 17, [], {}, "", "   ", "x" * 257,
                      "LitLink\x00", "LitLink\n", "\tLitLink", "LitLink\x7f", "LitLink\x85"]:
            with self.subTest(value=value):
                self.write({"input_device": value})
                with self.assertRaisesRegex(ValueError, "input_device must be null or"):
                    load_input_preference(self.path)

    def test_unstable_numeric_indices_cannot_be_saved_as_preferences(self):
        for value in ["17", "-1", "+2", " 0 ", "１２"]:
            with self.subTest(value=value):
                self.write({"input_device": value})
                with self.assertRaisesRegex(ValueError, "name, not a numeric index"):
                    load_input_preference(self.path)

    def test_unreadable_file_and_invalid_encoding_fail_explicitly(self):
        self.path.mkdir()
        with self.assertRaisesRegex(ValueError, "Cannot read local settings"):
            load_input_preference(self.path)
        self.path.rmdir()
        self.path.write_bytes(b"\xff")
        with self.assertRaisesRegex(ValueError, "Cannot read local settings"):
            load_input_preference(self.path)

    def test_reading_preferences_does_not_import_audio_or_models(self):
        self.write({"input_device": "LitLink Audio Bridge"})
        original_import = builtins.__import__
        def checked_import(name, *args, **kwargs):
            if name.split(".", 1)[0] in {"sounddevice", "torch", "mlx", "mlx_whisper", "transformers"}:
                self.fail(f"Preference loading imported {name}")
            return original_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=checked_import):
            self.assertEqual(load_input_preference(self.path), "LitLink Audio Bridge")
