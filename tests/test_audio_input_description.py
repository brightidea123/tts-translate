"""Resolved capture diagnostics use supplied metadata, never hardware discovery."""
from copy import deepcopy
import json
import math
import unittest

from tts_runtime.audio_devices import describe_input_device


class AudioInputDescriptionTests(unittest.TestCase):
    def setUp(self):
        self.info = {"index": 7, "name": "LitLink Audio Bridge", "hostapi": 0,
                     "max_input_channels": 2, "default_samplerate": 48000.0}

    def describe(self, requested=None, **kwargs):
        return describe_input_device(self.info, requested, samplerate=16000, channels=1,
                                     dtype="int16", **kwargs)

    def test_default_request_records_actual_resolved_input_without_assuming_a_microphone(self):
        description = self.describe(default_index=7, hostapi_info={"name": "Core Audio"})
        self.assertIsNone(description["requested"])
        self.assertTrue(description["requested_default"])
        self.assertEqual(description["name"], "LitLink Audio Bridge")
        self.assertEqual(description["index"], 7)
        self.assertTrue(description["is_default"])
        self.assertEqual(description["hostapi_index"], 0)
        self.assertEqual(description["hostapi_name"], "Core Audio")
        self.assertEqual(description["sample_rate"], 16000)
        self.assertEqual(description["default_samplerate"], 48000)
        self.assertEqual(description["channels"], 1)
        self.assertEqual(description["max_input_channels"], 2)
        self.assertEqual(description["dtype"], "int16")

    def test_explicit_name_or_index_keeps_request_and_does_not_select_another_device(self):
        for requested in ["LitLink Audio Bridge", 7, "7"]:
            with self.subTest(requested=requested):
                description = self.describe(requested, default_index=2)
                self.assertEqual(description["requested"], requested)
                self.assertFalse(description["requested_default"])
                self.assertEqual(description["index"], 7)
                self.assertEqual(description["name"], "LitLink Audio Bridge")
                self.assertFalse(description["is_default"])

    def test_unknown_default_or_host_api_stays_unknown(self):
        description = self.describe()
        self.assertIsNone(description["default_input_index"])
        self.assertIsNone(description["is_default"])
        self.assertIsNone(description["hostapi_name"])
        self.assertTrue(description["requested_default"])

    def test_explicit_device_can_also_be_the_current_default(self):
        description = self.describe("LitLink Audio Bridge", default_index=7)
        self.assertFalse(description["requested_default"])
        self.assertTrue(description["is_default"])

    def test_description_is_a_snapshot_and_does_not_mutate_queried_metadata(self):
        original = deepcopy(self.info)
        api = {"name": "Core Audio", "devices": [0, 7]}
        saved_api = deepcopy(api)
        description = self.describe(default_index=7, hostapi_info=api)
        self.assertEqual(self.info, original)
        self.assertEqual(api, saved_api)
        self.info.update(index=2, name="Another input")
        self.assertEqual(description["index"], 7)
        self.assertEqual(description["name"], "LitLink Audio Bridge")

    def test_malformed_optional_metadata_is_json_safe_and_never_invented(self):
        info = {"index": -1, "name": [], "hostapi": math.nan,
                "max_input_channels": "invalid", "default_samplerate": math.inf}
        description = describe_input_device(info, None, samplerate=16000, channels=1,
            dtype="int16", default_index=True, hostapi_info={"name": object()})
        for key in ["index", "name", "hostapi_index", "hostapi_name", "max_input_channels",
                    "default_samplerate", "default_input_index", "is_default"]:
            self.assertIsNone(description[key], key)
        json.dumps(description, allow_nan=False)

    def test_zero_index_is_valid_and_fractional_index_is_not_truncated(self):
        self.info["index"] = 0
        self.assertTrue(self.describe(default_index=0)["is_default"])
        self.info["index"] = 2.5
        self.assertIsNone(self.describe(default_index=2)["index"])
        self.assertIsNone(self.describe(default_index=2)["is_default"])

    def test_invalid_format_metadata_remains_unknown_instead_of_claiming_a_format(self):
        description = describe_input_device({}, None, samplerate=math.nan, channels=0, dtype="")
        self.assertIsNone(description["sample_rate"])
        self.assertIsNone(description["channels"])
        self.assertIsNone(description["dtype"])
        json.dumps(description, allow_nan=False)
