"""Model presence checks distinguish partial shards from complete local assets."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts import doctor


class ModelShardStatusTests(unittest.TestCase):
    def test_index_requires_every_unique_shard(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder)
            (model / "config.json").write_text("{}")
            (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
                "layer.0.weight": "model-00001-of-00002.safetensors",
                "layer.0.bias": "model-00001-of-00002.safetensors",
                "layer.1.weight": "model-00002-of-00002.safetensors",
            }}))
            (model / "model-00001-of-00002.safetensors").write_bytes(b"fixture shard one")
            partial = doctor.model_status(model)
            self.assertFalse(partial["appears_complete"])
            self.assertTrue(partial["weight_index"]["valid"])
            self.assertEqual(partial["weight_index"]["missing_shards"], ["model-00002-of-00002.safetensors"])
            self.assertEqual(len(partial["weight_index"]["shard_files"]), 2)

            (model / "model-00002-of-00002.safetensors").write_bytes(b"fixture shard two")
            complete = doctor.model_status(model)
        self.assertTrue(complete["appears_complete"])
        self.assertEqual(complete["weight_index"]["missing_shards"], [])
        self.assertIn("inference not tested", complete["validation"])

    def test_invalid_index_prevents_complete_status_even_with_weight_file(self):
        invalid_indexes = ["{", "[]", "{}", '{"weight_map": {}}', '{"weight_map": {"layer": 42}}']
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder)
            (model / "config.json").write_text("{}")
            (model / "model.safetensors").write_bytes(b"fixture")
            for invalid in invalid_indexes:
                with self.subTest(index=invalid):
                    (model / "model.safetensors.index.json").write_text(invalid)
                    status = doctor.model_status(model)
                    self.assertFalse(status["appears_complete"])
                    self.assertTrue(status["weight_index"]["present"])
                    self.assertFalse(status["weight_index"]["valid"])
                    self.assertIn("error", status["weight_index"])

    def test_indexed_directory_does_not_count_as_a_shard_file(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder)
            (model / "config.json").write_text("{}")
            (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"layer": "model.safetensors"}}))
            (model / "model.safetensors").mkdir()
            status = doctor.model_status(model)
        self.assertFalse(status["appears_complete"])
        self.assertEqual(status["weight_files"], [])
        self.assertEqual(status["weight_index"]["missing_shards"], ["model.safetensors"])

    def test_unsharded_model_still_uses_config_and_weight_presence(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder)
            (model / "config.json").write_text("{}")
            (model / "weights.npz").write_bytes(b"fixture")
            status = doctor.model_status(model)
            self.assertTrue(status["appears_complete"])
            self.assertFalse(status["weight_index"]["present"])
            (model / "config.json").unlink()
            self.assertFalse(doctor.model_status(model)["appears_complete"])


if __name__ == "__main__":
    unittest.main()
