"""Bounded tooltip admission/cache/direction without models or network."""
from __future__ import annotations

import unittest

from tts_runtime.selection_requests import SelectionRequests


def config(text="שלום", selection="selected-one", source="HE", target="AR"):
    return {"text": text, "selection_id": selection, "source_language": source, "target_language": target}


def packet(row, translation="مرحباً"):
    return {"lookup_id": row["lookup_id"], "selection_id": row["selection_id"],
            "source_language": row["source_language"], "target_language": row["target_language"],
            "translation": translation}


class SelectionRequestTests(unittest.TestCase):
    def setUp(self):
        self.clock = [10.0]
        self.requests = SelectionRequests(timeout=2, clock=lambda: self.clock[0])
        self.requests.reset("generation-one")

    def test_identical_pending_phrases_coalesce_and_cache_preserves_caller_identity(self):
        first, send, normalized = self.requests.reserve(config())
        self.assertTrue(send)
        second, send, _ = self.requests.reserve(config(selection="selected-two"))
        self.assertFalse(send)
        self.assertNotEqual(first["lookup_id"], second["lookup_id"])
        self.assertTrue(self.requests.complete("generation-one", packet(first)))
        self.assertEqual(self.requests.result(second["lookup_id"])["translation"], "مرحباً")
        cached, send, _ = self.requests.reserve(config(selection="selected-three"))
        self.assertFalse(send)
        self.assertEqual(cached["status"], "complete")
        self.assertEqual(cached["selection_id"], "selected-three")
        self.assertNotIn("text", cached)
        reverse, send, _ = self.requests.reserve(config(source="AR", target="HE"))
        self.assertTrue(send)
        self.assertEqual(reverse["status"], "pending")

    def test_results_need_exact_direction_selection_and_current_generation(self):
        row, _, _ = self.requests.reserve(config())
        for override in [{"source_language": "AR"}, {"target_language": "EN"}, {"selection_id": "wrong"}]:
            self.assertFalse(self.requests.complete("generation-one", {**packet(row), **override}))
        self.assertFalse(self.requests.complete("old-generation", packet(row)))
        self.assertEqual(self.requests.result(row["lookup_id"])["status"], "pending")
        self.requests.reset("generation-two")
        self.assertIsNone(self.requests.result(row["lookup_id"]))
        self.assertFalse(self.requests.complete("generation-one", packet(row)))
        uncached, send, _ = self.requests.reserve(config())
        self.assertTrue(send)

    def test_pending_admission_and_timeout_are_bounded_and_do_not_strand_aliases(self):
        first, _, _ = self.requests.reserve(config())
        alias, _, _ = self.requests.reserve(config(selection="alias"))
        self.requests.reserve(config(text="two"))
        self.requests.reserve(config(text="three"))
        with self.assertRaisesRegex(RuntimeError, "busy"):
            self.requests.reserve(config(text="five"))
        self.clock[0] = 12.1
        self.assertEqual(self.requests.result(first["lookup_id"])["status"], "error")
        self.assertEqual(self.requests.result(alias["lookup_id"])["status"], "error")
        fresh, send, _ = self.requests.reserve(config())
        self.assertTrue(send)
        self.assertEqual(fresh["status"], "pending")
        self.assertFalse(self.requests.complete("generation-one", packet(first)))

    def test_lru_history_cache_and_pending_records_remain_bounded(self):
        requests = SelectionRequests(max_pending=1, max_records=2, max_cache=1)
        requests.reset("run")
        one, _, _ = requests.reserve(config(text="one"))
        requests.complete("run", packet(one, "واحد"))
        two, _, _ = requests.reserve(config(text="two"))
        requests.complete("run", packet(two, "اثنان"))
        three, _, _ = requests.reserve(config(text="three"))
        self.assertEqual(len(requests.records), 2)
        self.assertEqual(len(requests.cache), 1)
        self.assertIsNotNone(requests.result(three["lookup_id"]))
        self.assertIsNone(requests.result(one["lookup_id"]))

    def test_errors_are_propagated_to_coalesced_callers(self):
        row, _, _ = self.requests.reserve(config())
        alias, _, _ = self.requests.reserve(config(selection="alias"))
        self.requests.complete("generation-one", {**packet(row), "error": "Model is busy"})
        self.assertEqual(self.requests.result(alias["lookup_id"])["error"], "Model is busy")
        retry, send, _ = self.requests.reserve(config())
        self.assertTrue(send)

    def test_malformed_or_oversized_input_never_enters_the_broker(self):
        for override in [{"text": ""}, {"text": " "}, {"text": "x" * 1501}, {"text": "bad\x00"},
                         {"source_language": "XX"}, {"target_language": "HE"},
                         {"selection_id": "\n"}, {"selection_id": "x" * 129}]:
            with self.subTest(override=str(override)[:50]), self.assertRaises(ValueError):
                self.requests.reserve({**config(), **override})
        self.assertEqual(len(self.requests.records), 0)
        self.requests.reset()
        with self.assertRaisesRegex(RuntimeError, "Start a session"):
            self.requests.reserve(config())


if __name__ == "__main__":
    unittest.main()
