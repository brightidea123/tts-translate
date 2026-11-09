"""Recovery uses the same captured samples and assigns overlap by timestamps."""
from dataclasses import replace
import math
import unittest

from tts_runtime.asr_recovery import RecoveryTranscript, recovery_windows, stitch_recovery


def summary(*words):
    text = " ".join(word[0].strip() for word in words).strip()
    return {
        "raw_text": text,
        "segments": [{
            "text": text,
            "words": [{"word": token, "start": start, "end": end} for token, start, end in words],
        }],
    }


class AsrRecoveryTests(unittest.TestCase):
    def test_cores_partition_exact_capture_without_gaps_or_padding(self):
        for count in [1, 128000, 128001, 394240, 448000, 512000]:
            with self.subTest(count=count):
                windows = recovery_windows(count)
                self.assertLessEqual(len(windows), 4)
                self.assertEqual(windows[0].core_start_sample, 0)
                self.assertEqual(windows[-1].core_end_sample, count)
                previous_end = 0
                for window in windows:
                    self.assertEqual(window.core_start_sample, previous_end)
                    self.assertLessEqual(window.core_end_sample - window.core_start_sample, 128000)
                    self.assertGreaterEqual(window.start_sample, 0)
                    self.assertLessEqual(window.end_sample, count)
                    self.assertLessEqual(window.core_start_sample - window.start_sample, 8000)
                    self.assertLessEqual(window.end_sample - window.core_end_sample, 8000)
                    previous_end = window.core_end_sample

    def test_unbounded_or_invalid_capture_requests_are_refused(self):
        for count, rate in [(512001, 16000), (0, 16000), (-1, 16000), (1.5, 16000),
                            (True, 16000), (10, 0), (10, -1), (10, 1.5)]:
            with self.subTest(count=count, rate=rate), self.assertRaises(ValueError):
                recovery_windows(count, rate)

    def test_tiny_tail_is_balanced_across_cores_instead_of_becoming_a_silent_last_core(self):
        for count in [128001, 256001, 384001, 394240]:
            with self.subTest(count=count):
                windows = recovery_windows(count)
                lengths = [window.core_end_sample - window.core_start_sample for window in windows]
                self.assertGreaterEqual(min(lengths), 4 * 16000)
                self.assertLessEqual(max(lengths) - min(lengths), 1)
                self.assertEqual(sum(lengths), count)
        self.assertEqual([window.core_end_sample - window.core_start_sample
                          for window in recovery_windows(394240)], [98560] * 4)

    def test_overlap_is_owned_by_core_without_text_deduping_repeated_words(self):
        windows = recovery_windows(16 * 16000)
        # Window 2 begins at 7.5 s. Both decodes see the same boundary words;
        # their absolute midpoints are 7.8 and 8.2 s, on opposite sides.
        decoded = [
            summary((" no", 7.6, 8.0), (" no", 8.0, 8.4)),
            summary((" no", 0.1, 0.5), (" no", 0.5, 0.9), (" 42", 1.0, 1.3)),
        ]
        result = stitch_recovery(windows, decoded)
        self.assertTrue(result.valid, result.error)
        self.assertEqual(result.text, "no no 42")

    def test_word_exactly_on_boundary_has_one_owner(self):
        windows = recovery_windows(16 * 16000)
        result = stitch_recovery(windows, [
            summary(("first", 7.0, 7.2), (" boundary", 7.9, 8.1)),
            summary((" boundary", 0.4, 0.6), (" end", 7.5, 7.7)),
        ])
        self.assertTrue(result.valid, result.error)
        self.assertEqual(result.text, "first boundary end")

    def test_rtl_numbers_and_meaningful_repetition_survive(self):
        windows = recovery_windows(4 * 16000)
        result = stitch_recovery(windows, [summary(
            ("לא", 0.1, 0.3), ("לא", 0.4, 0.6), ("42", 0.7, 1.0),
            ("نعم", 1.2, 1.5), ("نعم", 1.6, 1.9), ("!", 2.0, 2.0),
        )])
        self.assertTrue(result.valid, result.error)
        self.assertEqual(result.text, "לא לא 42 نعم نعم!")

    def test_existing_word_spacing_and_punctuation_are_preserved(self):
        result = stitch_recovery(recovery_windows(4 * 16000), [summary(
            (" Hello,", 0.0, 0.5), ("  world", 0.6, 1.0), ("!", 1.0, 1.0),
        )])
        self.assertEqual(result.text, "Hello,  world!")

    def test_missing_timestamps_or_any_empty_window_invalidates_whole_candidate(self):
        windows = recovery_windows(16 * 16000)
        good = summary((" early words", 1.0, 2.0))
        for bad in [None, {"raw_text": ""}, {"raw_text": "words", "segments": []},
                    {"raw_text": "words", "segments": [{"text": "words"}]},
                    {"raw_text": "words", "segments": [{"text": "words", "words": []}]},
                    {"raw_text": "words", "segments": [{"text": None, "words": []}]},
                    {**summary(("word", 1, 2)), "error": "decoder failed"}]:
            with self.subTest(bad=bad):
                result = stitch_recovery(windows, [good, bad])
                self.assertFalse(result.valid)
                self.assertEqual(result.text, "")
                self.assertTrue(result.error)

    def test_nonfinite_reversed_or_out_of_capture_word_times_are_rejected(self):
        windows = recovery_windows(4 * 16000)
        for start, end in [(math.nan, 1), (0, math.inf), (None, 1), (2, 1),
                           (-0.1, 1), (1, 4.1), (True, 2)]:
            with self.subTest(start=start, end=end):
                result = stitch_recovery(windows, [summary(("word", start, end))])
                self.assertFalse(result.valid)
                self.assertEqual(result.text, "")

    def test_regressing_word_times_are_not_reordered_into_a_transcript(self):
        result = stitch_recovery(recovery_windows(4 * 16000), [summary(
            (" first", 1.0, 1.4), (" older", 0.2, 0.8),
        )])
        self.assertFalse(result.valid)
        self.assertIn("unordered", result.error)

    def test_incomplete_or_reordered_word_alignment_invalidates_the_whole_candidate(self):
        windows = recovery_windows(16 * 16000)
        for raw_text, words in [
            ("first second", [(" first", 1, 2)]),
            ("first second", [(" second", 1, 2), (" first", 2, 3)]),
            ("first 43", [(" first", 1, 2), (" 42", 2, 3)]),
        ]:
            with self.subTest(raw_text=raw_text, words=words):
                incomplete = summary(*words)
                incomplete["raw_text"] = raw_text
                incomplete["segments"][0]["text"] = raw_text
                result = stitch_recovery(windows, [summary(("complete", 1, 2)), incomplete])
                self.assertFalse(result.valid)
                self.assertEqual(result.text, "")
                self.assertIn("alignment", result.error)

    def test_alignment_accepts_unicode_case_spacing_and_punctuation_variants(self):
        decoded = summary((" hello,", 0, 0.5), (" cafe\u0301", 0.6, 1), (" 42", 1.2, 1.5), ("!", 1.5, 1.5))
        decoded["raw_text"] = "HELLO café ４２!"
        result = stitch_recovery(recovery_windows(4 * 16000), [decoded])
        self.assertTrue(result.valid, result.error)
        self.assertEqual(result.text, "hello, cafe\u0301 42!")

    def test_small_edge_rounding_is_clamped_to_capture_and_end_punctuation_survives(self):
        result = stitch_recovery(recovery_windows(4 * 16000), [summary(
            (" word", -0.02, 0.4), ("!", 4.0, 4.02),
        )])
        self.assertTrue(result.valid, result.error)
        self.assertEqual(result.text, "word!")

    def test_an_overlap_only_decode_cannot_claim_its_whole_core_was_recovered(self):
        result = stitch_recovery(recovery_windows(16 * 16000), [
            summary(("before", 1, 2)), summary(("only left context", 0.1, 0.2)),
        ])
        self.assertFalse(result.valid)
        self.assertIn("owned", result.error)
        self.assertEqual(result.text, "")

    def test_missing_or_changed_core_cannot_claim_contiguous_recovery(self):
        windows = recovery_windows(16 * 16000)
        decoded = [summary(("one", 1, 2)), summary(("two", 1, 2))]
        for plan in [(windows[0], replace(windows[1], core_start_sample=128001)),
                     (windows[0], replace(windows[1], start_sample=0))]:
            with self.subTest(plan=plan):
                result = stitch_recovery(plan, decoded)
                self.assertFalse(result.valid)
        self.assertFalse(stitch_recovery(windows, decoded[:1]).valid)
        self.assertFalse(RecoveryTranscript().valid)


if __name__ == "__main__":
    unittest.main()
