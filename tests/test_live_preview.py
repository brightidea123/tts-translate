"""Rolling preview alignment and retained-prefix boundaries, without inference."""
import unittest

from tts_runtime.live_preview import merge_preview, word_key


FIRST = ("session", "TR", 0, 6)
GROWN = ("session", "TR", 0, 7)
SHIFTED = ("session", "TR", 1, 7)


class LivePreviewTests(unittest.TestCase):
    def test_growing_reading_corrects_only_the_provisional_tail(self):
        result = merge_preview("we need five blue chairs", "we need six red chairs", FIRST, GROWN, "we need")
        self.assertEqual(result.text, "we need six red chairs")
        self.assertTrue(result.accepted)

    def test_transient_shorter_reading_does_not_remove_visible_words(self):
        result = merge_preview("we need six red chairs", "we need", FIRST, GROWN, "we need")
        self.assertEqual(result.text, "we need six red chairs")
        self.assertFalse(result.accepted)
        inner_slice = merge_preview("keep the earlier words and recent words visible", "earlier words and recent",
                                    FIRST, SHIFTED, "keep the earlier")
        self.assertFalse(inner_slice.accepted)
        self.assertEqual(inner_slice.text, "keep the earlier words and recent words visible")

    def test_shifted_window_keeps_earlier_text_despite_punctuation_and_casing(self):
        previous = "Our earlier plan is to travel through Berlin, then meet the team near the station."
        current = "BERLIN then meet the team near the station before taking the train to Paris."
        stable = "Our earlier plan is to travel through Berlin,"
        result = merge_preview(previous, current, FIRST, SHIFTED, stable)
        self.assertEqual(result.text, "Our earlier plan is to travel through Berlin, then meet the team near the station before taking the train to Paris.")
        self.assertTrue(result.accepted)

    def test_exact_retained_spelling_survives_an_anchor_at_its_boundary(self):
        result = merge_preview("Hello, world old tail", "hello world new tail", FIRST, SHIFTED, "Hello, world")
        self.assertEqual(result.text, "Hello, world new tail")

    def test_correction_before_frozen_prefix_waits_for_final_but_new_tail_is_live(self):
        result = merge_preview("please keep this confirmed phrase and buy five blue chairs",
                               "please drop that shaky wording and buy six blue chairs tomorrow",
                               FIRST, GROWN, "please keep this confirmed phrase")
        self.assertEqual(result.text, "please keep this confirmed phrase and buy six blue chairs tomorrow")

    def test_repeated_spoken_phrases_attach_to_the_recent_matching_suffix(self):
        result = merge_preview("one two one two", "one two three four", FIRST, SHIFTED, "one two")
        self.assertEqual(result.text, "one two one two three four")

    def test_no_anchor_holds_previous_words_without_appending_overlapping_audio(self):
        result = merge_preview("keep these preceding words visible", "an entirely different overlapping reading",
                               FIRST, SHIFTED, "keep these preceding words")
        self.assertFalse(result.accepted)
        self.assertEqual(result.text, "keep these preceding words visible")
        self.assertEqual(result.reason, "no reliable overlap anchor")

    def test_older_or_different_source_windows_cannot_change_current_preview(self):
        for span in [("session", "TR", 0, 5), ("session", "TR", 0, 8), ("other", "TR", 2, 8)]:
            with self.subTest(span=span):
                result = merge_preview("keep current words", "stale words replace current words", SHIFTED, span, "keep")
                self.assertFalse(result.accepted)
                self.assertEqual(result.text, "keep current words")

    def test_proven_disjoint_audio_can_append_without_a_text_anchor(self):
        result = merge_preview("a completed earlier span", "new words arrive", FIRST, ("session", "TR", 6, 8), "a completed")
        self.assertEqual(result.text, "a completed earlier span new words arrive")

    def test_comparison_changes_only_punctuation_and_case_not_words_or_numbers(self):
        self.assertEqual(word_key("Berlin,"), word_key("BERLIN"))
        self.assertNotEqual(word_key("five"), word_key("six"))
        self.assertNotEqual(word_key("לא"), word_key("כן"))
        self.assertNotEqual(word_key("12"), word_key("21"))
        self.assertNotEqual(word_key("12.5"), word_key("125"))
        self.assertNotEqual(word_key("-5"), word_key("5"))
        self.assertNotEqual(word_key("5%"), word_key("5"))
