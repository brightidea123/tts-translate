"""Strong decoder loops trigger acoustic review without deleting real speech."""
from __future__ import annotations

import unittest

from tts_runtime.asr_consistency import decoder_repetition_loop


class DecoderRepetitionTests(unittest.TestCase):
    def test_single_word_loop_requires_substantial_contiguous_repetition(self):
        self.assertTrue(decoder_repetition_loop("ah, " * 74 + "ending", 23.47))
        self.assertTrue(decoder_repetition_loop("no " * 6, 3.5))
        self.assertFalse(decoder_repetition_loop("no no no please stop", 23.47))
        self.assertFalse(decoder_repetition_loop("well " * 5, 3.5))

    def test_multiword_sentence_loops_survive_punctuation_and_direction_mark_variations(self):
        text = "\u202bPlease, repeat this sentence!\u202c " * 6
        self.assertTrue(decoder_repetition_loop(text, 5.2))
        self.assertTrue(decoder_repetition_loop("\u202bאנחנו צריכים לקבוע פגישה.\u202c " * 5, 5.2))
        self.assertTrue(decoder_repetition_loop("نُرِيدُ مَوْعِدًا جَدِيدًا، " * 4, 5.2))

    def test_a_changed_prefix_does_not_hide_a_dominant_periodic_core(self):
        text = "and need blue chairs, " + "we need blue chairs, " * 4 + "they need blue chairs, " * 4
        self.assertTrue(decoder_repetition_loop(text, 5.2))

    def test_two_sentence_copies_and_meaningful_emphasis_are_not_decoder_loops(self):
        for text in ["Please send the package today. " * 2,
                     "no no no please stop", "very very very good", "I said yes, yes, yes."]:
            with self.subTest(text=text):
                self.assertFalse(decoder_repetition_loop(text, 8.5))

    def test_changed_numbers_and_negation_are_not_equated(self):
        for text in [
            "Send 5 blue chairs. Send 6 blue chairs. Send 7 blue chairs.",
            "We should send it. We should not send it. We must send it.",
            "The appointment is on Monday. The appointment is on Tuesday. The appointment is on Wednesday.",
        ]:
            with self.subTest(text=text):
                self.assertFalse(decoder_repetition_loop(text, 9.0))

    def test_decoder_compression_is_required_and_must_be_finite(self):
        text = "Please repeat this sentence. " * 10
        for compression in [None, False, True, "invalid", float("nan"), float("inf"), -1, 0, 2.4, 2.9]:
            with self.subTest(compression=compression):
                self.assertFalse(decoder_repetition_loop(text, compression))
        self.assertTrue(decoder_repetition_loop(text, 2.9001))

    def test_high_compression_without_a_tandem_loop_is_insufficient(self):
        text = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar"
        self.assertFalse(decoder_repetition_loop(text, 15.0))

    def test_long_sentence_cycle_is_detected_without_requiring_short_ngrams(self):
        sentence = " ".join(f"word{number}" for number in range(40))
        self.assertTrue(decoder_repetition_loop((sentence + ". ") * 3, 5.0))

    def test_small_repeated_fragment_does_not_condemn_a_long_transcript(self):
        text = " ".join(f"word{number}" for number in range(50)) + " " + "please wait here " * 3
        self.assertFalse(decoder_repetition_loop(text, 4.0))
        # A large contiguous core can be significant even with a longer intro.
        text = " ".join(f"word{number}" for number in range(40)) + " " + "please wait here " * 8
        self.assertTrue(decoder_repetition_loop(text, 4.0))
