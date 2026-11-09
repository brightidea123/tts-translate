"""Language and text agreement gates without model inference."""
from __future__ import annotations

import unittest

from tts_runtime.asr_verification import assess_verification, equivalent_text, lexical_words


def summary(text="In Dearborn Michigan on election day just now", language="en", **overrides):
    return {"raw_text": text, "detected_language": language, "language_auto_detected": True,
            "mean_avg_logprob": -.25, "min_avg_logprob": -.35,
            "max_no_speech_prob": .15, "max_compression_ratio": 1.1, **overrides}


class VerificationTests(unittest.TestCase):
    def test_foreign_script_primary_requires_fresh_automatic_language_agreement(self):
        candidate = "In Dearborn, Michigan, on election day, just now."
        result = assess_verification(candidate, summary())
        self.assertTrue(result.verified, result.reason)
        self.assertEqual(result.language, "EN")
        self.assertIn("Dearborn", result.text)
        self.assertFalse(assess_verification(candidate, summary(language_auto_detected=False)).verified)
        self.assertFalse(assess_verification(candidate, summary(language="he")).verified)

    def test_legitimate_short_hebrew_thanks_and_responses_are_allowed(self):
        for text in ["תודה רבה.", "תודה", "כן", "לא"]:
            with self.subTest(text=text):
                result = assess_verification(text, summary(text, "he"))
                self.assertTrue(result.verified, result.reason)

    def test_supported_language_scripts_pass_and_wrong_scripts_do_not(self):
        examples = {"AR": "شكرا جزيلا لكم", "HE": "שלום חברים", "DE": "Vielen Dank", "TR": "Çok teşekkür ederim",
                    "FR": "Merci beaucoup", "RU": "Спасибо большое", "EN": "Thank you", "ES": "Muchas gracias"}
        for language, text in examples.items():
            with self.subTest(language=language):
                self.assertTrue(assess_verification(text, summary(text, language)).verified)
        for language, text in [("AR", "hello everyone"), ("HE", "hello everyone"), ("RU", "hello everyone"), ("EN", "שלום חברים")]:
            with self.subTest(language=language, text=text):
                self.assertFalse(assess_verification(text, summary(text, language)).verified)

    def test_poor_decoder_quality_and_invalid_metrics_are_refused(self):
        for changes in [{"mean_avg_logprob": -.81}, {"max_no_speech_prob": .51}, {"max_compression_ratio": 2.41},
                        {"max_no_speech_prob": -1}, {"max_compression_ratio": 0}, {"mean_avg_logprob": float("nan")},
                        {"min_avg_logprob": float("inf")}, {"max_no_speech_prob": None}, {"mean_avg_logprob": True}]:
            with self.subTest(changes=changes):
                result = assess_verification(summary()["raw_text"], summary(**changes))
                self.assertFalse(result.verified)
                self.assertEqual(result.text, "")
                self.assertTrue(result.reason)

    def test_missing_metrics_and_unsupported_or_invented_language_are_refused(self):
        for field in ["mean_avg_logprob", "min_avg_logprob", "max_no_speech_prob", "max_compression_ratio"]:
            incomplete = summary()
            incomplete.pop(field)
            self.assertFalse(assess_verification(incomplete["raw_text"], incomplete).verified)
        for language in ["ja", "", None, "english"]:
            self.assertFalse(assess_verification(summary()["raw_text"], summary(language=language)).verified)
        self.assertFalse(assess_verification("words", None).verified)
        self.assertFalse(assess_verification(None, summary("None", "en")).verified)
        self.assertFalse(assess_verification("words", summary("", "en")).verified)
        self.assertFalse(assess_verification("words", summary("abc", "en")).verified)

    def test_changed_or_omitted_numbers_override_otherwise_high_agreement(self):
        candidate = "Please send the package to number 42 tomorrow morning"
        for text in [candidate.replace("42", "43"), candidate.replace("42", ""), candidate + " 7"]:
            with self.subTest(text=text):
                result = assess_verification(candidate, summary(text))
                self.assertFalse(result.verified)
                self.assertIn("numbers", result.reason)
        self.assertTrue(assess_verification(candidate, summary(candidate.replace("42", "４２"))).verified)

    def test_agreement_tolerates_minor_corrections_but_not_collapse_or_new_unrelated_text(self):
        candidate = "In Dearborn Michigan on election day just now an election worker speaks"
        self.assertTrue(assess_verification(candidate, summary(candidate.replace("speaks", "talks"))).verified)
        for text in ["Thank you very much", "In Dearborn", "We need six blue chairs for the afternoon meeting"]:
            with self.subTest(text=text):
                self.assertFalse(assess_verification(candidate, summary(text)).verified)

    def test_second_forced_confirmation_uses_prior_language_and_agreement(self):
        text = "The meeting begins tomorrow morning at nine"
        forced = summary(text, "en", language_auto_detected=False)
        self.assertTrue(assess_verification(text, forced, require_auto=False, expected_language="EN").verified)
        self.assertFalse(assess_verification(text, forced, require_auto=False, expected_language="he").verified)
        other = "The package arrives tomorrow morning at nine"
        self.assertFalse(assess_verification(other, forced, require_auto=False, expected_language="en").verified)

    def test_first_automatic_alternative_can_be_assessed_without_copying_primary_words(self):
        text = "The meeting begins tomorrow morning at nine"
        result = assess_verification("תודה רבה", summary(text), require_agreement=False)
        self.assertTrue(result.verified, result.reason)
        self.assertEqual(result.text, text)
        self.assertFalse(assess_verification("תודה רבה", summary(text, max_no_speech_prob=.9), require_agreement=False).verified)

    def test_unicode_normalization_and_punctuation_preserve_lexical_content(self):
        self.assertTrue(equivalent_text(" HELLO, world! ４２ ", "hello world 42"))
        self.assertTrue(equivalent_text("number ٤٢", "number 42"))
        self.assertTrue(equivalent_text("שָׁלוֹם עולם", "שלום עולם"))
        self.assertFalse(equivalent_text("number 42", "number 43"))
        self.assertFalse(equivalent_text("", ""))
        self.assertFalse(equivalent_text(None, "None"))
        self.assertEqual(lexical_words("hello\u200f world"), ("hello", "world"))


if __name__ == "__main__":
    unittest.main()
