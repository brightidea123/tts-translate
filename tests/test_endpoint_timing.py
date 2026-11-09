"""Pause observations and ASR agreement age use elapsed captured time."""
import unittest

from tts_runtime.endpoint_timing import PartialTextTiming, completed_pause_estimate


class EndpointTimingTests(unittest.TestCase):
    def test_completed_pauses_learn_once_and_brief_closures_do_not_lower_the_baseline(self):
        def observe(previous, pause):
            return completed_pause_estimate(previous, pause, base_ms=360, maximum_ms=820, alpha=0.32)
        self.assertEqual(observe(360, 0), 360)
        self.assertEqual(observe(360, 80), 360)
        self.assertAlmostEqual(observe(360, 700), 468.8)
        self.assertLess(observe(468.8, 100), 468.8)
        self.assertLessEqual(observe(800, 20000), 820)

    def test_source_time_age_is_independent_of_result_processing_delay(self):
        timing = PartialTextTiming()
        timing.update("unchanged words", 1_000_000_000, 10)
        timing.update("unchanged words", 2_250_000_000, 500)
        self.assertEqual(timing.age_ms(500), 1250)
        self.assertEqual(timing.age_ms(900, 2_500_000_000), 1500)

    def test_changed_words_reset_the_age_and_older_audio_cannot_roll_it_back(self):
        timing = PartialTextTiming()
        timing.update("five chairs", 1_000_000_000, 1)
        timing.update("six chairs", 2_000_000_000, 2)
        self.assertEqual(timing.age_ms(2), 0)
        timing.update("five chairs", 1_500_000_000, 3)
        self.assertEqual(timing.text, "six chairs")
        self.assertEqual(timing.capture_end_ns, 2_000_000_000)
        self.assertEqual(timing.age_ms(3, 3_000_000_000), 1000)

    def test_missing_provenance_uses_elapsed_monotonic_time(self):
        timing = PartialTextTiming()
        self.assertIsNone(timing.age_ms(1))
        timing.update("same words", None, 10)
        timing.update("same words", None, 11.5)
        self.assertEqual(timing.age_ms(11.5), 1500)
        timing.update("new words", None, 12)
        self.assertEqual(timing.age_ms(12), 0)

    def test_late_provenance_does_not_invent_an_origin_for_earlier_words(self):
        timing = PartialTextTiming()
        timing.update("same words", None, 10)
        timing.update("same words", 50_000_000_000, 11)
        self.assertEqual(timing.age_ms(12, 51_000_000_000), 2000)
