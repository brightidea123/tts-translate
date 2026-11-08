"""Actual framing families and every streamed split, without importing MLX."""
from __future__ import annotations

import unittest

from tts_runtime.text_output import clean_model_output, configure_translation_stop_tokens


class ModelOutputTests(unittest.TestCase):
    def test_repeated_gemma_end_turn_finishes_the_answer_once(self):
        text = "שלום עולם<end_of_turn>" + "<end_of_turn>" * 256 + "extra wrong answer"
        self.assertEqual(clean_model_output(text), "שלום עולם")
        self.assertEqual(clean_model_output("<bos><start_of_turn>model\nMerhaba<end_of_turn><eos>"), "Merhaba")

    def test_known_header_families_and_reasoning_channels(self):
        cases = [
            ("<|im_start|>assistant\nBonjour<|im_end|>", "Bonjour"),
            ("<|begin_of_text|><|start_header_id|>assistant<|end_header_id|>\nПривет<|eot_id|>", "Привет"),
            ("<|channel>thought private plan<channel|><|channel>final مرحباً<end_of_turn>", "مرحباً"),
            ("<|start|>assistant<|meta_sep|>analysis private plan<|end|><|start|>assistant<|meta_sep|>final<|message|>Hallo<|return|>", "Hallo"),
            ("<think>private<think>nested</think>still private</think>שלום", "שלום"),
        ]
        for raw, visible in cases:
            with self.subTest(raw=raw):
                self.assertEqual(clean_model_output(raw), visible)

    def test_unclosed_reasoning_never_becomes_visible(self):
        for raw in ["<think>private", "<thinking>private", "<|channel>thought private", "<|meta_sep|>analysis private"]:
            with self.subTest(raw=raw):
                self.assertEqual(clean_model_output(raw), "")
        self.assertEqual(clean_model_output("Public <think>private"), "Public")
        self.assertEqual(clean_model_output("Hello<think>private</think>world"), "Hello world")

    def test_every_streaming_character_prefix_hides_controls_and_private_text(self):
        raw = "<bos><start_of_turn>model\n<think>PRIVATE</think>שלום مرحباً<end_of_turn>EXTRA"
        answer = "שלום مرحباً"
        for boundary in range(len(raw) + 1):
            with self.subTest(boundary=boundary):
                snapshot = clean_model_output(raw[:boundary], final=False)
                self.assertNotIn("<", snapshot)
                self.assertNotIn("PRIVATE", snapshot)
                self.assertNotIn("EXTRA", snapshot)
                self.assertTrue(answer.startswith(snapshot), snapshot)
        for marker in ["<|im_start|>assistant\n", "<|start_header_id|>assistant<|end_header_id|>\n", "<|channel>final "]:
            for boundary in range(len(marker) + 1):
                self.assertEqual(clean_model_output(marker[:boundary], final=False), "")

    def test_visible_ordinary_markup_and_comparisons_are_preserved(self):
        text = 'שלום <b>עולם</b> — x < 5, y > 2; <custom-name>مرحبا</custom-name> &lt;test&gt;'
        self.assertEqual(clean_model_output(text), text)
        self.assertEqual(clean_model_output("Calculate x <", final=True), "Calculate x <")
        self.assertEqual(clean_model_output("Calculate x <", final=False), "Calculate x")
        self.assertEqual(clean_model_output("assistant: The word model is ordinary prose."), "assistant: The word model is ordinary prose.")

    def test_orphaned_control_prefixes_are_not_final_speech(self):
        for partial in ["<end_of_t", "<|im_en", "<|eot_", "</thi", "<think lang=x"]:
            self.assertEqual(clean_model_output("Hallo " + partial), "Hallo")

    def test_whisper_language_task_and_timestamp_controls_are_removed(self):
        self.assertEqual(clean_model_output("האזרח הזה<|lt|>"), "האזרח הזה")
        self.assertEqual(clean_model_output("<|startoftranscript|><|he|><|transcribe|><|0.00|>שלום<|1.24|><|endoftext|>extra"), "שלום")
        self.assertEqual(clean_model_output("<|ar|>مرحبا <b>بكم</b>"), "مرحبا <b>بكم</b>")
        self.assertEqual(clean_model_output("<|yue|>你好"), "你好")
        for fragment in ["<|l", "<|lt", "<|0.", "<|1.24|"]:
            self.assertEqual(clean_model_output("שלום " + fragment, final=False), "שלום")


class FakeTokenizer:
    eos_token_id = 1
    unk_token_id = 0

    def __init__(self):
        self.eos_token_ids = {1}
        self.added = []

    def convert_tokens_to_ids(self, token):
        return 106 if token == "<end_of_turn>" else 0

    def convert_ids_to_tokens(self, token):
        return "<end_of_turn>" if token == 106 else "<unk>"

    def encode(self, text, add_special_tokens=False):
        return [self.convert_tokens_to_ids(text)]

    def add_eos_token(self, token):
        self.added.append(token)
        self.eos_token_ids.add(self.convert_tokens_to_ids(token))


class TranslationStopTests(unittest.TestCase):
    def test_gemma_registers_actual_turn_end_in_wrapper_stop_set(self):
        tokenizer = FakeTokenizer()
        self.assertEqual(configure_translation_stop_tokens(tokenizer, "models/translategemma-12b-it-4bit"), frozenset({1, 106}))
        self.assertEqual(tokenizer.eos_token_ids, {1, 106})
        self.assertEqual(tokenizer.added, ["<end_of_turn>"])

    def test_non_gemma_and_invalid_unknown_ids_do_not_mutate_stop_configuration(self):
        tokenizer = FakeTokenizer()
        self.assertEqual(configure_translation_stop_tokens(tokenizer, "another-model"), frozenset({1}))
        self.assertEqual(tokenizer.added, [])
        tokenizer.convert_tokens_to_ids = lambda token: 0
        with self.assertRaisesRegex(ValueError, "valid.*stop token"):
            configure_translation_stop_tokens(tokenizer, "gemma")
        self.assertEqual(tokenizer.eos_token_ids, {1})

    def test_multi_token_or_mismatched_marker_is_rejected(self):
        tokenizer = FakeTokenizer()
        tokenizer.encode = lambda *args, **kwargs: [106, 42]
        with self.assertRaisesRegex(ValueError, "round-trip"):
            configure_translation_stop_tokens(tokenizer, "gemma")
        self.assertEqual(tokenizer.added, [])


if __name__ == "__main__":
    unittest.main()
