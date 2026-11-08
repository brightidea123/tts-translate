"""Prompt contract checks independent of model downloads or MLX availability."""

import itertools
import unittest

from translation_prompts import (
    DEFAULT_GENERIC_SYSTEM_PROMPT,
    build_translation_messages,
    is_translategemma_model,
    normalize_translation_language,
    translategemma_messages,
)


class TranslateGemmaPromptTests(unittest.TestCase):
    def test_requested_language_pairs_keep_exact_text_and_schema(self):
        samples = {
            "ar": "  أريدُ حجز موعد غدًا.\nالساعة ١٠:٣٠  ",
            "he": "  אני רוצה לקבוע תור מחר.\nבשעה 10:30  ",
            "de": "  Ich möchte morgen einen Termin.\nUm 10:30 Uhr.  ",
            "tr": "  Yarın bir randevu istiyorum.\nSaat 10:30’da.  ",
            "fr": "  Je voudrais un rendez-vous demain.\nÀ 10 h 30.  ",
            "ru": "  Я хочу записаться на завтра.\nВ 10:30.  ",
        }
        for source, target in itertools.permutations(samples, 2):
            with self.subTest(source=source, target=target):
                self.assertEqual(
                    build_translation_messages(
                        samples[source], source.upper(), target.upper(),
                        "mlx-community/translategemma-12b-it-4bit",
                        generic_system_prompt="THIS MUST NOT ENTER THE TRANSLATION",
                        generic_user_prompt="NEITHER MUST THIS",
                    ),
                    [{"role": "user", "content": [{
                        "type": "text", "source_lang_code": source,
                        "target_lang_code": target, "text": samples[source],
                    }]}],
                )

    def test_aliases_normalize_without_restricting_other_model_languages(self):
        aliases = {
            "ARB_ARAB": "ar", "Hebrew": "he", "iw": "he", "heb_Hebr": "he",
            "DEU_LATN": "de", "tur_Latn": "tr", "fra_Latn": "fr",
            "rus_Cyrl": "ru", "SP": "es", "eng_Latn": "en",
            "ja": "ja", "pt-br": "pt_BR", "fr_CA": "fr_CA",
        }
        for alias, expected in aliases.items():
            with self.subTest(alias=alias):
                self.assertEqual(normalize_translation_language(alias), expected)
        self.assertEqual(
            translategemma_messages("こんにちは", "ja", "pt-BR")[0]["content"][0],
            {"type": "text", "source_lang_code": "ja", "target_lang_code": "pt_BR", "text": "こんにちは"},
        )

    def test_translategemma_detection_works_for_hub_ids_and_local_directories(self):
        for model in [
            "mlx-community/translategemma-27b-it-4bit",
            "GOOGLE/TranslateGemma-12b-it",
            "/Users/example/models/translate_gemma_12b",
            "translate-gemma-4b",
        ]:
            with self.subTest(model=model):
                self.assertTrue(is_translategemma_model(model))
        for model in ["mlx-community/gemma-4-e4b-it-4bit", "Meta-Llama-3.1-8B", "", "translation-gemma"]:
            with self.subTest(model=model):
                self.assertFalse(is_translategemma_model(model))

    def test_same_language_is_left_for_caller_to_bypass(self):
        message = translategemma_messages("שלום", "HE", "he")[0]["content"][0]
        self.assertEqual(message["source_lang_code"], message["target_lang_code"])
        self.assertEqual(message["text"], "שלום")

    def test_invalid_inputs_fail_before_tokenizer_or_model_loading(self):
        for text in ["", " \n\t"]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                translategemma_messages(text, "ar", "he")
        for code in ["", " ", "arab_Arab", "en_US_extra", "english please", "<|he|>"]:
            with self.subTest(code=code), self.assertRaises(ValueError):
                translategemma_messages("text", code, "en")
        with self.assertRaises(TypeError):
            translategemma_messages(None, "ar", "he")
        with self.assertRaises(TypeError):
            normalize_translation_language(None)


class GenericPromptCompatibilityTests(unittest.TestCase):
    def test_default_generic_prompt_matches_existing_pipeline(self):
        self.assertEqual(
            build_translation_messages("Guten Morgen", "de", "en", "generic/model"),
            [
                {"role": "system", "content": DEFAULT_GENERIC_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    "Translate the following German text into natural English. "
                    "Return only English translation text. Do not answer in German.\n\n"
                    "Guten Morgen"
                )},
            ],
        )

    def test_rendered_custom_prompt_and_system_are_kept_verbatim(self):
        prompt = "Use formal Hebrew:\n\n  مرحبًا  "
        self.assertEqual(
            build_translation_messages(
                "  مرحبًا  ", "ar", "he", "/models/llama",
                generic_system_prompt="Custom system\n",
                generic_user_prompt=prompt,
            ),
            [
                {"role": "system", "content": "Custom system\n"},
                {"role": "user", "content": prompt},
            ],
        )


if __name__ == "__main__":
    unittest.main()
