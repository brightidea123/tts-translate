"""Build model-specific translation messages without loading an ML runtime.

TranslateGemma's official template takes one user content item containing the
unmodified source text and source/target ISO language codes. Generic instruct
models retain the project's existing system/user prompt structure.

Schema: https://huggingface.co/google/translategemma-12b-it#usage
"""

from __future__ import annotations

import re
from typing import Any


DEFAULT_GENERIC_SYSTEM_PROMPT = (
    "You are a strict real-time translator. Output only the requested target-language "
    "translation with no commentary, no quotes, and no extra notes."
)

_LANGUAGE_NAMES = {
    "ar": "Arabic",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "he": "Hebrew",
    "ru": "Russian",
    "tr": "Turkish",
}

# These are aliases used by the recovered application, not an allowlist of
# TranslateGemma's supported languages. Other ISO alpha-2 codes pass through.
_LANGUAGE_ALIASES = {
    "arabic": "ar", "arb": "ar", "arb_arab": "ar",
    "german": "de", "deu": "de", "deu_latn": "de",
    "english": "en", "eng": "en", "eng_latn": "en",
    "spanish": "es", "sp": "es", "spa": "es", "spa_latn": "es",
    "french": "fr", "fra": "fr", "fra_latn": "fr",
    "hebrew": "he", "iw": "he", "heb": "he", "heb_hebr": "he",
    "russian": "ru", "rus": "ru", "rus_cyrl": "ru",
    "turkish": "tr", "tur": "tr", "tur_latn": "tr",
}


def normalize_translation_language(code: str) -> str:
    """Normalize application aliases to ISO alpha-2, optionally with a region.

    Accepted examples include ``AR``, ``Hebrew``, ``tur_Latn``, ``fr-CA`` and
    ``pt_BR``. This validates the code's shape, not the model's language coverage;
    the model's own chat template remains authoritative for supported codes.
    Same-language source and target values are intentionally allowed so the
    caller can decide whether to bypass inference for identity translations.
    """
    if not isinstance(code, str):
        raise TypeError("language code must be a string")
    value = code.strip()
    if not value:
        raise ValueError("language code must not be empty")
    alias = _LANGUAGE_ALIASES.get(value.casefold())
    if alias is not None:
        return alias
    if re.fullmatch(r"[A-Za-z]{2}", value):
        return value.lower()
    regional = re.fullmatch(r"([A-Za-z]{2})[-_]([A-Za-z]{2})", value)
    if regional:
        language, region = regional.groups()
        language = _LANGUAGE_ALIASES.get(language.casefold(), language.lower())
        return f"{language}_{region.upper()}"
    raise ValueError(f"expected an ISO alpha-2 language code or known alias, got {code!r}")


def is_translategemma_model(model_ref: str) -> bool:
    """Recognize TranslateGemma Hub IDs or local paths, including word separators."""
    if not isinstance(model_ref, str):
        raise TypeError("model reference must be a string")
    return re.search(r"translate[-_\s]*gemma", model_ref, re.IGNORECASE) is not None


def translategemma_messages(
    text: str, source_code: str, target_code: str
) -> list[dict[str, Any]]:
    """Return the officially supported text-only TranslateGemma message schema.

    Preserve whitespace, newlines and Unicode exactly: normalization belongs to
    the application's transcription pipeline, rather than its prompt formatter.
    """
    if not isinstance(text, str):
        raise TypeError("translation text must be a string")
    if not text.strip():
        raise ValueError("translation text must not be empty")
    return [{
        "role": "user",
        "content": [{
            "type": "text",
            "source_lang_code": normalize_translation_language(source_code),
            "target_lang_code": normalize_translation_language(target_code),
            "text": text,
        }],
    }]


def _language_name(code: str) -> str:
    base = code.split("_", 1)[0]
    name = _LANGUAGE_NAMES.get(base, base)
    return f"{name} ({code})" if "_" in code else name


def build_translation_messages(
    text: str,
    source_code: str,
    target_code: str,
    model_ref: str,
    generic_system_prompt: str | None = None,
    *,
    generic_user_prompt: str | None = None,
) -> list[dict[str, Any]]:
    """Select TranslateGemma's schema or the existing generic instruct schema.

    Pass the application's already-rendered ``generic_user_prompt`` to preserve
    its custom ``--mt-prompt`` behavior verbatim for generic models. TranslateGemma
    deliberately receives only the original text, without generic instructions
    or a system role, because its supported template carries language metadata.
    """
    if is_translategemma_model(model_ref):
        return translategemma_messages(text, source_code, target_code)
    if not isinstance(text, str):
        raise TypeError("translation text must be a string")
    if not text.strip():
        raise ValueError("translation text must not be empty")
    if generic_user_prompt is None:
        source_name = _language_name(normalize_translation_language(source_code))
        target_name = _language_name(normalize_translation_language(target_code))
        generic_user_prompt = (
            f"Translate the following {source_name} text into natural {target_name}. "
            f"Return only {target_name} translation text. Do not answer in {source_name}.\n\n"
            f"{text}"
        )
    elif not isinstance(generic_user_prompt, str):
        raise TypeError("generic user prompt must be a string")
    if generic_system_prompt is None:
        generic_system_prompt = DEFAULT_GENERIC_SYSTEM_PROMPT
    elif not isinstance(generic_system_prompt, str):
        raise TypeError("generic system prompt must be a string")
    return [
        {"role": "system", "content": generic_system_prompt},
        {"role": "user", "content": generic_user_prompt},
    ]
