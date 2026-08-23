from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from app.features.language_learning.listening.policy import (
    LISTENING_NORMALIZATION_VERSION,
)

_PUNCTUATION_RE = re.compile(r"[^\w\s'’-]", flags=re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")
_LATIN_TOKEN_RE = re.compile(r"[\w]+(?:'[\w]+)?", flags=re.UNICODE)
_JAPANESE_TOKEN_RE = re.compile(
    r"[一-龯々〆ヵヶ]|[ぁ-んァ-ンー]|[A-Za-z]+|[0-9]+",
    flags=re.UNICODE,
)

_ENGLISH_CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "isn't": "is not",
    "aren't": "are not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "i'm": "i am",
    "it's": "it is",
    "that's": "that is",
}


@dataclass(frozen=True)
class NormalizedText:
    text: str
    tokens: list[str]
    version: str = LISTENING_NORMALIZATION_VERSION


@dataclass(frozen=True)
class CanonicalizedAnswer:
    text: str
    accepted_tokens: set[str]


def normalize_text(text: str, locale: str) -> NormalizedText:
    language = locale.lower().split("-")[0]
    value = unicodedata.normalize("NFKC", text).casefold()
    value = value.replace("’", "'").replace("‐", "-").replace("–", "-")
    value = _PUNCTUATION_RE.sub(" ", value)
    value = _WHITESPACE_RE.sub(" ", value).strip()

    if language == "en":
        for contraction, expanded in _ENGLISH_CONTRACTIONS.items():
            value = re.sub(rf"\b{re.escape(contraction)}\b", expanded, value)
        value = _WHITESPACE_RE.sub(" ", value).strip()

    tokens = _tokenize(value, language)
    canonical = "".join(tokens) if language in {"ja", "zh"} else " ".join(tokens)
    return NormalizedText(text=canonical, tokens=tokens)


def canonicalize_accepted_variants(
    answer: NormalizedText,
    accepted_variants: dict[str, list[str]],
    locale: str,
) -> CanonicalizedAnswer:
    value = answer.text
    accepted_tokens: set[str] = set()
    replacements: list[tuple[str, str]] = []
    for source, variants in accepted_variants.items():
        canonical = normalize_text(source, locale)
        if not canonical.text:
            continue
        for variant in variants:
            normalized_variant = normalize_text(variant, locale).text
            if normalized_variant:
                replacements.append((normalized_variant, canonical.text))
                accepted_tokens.update(canonical.tokens)

    for variant, canonical in sorted(
        replacements, key=lambda item: len(item[0]), reverse=True
    ):
        value = value.replace(variant, canonical)
    return CanonicalizedAnswer(text=value, accepted_tokens=accepted_tokens)


def tokenize_normalized(text: str, locale: str) -> list[str]:
    return _tokenize(text, locale.lower().split("-")[0])


def similarity_key(text: str, locale: str) -> str:
    return "".join(normalize_text(text, locale).tokens)


def _tokenize(value: str, language: str) -> list[str]:
    if language in {"ja", "zh"}:
        return _JAPANESE_TOKEN_RE.findall(value)
    return _LATIN_TOKEN_RE.findall(value)
