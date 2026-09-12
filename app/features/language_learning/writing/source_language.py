"""Writing-only source language contract and privacy-safe Unicode measurements.

Script presence is a necessary surface check, NOT language detection. In particular,
Han does not distinguish Chinese from Japanese, and a single Hangul letter does not
make a sentence Korean. Actual language and answer leakage stay in semantic review.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

SOURCE_LANGUAGE_POLICY_VERSION = "writing-source-language-v1"

# Match the existing Writing necessary-script policy. No lookarounds / provider-
# specific regex features; the same primitive is used locally and in JSON Schema.
_SCRIPT_PATTERNS = {
    "ko": r"[\uac00-\ud7a3\u1100-\u11ff]",
    "ja": r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]",
    "en": r"[A-Za-z]",
}


def required_script_pattern(language: str) -> str | None:
    base = language.replace("_", "-").split("-", 1)[0].lower()
    return _SCRIPT_PATTERNS.get(base)


def missing_expected_script(language: str, text: str) -> bool:
    pattern = required_script_pattern(language)
    return pattern is not None and re.search(pattern, text) is None


def script_statistics(language: str, text: str) -> dict[str, Any]:
    """No source text, excerpts or codepoint samples enter diagnostics.

    Counts are of the original code points; NFC length is reported separately.
    Kana presence is diagnostic, not an automatic ban on foreign proper names.
    """
    counts = dict.fromkeys(("hangul_count", "hiragana_count", "katakana_count", "han_count",
                           "latin_count", "digit_count", "other_letter_count"), 0)
    for char in text:
        cp = ord(char)
        name = unicodedata.name(char, "")
        if 0xAC00 <= cp <= 0xD7A3 or 0x1100 <= cp <= 0x11FF or 0x3130 <= cp <= 0x318F:
            counts["hangul_count"] += 1
        elif 0x3040 <= cp <= 0x309F:
            counts["hiragana_count"] += 1
        elif 0x30A0 <= cp <= 0x30FF or 0xFF66 <= cp <= 0xFF9D:
            counts["katakana_count"] += 1
        elif name.startswith("CJK UNIFIED IDEOGRAPH") or name.startswith("CJK COMPATIBILITY IDEOGRAPH"):
            counts["han_count"] += 1
        elif name.startswith("LATIN "):
            counts["latin_count"] += 1
        elif char.isdecimal():
            counts["digit_count"] += 1
        elif char.isalpha():
            counts["other_letter_count"] += 1
    return {
        **counts,
        "characters": len(text),
        "nfc_characters": len(unicodedata.normalize("NFC", text)),
        "script_check_supported": required_script_pattern(language) is not None,
        "expected_script_present": (not missing_expected_script(language, text)
                                    if required_script_pattern(language) else None),
        "kana_present": bool(counts["hiragana_count"] + counts["katakana_count"]),
    }


def source_language_contract(origin_language: str, learning_language: str) -> dict[str, Any]:
    return {
        "policyVersion": SOURCE_LANGUAGE_POLICY_VERSION,
        "field": "originText",
        "sourceLanguage": origin_language,
        "learnerAnswerLanguage": learning_language,
        "fieldRole": "SOURCE_SENTENCE_NOT_TRANSLATED_ANSWER",
        "requiredScriptPattern": required_script_pattern(origin_language),
        "technicalNamesAllowed": True,
        "scriptPresenceIsNotLanguageProof": True,
    }
