from copy import deepcopy

from app.features.language_learning.level_test.normalizer import LevelTestGenerationNormalizer
from app.schemas.language_learning_level_test import LevelTestItemType


_ALL_ITEM_TYPES = [item.value for item in LevelTestItemType]


def _candidate(*, item_type: str, instruction: str, instruction_language: str) -> dict:
    return {
        "domain": "VOCABULARY",
        "itemType": item_type,
        "instruction": instruction,
        "instructionLanguage": instruction_language,
        "promptText": "これは学習者向けの問題本文です。",
        "options": [
            {"key": "A", "text": "甲"},
            {"key": "B", "text": "乙"},
            {"key": "C", "text": "丙"},
            {"key": "D", "text": "丁"},
        ],
        "internalAnswerKey": {"correctOptionKey": "A", "correctOrder": []},
        "referencePayload": {"referenceMeanings": ["meaning"]},
    }


def test_every_level_test_item_has_ko_ja_en_instruction_template():
    for item_type in _ALL_ITEM_TYPES:
        for language in ("ko", "ja", "en"):
            assert LevelTestGenerationNormalizer.instruction_template(
                item_type,
                language,
            )


def test_normalizer_repairs_instruction_text_and_language_together():
    candidate = _candidate(
        item_type="VOCAB_CONTEXT_CHOICE",
        instruction="文脈に最も適切な表現を選んでください。",
        instruction_language="ja",
    )
    semantic_before = {
        "promptText": candidate["promptText"],
        "options": deepcopy(candidate["options"]),
        "internalAnswerKey": deepcopy(candidate["internalAnswerKey"]),
        "referenceMeanings": deepcopy(candidate["referencePayload"]["referenceMeanings"]),
    }

    normalized, stats = LevelTestGenerationNormalizer.normalize(
        {"candidates": [candidate]},
        origin_language="ko",
        learning_language="ja",
    )
    repaired = normalized["candidates"][0]

    assert repaired["instructionLanguage"] == "ja"
    assert repaired["instruction"] == "文脈に最も適切な表現を選んでください。"
    assert stats.instruction_language_repairs == 0
    assert repaired["promptText"] == semantic_before["promptText"]
    assert repaired["options"] == semantic_before["options"]
    assert repaired["internalAnswerKey"] == semantic_before["internalAnswerKey"]
    assert repaired["referencePayload"]["referenceMeanings"] == semantic_before["referenceMeanings"]


def test_normalizer_repairs_missing_instruction_with_known_item_type():
    candidate = _candidate(
        item_type="LISTENING_DICTATION",
        instruction="",
        instruction_language="ko",
    )

    normalized, stats = LevelTestGenerationNormalizer.normalize(
        {"candidates": [candidate]},
        origin_language="ko",
        learning_language="ja",
    )
    repaired = normalized["candidates"][0]

    assert repaired["instruction"] == "音声を聞き、聞こえた内容を正確に書き取ってください。"
    assert repaired["instructionLanguage"] == "ja"
    assert stats.instruction_language_repairs == 1


def test_normalizer_uses_learning_language_template_for_ja_and_en():
    ja_candidate = _candidate(
        item_type="GRAMMAR_SENTENCE_ORDER",
        instruction="wrong",
        instruction_language="en",
    )
    en_candidate = _candidate(
        item_type="WRITING_TRANSLATION",
        instruction="잘못된 안내",
        instruction_language="ko",
    )

    ja_normalized, _ = LevelTestGenerationNormalizer.normalize(
        {"candidates": [ja_candidate]},
        origin_language="ko",
        learning_language="ja",
    )
    en_normalized, _ = LevelTestGenerationNormalizer.normalize(
        {"candidates": [en_candidate]},
        origin_language="ko",
        learning_language="en",
    )

    assert ja_normalized["candidates"][0]["instructionLanguage"] == "ja"
    assert ja_normalized["candidates"][0]["instruction"] == "与えられた要素を正しい順序に並べてください。"
    assert en_normalized["candidates"][0]["instructionLanguage"] == "en"
    assert en_normalized["candidates"][0]["instruction"] == "Translate the following content into the learning language."


def test_unknown_learning_language_preserves_instruction_but_canonicalizes_language_tag():
    candidate = _candidate(
        item_type="VOCAB_CONTEXT_CHOICE",
        instruction="Choisissez la meilleure réponse.",
        instruction_language="en",
    )

    normalized, stats = LevelTestGenerationNormalizer.normalize(
        {"candidates": [candidate]},
        origin_language="ko",
        learning_language="fr",
    )
    repaired = normalized["candidates"][0]

    assert repaired["instruction"] == "Choisissez la meilleure réponse."
    assert repaired["instructionLanguage"] == "fr"
    assert stats.instruction_language_repairs == 1
