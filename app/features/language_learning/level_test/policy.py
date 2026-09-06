from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

from app.schemas.language_learning_level_test import LevelTestDomain, LevelTestItemType

LEVEL_TEST_ASSESSMENT_VERSION = "level-test-v2-multiskill"
LEVEL_TEST_GENERATION_VERSION = "level-test-generation-v2"
LEVEL_TEST_EVALUATION_VERSION = "level-test-evaluation-v2"
LEVEL_TEST_SPEAKING_EVALUATION_VERSION = "level-test-speaking-eval-v2"
LEVEL_TEST_SCORING_POLICY_VERSION = "level-test-scoring-v2"
LEVEL_TEST_PROMPT_VERSION = "level-test-multiskill-prompt-v9"
LEVEL_TEST_SPEAKING_PROMPT_VERSION = "level-test-speaking-eval-prompt-v2"

LEVEL_TEST_RECIPE: dict[int, tuple[LevelTestDomain, LevelTestItemType]] = {
    1: (LevelTestDomain.VOCABULARY, LevelTestItemType.VOCAB_CONTEXT_CHOICE),
    2: (LevelTestDomain.VOCABULARY, LevelTestItemType.VOCAB_CONTEXT_CHOICE),
    3: (LevelTestDomain.VOCABULARY, LevelTestItemType.VOCAB_PARAPHRASE_CHOICE),
    4: (LevelTestDomain.GRAMMAR, LevelTestItemType.GRAMMAR_FORM_CHOICE),
    5: (LevelTestDomain.GRAMMAR, LevelTestItemType.GRAMMAR_FORM_CHOICE),
    6: (LevelTestDomain.GRAMMAR, LevelTestItemType.GRAMMAR_SENTENCE_ORDER),
    7: (LevelTestDomain.READING, LevelTestItemType.READING_GIST),
    8: (LevelTestDomain.READING, LevelTestItemType.READING_DETAIL),
    9: (LevelTestDomain.READING, LevelTestItemType.READING_DISCOURSE_FUNCTION),
    10: (LevelTestDomain.READING, LevelTestItemType.READING_TEXT_INFERENCE),
    11: (LevelTestDomain.LISTENING, LevelTestItemType.LISTENING_GIST_CHOICE),
    12: (LevelTestDomain.LISTENING, LevelTestItemType.LISTENING_DETAIL_CHOICE),
    13: (LevelTestDomain.LISTENING, LevelTestItemType.LISTENING_DICTATION),
    14: (LevelTestDomain.LISTENING, LevelTestItemType.LISTENING_INTERPRETATION),
    # Writing Level Test intentionally favors translation (2/3 items).  Translation
    # isolates language ability with low non-linguistic reasoning load, while the
    # final guided paragraph checks productive organization with supplied content.
    15: (LevelTestDomain.WRITING, LevelTestItemType.WRITING_TRANSLATION),
    16: (LevelTestDomain.WRITING, LevelTestItemType.WRITING_TRANSLATION),
    17: (LevelTestDomain.WRITING, LevelTestItemType.WRITING_SHORT_PARAGRAPH),
    # Speaking intentionally separates pronunciation from free production:
    # 18 = text-assisted repeat (pronunciation/intonation with low memory load),
    # 19 = audio-only repeat (up to three listens; shorter utterance),
    # 20 = one guided open response.
    18: (LevelTestDomain.SPEAKING, LevelTestItemType.SPEAKING_REPEAT),
    19: (LevelTestDomain.SPEAKING, LevelTestItemType.SPEAKING_REPEAT),
    20: (LevelTestDomain.SPEAKING, LevelTestItemType.SPEAKING_GUIDED_RESPONSE),
}

DOMAIN_WEIGHTS: dict[LevelTestDomain, Decimal] = {
    LevelTestDomain.VOCABULARY: Decimal("0.10"),
    LevelTestDomain.GRAMMAR: Decimal("0.10"),
    LevelTestDomain.READING: Decimal("0.20"),
    LevelTestDomain.LISTENING: Decimal("0.20"),
    LevelTestDomain.WRITING: Decimal("0.20"),
    LevelTestDomain.SPEAKING: Decimal("0.20"),
}

SPEAKING_RESPONSE_WEIGHTS: dict[str, Decimal] = {
    "PRONUNCIATION": Decimal("0.20"),
    "FLUENCY": Decimal("0.20"),
    "GRAMMAR": Decimal("0.20"),
    "VOCABULARY": Decimal("0.15"),
    "TASK_FULFILLMENT": Decimal("0.25"),
}

SPEAKING_REPEAT_WEIGHTS: dict[str, Decimal] = {
    "PRONUNCIATION": Decimal("0.60"),
    "FLUENCY": Decimal("0.40"),
}

# A fluent sentence that refuses or avoids the assigned task still demonstrates
# some target-language production, but it must not receive a placement score from
# form metrics alone.
SPEAKING_NON_RESPONSE_SCORE_CAP = 10


def validate_recipe(question_number: int, domain: LevelTestDomain, item_type: LevelTestItemType) -> None:
    expected = LEVEL_TEST_RECIPE.get(question_number)
    if expected is None or expected != (domain, item_type):
        raise ValueError(
            "Level Test Recipe와 요청 Domain/ItemType이 일치하지 않습니다: "
            f"question={question_number} expected={expected} actual={(domain, item_type)}"
        )


def round_half_up(value: float) -> int:
    return int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
