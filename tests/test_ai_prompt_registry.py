from app.ai.prompt_registry import get_prompt_rule
from app.features.language_learning.listening.prompts import (
    LISTENING_SUMMARY_SYSTEM_PROMPT,
)
from app.features.language_learning.speaking.prompts import (
    SPEAKING_ASSISTANCE_SYSTEM_PROMPT,
)


def test_listening_summary_evaluation_prompt_is_registered():
    assert (
        get_prompt_rule("LANGUAGE_LEARNING_LISTENING_SUMMARY_EVALUATION")
        == LISTENING_SUMMARY_SYSTEM_PROMPT
    )


def test_speaking_assistance_prompt_is_registered():
    assert (
        get_prompt_rule("LANGUAGE_LEARNING_SPEAKING_ASSISTANCE")
        == SPEAKING_ASSISTANCE_SYSTEM_PROMPT
    )


def test_reading_vocabulary_pipeline_prompts_are_registered():
    for type_name in [
        "LANGUAGE_LEARNING_READING_VOCABULARY_GENERATION",
        "LANGUAGE_LEARNING_READING_PASSAGE_GENERATION",
        "LANGUAGE_LEARNING_READING_VOCABULARY_USAGE_PRESCREEN",
        "LANGUAGE_LEARNING_READING_VOCABULARY_VERIFICATION",
        "LANGUAGE_LEARNING_READING_VOCABULARY_ORIGIN_EXPLANATION",
        "LANGUAGE_LEARNING_READING_VOCABULARY_ORIGIN_EXPLANATION_FALLBACK",
    ]:
        assert get_prompt_rule(type_name)
