from app.ai.prompt_registry import get_prompt_rule
from app.features.language_learning.listening.prompts import (
    LISTENING_SUMMARY_SYSTEM_PROMPT,
)


def test_listening_summary_evaluation_prompt_is_registered():
    assert (
        get_prompt_rule("LANGUAGE_LEARNING_LISTENING_SUMMARY_EVALUATION")
        == LISTENING_SUMMARY_SYSTEM_PROMPT
    )
