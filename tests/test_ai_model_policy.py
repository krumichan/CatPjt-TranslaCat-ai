from app.ai.model_policy import AiModelTier, get_model_name_for_task, get_task_model_policy
from app.ai.prompt_registry import get_prompt_rule
from app.core.config import settings
from app.features.language_learning.reading_vocabulary.service import ReadingVocabularyGenerationService


def test_generation_uses_luna_and_evaluation_uses_mini():
    assert (
        get_task_model_policy("LANGUAGE_LEARNING_DAILY_WRITING_GENERATION").tier
        == AiModelTier.LUNA
    )
    assert (
        get_task_model_policy("LANGUAGE_LEARNING_WRITING_EVALUATION").tier
        == AiModelTier.MINI
    )
    assert (
        get_task_model_policy("LANGUAGE_LEARNING_LEVEL_TEST_CHOICE_VERIFICATION").tier
        == AiModelTier.MINI
    )
    assert (
        get_task_model_policy("LANGUAGE_LEARNING_LISTENING_SUMMARY_EVALUATION").tier
        == AiModelTier.MINI
    )
    assistance_policy = get_task_model_policy(
        "LANGUAGE_LEARNING_SPEAKING_ASSISTANCE"
    )
    assert assistance_policy.tier == AiModelTier.LUNA
    assert assistance_policy.reasoning_effort == "none"
    assert assistance_policy.max_output_tokens == 2048


def test_receipt_and_voice_translation_start_on_luna():
    assert get_task_model_policy("RECEIPT_ANALYSIS").tier == AiModelTier.LUNA
    assert get_task_model_policy("VOICE_TRANSLATION").tier == AiModelTier.LUNA


def test_unknown_task_fails_quality_safe_to_mini():
    assert get_task_model_policy("NEW_UNCLASSIFIED_TASK").tier == AiModelTier.MINI


def test_selected_reading_passage_uses_sol_and_vocab_and_explanations_stay_scoped(monkeypatch):
    monkeypatch.setattr(settings, "AI_READING_GENERATION_MODEL", "SOL")
    assert (
        get_task_model_policy("LANGUAGE_LEARNING_READING_VOCABULARY_GENERATION").tier
        == AiModelTier.LUNA
    )
    assert (
        get_task_model_policy("LANGUAGE_LEARNING_READING_PASSAGE_GENERATION").tier
        == AiModelTier.SOL
    )
    assert (
        get_task_model_policy("LANGUAGE_LEARNING_READING_VOCABULARY_USAGE_PRESCREEN").tier
        == AiModelTier.NANO
    )
    assert (
        get_task_model_policy("LANGUAGE_LEARNING_READING_VOCABULARY_VERIFICATION").tier
        == AiModelTier.MINI
    )
    nano_policy = get_task_model_policy(
        "LANGUAGE_LEARNING_READING_VOCABULARY_ORIGIN_EXPLANATION"
    )
    assert nano_policy.tier == AiModelTier.NANO
    assert nano_policy.max_output_tokens == 2048
    assert (
        get_task_model_policy(
            "LANGUAGE_LEARNING_READING_VOCABULARY_ORIGIN_EXPLANATION_FALLBACK"
        ).tier
        == AiModelTier.MINI
    )


def test_sol_comparison_changes_only_reading_generation_and_existing_repair(monkeypatch):
    monkeypatch.setattr(settings, "AI_READING_GENERATION_MODEL", "SOL")
    reading = ReadingVocabularyGenerationService
    tasks = (reading.PASSAGE_TYPE_NAME, reading.READING_SOL_TYPE_NAME,
             reading.DISTRACTOR_REPAIR_TYPE_NAME)
    for task in tasks:
        policy = get_task_model_policy(task)
        assert (policy.tier, policy.reasoning_effort, policy.max_output_tokens) == (
            AiModelTier.SOL, "high", 8192,
        )
        assert get_model_name_for_task(task) == "gpt-5.6-sol"
        assert get_prompt_rule(task)
    assert get_prompt_rule(reading.READING_SOL_TYPE_NAME) == get_prompt_rule(reading.TYPE_NAME)
    assert get_task_model_policy(reading.TYPE_NAME).tier == AiModelTier.LUNA
    assert get_task_model_policy(reading.VERIFICATION_TYPE_NAME).tier == AiModelTier.MINI
    assert get_task_model_policy("LANGUAGE_LEARNING_SPEAKING_CONVERSATION").tier == AiModelTier.LUNA
    assert get_task_model_policy("LANGUAGE_LEARNING_LISTENING_GENERATION").tier == AiModelTier.LUNA
    monkeypatch.setattr(settings, "AI_READING_GENERATION_MODEL", "LUNA")
    assert get_task_model_policy(reading.PASSAGE_TYPE_NAME).tier == AiModelTier.LUNA
    assert get_task_model_policy(reading.DISTRACTOR_REPAIR_TYPE_NAME).tier == AiModelTier.LUNA
