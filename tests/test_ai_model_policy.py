from app.ai.model_policy import AiModelTier, get_task_model_policy


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


def test_reading_vocabulary_uses_nano_only_for_low_risk_origin_explanations():
    assert (
        get_task_model_policy("LANGUAGE_LEARNING_READING_VOCABULARY_GENERATION").tier
        == AiModelTier.LUNA
    )
    assert (
        get_task_model_policy("LANGUAGE_LEARNING_READING_PASSAGE_GENERATION").tier
        == AiModelTier.LUNA
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
