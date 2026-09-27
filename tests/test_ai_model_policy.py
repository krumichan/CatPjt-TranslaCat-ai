from app.ai.model_policy import AiModelTier, get_task_model_policy


def test_receipt_and_voice_translation_start_on_luna():
    assert get_task_model_policy("RECEIPT_ANALYSIS").tier == AiModelTier.LUNA
    assert get_task_model_policy("VOICE_TRANSLATION").tier == AiModelTier.LUNA


def test_unknown_task_fails_quality_safe_to_mini():
    assert get_task_model_policy("NEW_UNCLASSIFIED_TASK").tier == AiModelTier.MINI
