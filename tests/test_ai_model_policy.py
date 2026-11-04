from app.ai.model_policy import AiModelTier, get_task_model_policy, get_model_name_for_task


def test_receipt_and_voice_translation_start_on_luna():
    assert get_task_model_policy("RECEIPT_ANALYSIS").tier == AiModelTier.LUNA
    assert get_task_model_policy("VOICE_TRANSLATION").tier == AiModelTier.LUNA


def test_unknown_task_fails_quality_safe_to_mini():
    assert get_task_model_policy("NEW_UNCLASSIFIED_TASK").tier == AiModelTier.MINI


def test_purchase_context_work_uses_the_verified_coaching_model_only():
    # 준비 / 실행 / 검증: 가계부 생성·검토만 강화하고 일반 작업의 비용 정책은 보존한다.
    for name in ("ACCOUNT_BOOK_SPENDING_ANALYSIS", "ACCOUNT_BOOK_SPENDING_REVIEW"):
        assert get_task_model_policy(name).tier == AiModelTier.SPENDING_COACH
        assert get_task_model_policy(name).reasoning_effort == "medium"
        assert get_model_name_for_task(name) == "gpt-6.1-sol"
    assert get_task_model_policy("RECEIPT_ANALYSIS").tier == AiModelTier.LUNA
