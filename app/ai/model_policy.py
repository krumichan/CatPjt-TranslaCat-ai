from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.core.config import settings


class AiModelTier(str, Enum):
    NANO = "NANO"
    LUNA = "LUNA"
    MINI = "MINI"
    SOL = "SOL"
    SPENDING_COACH = "SPENDING_COACH"


@dataclass(frozen=True)
class AiTaskModelPolicy:
    tier: AiModelTier
    reasoning_effort: str
    max_output_tokens: int
    verbosity: str = "low"


# Model selection is application policy, not deployment configuration.
# Environment variables only map logical tiers to concrete OpenAI model IDs.
_TASK_POLICIES: dict[str, AiTaskModelPolicy] = {
    # General translation / generation -> Luna.
    "RANK": AiTaskModelPolicy(AiModelTier.LUNA, "none", 4096),
    "NOVEL": AiTaskModelPolicy(AiModelTier.LUNA, "none", 4096),
    "EPISODE": AiTaskModelPolicy(AiModelTier.LUNA, "none", 8192),
    "VOICE": AiTaskModelPolicy(AiModelTier.LUNA, "none", 2048),
    "VOICE_TRANSLATION": AiTaskModelPolicy(AiModelTier.LUNA, "none", 2048),
    "VOICE_TRANSLATION_TEXT": AiTaskModelPolicy(AiModelTier.LUNA, "none", 768),
    "VOICE_TRANSLATION_PREVIEW": AiTaskModelPolicy(AiModelTier.LUNA, "none", 768),
    "RECEIPT_ANALYSIS": AiTaskModelPolicy(AiModelTier.LUNA, "none", 16384),
    # 구매 역할·과거 맥락을 해석하는 가계부 분석만 강화한다. 다른 작업의 모델은 유지한다.
    "ACCOUNT_BOOK_SPENDING_ANALYSIS": AiTaskModelPolicy(AiModelTier.SPENDING_COACH, "medium", 12288, "medium"),
    "ACCOUNT_BOOK_SPENDING_REVIEW": AiTaskModelPolicy(AiModelTier.SPENDING_COACH, "medium", 12288, "medium"),
}

# Unknown/new tasks fail quality-safe: use Mini until explicitly classified.
_DEFAULT_POLICY = AiTaskModelPolicy(AiModelTier.MINI, "low", 8192)


def get_task_model_policy(type_name: str) -> AiTaskModelPolicy:
    return _TASK_POLICIES.get(type_name, _DEFAULT_POLICY)


def get_model_name_for_tier(tier: AiModelTier) -> str:
    if tier == AiModelTier.NANO:
        return settings.OPENAI_MODEL_NANO
    if tier == AiModelTier.LUNA:
        return settings.OPENAI_MODEL_LUNA
    if tier == AiModelTier.MINI:
        return settings.OPENAI_MODEL_MINI
    if tier == AiModelTier.SOL:
        return "gpt-5.6-sol"
    if tier == AiModelTier.SPENDING_COACH:
        return "gpt-6.1-sol"
    raise ValueError(f"Unsupported AI model tier: {tier}")


def get_model_name_for_task(type_name: str) -> str:
    return get_model_name_for_tier(get_task_model_policy(type_name).tier)
