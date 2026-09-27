from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.core.config import settings


class AiModelTier(str, Enum):
    NANO = "NANO"
    LUNA = "LUNA"
    MINI = "MINI"
    SOL = "SOL"


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
    "RECEIPT_ANALYSIS": AiTaskModelPolicy(AiModelTier.LUNA, "none", 16384),
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
    raise ValueError(f"Unsupported AI model tier: {tier}")


def get_model_name_for_task(type_name: str) -> str:
    return get_model_name_for_tier(get_task_model_policy(type_name).tier)
