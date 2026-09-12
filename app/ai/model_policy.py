from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.core.config import settings


class AiModelTier(str, Enum):
    NANO = "NANO"
    LUNA = "LUNA"
    MINI = "MINI"


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
    "CHAT_MESSAGE_TRANSLATION": AiTaskModelPolicy(AiModelTier.LUNA, "none", 1024),
    "VOICE_TRANSLATION": AiTaskModelPolicy(AiModelTier.LUNA, "none", 2048),
    "AI_CHAT_REPLY": AiTaskModelPolicy(AiModelTier.LUNA, "none", 2048, "medium"),
    "RECEIPT_ANALYSIS": AiTaskModelPolicy(AiModelTier.LUNA, "none", 4096),
    # Language-learning generation -> Luna.
    "LANGUAGE_LEARNING_DAILY_WRITING_GENERATION": AiTaskModelPolicy(
        AiModelTier.LUNA, "none", 8192
    ),
    # Writing: content generation is separate from blind difficulty/task checks.
    # Nano never issues the final reject/accept; Mini task review is mandatory.
    "LANGUAGE_LEARNING_WRITING_DIFFICULTY_PRESCREEN": AiTaskModelPolicy(
        AiModelTier.NANO, "low", 2048
    ),
    "LANGUAGE_LEARNING_WRITING_TASK_VERIFICATION": AiTaskModelPolicy(
        AiModelTier.MINI, "low", 4096
    ),
    "LANGUAGE_LEARNING_WRITING_DIFFICULTY_VERIFICATION": AiTaskModelPolicy(
        AiModelTier.MINI, "low", 2048
    ),
    # Note localization proposes text only; independent Mini must still approve it.
    "LANGUAGE_LEARNING_WRITING_SOURCE_LOCALIZATION": AiTaskModelPolicy(
        AiModelTier.NANO, "low", 3072
    ),
    "LANGUAGE_LEARNING_WRITING_NOTE_LOCALIZATION": AiTaskModelPolicy(
        AiModelTier.NANO, "low", 2048
    ),
    "LANGUAGE_LEARNING_WRITING_NOTE_VERIFICATION": AiTaskModelPolicy(
        AiModelTier.MINI, "low", 2048
    ),
    "LANGUAGE_LEARNING_READING_VOCABULARY_GENERATION": AiTaskModelPolicy(
        AiModelTier.LUNA, "none", 4096
    ),
    "LANGUAGE_LEARNING_READING_PASSAGE_GENERATION": AiTaskModelPolicy(
        AiModelTier.LUNA, "none", 4096
    ),
    "LANGUAGE_LEARNING_READING_VOCABULARY_USAGE_PRESCREEN": AiTaskModelPolicy(
        AiModelTier.NANO, "low", 2048
    ),
    "LANGUAGE_LEARNING_READING_VOCABULARY_VERIFICATION": AiTaskModelPolicy(
        AiModelTier.MINI, "low", 4096
    ),
    # Lightweight post-validation explanation localization is intentionally Nano.
    # It never decides correctness; Mini is the quality-safe fallback.
    "LANGUAGE_LEARNING_READING_VOCABULARY_ORIGIN_EXPLANATION": AiTaskModelPolicy(
        AiModelTier.NANO, "low", 2048
    ),
    "LANGUAGE_LEARNING_READING_VOCABULARY_ORIGIN_EXPLANATION_FALLBACK": AiTaskModelPolicy(
        AiModelTier.MINI, "low", 2048
    ),
    "LANGUAGE_LEARNING_LEVEL_TEST_QUESTION": AiTaskModelPolicy(
        AiModelTier.LUNA, "none", 4096
    ),
    "LANGUAGE_LEARNING_LEVEL_TEST_GENERATION": AiTaskModelPolicy(
        AiModelTier.LUNA, "none", 8192
    ),
    "LANGUAGE_LEARNING_LEVEL_TEST_VOCAB_CONTEXT_DESIGN": AiTaskModelPolicy(
        AiModelTier.LUNA, "none", 4096
    ),
    "LANGUAGE_LEARNING_SPEAKING_CONVERSATION": AiTaskModelPolicy(
        AiModelTier.LUNA, "none", 4096
    ),
    "LANGUAGE_LEARNING_SPEAKING_ASSISTANCE": AiTaskModelPolicy(
        AiModelTier.LUNA, "none", 2048
    ),
    "LANGUAGE_LEARNING_LISTENING_GENERATION": AiTaskModelPolicy(
        AiModelTier.LUNA, "none", 8192
    ),
    "LANGUAGE_LEARNING_LISTENING_EXPLANATION": AiTaskModelPolicy(
        AiModelTier.LUNA, "none", 4096
    ),
    # Quality-critical repair / verification / evaluation -> Mini.
    "LANGUAGE_LEARNING_LEVEL_TEST_VOCAB_CONTEXT_REPAIR": AiTaskModelPolicy(
        AiModelTier.MINI, "low", 4096
    ),
    "LANGUAGE_LEARNING_LEVEL_TEST_CHOICE_VERIFICATION": AiTaskModelPolicy(
        AiModelTier.MINI, "low", 2048
    ),
    "LANGUAGE_LEARNING_LEVEL_TEST_TASK_SUFFICIENCY_VERIFICATION": AiTaskModelPolicy(
        AiModelTier.MINI, "low", 2048
    ),
    "LANGUAGE_LEARNING_WRITING_EVALUATION": AiTaskModelPolicy(
        AiModelTier.MINI, "low", 8192
    ),
    "LANGUAGE_LEARNING_SPEAKING_EVALUATION": AiTaskModelPolicy(
        AiModelTier.MINI, "low", 8192
    ),
    "LANGUAGE_LEARNING_LISTENING_INTERPRETATION": AiTaskModelPolicy(
        AiModelTier.MINI, "low", 8192
    ),
    "LANGUAGE_LEARNING_LISTENING_SUMMARY_EVALUATION": AiTaskModelPolicy(
        AiModelTier.MINI, "low", 8192
    ),
    "LANGUAGE_LEARNING_LEVEL_TEST_SPEAKING_EVALUATION": AiTaskModelPolicy(
        AiModelTier.MINI, "low", 8192
    ),
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
    raise ValueError(f"Unsupported AI model tier: {tier}")


def get_model_name_for_task(type_name: str) -> str:
    return get_model_name_for_tier(get_task_model_policy(type_name).tier)
