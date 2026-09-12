from __future__ import annotations

from dataclasses import dataclass

from app.features.language_learning.difficulty.contracts import DifficultyTarget


SPEAKING_GENERATION_DIFFICULTY_SPEC_VERSION = "speaking-generation-difficulty-contract-v1"


@dataclass(frozen=True)
class SpeakingGenerationDifficultyTargetValue:
    target_level: str | None
    practice_mode: str


@dataclass(frozen=True)
class SpeakingGenerationDifficultySpec:
    target: DifficultyTarget[SpeakingGenerationDifficultyTargetValue]
    version: str = SPEAKING_GENERATION_DIFFICULTY_SPEC_VERSION
