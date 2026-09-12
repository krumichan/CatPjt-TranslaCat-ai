from __future__ import annotations

from dataclasses import dataclass

from app.features.language_learning.difficulty.contracts import DifficultyTarget


LISTENING_DIFFICULTY_SPEC_VERSION = "listening-difficulty-contract-v1"


@dataclass(frozen=True)
class ListeningDifficultyTargetValue:
    text_band: int
    mode: str


@dataclass(frozen=True)
class ListeningDifficultySpec:
    target: DifficultyTarget[ListeningDifficultyTargetValue]
    duration_min_seconds: float
    duration_max_seconds: float
    version: str = LISTENING_DIFFICULTY_SPEC_VERSION
