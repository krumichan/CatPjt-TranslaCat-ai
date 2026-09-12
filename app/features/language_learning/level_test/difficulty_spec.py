from __future__ import annotations

from dataclasses import dataclass

from app.features.language_learning.difficulty.contracts import DifficultyTarget


LEVEL_TEST_DIFFICULTY_SPEC_VERSION = "level-test-difficulty-contract-v1"


@dataclass(frozen=True)
class LevelTestDifficultyTargetValue:
    domain: str
    item_type: str
    complexity_band: int


@dataclass(frozen=True)
class LevelTestDifficultySpec:
    target: DifficultyTarget[LevelTestDifficultyTargetValue]
    question_number: int
    version: str = LEVEL_TEST_DIFFICULTY_SPEC_VERSION
