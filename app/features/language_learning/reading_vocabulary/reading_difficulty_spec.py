from __future__ import annotations

from dataclasses import dataclass

from app.features.language_learning.difficulty.contracts import DifficultyTarget
from app.features.language_learning.reading_vocabulary.reading_difficulty_recipe import (
    ReadingDifficultyRecipe,
)


READING_DIFFICULTY_SPEC_VERSION = "reading-difficulty-contract-v1"


@dataclass(frozen=True)
class ReadingPassageDifficultyTargetValue:
    complexity_band: int
    mode: str


@dataclass(frozen=True)
class ReadingPassageDifficultySpec:
    target: DifficultyTarget[ReadingPassageDifficultyTargetValue]
    passage_id: str
    recipe: ReadingDifficultyRecipe
    version: str = READING_DIFFICULTY_SPEC_VERSION


@dataclass(frozen=True)
class ReadingQuestionDifficultyTargetValue:
    difficulty: str
    complexity_band: int
    skill_tag: str


@dataclass(frozen=True)
class ReadingQuestionDifficultySpec:
    target: DifficultyTarget[ReadingQuestionDifficultyTargetValue]
    passage_id: str
    recipe: ReadingDifficultyRecipe
    version: str = READING_DIFFICULTY_SPEC_VERSION
