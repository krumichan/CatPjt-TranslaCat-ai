from __future__ import annotations

from dataclasses import dataclass

from app.features.language_learning.difficulty.contracts import DifficultyTarget
from app.features.language_learning.reading_vocabulary.vocabulary_difficulty_recipe import (
    VocabularyDifficultyRecipe,
)


VOCABULARY_DIFFICULTY_SPEC_VERSION = "vocabulary-difficulty-contract-v1"


@dataclass(frozen=True)
class VocabularyDifficultyTargetValue:
    mode: str
    difficulty: str
    complexity_band: int
    skill_tag: str


@dataclass(frozen=True)
class VocabularyDifficultySpec:
    target: DifficultyTarget[VocabularyDifficultyTargetValue]
    question_type: str
    target_expression: str
    usage_intent: str | None
    recipe: VocabularyDifficultyRecipe
    version: str = VOCABULARY_DIFFICULTY_SPEC_VERSION
