from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class CamelCaseModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        str_strip_whitespace=True,
        extra="forbid",
    )


class KeywordSource(str, Enum):
    SYSTEM = "SYSTEM"
    CUSTOM = "CUSTOM"


class KeywordType(str, Enum):
    TOPIC = "TOPIC"
    VOCABULARY = "VOCABULARY"


class WritingMetric(str, Enum):
    MEANING = "MEANING"
    GRAMMAR = "GRAMMAR"
    VOCABULARY = "VOCABULARY"
    NATURALNESS = "NATURALNESS"
    EXPRESSION = "EXPRESSION"


class SelectedKeyword(CamelCaseModel):
    key: str = Field(..., min_length=1, max_length=200)
    text: str = Field(..., min_length=1, max_length=200)
    source: KeywordSource
    type: KeywordType
    canonical_key: str | None = Field(default=None, max_length=200)
    selection_weight: float | None = Field(default=None, ge=0, le=1)


class WritingSkillScores(CamelCaseModel):
    meaning: float = Field(..., ge=0, le=100)
    grammar: float = Field(..., ge=0, le=100)
    vocabulary: float = Field(..., ge=0, le=100)
    naturalness: float = Field(..., ge=0, le=100)
    expression: float = Field(..., ge=0, le=100)


class DifficultyPerformance(CamelCaseModel):
    review: float | None = Field(default=None, ge=0, le=100)
    normal: float | None = Field(default=None, ge=0, le=100)
    challenge: float | None = Field(default=None, ge=0, le=100)


class KeywordMastery(CamelCaseModel):
    canonical_key: str = Field(..., min_length=1, max_length=200)
    score: float = Field(..., ge=0, le=100)


class LearningProfileSummary(CamelCaseModel):
    profile_version: str | None = Field(default=None, max_length=100)
    base_level_score: float | None = Field(default=None, ge=0, le=100)
    skill_scores: WritingSkillScores | None = None
    grammar_weaknesses: list[str] = Field(default_factory=list, max_length=50)
    keyword_masteries: list[KeywordMastery] = Field(default_factory=list, max_length=100)
    difficulty_performance: DifficultyPerformance | None = None
    error_patterns: list[str] = Field(default_factory=list, max_length=50)
    trend: str | None = Field(default=None, max_length=50)
    confidence: float | None = Field(default=None, ge=0, le=1)
    strengths: list[str] = Field(default_factory=list, max_length=30)
    weaknesses: list[str] = Field(default_factory=list, max_length=30)
    recommended_focus: list[str] = Field(default_factory=list, max_length=30)
    additional_signals: dict[str, Any] = Field(default_factory=dict)
