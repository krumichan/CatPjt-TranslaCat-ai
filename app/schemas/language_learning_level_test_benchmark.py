from __future__ import annotations

from pydantic import Field, model_validator

from app.schemas.language_learning_level_test import (
    CamelCaseModel,
    LevelTestDomain,
    LevelTestItemType,
)


class HumanLevelTestScore(CamelCaseModel):
    evaluator_id: str = Field(..., min_length=1, max_length=100)
    score: float = Field(..., ge=0, le=100)


class LevelTestBenchmarkSample(CamelCaseModel):
    sample_id: str = Field(..., min_length=1, max_length=100)
    domain: LevelTestDomain
    item_type: LevelTestItemType
    language: str = Field(..., min_length=2, max_length=20)
    complexity_band: int = Field(..., ge=1, le=5)
    ai_score: float = Field(..., ge=0, le=100)
    human_scores: list[HumanLevelTestScore] = Field(..., min_length=2)

    @model_validator(mode="after")
    def validate_distinct_evaluators(self) -> "LevelTestBenchmarkSample":
        evaluator_ids = [score.evaluator_id for score in self.human_scores]
        if len(evaluator_ids) != len(set(evaluator_ids)):
            raise ValueError("Human 평가자는 서로 달라야 합니다.")
        return self


class LevelTestBenchmarkSegment(CamelCaseModel):
    segment: str
    sample_count: int = Field(..., ge=1)
    matched_count: int = Field(..., ge=0)
    agreement_rate: float = Field(..., ge=0, le=1)
    passed: bool


class LevelTestBenchmarkResult(CamelCaseModel):
    sample_count: int = Field(..., ge=1)
    matched_count: int = Field(..., ge=0)
    agreement_rate: float = Field(..., ge=0, le=1)
    threshold_points: float = Field(default=10, ge=0)
    required_agreement_rate: float = Field(default=0.70, ge=0, le=1)
    segments: list[LevelTestBenchmarkSegment]
    passed: bool
