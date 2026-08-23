from __future__ import annotations

from pydantic import Field, model_validator

from app.schemas.language_learning_listening.common import (
    CamelCaseModel,
    ListeningTaskType,
)


class HumanListeningScore(CamelCaseModel):
    evaluator_id: str = Field(..., min_length=1, max_length=100)
    score: float = Field(..., ge=0, le=100)


class ListeningBenchmarkSample(CamelCaseModel):
    sample_id: str = Field(..., min_length=1, max_length=100)
    language: str = Field(..., min_length=2, max_length=20)
    difficulty: str = Field(..., min_length=1, max_length=50)
    task_type: ListeningTaskType
    ai_score: float = Field(..., ge=0, le=100)
    human_scores: list[HumanListeningScore] = Field(..., min_length=2)

    @model_validator(mode="after")
    def validate_distinct_evaluators(self) -> "ListeningBenchmarkSample":
        evaluator_ids = [score.evaluator_id for score in self.human_scores]
        if len(evaluator_ids) != len(set(evaluator_ids)):
            raise ValueError("Human 평가자는 서로 달라야 합니다.")
        return self


class ListeningBenchmarkSegment(CamelCaseModel):
    segment: str
    sample_count: int = Field(..., ge=1)
    matched_count: int = Field(..., ge=0)
    agreement_rate: float = Field(..., ge=0, le=1)
    passed: bool


class ListeningBenchmarkResult(CamelCaseModel):
    sample_count: int = Field(..., ge=1)
    matched_count: int = Field(..., ge=0)
    agreement_rate: float = Field(..., ge=0, le=1)
    threshold_points: float = 10
    required_agreement_rate: float = 0.70
    segments: list[ListeningBenchmarkSegment]
    passed: bool
