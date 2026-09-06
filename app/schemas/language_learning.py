from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.alias_generators import to_camel

from app.core.config import settings
from app.schemas.language_learning_quality import (
    DiversityContext,
    DiversityMetadata,
    DiversitySummary,
    LanguageComplexityContext,
)


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


class DailyWritingDifficulty(str, Enum):
    REVIEW = "REVIEW"
    NORMAL = "NORMAL"
    CHALLENGE = "CHALLENGE"


class DailyWritingType(str, Enum):
    TRANSLATION = "TRANSLATION"
    GUIDED = "GUIDED"
    FREE = "FREE"


class LevelTestDifficulty(str, Enum):
    EASY = "EASY"
    NORMAL = "NORMAL"
    CHALLENGE = "CHALLENGE"


class WritingEvaluationContext(str, Enum):
    DAILY = "DAILY"
    LEVEL_TEST = "LEVEL_TEST"


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


class RecentEvaluationSummary(CamelCaseModel):
    sample_count: int = Field(default=0, ge=0)
    overall_average: float | None = Field(default=None, ge=0, le=100)
    skill_scores: WritingSkillScores | None = None
    recent_focus: list[str] = Field(default_factory=list, max_length=30)


class DifficultyDistribution(CamelCaseModel):
    review: int = Field(..., ge=0)
    normal: int = Field(..., ge=0)
    challenge: int = Field(..., ge=0)

    @property
    def total(self) -> int:
        return self.review + self.normal + self.challenge


class DailyWritingGenerationRequest(CamelCaseModel):
    request_id: str = Field(..., min_length=1, max_length=100)
    origin_language: str = Field(..., min_length=2, max_length=20)
    learning_language: str = Field(..., min_length=2, max_length=20)
    writing_type: DailyWritingType = DailyWritingType.FREE
    sentence_count: int = Field(
        ...,
        ge=1,
        le=settings.AI_LANGUAGE_LEARNING_HARD_MAX_SENTENCE_COUNT,
    )
    difficulty_distribution: DifficultyDistribution
    selected_keywords: list[SelectedKeyword] = Field(
        default_factory=list,
        max_length=settings.AI_LANGUAGE_LEARNING_HARD_MAX_SELECTED_KEYWORDS,
    )
    learning_profile: LearningProfileSummary | None = None
    recent_evaluation_summary: RecentEvaluationSummary | None = None
    recent_mistakes: list[str] = Field(default_factory=list, max_length=50)
    recently_learned_expressions: list[str] = Field(default_factory=list, max_length=50)
    generation_date: date
    snapshot_id: str | None = Field(default=None, max_length=100)
    language_complexity: LanguageComplexityContext | None = None
    diversity_context: DiversityContext = Field(default_factory=DiversityContext)
    content_diversity_policy_version: str | None = Field(default=None, max_length=100)

    @model_validator(mode="after")
    def validate_distribution(self) -> "DailyWritingGenerationRequest":
        if self.difficulty_distribution.total != self.sentence_count:
            raise ValueError(
                "difficultyDistribution 합계는 sentenceCount와 일치해야 합니다."
            )
        return self


class DailyWritingItem(CamelCaseModel):
    order: int = Field(..., ge=1)
    difficulty: DailyWritingDifficulty
    origin_text: str = Field(..., min_length=1, max_length=2000)
    keywords: list[str] = Field(default_factory=list, max_length=20)
    focus_metrics: list[WritingMetric] = Field(default_factory=list, max_length=5)
    focus_reason: str = Field(..., min_length=1, max_length=1000)
    provided_facts: list[str] = Field(default_factory=list, max_length=12)
    required_intents: list[str] = Field(default_factory=list, max_length=12)
    response_constraints: list[str] = Field(default_factory=list, max_length=12)
    language_complexity_band: int | None = Field(default=None, ge=1, le=5)
    diversity_metadata: DiversityMetadata | None = None


class DailyWritingGenerationResponse(CamelCaseModel):
    request_id: str
    prompt_version: str
    items: list[DailyWritingItem]
    content_diversity_policy_version: str | None = None
    language_complexity_policy_version: str | None = None
    diversity_summary: DiversitySummary | None = None


class BilingualMessage(CamelCaseModel):
    origin_text: str = Field(..., min_length=1, max_length=4000)
    learning_text: str = Field(..., min_length=1, max_length=4000)


class WritingCorrection(CamelCaseModel):
    original: str = Field(..., min_length=1, max_length=1000)
    corrected: str = Field(..., min_length=1, max_length=1000)
    category: str = Field(..., min_length=1, max_length=100)
    explanation: BilingualMessage


class ProfileSignals(CamelCaseModel):
    strength_tags: list[str] = Field(default_factory=list, max_length=30)
    weakness_tags: list[str] = Field(default_factory=list, max_length=30)
    grammar_patterns: list[str] = Field(default_factory=list, max_length=30)
    vocabulary_patterns: list[str] = Field(default_factory=list, max_length=30)
    naturalness_patterns: list[str] = Field(default_factory=list, max_length=30)
    expression_patterns: list[str] = Field(default_factory=list, max_length=30)
    meaning_patterns: list[str] = Field(default_factory=list, max_length=30)
    recommended_focus: list[str] = Field(default_factory=list, max_length=30)


class WritingEvaluationRequest(CamelCaseModel):
    request_id: str = Field(..., min_length=1, max_length=100)
    context: WritingEvaluationContext = WritingEvaluationContext.DAILY
    writing_type: DailyWritingType | None = None
    origin_language: str = Field(..., min_length=2, max_length=20)
    learning_language: str = Field(..., min_length=2, max_length=20)
    origin_sentence: str = Field(..., min_length=1, max_length=4000)
    user_answer: str = Field(..., min_length=1, max_length=4000)
    difficulty: str = Field(..., min_length=1, max_length=30)
    keywords: list[SelectedKeyword] = Field(default_factory=list, max_length=20)
    focus_metrics: list[WritingMetric] = Field(default_factory=list, max_length=5)
    learning_profile_summary: LearningProfileSummary | None = None
    # Structured task contract used by guided Daily Writing and Level Test writing tasks.
    task_type: str | None = Field(default=None, max_length=100)
    translation_source_text: str | None = Field(default=None, max_length=4000)
    provided_facts: list[str] = Field(default_factory=list, max_length=12)
    required_intents: list[str] = Field(default_factory=list, max_length=12)
    response_constraints: list[str] = Field(default_factory=list, max_length=12)


class AiWritingEvaluationPayload(CamelCaseModel):
    scores: WritingSkillScores
    strengths: list[BilingualMessage] = Field(default_factory=list, max_length=20)
    weaknesses: list[BilingualMessage] = Field(default_factory=list, max_length=20)
    corrections: list[WritingCorrection] = Field(default_factory=list, max_length=30)
    recommended_answers: list[str] = Field(..., min_length=2, max_length=3)
    explanation: BilingualMessage
    profile_signals: ProfileSignals


class WritingEvaluationScores(CamelCaseModel):
    overall: int = Field(..., ge=0, le=100)
    meaning: int = Field(..., ge=0, le=100)
    grammar: int = Field(..., ge=0, le=100)
    vocabulary: int = Field(..., ge=0, le=100)
    naturalness: int = Field(..., ge=0, le=100)
    expression: int = Field(..., ge=0, le=100)


class WritingEvaluationResponse(CamelCaseModel):
    request_id: str
    scores: WritingEvaluationScores
    strengths: list[BilingualMessage]
    weaknesses: list[BilingualMessage]
    corrections: list[WritingCorrection]
    recommended_answers: list[str]
    explanation: BilingualMessage
    profile_signals: ProfileSignals
    evaluation_rubric_version: str
    scoring_policy_version: str
    prompt_version: str


class LevelTestPreviousEvaluation(CamelCaseModel):
    question_number: int = Field(..., ge=1)
    difficulty: LevelTestDifficulty
    scores: WritingEvaluationScores


class LevelTestQuestionRequest(CamelCaseModel):
    request_id: str = Field(..., min_length=1, max_length=100)
    origin_language: str = Field(..., min_length=2, max_length=20)
    learning_language: str = Field(..., min_length=2, max_length=20)
    question_number: int = Field(..., ge=1)
    total_questions: int = Field(default=12, ge=1, le=30)
    previous_evaluations: list[LevelTestPreviousEvaluation] = Field(
        default_factory=list,
        max_length=29,
    )

    @model_validator(mode="after")
    def validate_progress(self) -> "LevelTestQuestionRequest":
        if self.question_number > self.total_questions:
            raise ValueError("questionNumber는 totalQuestions를 초과할 수 없습니다.")
        if len(self.previous_evaluations) >= self.question_number:
            raise ValueError(
                "previousEvaluations 개수는 현재 questionNumber보다 작아야 합니다."
            )
        return self


class LevelTestQuestionPayload(CamelCaseModel):
    difficulty: LevelTestDifficulty
    origin_text: str = Field(..., min_length=1, max_length=2000)
    focus_metrics: list[WritingMetric] = Field(..., min_length=1, max_length=5)
    focus_reason: str = Field(..., min_length=1, max_length=1000)


class LevelTestQuestionResponse(CamelCaseModel):
    request_id: str
    question_number: int
    total_questions: int
    difficulty: LevelTestDifficulty
    origin_text: str
    focus_metrics: list[WritingMetric]
    focus_reason: str
    prompt_version: str
