from __future__ import annotations

from pydantic import Field, model_validator

from app.schemas.language_learning import LearningProfileSummary
from app.schemas.language_learning_speaking.common import (
    AssistanceUsage,
    CamelCaseModel,
    MetricEvaluationState,
    SpeakingEvaluationStatus,
    SpeakingMetricType,
    SpeakingPracticeMode,
    SpeakingUsage,
)
from app.schemas.language_learning_speaking.turn import AudioQualitySignals, SttSegment


class SpeakingEvaluationTurn(CamelCaseModel):
    turn_id: str = Field(..., min_length=1, max_length=100)
    turn_index: int = Field(..., ge=1, le=20)
    transcript: str = Field(default="", max_length=4000)
    stt_confidence: float = Field(..., ge=0, le=1)
    duration_seconds: float = Field(..., ge=0, le=60)
    segments: list[SttSegment] = Field(default_factory=list, max_length=100)
    audio_reference: str | None = Field(default=None, max_length=500)
    audio_available: bool = False
    audio_quality_signals: AudioQualitySignals | None = None
    excluded_from_evaluation: bool = False
    assistance_usage: list[AssistanceUsage] = Field(default_factory=list, max_length=20)


class AssistantEvaluationTurn(CamelCaseModel):
    turn_id: str = Field(..., min_length=1, max_length=100)
    turn_index: int = Field(..., ge=0, le=20)
    text: str = Field(..., min_length=1, max_length=4000)
    script_text: str | None = Field(default=None, max_length=4000)
    provided_facts: list[str] = Field(default_factory=list, max_length=12)
    required_intents: list[str] = Field(default_factory=list, max_length=12)
    response_constraints: list[str] = Field(default_factory=list, max_length=12)


class SpeakingEvaluationRequest(CamelCaseModel):
    request_id: str = Field(..., min_length=1, max_length=100)
    idempotency_key: str = Field(..., min_length=1, max_length=200)
    session_id: str = Field(..., min_length=1, max_length=100)
    topic: str = Field(..., min_length=1, max_length=500)
    practice_mode: SpeakingPracticeMode = SpeakingPracticeMode.FREE
    goal: str | None = Field(default=None, max_length=1000)
    target_level: str | None = Field(default=None, max_length=50)
    origin_language: str = Field(..., min_length=2, max_length=20)
    learning_language: str = Field(..., min_length=2, max_length=20)
    user_turns: list[SpeakingEvaluationTurn] = Field(..., min_length=1, max_length=20)
    assistant_turns: list[AssistantEvaluationTurn] = Field(
        default_factory=list,
        max_length=21,
    )
    session_summary: str | None = Field(default=None, max_length=6000)
    prior_profile_summary: LearningProfileSummary | None = None
    evaluation_policy_version: str = Field(default="speaking-evaluation-policy-v1")
    manual_retry_attempt: int = Field(default=0, ge=0, le=1)

    @model_validator(mode="after")
    def validate_turn_identity(self) -> "SpeakingEvaluationRequest":
        user_turn_ids = [turn.turn_id for turn in self.user_turns]
        if len(user_turn_ids) != len(set(user_turn_ids)):
            raise ValueError("userTurns의 turnId는 중복될 수 없습니다.")

        assistant_turn_ids = [turn.turn_id for turn in self.assistant_turns]
        if len(assistant_turn_ids) != len(set(assistant_turn_ids)):
            raise ValueError("assistantTurns의 turnId는 중복될 수 없습니다.")

        return self


class MetricEvidence(CamelCaseModel):
    turn_id: str = Field(..., min_length=1, max_length=100)
    start_ms: int | None = Field(default=None, ge=0)
    end_ms: int | None = Field(default=None, ge=0)
    message: str = Field(..., min_length=1, max_length=2000)

    @model_validator(mode="after")
    def validate_timestamps(self) -> "MetricEvidence":
        if (
            self.start_ms is not None
            and self.end_ms is not None
            and self.end_ms < self.start_ms
        ):
            raise ValueError("endMs는 startMs 이상이어야 합니다.")
        return self


class SpeakingMetricPayload(CamelCaseModel):
    type: SpeakingMetricType
    state: MetricEvaluationState = MetricEvaluationState.EVALUATED
    score: float | None = Field(default=None, ge=0, le=100)
    confidence: float = Field(..., ge=0, le=1)
    summary: str = Field(..., min_length=1, max_length=2000)
    evidence: list[MetricEvidence] = Field(default_factory=list, max_length=30)
    not_evaluable_reason: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def validate_evaluability(self) -> "SpeakingMetricPayload":
        if self.state == MetricEvaluationState.EVALUATED:
            if self.score is None:
                raise ValueError("EVALUATED Metric에는 score가 필요합니다.")
            if not self.evidence:
                raise ValueError("EVALUATED Metric에는 최소 1개의 Evidence가 필요합니다.")

        if self.state == MetricEvaluationState.NOT_EVALUABLE:
            if self.score is not None:
                raise ValueError("NOT_EVALUABLE Metric에는 score를 설정할 수 없습니다.")
            if not self.not_evaluable_reason:
                raise ValueError("NOT_EVALUABLE Metric에는 사유가 필요합니다.")

        return self


class SpeakingProfileSignal(CamelCaseModel):
    metric_type: SpeakingMetricType
    source: str = Field(default="SPEAKING", pattern="^SPEAKING$")
    direction: str = Field(pattern="^(STRENGTH|WEAKNESS|IMPROVING)$")
    confidence: float = Field(..., ge=0, le=1)
    evidence_turn_ids: list[str] = Field(default_factory=list, max_length=30)
    pattern_key: str = Field(..., min_length=1, max_length=200)
    recommended_focus: str = Field(..., min_length=1, max_length=1000)


class RecommendedExpression(CamelCaseModel):
    original: str | None = Field(default=None, max_length=2000)
    recommended: str = Field(..., min_length=1, max_length=2000)
    explanation: str = Field(..., min_length=1, max_length=2000)
    evidence_turn_ids: list[str] = Field(default_factory=list, max_length=20)


class PronunciationPractice(CamelCaseModel):
    target: str = Field(..., min_length=1, max_length=500)
    practice_phrase: str = Field(..., min_length=1, max_length=2000)
    reason: str = Field(..., min_length=1, max_length=2000)
    evidence_turn_ids: list[str] = Field(default_factory=list, max_length=20)


class SpeakingEvaluationPayload(CamelCaseModel):
    evaluation_confidence: float = Field(..., ge=0, le=1)
    metrics: list[SpeakingMetricPayload] = Field(..., min_length=8, max_length=8)
    strengths: list[str] = Field(default_factory=list, max_length=20)
    improvements: list[str] = Field(default_factory=list, max_length=20)
    recommended_expressions: list[RecommendedExpression] = Field(default_factory=list, max_length=20)
    pronunciation_practice: list[PronunciationPractice] = Field(default_factory=list, max_length=20)
    profile_signals: list[SpeakingProfileSignal] = Field(default_factory=list, max_length=30)


class EvaluationEligibility(CamelCaseModel):
    valid_user_turns: int = Field(..., ge=0)
    valid_user_speech_seconds: float = Field(..., ge=0)
    valid_stt_turn_ratio: float = Field(..., ge=0, le=1)
    required_user_turns: int = 5
    required_speech_seconds: float = 60
    required_stt_turn_ratio: float = 0.8
    required_evaluation_confidence: float = 0.7
    eligible_before_ai: bool
    missing_requirements: list[str] = Field(default_factory=list)


class SpeakingEvaluationResponse(CamelCaseModel):
    request_id: str
    session_id: str
    status: SpeakingEvaluationStatus
    overall_score: int | None = Field(default=None, ge=0, le=100)
    evaluation_confidence: float | None = Field(default=None, ge=0, le=1)
    metrics: list[SpeakingMetricPayload] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    recommended_expressions: list[RecommendedExpression] = Field(default_factory=list)
    pronunciation_practice: list[PronunciationPractice] = Field(default_factory=list)
    profile_signals: list[SpeakingProfileSignal] = Field(default_factory=list)
    eligibility: EvaluationEligibility
    evaluation_version: str
    scoring_policy_version: str
    prompt_version: str
    usage: SpeakingUsage


class HumanMetricScore(CamelCaseModel):
    metric_type: SpeakingMetricType
    score: float = Field(..., ge=0, le=100)


def _validate_benchmark_metric_scores(scores: list[HumanMetricScore]) -> None:
    metric_types = [score.metric_type for score in scores]
    expected = set(SpeakingMetricType)
    if len(metric_types) != len(set(metric_types)) or set(metric_types) != expected:
        raise ValueError("Benchmark Score는 Speaking 8축을 중복 없이 포함해야 합니다.")


class HumanEvaluatorScoreSet(CamelCaseModel):
    evaluator_id: str
    scores: list[HumanMetricScore] = Field(..., min_length=8, max_length=8)

    @model_validator(mode="after")
    def validate_metric_set(self) -> "HumanEvaluatorScoreSet":
        _validate_benchmark_metric_scores(self.scores)
        return self


class BenchmarkSample(CamelCaseModel):
    sample_id: str
    ai_scores: list[HumanMetricScore] = Field(..., min_length=8, max_length=8)
    human_evaluators: list[HumanEvaluatorScoreSet] = Field(..., min_length=2)

    @model_validator(mode="after")
    def validate_ai_metric_set(self) -> "BenchmarkSample":
        _validate_benchmark_metric_scores(self.ai_scores)
        evaluator_ids = [item.evaluator_id for item in self.human_evaluators]
        if len(evaluator_ids) != len(set(evaluator_ids)):
            raise ValueError("Human Benchmark에는 서로 다른 2명 이상의 평가자가 필요합니다.")
        return self


class BenchmarkResult(CamelCaseModel):
    total_metrics: int
    matched_metrics: int
    agreement_rate: float = Field(..., ge=0, le=1)
    threshold_points: float = 10
    required_agreement_rate: float = 0.7
    passed: bool
