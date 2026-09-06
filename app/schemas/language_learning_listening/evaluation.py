from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import Field, model_validator

from app.schemas.language_learning_listening.generation import ListeningChoiceOption

from app.schemas.language_learning_listening.common import (
    AssistanceUsage,
    CamelCaseModel,
    EvaluationPurpose,
    ListeningAssistanceLevel,
    ListeningTaskStatus,
    ListeningTaskType,
    ListeningUsage,
    MetricEvaluationState,
)


class DictationMetricType(str, Enum):
    TOKEN_RECOGNITION = "TOKEN_RECOGNITION"
    OMISSION_ADDITION_ORDER = "OMISSION_ADDITION_ORDER"
    ORTHOGRAPHY = "ORTHOGRAPHY"


class InterpretationMetricType(str, Enum):
    MEANING_FIDELITY = "MEANING_FIDELITY"
    DETAIL_AND_NUANCE = "DETAIL_AND_NUANCE"
    ORIGIN_NATURALNESS = "ORIGIN_NATURALNESS"


class ComprehensionMetricType(str, Enum):
    ANSWER_ACCURACY = "ANSWER_ACCURACY"


class SummaryMetricType(str, Enum):
    GIST_COVERAGE = "GIST_COVERAGE"
    KEY_POINT_COVERAGE = "KEY_POINT_COVERAGE"
    LANGUAGE_CLARITY = "LANGUAGE_CLARITY"


class RepeatMetricType(str, Enum):
    PRONUNCIATION = "PRONUNCIATION"
    PROSODY_RHYTHM = "PROSODY_RHYTHM"
    FLUENCY = "FLUENCY"
    COMPLETENESS = "COMPLETENESS"


class ProfileMetric(str, Enum):
    LISTENING_RECOGNITION = "LISTENING_RECOGNITION"
    VOCABULARY = "VOCABULARY"
    ORTHOGRAPHY = "ORTHOGRAPHY"
    MEANING = "MEANING"
    ORIGIN_NATURALNESS = "ORIGIN_NATURALNESS"
    PRONUNCIATION = "PRONUNCIATION"
    FLUENCY = "FLUENCY"


class AlignmentStatus(str, Enum):
    MATCH = "MATCH"
    ACCEPTED_VARIANT = "ACCEPTED_VARIANT"
    SUBSTITUTION = "SUBSTITUTION"
    OMISSION = "OMISSION"
    ADDITION = "ADDITION"
    ORDER = "ORDER"


class EvaluationBaseRequest(CamelCaseModel):
    request_id: str = Field(..., min_length=1, max_length=100)
    idempotency_key: str = Field(..., min_length=1, max_length=200)
    item_id: int | str
    attempt_id: int | str
    evaluation_purpose: EvaluationPurpose
    answer_revealed: bool = False
    assistance_usage: list[AssistanceUsage] = Field(default_factory=list, max_length=20)
    policy_version: str = Field(default="listening-profile-v1", max_length=100)
    model_config_version: str = Field(
        default="listening-model-config-v1", max_length=100
    )
    manual_retry_attempt: int = Field(default=0, ge=0, le=1)


class DictationEvaluationRequest(EvaluationBaseRequest):
    source_text: str = Field(..., min_length=1, max_length=4000)
    answer: str = Field(..., min_length=1, max_length=4000)
    learning_language: str = Field(..., min_length=2, max_length=20)
    accepted_variants: dict[str, list[str]] = Field(default_factory=dict)


class InterpretationEvaluationRequest(EvaluationBaseRequest):
    source_text: str = Field(..., min_length=1, max_length=4000)
    reference_meanings: list[str] = Field(..., min_length=2, max_length=3)
    key_meaning_units: list[str] = Field(..., min_length=1, max_length=30)
    answer: str = Field(..., min_length=1, max_length=4000)
    origin_language: str = Field(..., min_length=2, max_length=20)
    learning_language: str = Field(..., min_length=2, max_length=20)


class ComprehensionEvaluationRequest(EvaluationBaseRequest):
    question: str = Field(..., min_length=1, max_length=2000)
    options: list[ListeningChoiceOption] = Field(..., min_length=2, max_length=4)
    selected_option_key: str = Field(..., min_length=1, max_length=12)
    correct_option_key: str = Field(..., min_length=1, max_length=12)
    comprehension_focus: str = Field(
        ..., pattern="^(GIST|DETAIL|INTENT|INFERENCE|NEXT_ACTION)$"
    )
    origin_language: str = Field(..., min_length=2, max_length=20)
    learning_language: str = Field(..., min_length=2, max_length=20)

    @model_validator(mode="after")
    def validate_options(self) -> "ComprehensionEvaluationRequest":
        keys = [option.key for option in self.options]
        if len(keys) != len(set(keys)):
            raise ValueError("Listening 객관식 option key는 중복될 수 없습니다.")
        if self.correct_option_key not in set(keys):
            raise ValueError("correctOptionKey는 options에 포함되어야 합니다.")
        if self.selected_option_key not in set(keys):
            raise ValueError("selectedOptionKey는 options에 포함되어야 합니다.")
        return self


class SummaryEvaluationRequest(EvaluationBaseRequest):
    source_text: str = Field(..., min_length=1, max_length=4000)
    summary_key_points: list[str] = Field(..., min_length=2, max_length=8)
    answer: str = Field(..., min_length=1, max_length=4000)
    origin_language: str = Field(..., min_length=2, max_length=20)
    learning_language: str = Field(..., min_length=2, max_length=20)


class RepeatEvaluationContext(EvaluationBaseRequest):
    source_text: str = Field(..., min_length=1, max_length=4000)
    source_duration_seconds: float = Field(..., gt=0, le=60)
    learning_language: str = Field(..., min_length=2, max_length=20)
    phrase_hints: list[str] = Field(default_factory=list, max_length=30)


class MetricEvidence(CamelCaseModel):
    start_ms: int | None = Field(default=None, ge=0)
    end_ms: int | None = Field(default=None, ge=0)
    reference: str | None = Field(default=None, max_length=1000)
    recognized: str | None = Field(default=None, max_length=1000)
    metric: str
    severity: str = Field(default="INFO", pattern="^(INFO|LOW|MEDIUM|HIGH)$")
    feedback: str = Field(..., min_length=1, max_length=2000)

    @model_validator(mode="after")
    def validate_timestamp_range(self) -> "MetricEvidence":
        if (
            self.start_ms is not None
            and self.end_ms is not None
            and self.end_ms < self.start_ms
        ):
            raise ValueError("endMs는 startMs 이상이어야 합니다.")
        return self


class AlignmentEntry(CamelCaseModel):
    source: str | None = None
    answer: str | None = None
    status: AlignmentStatus
    source_index: int | None = Field(default=None, ge=0)
    answer_index: int | None = Field(default=None, ge=0)
    start_ms: int | None = Field(default=None, ge=0)
    end_ms: int | None = Field(default=None, ge=0)


class ListeningMetric(CamelCaseModel):
    type: str
    state: MetricEvaluationState = MetricEvaluationState.EVALUATED
    score: float | None = Field(default=None, ge=0, le=100)
    weight: float = Field(..., gt=0, le=1)
    confidence: float = Field(..., ge=0, le=1)
    evidence: list[MetricEvidence] = Field(default_factory=list, max_length=100)
    not_evaluable_reason: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def validate_evaluability(self) -> "ListeningMetric":
        if self.state == MetricEvaluationState.EVALUATED and self.score is None:
            raise ValueError("EVALUATED Metric에는 score가 필요합니다.")
        if self.state == MetricEvaluationState.NOT_EVALUABLE:
            if self.score is not None:
                raise ValueError("NOT_EVALUABLE Metric에는 score를 둘 수 없습니다.")
            if not self.not_evaluable_reason:
                raise ValueError("NOT_EVALUABLE Metric에는 사유가 필요합니다.")
        return self


class ListeningProfileSignal(CamelCaseModel):
    metric: ProfileMetric
    score: float = Field(..., ge=0, le=100)
    confidence: float = Field(..., ge=0, le=1)
    evidence_weight: float = Field(..., gt=0, le=1)
    evidence_ids: list[str] = Field(default_factory=list, max_length=50)
    source_task: ListeningTaskType
    policy_version: str = "listening-profile-v1"


class ListeningTaskResult(CamelCaseModel):
    task_type: ListeningTaskType
    status: ListeningTaskStatus
    evaluable: bool
    score: int | None = Field(default=None, ge=0, le=100)
    confidence: float | None = Field(default=None, ge=0, le=1)
    reason_code: str | None = None
    assistance_level: ListeningAssistanceLevel
    metrics: list[ListeningMetric] = Field(default_factory=list)
    alignment: list[AlignmentEntry] = Field(default_factory=list)
    evidence: list[MetricEvidence] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list, max_length=20)
    improvements: list[str] = Field(default_factory=list, max_length=20)
    recommended_interpretations: list[str] = Field(default_factory=list, max_length=3)
    delivered_meaning_units: list[str] = Field(default_factory=list, max_length=30)
    omitted_meaning_units: list[str] = Field(default_factory=list, max_length=30)
    misunderstood_meaning_units: list[str] = Field(default_factory=list, max_length=30)
    added_information: list[str] = Field(default_factory=list, max_length=30)
    profile_signals: list[ListeningProfileSignal] = Field(default_factory=list)
    profile_eligible: bool = False
    assistance_usage: list[AssistanceUsage] = Field(default_factory=list)
    debug_metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_status(self) -> "ListeningTaskResult":
        if not self.evaluable and self.score is not None:
            raise ValueError("evaluable=false일 때 score는 null이어야 합니다.")
        if self.status != ListeningTaskStatus.EVALUATED and self.score is not None:
            raise ValueError("평가 완료가 아닌 Task에는 score를 둘 수 없습니다.")
        if not self.profile_eligible and self.profile_signals:
            raise ValueError("Profile 비대상 Task에는 Signal을 둘 수 없습니다.")
        if self.status == ListeningTaskStatus.NOT_SELECTED:
            if self.metrics or self.evidence or self.profile_signals:
                raise ValueError("NOT_SELECTED Task에는 평가 결과를 둘 수 없습니다.")
        return self


class ListeningEvaluationOverall(CamelCaseModel):
    score: int | None = Field(default=None, ge=0, le=100)
    evaluated_task_count: int = Field(..., ge=0, le=5)
    total_task_count: int = Field(default=5, ge=5, le=5)


class ListeningEvaluationResponse(CamelCaseModel):
    request_id: str
    item_id: int | str
    attempt_id: int | str
    evaluation_version: str
    scoring_policy_version: str
    profile_policy_version: str
    tasks: list[ListeningTaskResult] = Field(..., min_length=5, max_length=5)
    overall: ListeningEvaluationOverall
    usage: ListeningUsage

    @model_validator(mode="after")
    def validate_task_set(self) -> "ListeningEvaluationResponse":
        types = [task.task_type for task in self.tasks]
        if len(types) != len(set(types)) or set(types) != set(ListeningTaskType):
            raise ValueError("tasks는 Listening Task 전체를 중복 없이 포함해야 합니다.")
        return self


class InterpretationMetricPayload(CamelCaseModel):
    type: InterpretationMetricType
    score: float = Field(..., ge=0, le=100)
    confidence: float = Field(..., ge=0, le=1)
    evidence: list[MetricEvidence] = Field(..., min_length=1, max_length=30)


class InterpretationEvaluationPayload(CamelCaseModel):
    evaluation_confidence: float = Field(..., ge=0, le=1)
    metrics: list[InterpretationMetricPayload] = Field(..., min_length=3, max_length=3)
    delivered_meaning_units: list[str] = Field(default_factory=list, max_length=30)
    omitted_meaning_units: list[str] = Field(default_factory=list, max_length=30)
    misunderstood_meaning_units: list[str] = Field(default_factory=list, max_length=30)
    added_information: list[str] = Field(default_factory=list, max_length=30)
    recommended_interpretations: list[str] = Field(..., min_length=2, max_length=3)
    strengths: list[str] = Field(default_factory=list, max_length=20)
    improvements: list[str] = Field(default_factory=list, max_length=20)


class SummaryMetricPayload(CamelCaseModel):
    type: SummaryMetricType
    score: float = Field(..., ge=0, le=100)
    confidence: float = Field(..., ge=0, le=1)
    evidence: list[MetricEvidence] = Field(..., min_length=1, max_length=30)


class SummaryEvaluationPayload(CamelCaseModel):
    evaluation_confidence: float = Field(..., ge=0, le=1)
    metrics: list[SummaryMetricPayload] = Field(..., min_length=3, max_length=3)
    strengths: list[str] = Field(default_factory=list, max_length=20)
    improvements: list[str] = Field(default_factory=list, max_length=20)
    recommended_summaries: list[str] = Field(..., min_length=2, max_length=3)
    delivered_key_points: list[str] = Field(default_factory=list, max_length=30)
    omitted_key_points: list[str] = Field(default_factory=list, max_length=30)


class RecommendationEvidenceSummary(CamelCaseModel):
    sources: list[str] = Field(..., min_length=1, max_length=10)
    count: int = Field(..., ge=1)
    recent_average: float = Field(..., ge=0, le=100)


class RecommendationExplanationRequest(CamelCaseModel):
    request_id: str = Field(..., min_length=1, max_length=100)
    idempotency_key: str = Field(..., min_length=1, max_length=200)
    type: str = Field(default="RECOMMENDATION", pattern="^RECOMMENDATION$")
    target_metric: ProfileMetric
    recommended_activity: str = Field(..., min_length=1, max_length=100)
    recommended_task: str = Field(..., min_length=1, max_length=100)
    evidence_summary: RecommendationEvidenceSummary
    origin_language: str = Field(..., min_length=2, max_length=20)
    policy_version: str = Field(default="listening-profile-v1", max_length=100)
    model_config_version: str = Field(
        default="listening-model-config-v1", max_length=100
    )


class RecommendationExplanationPayload(CamelCaseModel):
    explanation: str = Field(..., min_length=1, max_length=1000)
    cta_label: str = Field(..., min_length=1, max_length=80)


class RecommendationExplanationResponse(CamelCaseModel):
    request_id: str
    target_metric: ProfileMetric
    recommended_activity: str
    recommended_task: str
    explanation: str
    cta_label: str
    explanation_version: str
    usage: ListeningUsage
