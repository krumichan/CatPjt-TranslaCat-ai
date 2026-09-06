from __future__ import annotations

from enum import Enum
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.alias_generators import to_camel

from app.schemas.language_learning import WritingMetric
from app.schemas.language_learning_quality import (
    CommunicativeIntent,
    DiversityContext,
    DiversityMetadata,
    DiversitySummary,
    ScenarioCategory,
)


class CamelCaseModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        str_strip_whitespace=True,
        extra="forbid",
    )


class LevelTestDomain(str, Enum):
    VOCABULARY = "VOCABULARY"
    GRAMMAR = "GRAMMAR"
    READING = "READING"
    LISTENING = "LISTENING"
    WRITING = "WRITING"
    SPEAKING = "SPEAKING"


class LevelTestItemType(str, Enum):
    VOCAB_CONTEXT_CHOICE = "VOCAB_CONTEXT_CHOICE"
    VOCAB_PARAPHRASE_CHOICE = "VOCAB_PARAPHRASE_CHOICE"
    GRAMMAR_FORM_CHOICE = "GRAMMAR_FORM_CHOICE"
    GRAMMAR_SENTENCE_ORDER = "GRAMMAR_SENTENCE_ORDER"
    READING_GIST = "READING_GIST"
    READING_DETAIL = "READING_DETAIL"
    READING_DISCOURSE_FUNCTION = "READING_DISCOURSE_FUNCTION"
    READING_TEXT_INFERENCE = "READING_TEXT_INFERENCE"
    LISTENING_GIST_CHOICE = "LISTENING_GIST_CHOICE"
    LISTENING_DETAIL_CHOICE = "LISTENING_DETAIL_CHOICE"
    LISTENING_DICTATION = "LISTENING_DICTATION"
    LISTENING_INTERPRETATION = "LISTENING_INTERPRETATION"
    WRITING_TRANSLATION = "WRITING_TRANSLATION"
    WRITING_GUIDED_SENTENCE = "WRITING_GUIDED_SENTENCE"
    WRITING_SCENARIO_RESPONSE = "WRITING_SCENARIO_RESPONSE"
    WRITING_SHORT_PARAGRAPH = "WRITING_SHORT_PARAGRAPH"
    SPEAKING_REPEAT = "SPEAKING_REPEAT"
    SPEAKING_GUIDED_RESPONSE = "SPEAKING_GUIDED_RESPONSE"
    SPEAKING_SHORT_RESPONSE = "SPEAKING_SHORT_RESPONSE"


class LevelTestAnswerMode(str, Enum):
    CHOICE = "CHOICE"
    TEXT = "TEXT"
    AUDIO = "AUDIO"


class LevelTestMetricState(str, Enum):
    EVALUATED = "EVALUATED"
    NOT_EVALUABLE = "NOT_EVALUABLE"


class LevelTestSpeakingTaskResponseStatus(str, Enum):
    FULFILLED = "FULFILLED"
    PARTIAL = "PARTIAL"
    META_REFUSAL = "META_REFUSAL"
    OFF_TOPIC = "OFF_TOPIC"
    EMPTY_CONTENT = "EMPTY_CONTENT"


class LevelTestPreviousResult(CamelCaseModel):
    question_number: int = Field(..., ge=1, le=20)
    domain: LevelTestDomain
    item_type: LevelTestItemType
    score: int | None = Field(default=None, ge=0, le=100)
    complexity_band: int = Field(..., ge=1, le=5)
    evaluable: bool = True


class LevelTestReferenceAudioUpload(CamelCaseModel):
    upload_url: str = Field(..., min_length=1, max_length=12000)
    object_key: str = Field(..., min_length=1, max_length=1000)
    content_type: str = Field(default="audio/wav", min_length=1, max_length=100)
    voice: str = Field(default="Kore", min_length=1, max_length=100)
    playback_speed: str = Field(default="NORMAL", min_length=1, max_length=30)


class LevelTestReferenceAudio(CamelCaseModel):
    object_key: str = Field(..., min_length=1, max_length=1000)
    content_type: str = Field(..., min_length=1, max_length=100)
    duration_ms: int | None = Field(default=None, ge=0)
    checksum_sha256: str = Field(..., pattern="^[0-9a-f]{64}$")


class LevelTestQuestionGenerationRequest(CamelCaseModel):
    request_id: str = Field(..., min_length=1, max_length=100)
    idempotency_key: str = Field(..., min_length=1, max_length=200)
    session_id: int | str
    question_number: int = Field(..., ge=1, le=20)
    total_questions: int = Field(default=20, ge=20, le=20)
    domain: LevelTestDomain
    item_type: LevelTestItemType
    origin_language: str = Field(..., min_length=2, max_length=20)
    learning_language: str = Field(..., min_length=2, max_length=20)
    target_complexity_band: int = Field(..., ge=1, le=5)
    previous_results: list[LevelTestPreviousResult] = Field(default_factory=list, max_length=19)
    diversity_context: DiversityContext = Field(default_factory=DiversityContext)
    preferred_scenario_categories: list[ScenarioCategory] = Field(default_factory=list, max_length=4)
    policy_version: str = Field(default="level-test-v2-multiskill", min_length=1, max_length=100)
    model_config_version: str = Field(default="level-test-model-config-v1", min_length=1, max_length=100)
    reference_audio_upload: LevelTestReferenceAudioUpload | None = None

    @model_validator(mode="after")
    def validate_progress(self) -> "LevelTestQuestionGenerationRequest":
        if len(self.previous_results) >= self.question_number:
            raise ValueError("previousResults 개수는 questionNumber보다 작아야 합니다.")
        return self


class LevelTestOption(CamelCaseModel):
    key: str = Field(..., pattern="^[A-Z][A-Z0-9_-]{0,15}$")
    text: str = Field(..., min_length=1, max_length=2000)


class LevelTestChoiceSelectionPolicy(str, Enum):
    UNIQUE_ANSWER = "UNIQUE_ANSWER"
    BEST_ANSWER = "BEST_ANSWER"


class LevelTestInternalAnswerKey(CamelCaseModel):
    correct_option_key: str | None = Field(default=None, pattern="^[A-D]$")
    correct_order: list[str] = Field(default_factory=list, max_length=30)


class LevelTestScoredInternalAnswerKey(LevelTestInternalAnswerKey):
    """Server-authored deterministic scoring metadata for accepted Choice items."""

    selection_policy: LevelTestChoiceSelectionPolicy | None = None
    option_scores: dict[str, int] = Field(default_factory=dict)


class LevelTestChoiceQualityAudit(CamelCaseModel):
    # Legacy generation-side self-audit. Kept for backward compatibility only.
    # Server acceptance does not trust this field; semantic verification is independent.
    unique_correct_option: bool
    directly_compatible_option_keys: list[str] = Field(default_factory=list, max_length=4)


class LevelTestReferencePayload(TypedDict):
    """Provider-visible structured reference fields.

    All keys are required in the provider schema so Gemini sees the concrete shape.
    The normalizer fills neutral defaults before Pydantic validation; domain-specific
    validators still require non-empty values where the item type needs them.
    """

    sourceText: str | None
    referenceMeanings: list[str]
    keyMeaningUnits: list[str]
    referenceText: str | None
    translationSourceText: str | None
    emphasisText: str | None
    readingPassage: str | None
    readingQuestion: str | None
    listeningQuestion: str | None
    providedFacts: list[str]
    requiredIntents: list[str]
    responseConstraints: list[str]


class LevelTestBestOptionAdvantage(str, Enum):
    CLEAR = "CLEAR"
    WEAK = "WEAK"
    NONE = "NONE"


class LevelTestChoiceSemanticVerificationPayload(CamelCaseModel):
    verifiable: bool
    plausible_option_keys: list[str] = Field(default_factory=list, max_length=4)
    near_equivalent_option_keys: list[str] = Field(default_factory=list, max_length=4)
    best_option_key: Literal["A", "B", "C", "D"]
    best_option_advantage: LevelTestBestOptionAdvantage


class LevelTestTaskSufficiencyVerificationPayload(CamelCaseModel):
    sufficient: bool
    requires_external_knowledge: bool
    requires_problem_solving: bool
    provided_facts_sufficient: bool
    communicative_goals_clear: bool
    instruction_and_task_roles_separated: bool
    missing_information: list[str] = Field(default_factory=list, max_length=6)


class LevelTestVocabContextDesign(CamelCaseModel):
    """Small semantic contract for target-first vocabulary generation.

    Only information that must be decided before learner-facing wording is kept
    here. Presentation metadata and distractor wording belong to the generation
    stage, which materially reduces provider-schema failure surface.
    """

    design_id: Literal["A", "B"]
    target_expression: str = Field(..., min_length=1, max_length=120)
    target_meaning: str = Field(..., min_length=1, max_length=300)
    semantic_constraint: str = Field(..., min_length=1, max_length=800)
    scenario_category: ScenarioCategory
    communicative_intent: CommunicativeIntent


class LevelTestVocabContextDesignPayload(CamelCaseModel):
    designs: list[LevelTestVocabContextDesign] = Field(..., min_length=2, max_length=2)

    @model_validator(mode="after")
    def validate_designs(self) -> "LevelTestVocabContextDesignPayload":
        if {design.design_id for design in self.designs} != {"A", "B"}:
            raise ValueError("VOCAB_CONTEXT_CHOICE designId는 A/B를 정확히 한 번씩 포함해야 합니다.")
        normalized_targets = {
            "".join(design.target_expression.casefold().split())
            for design in self.designs
        }
        if len(normalized_targets) != len(self.designs):
            raise ValueError("VOCAB_CONTEXT_CHOICE 설계의 targetExpression은 서로 달라야 합니다.")
        return self


class LevelTestQuestionCandidate(CamelCaseModel):
    domain: LevelTestDomain
    item_type: LevelTestItemType
    complexity_band: int = Field(..., ge=1, le=5)
    instruction: str = Field(..., min_length=1, max_length=2000)
    instruction_language: str = Field(..., min_length=2, max_length=20)
    answer_mode: LevelTestAnswerMode
    answer_language: str | None = Field(default=None, min_length=2, max_length=20)
    prompt_text: str = Field(..., min_length=1, max_length=6000)
    options: list[LevelTestOption] = Field(default_factory=list, max_length=30)
    internal_answer_key: LevelTestInternalAnswerKey = Field(default_factory=LevelTestInternalAnswerKey)
    choice_quality_audit: LevelTestChoiceQualityAudit | None = None
    generation_plan_id: str | None = Field(default=None, pattern="^[AB]$")
    reference_payload: LevelTestReferencePayload
    diversity_metadata: DiversityMetadata
    max_answer_length: int | None = Field(default=None, ge=1, le=10000)
    max_audio_seconds: int | None = Field(default=None, ge=1, le=60)

    @model_validator(mode="after")
    def validate_answer_contract(self) -> "LevelTestQuestionCandidate":
        if self.answer_mode == LevelTestAnswerMode.CHOICE:
            if self.item_type == LevelTestItemType.GRAMMAR_SENTENCE_ORDER:
                if len(self.options) < 2:
                    raise ValueError("Sentence Order에는 최소 2개 Token Option이 필요합니다.")
                keys = [item.key for item in self.options]
                if len(keys) != len(set(keys)):
                    raise ValueError("Sentence Order Option key는 중복될 수 없습니다.")
                if sorted(self.internal_answer_key.correct_order) != sorted(keys):
                    raise ValueError("correctOrder는 Option key를 정확히 한 번씩 포함해야 합니다.")
            else:
                if len(self.options) != 4:
                    raise ValueError("Choice 문제는 정확히 4개의 Option이 필요합니다.")
                keys = {item.key for item in self.options}
                if self.internal_answer_key.correct_option_key not in keys:
                    raise ValueError("Choice 정답 Key가 Option에 존재하지 않습니다.")
        else:
            if self.options:
                raise ValueError("TEXT/AUDIO 문제에는 Choice Option을 둘 수 없습니다.")

        if self.answer_mode == LevelTestAnswerMode.TEXT and not self.answer_language:
            raise ValueError("TEXT 문제에는 answerLanguage가 필요합니다.")
        if self.answer_mode == LevelTestAnswerMode.AUDIO and not self.answer_language:
            raise ValueError("AUDIO 문제에는 answerLanguage가 필요합니다.")
        return self


class LevelTestQuestionGenerationPayload(CamelCaseModel):
    candidates: list[LevelTestQuestionCandidate] = Field(..., min_length=1, max_length=6)


class LevelTestVocabContextRepairPayload(CamelCaseModel):
    candidate: LevelTestQuestionCandidate


class LevelTestUsage(CamelCaseModel):
    latency_ms: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    provider: str | None = None
    model: str | None = None
    prompt_version: str | None = None
    evaluation_version: str | None = None


class LevelTestQuestionGenerationResponse(CamelCaseModel):
    request_id: str
    session_id: int | str
    question_number: int
    total_questions: int = 20
    domain: LevelTestDomain
    item_type: LevelTestItemType
    complexity_band: int
    instruction: str
    instruction_language: str
    answer_mode: LevelTestAnswerMode
    answer_language: str | None = None
    prompt_text: str
    options: list[LevelTestOption]
    internal_answer_key: LevelTestScoredInternalAnswerKey
    reference_payload: LevelTestReferencePayload
    diversity_metadata: DiversityMetadata
    max_answer_length: int | None = None
    max_audio_seconds: int | None = None
    generation_version: str
    prompt_version: str
    diversity_summary: DiversitySummary
    usage: LevelTestUsage
    reference_audio: LevelTestReferenceAudio | None = None


class LevelTestTextEvaluationRequest(CamelCaseModel):
    request_id: str = Field(..., min_length=1, max_length=100)
    idempotency_key: str = Field(..., min_length=1, max_length=200)
    session_id: int | str
    item_id: int | str
    domain: LevelTestDomain
    item_type: LevelTestItemType
    prompt_text: str = Field(..., min_length=1, max_length=6000)
    answer: str = Field(..., min_length=1, max_length=6000)
    origin_language: str = Field(..., min_length=2, max_length=20)
    learning_language: str = Field(..., min_length=2, max_length=20)
    complexity_band: int = Field(..., ge=1, le=5)
    source_text: str | None = Field(default=None, max_length=6000)
    reference_meanings: list[str] = Field(default_factory=list, max_length=3)
    key_meaning_units: list[str] = Field(default_factory=list, max_length=30)
    translation_source_text: str | None = Field(default=None, max_length=6000)
    provided_facts: list[str] = Field(default_factory=list, max_length=12)
    required_intents: list[str] = Field(default_factory=list, max_length=12)
    response_constraints: list[str] = Field(default_factory=list, max_length=12)
    focus_metrics: list[WritingMetric] = Field(default_factory=list, max_length=5)
    manual_retry_attempt: int = Field(default=0, ge=0, le=1)

    @model_validator(mode="after")
    def validate_writing_contract(self) -> "LevelTestTextEvaluationRequest":
        if self.domain != LevelTestDomain.WRITING:
            return self
        if self.item_type == LevelTestItemType.WRITING_TRANSLATION:
            if not self.translation_source_text or not self.translation_source_text.strip():
                raise ValueError("WRITING_TRANSLATION 평가에는 translationSourceText가 필요합니다.")
            return self
        if self.item_type in {
            LevelTestItemType.WRITING_GUIDED_SENTENCE,
            LevelTestItemType.WRITING_SCENARIO_RESPONSE,
            LevelTestItemType.WRITING_SHORT_PARAGRAPH,
        } and (
            not self.provided_facts
            or not self.required_intents
            or not self.response_constraints
        ):
            raise ValueError("Guided Writing 평가에는 task guidance가 필요합니다.")
        return self


class LevelTestMetricResult(CamelCaseModel):
    type: str = Field(..., min_length=1, max_length=100)
    state: LevelTestMetricState = LevelTestMetricState.EVALUATED
    score: float | None = Field(default=None, ge=0, le=100)
    confidence: float = Field(..., ge=0, le=1)
    summary: str | None = Field(default=None, max_length=2000)
    evidence: list[dict[str, Any]] = Field(default_factory=list, max_length=50)
    not_evaluable_reason: str | None = Field(default=None, max_length=1000)


class LevelTestAssessmentSignal(CamelCaseModel):
    domain: LevelTestDomain
    metric: str = Field(..., min_length=1, max_length=100)
    score: float = Field(..., ge=0, le=100)
    confidence: float = Field(..., ge=0, le=1)


class LevelTestFeedbackDetail(CamelCaseModel):
    category: str = Field(..., min_length=1, max_length=100)
    severity: Literal["INFO", "STRENGTH", "IMPROVEMENT", "CORRECTION", "OMISSION"] = "INFO"
    original: str | None = Field(default=None, max_length=2000)
    corrected: str | None = Field(default=None, max_length=2000)
    explanation: str = Field(..., min_length=1, max_length=3000)


class LevelTestEvaluationResponse(CamelCaseModel):
    request_id: str
    session_id: int | str
    item_id: int | str
    domain: LevelTestDomain
    item_type: LevelTestItemType
    evaluable: bool
    score: int | None = Field(default=None, ge=0, le=100)
    confidence: float | None = Field(default=None, ge=0, le=1)
    transcript: str | None = None
    metrics: list[LevelTestMetricResult] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list, max_length=20)
    improvements: list[str] = Field(default_factory=list, max_length=20)
    recommended_answers: list[str] = Field(default_factory=list, max_length=3)
    detailed_feedback: list[LevelTestFeedbackDetail] = Field(default_factory=list, max_length=50)
    assessment_signals: list[LevelTestAssessmentSignal] = Field(default_factory=list, max_length=30)
    reason_code: str | None = None
    evaluation_version: str
    prompt_version: str | None = None
    usage: LevelTestUsage = Field(default_factory=LevelTestUsage)


class LevelTestSpeakingEvaluationContext(CamelCaseModel):
    request_id: str = Field(..., min_length=1, max_length=100)
    idempotency_key: str = Field(..., min_length=1, max_length=200)
    session_id: int | str
    item_id: int | str
    item_type: LevelTestItemType
    prompt_text: str = Field(..., min_length=1, max_length=6000)
    reference_text: str | None = Field(default=None, max_length=6000)
    origin_language: str = Field(..., min_length=2, max_length=20)
    learning_language: str = Field(..., min_length=2, max_length=20)
    complexity_band: int = Field(..., ge=1, le=5)
    max_duration_seconds: int = Field(default=30, ge=3, le=60)
    phrase_hints: list[str] = Field(default_factory=list, max_length=30)
    provided_facts: list[str] = Field(default_factory=list, max_length=12)
    required_intents: list[str] = Field(default_factory=list, max_length=12)
    response_constraints: list[str] = Field(default_factory=list, max_length=12)
    manual_retry_attempt: int = Field(default=0, ge=0, le=1)

    @model_validator(mode="after")
    def validate_speaking_type(self) -> "LevelTestSpeakingEvaluationContext":
        if self.item_type not in {
            LevelTestItemType.SPEAKING_REPEAT,
            LevelTestItemType.SPEAKING_GUIDED_RESPONSE,
            LevelTestItemType.SPEAKING_SHORT_RESPONSE,
        }:
            raise ValueError("Speaking 평가에는 SPEAKING Item Type이 필요합니다.")
        if self.item_type == LevelTestItemType.SPEAKING_REPEAT and not self.reference_text:
            raise ValueError("SPEAKING_REPEAT에는 referenceText가 필요합니다.")
        if self.item_type in {
            LevelTestItemType.SPEAKING_GUIDED_RESPONSE,
            LevelTestItemType.SPEAKING_SHORT_RESPONSE,
        } and (
            not self.provided_facts
            or not self.required_intents
            or not self.response_constraints
        ):
            raise ValueError("Guided Speaking 평가에는 task guidance가 필요합니다.")
        return self


class LevelTestSpeakingMetricPayload(CamelCaseModel):
    type: str = Field(pattern="^(PRONUNCIATION|FLUENCY|GRAMMAR|VOCABULARY|TASK_FULFILLMENT)$")
    state: LevelTestMetricState = LevelTestMetricState.EVALUATED
    score: float | None = Field(default=None, ge=0, le=100)
    confidence: float = Field(..., ge=0, le=1)
    summary: str = Field(..., min_length=1, max_length=2000)
    evidence: list[str] = Field(default_factory=list, max_length=20)
    not_evaluable_reason: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def validate_state(self) -> "LevelTestSpeakingMetricPayload":
        if self.state == LevelTestMetricState.EVALUATED and self.score is None:
            raise ValueError("EVALUATED Speaking Metric에는 score가 필요합니다.")
        if self.state == LevelTestMetricState.NOT_EVALUABLE:
            if self.score is not None or not self.not_evaluable_reason:
                raise ValueError("NOT_EVALUABLE Speaking Metric 계약이 유효하지 않습니다.")
        return self


class LevelTestSpeakingEvaluationPayload(CamelCaseModel):
    evaluation_confidence: float = Field(..., ge=0, le=1)
    task_response_status: LevelTestSpeakingTaskResponseStatus
    metrics: list[LevelTestSpeakingMetricPayload] = Field(..., min_length=5, max_length=5)
    strengths: list[str] = Field(default_factory=list, max_length=20)
    improvements: list[str] = Field(default_factory=list, max_length=20)
    recommended_answers: list[str] = Field(..., max_length=2)

    @model_validator(mode="after")
    def validate_metric_set(self) -> "LevelTestSpeakingEvaluationPayload":
        expected = {"PRONUNCIATION", "FLUENCY", "GRAMMAR", "VOCABULARY", "TASK_FULFILLMENT"}
        actual = [metric.type for metric in self.metrics]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError("Level Test Speaking 5축 Metric이 정확히 필요합니다.")
        task_metric = next(metric for metric in self.metrics if metric.type == "TASK_FULFILLMENT")
        if (
            self.task_response_status
            in {
                LevelTestSpeakingTaskResponseStatus.META_REFUSAL,
                LevelTestSpeakingTaskResponseStatus.OFF_TOPIC,
                LevelTestSpeakingTaskResponseStatus.EMPTY_CONTENT,
            }
            and task_metric.state == LevelTestMetricState.EVALUATED
            and task_metric.score != 0
        ):
            raise ValueError("과제 비응답 Speaking 평가의 TASK_FULFILLMENT score는 0이어야 합니다.")
        return self
