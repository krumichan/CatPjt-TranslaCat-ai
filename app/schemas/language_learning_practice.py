from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.alias_generators import to_camel


class CamelCaseModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        str_strip_whitespace=True,
        extra="forbid",
    )


class PracticeDomain(str, Enum):
    READING = "READING"
    VOCABULARY = "VOCABULARY"


class ReadingMode(str, Enum):
    COMPREHENSION = "COMPREHENSION"
    STRUCTURE = "STRUCTURE"
    CONTEXT_INFERENCE = "CONTEXT_INFERENCE"


class VocabularyMode(str, Enum):
    CONTEXTUAL_CHOICE = "CONTEXTUAL_CHOICE"
    MEANING_RELATION = "MEANING_RELATION"
    USAGE_DISTINCTION = "USAGE_DISTINCTION"
    COMPOSITION = "COMPOSITION"


class PracticeQuestionType(str, Enum):
    SINGLE_CHOICE = "SINGLE_CHOICE"
    ORDERING = "ORDERING"


class PracticeDifficulty(str, Enum):
    EASIER = "EASIER"
    CURRENT = "CURRENT"
    CHALLENGE = "CHALLENGE"


class ReadingSkill(str, Enum):
    CONTENT = "CONTENT"
    DETAIL = "DETAIL"
    INFERENCE = "INFERENCE"
    GIST = "GIST"
    STRUCTURE = "STRUCTURE"
    CONTEXT_INFERENCE = "CONTEXT_INFERENCE"


class VocabularySkill(str, Enum):
    MEANING = "MEANING"
    NUANCE = "NUANCE"
    SYNONYM = "SYNONYM"
    ANTONYM = "ANTONYM"
    DISTINCTION = "DISTINCTION"
    COLLOCATION = "COLLOCATION"
    REGISTER = "REGISTER"
    PRAGMATIC_FIT = "PRAGMATIC_FIT"
    CONTEXT_USAGE = "CONTEXT_USAGE"
    COMPOSITION = "COMPOSITION"


class PracticeOption(CamelCaseModel):
    key: str = Field(..., min_length=1, max_length=40)
    text: str = Field(..., min_length=1, max_length=1000)


class PracticeReviewTarget(CamelCaseModel):
    canonical_key: str = Field(..., min_length=1, max_length=200)
    expression: str = Field(..., min_length=1, max_length=300)
    mastery_score: float | None = Field(default=None, ge=0, le=100)
    wrong_count: int = Field(default=0, ge=0)
    previous_question_types: list[PracticeQuestionType] = Field(
        default_factory=list,
        max_length=10,
    )
    preferred_skill: Literal[
        "MEANING",
        "COLLOCATION",
        "NUANCE",
        "REGISTER",
        "PRAGMATIC_FIT",
    ] | None = None


class VocabularyPlanAnchorType(str, Enum):
    SELECTED_KEYWORD = "SELECTED_KEYWORD"
    WEAK_SIGNAL = "WEAK_SIGNAL"
    RECENT_MISTAKE = "RECENT_MISTAKE"
    LEARNING_PROFILE = "LEARNING_PROFILE"


class VocabularyPlanItem(CamelCaseModel):
    global_order: int = Field(..., ge=1, le=10)
    review_target: bool
    target_expression: str = Field(..., min_length=1, max_length=300)
    canonical_key: str = Field(..., min_length=1, max_length=200)
    distractors: list[str] = Field(..., min_length=3, max_length=3)
    skill_tag: Literal[
        "MEANING",
        "COLLOCATION",
        "NUANCE",
        "REGISTER",
        "PRAGMATIC_FIT",
    ]
    difficulty: PracticeDifficulty
    complexity_band: int = Field(..., ge=1, le=5)
    scenario_family: str | None = Field(default=None, min_length=1, max_length=80)
    anchor_type: VocabularyPlanAnchorType | None = None
    anchor_value: str | None = Field(default=None, min_length=1, max_length=300)

    @model_validator(mode="after")
    def validate_binding_shape(self) -> "VocabularyPlanItem":
        if self.review_target:
            if any(
                value is not None
                for value in (self.scenario_family, self.anchor_type, self.anchor_value)
            ):
                raise ValueError("review plan items must not contain new-item metadata")
        elif any(
            value is None
            for value in (self.scenario_family, self.anchor_type, self.anchor_value)
        ):
            raise ValueError("new plan items require scenarioFamily and anchor metadata")
        return self


class PersonalizedVocabularyPlan(CamelCaseModel):
    version: str = Field(..., min_length=1, max_length=100)
    items: list[VocabularyPlanItem] = Field(..., min_length=10, max_length=10)

    @model_validator(mode="after")
    def validate_daily_shape(self) -> "PersonalizedVocabularyPlan":
        if [item.global_order for item in self.items] != list(range(1, 11)):
            raise ValueError("vocabularyPlan items must use contiguous globalOrder 1..10")
        if sum(item.review_target for item in self.items) > 2:
            raise ValueError("vocabularyPlan review count must not exceed 2")
        difficulty_counts = {
            difficulty: sum(item.difficulty == difficulty for item in self.items)
            for difficulty in PracticeDifficulty
        }
        if difficulty_counts != {
            PracticeDifficulty.EASIER: 2,
            PracticeDifficulty.CURRENT: 6,
            PracticeDifficulty.CHALLENGE: 2,
        }:
            raise ValueError("vocabularyPlan difficulty mix must be 2/6/2")
        expected_skills = {"MEANING", "COLLOCATION", "NUANCE", "REGISTER", "PRAGMATIC_FIT"}
        if {
            skill: sum(item.skill_tag == skill for item in self.items)
            for skill in expected_skills
        } != {skill: 2 for skill in expected_skills}:
            raise ValueError("vocabularyPlan must contain each skill exactly twice")
        return self


class PracticeGenerationRequest(CamelCaseModel):
    request_id: str = Field(..., min_length=1, max_length=120)
    domain: PracticeDomain
    mode: str = Field(..., min_length=1, max_length=40)
    origin_language: str = Field(..., min_length=2, max_length=20)
    learning_language: str = Field(..., min_length=2, max_length=20)
    question_count: int = Field(..., ge=1, le=20)
    complexity_band: int = Field(..., ge=1, le=5)
    easier_count: int = Field(default=0, ge=0, le=20)
    current_count: int = Field(default=0, ge=0, le=20)
    challenge_count: int = Field(default=0, ge=0, le=20)
    selected_keywords: list[str] = Field(default_factory=list, max_length=30)
    weak_signals: list[str] = Field(default_factory=list, max_length=30)
    recent_mistakes: list[str] = Field(default_factory=list, max_length=30)
    review_targets: list[PracticeReviewTarget] = Field(default_factory=list, max_length=30)
    review_question_count: int = Field(default=0, ge=0, le=20)
    generation_date: date
    previous_questions: list[PracticeGeneratedQuestion] = Field(
        default_factory=list,
        max_length=9,
        description=(
            "Previously committed questions in global order. For single-question requests, "
            "this prefix preserves the daily skill plan, passages and vocabulary uniqueness. "
            "The generated response still uses local order starting at 1."
        ),
    )
    vocabulary_plan: PersonalizedVocabularyPlan | None = None
    vocabulary_plan_only: bool = False

    @property
    def question_offset(self) -> int:
        return len(self.previous_questions)

    @model_validator(mode="after")
    def validate_contract(self) -> "PracticeGenerationRequest":
        if self.easier_count + self.current_count + self.challenge_count != self.question_count:
            raise ValueError("difficulty mix must equal questionCount")
        if self.domain == PracticeDomain.READING:
            ReadingMode(self.mode)
            if self.question_count not in (1, 5):
                raise ValueError("Reading questionCount must be 1 or 5")
            target_count = 5
        else:
            VocabularyMode(self.mode)
            if self.question_count not in (1, 10):
                raise ValueError("Vocabulary questionCount must be 1 or 10")
            target_count = 10
            if self.review_question_count > min(self.question_count, len(self.review_targets)):
                raise ValueError("reviewQuestionCount exceeds available reviewTargets")
            if (
                self.mode == VocabularyMode.CONTEXTUAL_CHOICE.value
                and self.review_question_count > 2
            ):
                raise ValueError("CONTEXTUAL_CHOICE reviewQuestionCount must not exceed 2")
        if self.previous_questions and self.question_count != 1:
            raise ValueError("previousQuestions requires single-question generation")
        if self.question_offset + self.question_count > target_count:
            raise ValueError("previousQuestions exceeds the daily question target")
        if [question.order for question in self.previous_questions] != list(
            range(1, self.question_offset + 1)
        ):
            raise ValueError("previousQuestions must be a contiguous global-order prefix")
        if self.domain == PracticeDomain.READING:
            passages: dict[str, str] = {}
            for question in self.previous_questions:
                expected_passage = "p1" if question.order <= 3 else "p2"
                if question.passage_id != expected_passage or not question.passage_text:
                    raise ValueError("previousQuestions must retain the planned Reading passages")
                if passages.setdefault(expected_passage, question.passage_text) != question.passage_text:
                    raise ValueError("previousQuestions must reuse identical passageText")
        elif any(
            not question.canonical_key or not question.target_expression
            for question in self.previous_questions
        ):
            raise ValueError("previousQuestions requires vocabulary identities")
        if self.vocabulary_plan is not None and (
            self.domain != PracticeDomain.VOCABULARY
            or self.mode != VocabularyMode.CONTEXTUAL_CHOICE.value
        ):
            raise ValueError("vocabularyPlan is only supported for CONTEXTUAL_CHOICE")
        if self.vocabulary_plan_only and (
            self.domain != PracticeDomain.VOCABULARY
            or self.mode != VocabularyMode.CONTEXTUAL_CHOICE.value
            or self.vocabulary_plan is not None
            or self.previous_questions
        ):
            raise ValueError(
                "vocabularyPlanOnly requires an initial CONTEXTUAL_CHOICE request"
            )
        return self


class PracticeGeneratedQuestion(CamelCaseModel):
    order: int = Field(..., ge=1, le=20)
    question_type: PracticeQuestionType
    difficulty: PracticeDifficulty
    complexity_band: int = Field(..., ge=1, le=5)
    passage_id: str | None = Field(default=None, max_length=80)
    passage_text: str | None = Field(default=None, max_length=12000)
    prompt: str = Field(..., min_length=1, max_length=3000)
    options: list[PracticeOption] = Field(..., min_length=3, max_length=12)
    correct_answer: list[str] = Field(..., min_length=1, max_length=12)
    skill_tag: str = Field(..., min_length=1, max_length=60)
    evidence_text: str | None = Field(default=None, max_length=3000)
    explanation_origin: str = Field(..., min_length=1, max_length=3000)
    explanation_learning: str = Field(..., min_length=1, max_length=3000)
    target_expression: str | None = Field(default=None, max_length=300)
    canonical_key: str | None = Field(default=None, max_length=200)
    review_target: bool = False
    vocabulary_candidates: list[str] = Field(default_factory=list, max_length=3)


class PracticeGenerationResponse(CamelCaseModel):
    request_id: str
    prompt_version: str
    domain: PracticeDomain
    mode: str
    complexity_band: int = Field(..., ge=1, le=5)
    questions: list[PracticeGeneratedQuestion]
    vocabulary_plan: PersonalizedVocabularyPlan | None = None
