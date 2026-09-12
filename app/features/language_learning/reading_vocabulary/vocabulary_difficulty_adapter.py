from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from app.features.language_learning.difficulty.contracts import (
    DifficultyAcceptanceDecision,
    DifficultyTarget,
    DifficultyValidationResult,
    SemanticAssessment,
)
from app.features.language_learning.reading_vocabulary.vocabulary_difficulty_spec import (
    VOCABULARY_DIFFICULTY_SPEC_VERSION,
    VocabularyDifficultySpec,
    VocabularyDifficultyTargetValue,
)
from app.schemas.language_learning_practice import VocabularyMode


def build_vocabulary_difficulty_spec(
    *,
    mode: str,
    difficulty: str,
    complexity_band: int,
    skill_tag: str,
    question_type: str,
    target_expression: str,
) -> VocabularyDifficultySpec:
    return VocabularyDifficultySpec(
        target=DifficultyTarget(
            service="vocabulary",
            scale=f"vocabulary-{mode.lower().replace('_', '-')}",
            value=VocabularyDifficultyTargetValue(
                mode=mode,
                difficulty=difficulty,
                complexity_band=complexity_band,
                skill_tag=skill_tag,
            ),
            policy_version=VOCABULARY_DIFFICULTY_SPEC_VERSION,
        ),
        question_type=question_type,
        target_expression=target_expression,
    )


def project_vocabulary_validation(
    spec: VocabularyDifficultySpec,
    validator: Callable[[], None],
) -> DifficultyValidationResult:
    try:
        validator()
    except ValueError as exc:
        return DifficultyValidationResult.reject(str(exc))
    return DifficultyValidationResult.accept(
        measurements={
            "mode": spec.target.value.mode,
            "complexityBand": spec.target.value.complexity_band,
            "skillTag": spec.target.value.skill_tag,
        }
    )


@dataclass(frozen=True)
class VocabularySemanticAssessment(SemanticAssessment[VocabularyDifficultyTargetValue]):
    best_answer_key: str = ""
    ambiguous: bool = False
    supported: bool = True
    mode_fit: bool = True
    answer_leakage: bool = False
    context_dependent: bool = True
    distractors_plausible: bool = True


def normalize_vocabulary_semantic_assessment(
    *,
    best_answer_key: str,
    ambiguous: bool,
    supported: bool,
    mode_fit: bool,
    answer_leakage: bool,
    context_dependent: bool,
    distractors_plausible: bool,
) -> VocabularySemanticAssessment:
    rejected = (
        ambiguous
        or not supported
        or not mode_fit
        or answer_leakage
        or not distractors_plausible
    )
    return VocabularySemanticAssessment(
        status="REJECT" if rejected else "PASS",
        difficulty_status="NOT_ASSESSED",
        best_answer_key=best_answer_key,
        ambiguous=ambiguous,
        supported=supported,
        mode_fit=mode_fit,
        answer_leakage=answer_leakage,
        context_dependent=context_dependent,
        distractors_plausible=distractors_plausible,
    )


@dataclass(frozen=True)
class VocabularyAcceptanceContext:
    expected_answer_key: str


class VocabularySemanticQualityPolicy(Protocol):
    def decide(
        self,
        *,
        assessment: VocabularySemanticAssessment,
        context: VocabularyAcceptanceContext,
    ) -> DifficultyAcceptanceDecision: ...


def _base_vocabulary_decision(
    assessment: VocabularySemanticAssessment,
    context: VocabularyAcceptanceContext,
    *,
    require_context: bool,
) -> DifficultyAcceptanceDecision:
    reason: str | None = None
    if assessment.ambiguous:
        reason = "ambiguous single-choice item"
    elif not assessment.supported:
        reason = "answer is not sufficiently supported"
    elif not assessment.mode_fit:
        reason = "question does not fit requested mode/skill"
    elif assessment.answer_leakage:
        reason = "semantic verifier detected answer leakage"
    elif not assessment.distractors_plausible:
        reason = "distractors are too weak or unrelated"
    elif require_context and not assessment.context_dependent:
        reason = "USAGE_DISTINCTION does not require context"
    elif assessment.best_answer_key != context.expected_answer_key:
        reason = (
            f"answer mismatch expected={context.expected_answer_key} "
            f"verifier={assessment.best_answer_key}"
        )
    if reason is not None:
        return DifficultyAcceptanceDecision("REJECT", reason)
    return DifficultyAcceptanceDecision("ACCEPT", "semantic verifier accepted candidate")


@dataclass(frozen=True)
class MeaningRelationSemanticQualityPolicy:
    def decide(
        self,
        *,
        assessment: VocabularySemanticAssessment,
        context: VocabularyAcceptanceContext,
    ) -> DifficultyAcceptanceDecision:
        return _base_vocabulary_decision(assessment, context, require_context=False)


@dataclass(frozen=True)
class UsageDistinctionSemanticQualityPolicy:
    def decide(
        self,
        *,
        assessment: VocabularySemanticAssessment,
        context: VocabularyAcceptanceContext,
    ) -> DifficultyAcceptanceDecision:
        return _base_vocabulary_decision(assessment, context, require_context=True)


@dataclass(frozen=True)
class CompositionSemanticQualityPolicy:
    def decide(
        self,
        *,
        assessment: VocabularySemanticAssessment,
        context: VocabularyAcceptanceContext,
    ) -> DifficultyAcceptanceDecision:
        return _base_vocabulary_decision(assessment, context, require_context=False)


def semantic_quality_policy_for_vocabulary_mode(
    mode: str,
) -> VocabularySemanticQualityPolicy:
    if mode == VocabularyMode.MEANING_RELATION.value:
        return MeaningRelationSemanticQualityPolicy()
    if mode == VocabularyMode.USAGE_DISTINCTION.value:
        return UsageDistinctionSemanticQualityPolicy()
    if mode == VocabularyMode.COMPOSITION.value:
        return CompositionSemanticQualityPolicy()
    raise ValueError(f"unsupported vocabulary mode: {mode}")
