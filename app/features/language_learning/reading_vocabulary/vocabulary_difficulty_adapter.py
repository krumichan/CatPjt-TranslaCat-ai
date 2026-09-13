from __future__ import annotations

import math
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
from app.features.language_learning.reading_vocabulary.vocabulary_difficulty_recipe import (
    validate_vocabulary_difficulty_recipe,
    vocabulary_difficulty_recipe,
)
from app.schemas.language_learning_practice import VocabularyMode


VOCABULARY_DIFFICULTY_ISSUE_CODES = (
    "MEANING_CONTRAST",
    "SEMANTIC_DISTANCE",
    "DISTRACTOR_PROXIMITY",
    "VISIBLE_CUE_INTEGRATION",
    "CONTEXT_DEPENDENCE",
    "REGISTER_DEPENDENCE",
    "COLLOCATION_DEPENDENCE",
    "PRAGMATIC_DEPENDENCE",
    "CONSTRAINT_INTEGRATION",
    "DEPENDENCY_SPAN",
    "CLAUSE_RELATION",
    "AMBIGUOUS_OR_NO_SINGLE_ANSWER",
    "INSUFFICIENT_VISIBLE_EVIDENCE",
    "EXTERNAL_KNOWLEDGE_DEPENDENCE",
    "BORDERLINE_ADJACENT_BANDS",
    "DEFINITION_MATCHING",
    "SEMANTIC_NEIGHBORHOOD_REPETITION",
)
_VOCABULARY_DIFFICULTY_ISSUE_CODE_SET = frozenset(
    VOCABULARY_DIFFICULTY_ISSUE_CODES
)
def build_vocabulary_difficulty_spec(
    *,
    mode: str,
    difficulty: str,
    complexity_band: int,
    skill_tag: str,
    question_type: str,
    target_expression: str,
    usage_intent: str | None = None,
) -> VocabularyDifficultySpec:
    recipe = vocabulary_difficulty_recipe(
        mode=mode,
        band=complexity_band,
        skill_tag=skill_tag,
        question_type=question_type,
        usage_intent=usage_intent,
    )
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
        usage_intent=usage_intent,
        recipe=recipe,
    )


def project_vocabulary_validation(
    spec: VocabularyDifficultySpec,
    validator: Callable[[], None],
) -> DifficultyValidationResult:
    try:
        validate_vocabulary_difficulty_recipe(
            mode=spec.target.value.mode,
            band=spec.target.value.complexity_band,
            skill_tag=spec.target.value.skill_tag,
            question_type=spec.question_type,
            usage_intent=spec.usage_intent,
            recipe=spec.recipe,
        )
        validator()
    except ValueError as exc:
        return DifficultyValidationResult.reject(str(exc))
    return DifficultyValidationResult.accept(
        measurements={
            "mode": spec.target.value.mode,
            "complexityBand": spec.target.value.complexity_band,
            "skillTag": spec.target.value.skill_tag,
            "questionType": spec.question_type,
            "recipeVersion": spec.recipe.version,
        }
    )


@dataclass(frozen=True, kw_only=True)
class VocabularySemanticDifficultyAssessment(SemanticAssessment[int]):
    """Blind Vocabulary difficulty classification for observational comparison."""


def normalize_vocabulary_semantic_difficulty_assessment(
    raw: object,
    *,
    allowed_evidence_refs: set[str] | None = None,
) -> VocabularySemanticDifficultyAssessment:
    if not isinstance(raw, dict):
        return VocabularySemanticDifficultyAssessment(
            status="UNSURE",
            difficulty_status="NOT_ASSESSED",
        )

    status = str(raw.get("difficultyStatus", "NOT_ASSESSED")).upper()
    observed = _difficulty_band(raw.get("observedBand"))
    alternative = _difficulty_band(raw.get("alternativeBand"))
    valid = (
        status == "ASSESSED"
        and observed is not None
        and alternative is None
    ) or (
        status == "BORDERLINE"
        and observed is not None
        and alternative is not None
        and abs(observed - alternative) == 1
    ) or (
        status in {"UNSURE", "NOT_ASSESSED"}
        and observed is None
        and alternative is None
    )
    if not valid:
        status = "NOT_ASSESSED"
        observed = None
        alternative = None

    raw_issue_codes = raw.get("issueCodes", [])
    issue_codes = (
        [
            value
            for value in raw_issue_codes
            if isinstance(value, str)
            and value in _VOCABULARY_DIFFICULTY_ISSUE_CODE_SET
        ]
        if isinstance(raw_issue_codes, list)
        else []
    )
    raw_evidence_refs = raw.get("evidenceSegmentIds", [])
    evidence_refs = (
        tuple(
            dict.fromkeys(
                value
                for value in raw_evidence_refs
                if isinstance(value, str)
                and value
                and (allowed_evidence_refs is None or value in allowed_evidence_refs)
            )
        )
        if isinstance(raw_evidence_refs, list)
        else ()
    )
    confidence = raw.get("difficultyConfidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0 <= confidence <= 1
    ):
        confidence = None
    return VocabularySemanticDifficultyAssessment(
        status="UNSURE" if status in {"UNSURE", "NOT_ASSESSED"} else "PASS",
        difficulty_status=status,
        observed_target=observed,
        alternative_target=alternative,
        issue_codes=tuple(dict.fromkeys(issue_codes)),
        difficulty_confidence=confidence,
        evidence_refs=evidence_refs,
    )


@dataclass(frozen=True)
class VocabularySemanticDifficultyPolicy:
    """Compare blind observations without granting quality or retry authority."""

    def decide(
        self,
        *,
        requested_band: int,
        assessment: VocabularySemanticDifficultyAssessment,
    ) -> DifficultyAcceptanceDecision:
        if assessment.difficulty_status == "ASSESSED":
            if assessment.observed_target == requested_band:
                return DifficultyAcceptanceDecision(
                    "SHADOW_MATCH", "observed band matches target"
                )
            return DifficultyAcceptanceDecision(
                "SHADOW_MISMATCH", "observed band does not match target"
            )
        if assessment.difficulty_status == "BORDERLINE":
            if requested_band in {
                assessment.observed_target,
                assessment.alternative_target,
            }:
                return DifficultyAcceptanceDecision(
                    "SHADOW_BORDERLINE_MATCH", "target is one adjacent observed band"
                )
            return DifficultyAcceptanceDecision(
                "SHADOW_MISMATCH", "target is outside adjacent observed bands"
            )
        return DifficultyAcceptanceDecision(
            "SHADOW_NOT_ASSESSED", "difficulty classifier was unsure"
        )


def _difficulty_band(value: object) -> int | None:
    return value if type(value) is int and 1 <= value <= 5 else None


@dataclass(frozen=True)
class VocabularySemanticAssessment(SemanticAssessment[VocabularyDifficultyTargetValue]):
    best_answer_key: str = ""
    ambiguous: bool = False
    supported: bool = True
    mode_fit: bool = True
    answer_leakage: bool = False
    context_dependent: bool = True
    distractors_plausible: bool = True
    definition_like: bool = False
    lexical_concept_repeated: bool = False
    genuine_competitor_keys: tuple[str, ...] = ()


def normalize_vocabulary_semantic_assessment(
    *,
    best_answer_key: str,
    ambiguous: bool,
    supported: bool,
    mode_fit: bool,
    answer_leakage: bool,
    context_dependent: bool,
    distractors_plausible: bool,
    definition_like: bool = False,
    lexical_concept_repeated: bool = False,
    genuine_competitor_keys: tuple[str, ...] = (),
) -> VocabularySemanticAssessment:
    rejected = (
        ambiguous
        or not supported
        or not mode_fit
        or answer_leakage
        or not distractors_plausible
        or definition_like
        or lexical_concept_repeated
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
        definition_like=definition_like,
        lexical_concept_repeated=lexical_concept_repeated,
        genuine_competitor_keys=genuine_competitor_keys,
    )


@dataclass(frozen=True)
class VocabularyAcceptanceContext:
    expected_answer_key: str
    minimum_close_distractors: int = 0
    review_target: bool = False


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
    context_failure_reason: str = "USAGE_DISTINCTION does not require context",
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
        reason = context_failure_reason
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
        return _base_vocabulary_decision(
            assessment,
            context,
            require_context=False,
        )


@dataclass(frozen=True)
class UsageDistinctionSemanticQualityPolicy:
    def decide(
        self,
        *,
        assessment: VocabularySemanticAssessment,
        context: VocabularyAcceptanceContext,
    ) -> DifficultyAcceptanceDecision:
        return _base_vocabulary_decision(
            assessment,
            context,
            require_context=True,
        )


@dataclass(frozen=True)
class CompositionSemanticQualityPolicy:
    def decide(
        self,
        *,
        assessment: VocabularySemanticAssessment,
        context: VocabularyAcceptanceContext,
    ) -> DifficultyAcceptanceDecision:
        return _base_vocabulary_decision(
            assessment,
            context,
            require_context=False,
        )


@dataclass(frozen=True)
class ContextualChoiceSemanticQualityPolicy:
    def decide(
        self,
        *,
        assessment: VocabularySemanticAssessment,
        context: VocabularyAcceptanceContext,
    ) -> DifficultyAcceptanceDecision:
        reason: str | None = None
        if assessment.ambiguous:
            reason = "ambiguous single-choice item"
        elif not assessment.supported:
            reason = "answer is not sufficiently supported"
        elif not assessment.mode_fit:
            reason = "question does not fit requested skill"
        elif assessment.best_answer_key != context.expected_answer_key:
            reason = (
                f"answer mismatch expected={context.expected_answer_key} "
                f"verifier={assessment.best_answer_key}"
            )
        elif assessment.definition_like:
            reason = "contextual choice collapsed into definition matching"
        elif assessment.lexical_concept_repeated and not context.review_target:
            reason = "new item repeats a same-day lexical concept"
        elif (
            len(assessment.genuine_competitor_keys)
            < context.minimum_close_distractors
        ):
            reason = "insufficient genuine distractor competitors"
        if reason is not None:
            return DifficultyAcceptanceDecision("REJECT", reason)
        return DifficultyAcceptanceDecision(
            "ACCEPT",
            (
                "semantic verifier accepted review candidate"
                if context.review_target
                else "semantic verifier accepted contextual choice candidate"
            ),
        )


def semantic_quality_policy_for_vocabulary_mode(
    mode: str,
) -> VocabularySemanticQualityPolicy:
    if mode == VocabularyMode.CONTEXTUAL_CHOICE.value:
        return ContextualChoiceSemanticQualityPolicy()
    if mode == VocabularyMode.MEANING_RELATION.value:
        return MeaningRelationSemanticQualityPolicy()
    if mode == VocabularyMode.USAGE_DISTINCTION.value:
        return UsageDistinctionSemanticQualityPolicy()
    if mode == VocabularyMode.COMPOSITION.value:
        return CompositionSemanticQualityPolicy()
    raise ValueError(f"unsupported vocabulary mode: {mode}")
