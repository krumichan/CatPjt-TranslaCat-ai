from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from app.features.language_learning.difficulty.contracts import (
    DifficultyAcceptanceDecision,
    DifficultyTarget,
    DifficultyValidationResult,
    SemanticAssessment,
)
from app.features.language_learning.reading_vocabulary.reading_difficulty_spec import (
    READING_DIFFICULTY_SPEC_VERSION,
    ReadingPassageDifficultySpec,
    ReadingPassageDifficultyTargetValue,
    ReadingQuestionDifficultySpec,
    ReadingQuestionDifficultyTargetValue,
)
from app.schemas.language_learning_practice import PracticeGenerationRequest


def build_reading_passage_difficulty_spec(
    request: PracticeGenerationRequest,
    *,
    passage_id: str,
    complexity_band: int,
) -> ReadingPassageDifficultySpec:
    return ReadingPassageDifficultySpec(
        target=DifficultyTarget(
            service="reading",
            scale="reading-passage-generation",
            value=ReadingPassageDifficultyTargetValue(
                complexity_band=complexity_band,
                mode=request.mode,
            ),
            policy_version=READING_DIFFICULTY_SPEC_VERSION,
        ),
        passage_id=passage_id,
    )


def build_reading_question_difficulty_spec(
    *,
    difficulty: str,
    complexity_band: int,
    skill_tag: str,
    passage_id: str,
) -> ReadingQuestionDifficultySpec:
    return ReadingQuestionDifficultySpec(
        target=DifficultyTarget(
            service="reading",
            scale="reading-question-generation",
            value=ReadingQuestionDifficultyTargetValue(
                difficulty=difficulty,
                complexity_band=complexity_band,
                skill_tag=skill_tag,
            ),
            policy_version=READING_DIFFICULTY_SPEC_VERSION,
        ),
        passage_id=passage_id,
    )


def validate_reading_passage(
    spec: ReadingPassageDifficultySpec,
    *,
    observed_passage_id: str,
    passage_text: str,
    language_validator: Callable[[], None],
) -> DifficultyValidationResult:
    if observed_passage_id != spec.passage_id:
        return DifficultyValidationResult.reject("reading passageId mismatch")
    if not passage_text:
        return DifficultyValidationResult.reject("reading passageText is blank")
    try:
        language_validator()
    except ValueError as exc:
        return DifficultyValidationResult.reject(str(exc))
    return DifficultyValidationResult.accept(
        measurements={
            "complexityBand": spec.target.value.complexity_band,
            "passageId": spec.passage_id,
        }
    )


def project_reading_question_validation(
    spec: ReadingQuestionDifficultySpec,
    validator: Callable[[], None],
) -> DifficultyValidationResult:
    try:
        validator()
    except ValueError as exc:
        return DifficultyValidationResult.reject(str(exc))
    return DifficultyValidationResult.accept(
        measurements={
            "complexityBand": spec.target.value.complexity_band,
            "skillTag": spec.target.value.skill_tag,
        }
    )


@dataclass(frozen=True)
class ReadingSemanticAssessment(SemanticAssessment[ReadingQuestionDifficultyTargetValue]):
    best_answer_key: str = ""
    ambiguous: bool = False
    supported: bool = True
    mode_fit: bool = True
    answer_leakage: bool = False
    distractors_plausible: bool = True


def normalize_reading_semantic_assessment(
    *,
    best_answer_key: str,
    ambiguous: bool,
    supported: bool,
    mode_fit: bool,
    answer_leakage: bool,
    distractors_plausible: bool,
) -> ReadingSemanticAssessment:
    rejected = (
        ambiguous
        or not supported
        or not mode_fit
        or answer_leakage
        or not distractors_plausible
    )
    return ReadingSemanticAssessment(
        status="REJECT" if rejected else "PASS",
        difficulty_status="NOT_ASSESSED",
        best_answer_key=best_answer_key,
        ambiguous=ambiguous,
        supported=supported,
        mode_fit=mode_fit,
        answer_leakage=answer_leakage,
        distractors_plausible=distractors_plausible,
    )


@dataclass(frozen=True)
class ReadingAcceptanceContext:
    expected_answer_key: str


@dataclass(frozen=True)
class ReadingSemanticQualityPolicy:
    def decide(
        self,
        *,
        assessment: ReadingSemanticAssessment,
        context: ReadingAcceptanceContext,
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
        elif assessment.best_answer_key != context.expected_answer_key:
            reason = (
                f"answer mismatch expected={context.expected_answer_key} "
                f"verifier={assessment.best_answer_key}"
            )
        if reason is not None:
            return DifficultyAcceptanceDecision("REJECT", reason)
        return DifficultyAcceptanceDecision("ACCEPT", "semantic verifier accepted candidate")
