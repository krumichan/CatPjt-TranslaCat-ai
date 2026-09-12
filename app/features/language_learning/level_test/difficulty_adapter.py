from __future__ import annotations

from collections.abc import Callable

from app.features.language_learning.difficulty.contracts import (
    DifficultyTarget,
    DifficultyValidationResult,
)
from app.features.language_learning.level_test.difficulty_spec import (
    LevelTestDifficultySpec,
    LevelTestDifficultyTargetValue,
)
from app.schemas.language_learning_level_test import LevelTestQuestionGenerationRequest


def build_level_test_difficulty_spec(
    request: LevelTestQuestionGenerationRequest,
) -> LevelTestDifficultySpec:
    return LevelTestDifficultySpec(
        target=DifficultyTarget(
            service="level-test",
            scale="level-test-question-generation",
            value=LevelTestDifficultyTargetValue(
                domain=request.domain.value,
                item_type=request.item_type.value,
                complexity_band=request.target_complexity_band,
            ),
            policy_version=request.policy_version,
        ),
        question_number=request.question_number,
    )


def project_level_test_candidate_validation(
    spec: LevelTestDifficultySpec,
    validator: Callable[[], None],
) -> DifficultyValidationResult:
    try:
        validator()
    except ValueError as exc:
        return DifficultyValidationResult.reject(str(exc))
    return DifficultyValidationResult.accept(
        measurements={
            "questionNumber": spec.question_number,
            "domain": spec.target.value.domain,
            "itemType": spec.target.value.item_type,
            "complexityBand": spec.target.value.complexity_band,
        }
    )
