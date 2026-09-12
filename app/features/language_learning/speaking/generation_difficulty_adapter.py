from __future__ import annotations

from collections.abc import Callable

from app.features.language_learning.difficulty.contracts import (
    DifficultyTarget,
    DifficultyValidationResult,
)
from app.features.language_learning.speaking.generation_difficulty_spec import (
    SPEAKING_GENERATION_DIFFICULTY_SPEC_VERSION,
    SpeakingGenerationDifficultySpec,
    SpeakingGenerationDifficultyTargetValue,
)
from app.schemas.language_learning_speaking import ConversationGenerationRequest


def build_speaking_generation_difficulty_spec(
    request: ConversationGenerationRequest,
) -> SpeakingGenerationDifficultySpec:
    return SpeakingGenerationDifficultySpec(
        target=DifficultyTarget(
            service="speaking-generation",
            scale="speaking-target-level-and-practice-mode",
            value=SpeakingGenerationDifficultyTargetValue(
                target_level=request.target_level,
                practice_mode=request.practice_mode.value,
            ),
            policy_version=SPEAKING_GENERATION_DIFFICULTY_SPEC_VERSION,
        )
    )


def project_speaking_generation_validation(
    spec: SpeakingGenerationDifficultySpec,
    validator: Callable[[], None],
) -> DifficultyValidationResult:
    try:
        validator()
    except ValueError as exc:
        return DifficultyValidationResult.reject(str(exc))
    return DifficultyValidationResult.accept(
        measurements={
            "targetLevel": spec.target.value.target_level,
            "practiceMode": spec.target.value.practice_mode,
        }
    )
