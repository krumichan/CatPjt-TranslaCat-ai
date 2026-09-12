from __future__ import annotations

from collections.abc import Callable

from app.features.language_learning.difficulty.contracts import (
    DifficultyTarget,
    DifficultyValidationResult,
)
from app.features.language_learning.listening.difficulty_spec import (
    LISTENING_DIFFICULTY_SPEC_VERSION,
    ListeningDifficultySpec,
    ListeningDifficultyTargetValue,
)
from app.features.language_learning.quality import (
    LANGUAGE_COMPLEXITY_POLICY_VERSION,
    resolve_listening_complexity_band,
)
from app.schemas.language_learning_listening import (
    GeneratedListeningItemPayload,
    ListeningSetGenerationRequest,
)


def build_listening_difficulty_spec(
    request: ListeningSetGenerationRequest,
    *,
    duration_range: tuple[float, float],
) -> ListeningDifficultySpec:
    minimum, maximum = duration_range
    value = ListeningDifficultyTargetValue(
        text_band=resolve_listening_complexity_band(
            request.set_context.difficulty.value,
            request.language_complexity,
        ),
        mode=request.set_context.learning_mode.value,
    )
    return ListeningDifficultySpec(
        target=DifficultyTarget(
            service="listening",
            scale="listening-generation",
            value=value,
            policy_version=LANGUAGE_COMPLEXITY_POLICY_VERSION,
        ),
        duration_min_seconds=minimum,
        duration_max_seconds=maximum,
        version=LISTENING_DIFFICULTY_SPEC_VERSION,
    )


def validate_listening_candidate(
    request: ListeningSetGenerationRequest,
    item: GeneratedListeningItemPayload,
    *,
    duration_range_resolver: Callable[[], tuple[float, float]],
    mode_payload_validator: Callable[[], bool],
) -> tuple[ListeningDifficultySpec | None, DifficultyValidationResult]:
    """Project the existing ordered Listening checks into the common result contract."""
    if item.diversity_metadata is None or item.language_complexity_band is None:
        return None, DifficultyValidationResult.reject(
            "LISTENING_REQUIRED_GENERATION_METADATA_MISSING"
        )

    expected_band = resolve_listening_complexity_band(
        request.set_context.difficulty.value,
        request.language_complexity,
    )
    if item.language_complexity_band != expected_band or not item.safety.passed:
        return None, DifficultyValidationResult.reject(
            "LISTENING_TEXT_BAND_OR_SAFETY_REJECTED",
            measurements={
                "expectedTextBand": expected_band,
                "observedTextBand": item.language_complexity_band,
                "safetyPassed": item.safety.passed,
            },
        )

    duration_range = duration_range_resolver()
    spec = build_listening_difficulty_spec(request, duration_range=duration_range)
    target = spec.target.value
    if not spec.duration_min_seconds <= item.estimated_audio_seconds <= spec.duration_max_seconds:
        return spec, DifficultyValidationResult.reject(
            "LISTENING_DURATION_OUT_OF_PROFILE",
            measurements={
                "estimatedAudioSeconds": item.estimated_audio_seconds,
                "durationMinSeconds": spec.duration_min_seconds,
                "durationMaxSeconds": spec.duration_max_seconds,
            },
        )
    if not mode_payload_validator():
        return spec, DifficultyValidationResult.reject(
            "LISTENING_MODE_PAYLOAD_INVALID",
            measurements={"mode": target.mode},
        )
    return spec, DifficultyValidationResult.accept(
        measurements={
            "textBand": target.text_band,
            "estimatedAudioSeconds": item.estimated_audio_seconds,
            "mode": target.mode,
        }
    )
