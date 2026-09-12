"""Writing compatibility façade over service-neutral stage failures."""
from __future__ import annotations

from dataclasses import dataclass

from pydantic import ValidationError

from app.features.language_learning.difficulty.failures import (
    RequiredStageUnavailableError,
    StageConfigurationError,
    StageFailure,
    classify_exception as classify_stage_exception,
    retry_after_seconds,
    validation_failure as classify_validation_failure,
)

@dataclass(frozen=True)
class ReviewFailure(StageFailure):
    """Writing-facing failure type retained for import and repr compatibility."""


class WritingProviderConfigurationError(StageConfigurationError):
    """Permanent provider/auth/schema-routing error; never silently accepted."""


class WritingVerificationUnavailableError(RequiredStageUnavailableError):
    """Required review exhausted its own budget, NOT a content-quality rejection."""

    def __init__(self, task: str, failure: ReviewFailure, attempts: int) -> None:
        super().__init__(task, failure, attempts)
        self.args = ("Writing verification could not produce a trustworthy result",)


_SAFE_FIELDS = frozenset({
    "candidateId", "candidate_id", "contentHash", "content_hash", "confidence",
    "evidence", "field", "quote", "feature", "estimatedBand", "estimated_band",
    "verdict", "observedWritingType", "observed_writing_type", "difficultyConfidence",
    "difficulty_confidence", "issues", "focusReason", "focus_reason",
    "difficultyStatus", "difficulty_status", "alternativeBand", "alternative_band",
    "difficultyEvidenceSegmentIds", "difficulty_evidence_segment_ids", "checks", "criterion",
    "status", "evidenceSegmentIds", "evidence_segment_ids", "items", "originText", "origin_text",
    "recoveryHash", "recovery_hash", "sourcePreservation", "source_preservation",
})


def validation_failure(exc: ValidationError) -> ReviewFailure:
    errors = exc.errors(include_input=False, include_context=False, include_url=False)
    paths = [error["loc"] for error in errors]
    fields = {str(part) for path in paths for part in path if isinstance(part, str)}
    if fields & {"confidence", "difficultyConfidence", "difficulty_confidence"}:
        suffix = "INVALID_CONFIDENCE"
    elif fields & {"estimatedBand", "estimated_band", "alternativeBand", "alternative_band"}:
        suffix = "INVALID_BAND"
    elif fields & {"candidateId", "candidate_id"}:
        suffix = "IDENTITY_MISMATCH"
    elif fields & {"contentHash", "content_hash"}:
        suffix = "CONTENT_HASH_MISMATCH"
    elif "verdict" in fields:
        suffix = "INVALID_VERDICT"
    elif any(not path for path in paths):
        suffix = "INCONSISTENT_RESULT"
    else:
        suffix = "SCHEMA_INVALID"
    failure = classify_validation_failure(
        exc,
        code=f"VERIFIER_{suffix}",
        safe_fields=_SAFE_FIELDS,
    )
    return _as_review_failure(failure)


def _as_review_failure(failure: StageFailure) -> ReviewFailure:
    return ReviewFailure(
        failure.code,
        failure.category,
        failure.retryable,
        failure.status_code,
        failure.exception_type,
        failure.schema_errors,
        failure.retry_after_seconds,
    )


def classify_exception(exc: Exception) -> ReviewFailure:
    if isinstance(exc, ValidationError):
        return validation_failure(exc)
    return _as_review_failure(
        classify_stage_exception(
            exc,
            code_namespace="VERIFIER",
            safe_fields=_SAFE_FIELDS,
        )
    )


__all__ = [
    "ReviewFailure",
    "WritingProviderConfigurationError",
    "WritingVerificationUnavailableError",
    "classify_exception",
    "retry_after_seconds",
    "validation_failure",
]
