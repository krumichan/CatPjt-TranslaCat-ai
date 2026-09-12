"""Writing compatibility façade over the common bounded stage runtime."""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from app.ai.model_policy import get_task_model_policy
from app.ai.ports import TextGenerationProvider
from app.features.language_learning.difficulty.diagnostics import (
    emit_structured_event,
    safe_identifier,
)
from app.features.language_learning.difficulty.runtime import (
    BoundedStageRuntime,
    StageContext,
    StageEventNames,
    StageMetrics,
)
from app.features.language_learning.writing.review_diagnostics import (
    ReviewFailure,
    WritingProviderConfigurationError,
    WritingVerificationUnavailableError,
    classify_exception,
)

logger = logging.getLogger(__name__)


def emit_event(event: str, *, level: int = logging.INFO, **fields: Any) -> None:
    """Preserve the existing writing event schema and LogRecord attribute."""
    emit_structured_event(
        logger,
        event,
        level=level,
        schema_version=1,
        record_attribute="writing_event",
        identifier_fields=(
            "request_id", "candidate_id", "model", "provider", "provider_route",
        ),
        **fields,
    )


@dataclass(frozen=True)
class ReviewContext(StageContext):
    """Writing-facing context retained while Common owns field mechanics."""


@dataclass
class ReviewMetrics(StageMetrics):
    """Writing diagnostic projection with the original summary key contract."""

    elapsed_ms: Counter[str] = field(default_factory=Counter)

    def summary(self) -> dict[str, Any]:
        return {
            "review_calls": dict(self.calls),
            "review_elapsed_ms": {
                task: round(value, 2) for task, value in self.elapsed_ms.items()
            },
            "review_failures": dict(self.failures),
            "review_retries": dict(self.retries),
            "review_backoff_ms": round(self.backoff_ms, 2),
            "peak_inflight_reviews": self.peak_active,
        }


def _writing_result_projection(result: BaseModel | None) -> dict[str, Any]:
    return {
        "verdict": getattr(result, "verdict", None),
        "confidence": getattr(result, "confidence", None),
        "estimated_band": getattr(result, "estimated_band", None),
        "difficulty_confidence": getattr(result, "difficulty_confidence", None),
        "difficulty_status": getattr(result, "difficulty_status", None),
        "confidence_used_for_acceptance": False,
    }


def _bind_writing_identity(schema: dict[str, Any], context: StageContext) -> None:
    for name, value in (
        ("candidateId", context.candidate_id),
        ("contentHash", context.content_hash),
    ):
        if name in schema.get("properties", {}):
            schema["properties"][name]["enum"] = [value]


class ReviewRuntime(BoundedStageRuntime):
    def __init__(
        self,
        provider: TextGenerationProvider,
        timeout_seconds: float,
        *,
        deadline: float | None = None,
        retry_base_seconds: float = 0.25,
        max_retry_delay_seconds: float = 5.0,
    ) -> None:
        super().__init__(
            provider,
            timeout_seconds,
            deadline=deadline,
            retry_base_seconds=retry_base_seconds,
            max_retry_delay_seconds=max_retry_delay_seconds,
            failure_classifier=classify_exception,
            event_emitter=emit_event,
            stage_metadata=lambda task: {
                "tier": get_task_model_policy(task).tier.value,
                "provider_route": getattr(self.provider, "provider_name", "unknown"),
            },
            event_names=StageEventNames(
                started="writing.verifier.started",
                finished="writing.verifier.finished",
                retry_wait="writing.verifier.retry_wait",
                exhausted="writing.verifier.exhausted",
            ),
            failure_code_namespace="VERIFIER",
            deadline_message="Writing request deadline expired before review",
            configuration_error_message="Writing reviewer configuration failure",
            timeout_error_message="Review timeout must be positive and finite",
            backoff_error_message="Review backoff must be nonnegative and finite",
            attempts_error_message="Review stage permits one initial call and at most one retry",
            configuration_error_type=WritingProviderConfigurationError,
            unavailable_error_type=WritingVerificationUnavailableError,
            failure_factory=ReviewFailure,
            metrics_factory=ReviewMetrics,
            result_projection=_writing_result_projection,
            schema_binding=_bind_writing_identity,
            attempts_validator=lambda max_attempts: max_attempts in {1, 2},
        )


__all__ = [
    "ReviewContext",
    "ReviewMetrics",
    "ReviewRuntime",
    "emit_event",
    "safe_identifier",
]
