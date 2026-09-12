"""Safe, machine-readable reviewer failures. Never retain prompt/response bodies."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import httpx
from pydantic import ValidationError


@dataclass(frozen=True)
class ReviewFailure:
    code: str
    category: str  # DEPENDENCY, PROTOCOL, SEMANTIC, CONFIGURATION, INTERNAL, CANCELLED
    retryable: bool = True
    status_code: int | None = None
    exception_type: str | None = None
    schema_errors: tuple[tuple[str, str], ...] = ()
    retry_after_seconds: float | None = None

    def fields(self) -> dict[str, Any]:
        return {
            "failure_code": self.code, "failure_category": self.category,
            "retryable": self.retryable, "status_code": self.status_code,
            "exception_type": self.exception_type,
            "schema_errors": [{"field": field, "type": kind} for field, kind in self.schema_errors],
            "retry_after_ms": (round(self.retry_after_seconds * 1000, 2)
                               if self.retry_after_seconds is not None else None),
        }


class WritingProviderConfigurationError(RuntimeError):
    """Permanent provider/auth/schema-routing error; never silently accepted."""


class WritingVerificationUnavailableError(RuntimeError):
    """Required review exhausted its own budget, NOT a content-quality rejection."""

    def __init__(self, task: str, failure: ReviewFailure, attempts: int) -> None:
        super().__init__("Writing verification could not produce a trustworthy result")
        self.task = task
        self.failure = failure
        self.attempts = attempts


# ValidationError locations can contain attacker-controlled extra-property names.
# Only schema-owned tokens/indices are logged; input, ctx and msg are never logged.
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
        code = "VERIFIER_INVALID_CONFIDENCE"
    elif fields & {"estimatedBand", "estimated_band", "alternativeBand", "alternative_band"}:
        code = "VERIFIER_INVALID_BAND"
    elif fields & {"candidateId", "candidate_id"}:
        code = "VERIFIER_IDENTITY_MISMATCH"
    elif fields & {"contentHash", "content_hash"}:
        code = "VERIFIER_CONTENT_HASH_MISMATCH"
    elif "verdict" in fields:
        code = "VERIFIER_INVALID_VERDICT"
    elif any(not path for path in paths):
        code = "VERIFIER_INCONSISTENT_RESULT"
    else:
        code = "VERIFIER_SCHEMA_INVALID"
    safe_errors = []
    for error in errors[:8]:
        parts = [str(part) if (isinstance(part, int) or part in _SAFE_FIELDS) else "<extra>"
                 for part in error["loc"]]
        safe_errors.append((".".join(parts) or "<root>", str(error["type"])[:60]))
    return ReviewFailure(code, "PROTOCOL", exception_type="ValidationError", schema_errors=tuple(safe_errors))


def _status_code(exc: Exception) -> int | None:
    values = (getattr(exc, "status_code", None),
              getattr(getattr(exc, "response", None), "status_code", None),
              getattr(exc, "code", None))  # google.genai APIError uses code.
    return next((value for value in values if type(value) is int and 100 <= value <= 599), None)


def retry_after_seconds(exc: Exception) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is None:
        headers = getattr(exc, "headers", None)
    if not hasattr(headers, "get"):
        return None
    try:
        milliseconds = headers.get("retry-after-ms")
        seconds = headers.get("retry-after")
        if milliseconds is not None:
            value = float(milliseconds) / 1000.0
        elif seconds is not None:
            try:
                value = float(seconds)
            except (ValueError, TypeError):
                date = parsedate_to_datetime(str(seconds))
                if date.tzinfo is None:
                    date = date.replace(tzinfo=timezone.utc)
                value = (date - datetime.now(timezone.utc)).total_seconds()
        else:
            return None
        return max(0.0, value) if math.isfinite(value) else None
    except (ValueError, TypeError, OverflowError, AttributeError):
        return None


def classify_exception(exc: Exception) -> ReviewFailure:
    name = type(exc).__name__
    names = {cls.__name__ for cls in type(exc).__mro__}
    status = _status_code(exc)
    detail = {"status_code": status, "exception_type": name[:80]}
    if isinstance(exc, ValidationError):
        return validation_failure(exc)
    # OpenAI adapter exposes enum-like reasons without exposing raw provider output.
    reason = getattr(exc, "reason_code", None)
    reasons = {
        "OUTPUT_TOKEN_LIMIT": "VERIFIER_OUTPUT_TOKEN_LIMIT",
        "RESPONSE_INCOMPLETE": "VERIFIER_RESPONSE_INCOMPLETE",
        "REFUSAL": "VERIFIER_PROVIDER_REFUSAL",
        "EMPTY_OUTPUT": "VERIFIER_EMPTY_RESPONSE",
        "JSON_INVALID": "VERIFIER_JSON_INVALID",
    }
    if isinstance(reason, str) and reason in reasons:
        return ReviewFailure(reasons[reason], "PROTOCOL",
                             retryable=getattr(exc, "retryable", True) is not False, **detail)
    if isinstance(exc, (TimeoutError, httpx.TimeoutException)) or names & {
        "APITimeoutError", "TimeoutException", "ConnectTimeout", "ReadTimeout", "WriteTimeout", "PoolTimeout",
    }:
        return ReviewFailure("VERIFIER_TIMEOUT", "DEPENDENCY", **detail)
    if isinstance(exc, (ConnectionError, httpx.TransportError)) or names & {
        "APIConnectionError", "ConnectError", "ReadError", "WriteError", "RemoteProtocolError",
    }:
        return ReviewFailure("VERIFIER_CONNECTION_ERROR", "DEPENDENCY", **detail)
    if status == 429:
        return ReviewFailure("VERIFIER_RATE_LIMITED", "DEPENDENCY", retry_after_seconds=retry_after_seconds(exc), **detail)
    if status in {408, 409} or (status is not None and status >= 500):
        return ReviewFailure("VERIFIER_PROVIDER_ERROR", "DEPENDENCY", retry_after_seconds=retry_after_seconds(exc), **detail)
    if status is not None and 400 <= status < 500:
        return ReviewFailure("VERIFIER_PROVIDER_CONFIGURATION", "CONFIGURATION", retryable=False, **detail)
    if isinstance(exc, ValueError):
        # Adapters may raise JSONDecodeError/ValueError before returning structured data.
        return ReviewFailure("VERIFIER_JSON_INVALID" if name == "JSONDecodeError" else "VERIFIER_INVALID_RESULT",
                             "PROTOCOL", **detail)
    return ReviewFailure("VERIFIER_UNEXPECTED_ERROR", "INTERNAL", retryable=False, **detail)
