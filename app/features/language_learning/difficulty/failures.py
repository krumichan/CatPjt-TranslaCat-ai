"""Service-neutral stage failure classification and safe validation diagnostics."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Collection, cast

import httpx
from pydantic import ValidationError


@dataclass(frozen=True)
class StageFailure:
    code: str
    category: str
    retryable: bool = True
    status_code: int | None = None
    exception_type: str | None = None
    schema_errors: tuple[tuple[str, str], ...] = ()
    retry_after_seconds: float | None = None

    def fields(self) -> dict[str, Any]:
        return {
            "failure_code": self.code,
            "failure_category": self.category,
            "retryable": self.retryable,
            "status_code": self.status_code,
            "exception_type": self.exception_type,
            "schema_errors": [
                {"field": field, "type": kind} for field, kind in self.schema_errors
            ],
            "retry_after_ms": (
                round(self.retry_after_seconds * 1000, 2)
                if self.retry_after_seconds is not None
                else None
            ),
        }


class StageConfigurationError(RuntimeError):
    """Permanent provider/auth/schema-routing failure."""


class RequiredStageUnavailableError(RuntimeError):
    """A required stage exhausted its budget without a trustworthy result."""

    def __init__(self, task: str, failure: StageFailure, attempts: int) -> None:
        super().__init__("Required stage could not produce a trustworthy result")
        self.task = task
        self.failure = failure
        self.attempts = attempts


def validation_failure(
    exc: ValidationError,
    *,
    code: str,
    safe_fields: Collection[str],
) -> StageFailure:
    errors = exc.errors(include_input=False, include_context=False, include_url=False)
    safe_errors = []
    for error in errors[:8]:
        parts = [
            str(part) if (isinstance(part, int) or part in safe_fields) else "<extra>"
            for part in error["loc"]
        ]
        safe_errors.append((".".join(parts) or "<root>", str(error["type"])[:60]))
    return StageFailure(
        code,
        "PROTOCOL",
        exception_type="ValidationError",
        schema_errors=tuple(safe_errors),
    )


def _status_code(exc: Exception) -> int | None:
    values = (
        getattr(exc, "status_code", None),
        getattr(getattr(exc, "response", None), "status_code", None),
        getattr(exc, "code", None),
    )
    return next(
        (value for value in values if type(value) is int and 100 <= value <= 599),
        None,
    )


def retry_after_seconds(exc: Exception) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is None:
        headers = getattr(exc, "headers", None)
    get_header = getattr(headers, "get", None)
    if not callable(get_header):
        return None
    try:
        milliseconds = get_header("retry-after-ms")
        seconds = get_header("retry-after")
        if milliseconds is not None:
            value = float(cast(Any, milliseconds)) / 1000.0
        elif seconds is not None:
            try:
                value = float(cast(Any, seconds))
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


def classify_exception(
    exc: Exception,
    *,
    code_namespace: str,
    safe_fields: Collection[str] = (),
) -> StageFailure:
    name = type(exc).__name__
    names = {cls.__name__ for cls in type(exc).__mro__}
    status = _status_code(exc)
    detail = {"status_code": status, "exception_type": name[:80]}
    if isinstance(exc, ValidationError):
        return validation_failure(
            exc, code=f"{code_namespace}_SCHEMA_INVALID", safe_fields=safe_fields,
        )
    reason = getattr(exc, "reason_code", None)
    reasons = {
        "OUTPUT_TOKEN_LIMIT": "OUTPUT_TOKEN_LIMIT",
        "RESPONSE_INCOMPLETE": "RESPONSE_INCOMPLETE",
        "REFUSAL": "PROVIDER_REFUSAL",
        "EMPTY_OUTPUT": "EMPTY_RESPONSE",
        "JSON_INVALID": "JSON_INVALID",
    }
    if isinstance(reason, str) and reason in reasons:
        return StageFailure(
            f"{code_namespace}_{reasons[reason]}",
            "PROTOCOL",
            retryable=getattr(exc, "retryable", True) is not False,
            **detail,
        )
    if isinstance(exc, (TimeoutError, httpx.TimeoutException)) or names & {
        "APITimeoutError", "TimeoutException", "ConnectTimeout", "ReadTimeout",
        "WriteTimeout", "PoolTimeout",
    }:
        return StageFailure(f"{code_namespace}_TIMEOUT", "DEPENDENCY", **detail)
    if isinstance(exc, (ConnectionError, httpx.TransportError)) or names & {
        "APIConnectionError", "ConnectError", "ReadError", "WriteError", "RemoteProtocolError",
    }:
        return StageFailure(f"{code_namespace}_CONNECTION_ERROR", "DEPENDENCY", **detail)
    if status == 429:
        return StageFailure(
            f"{code_namespace}_RATE_LIMITED",
            "DEPENDENCY",
            retry_after_seconds=retry_after_seconds(exc),
            **detail,
        )
    if status in {408, 409} or (status is not None and status >= 500):
        return StageFailure(
            f"{code_namespace}_PROVIDER_ERROR",
            "DEPENDENCY",
            retry_after_seconds=retry_after_seconds(exc),
            **detail,
        )
    if status is not None and 400 <= status < 500:
        return StageFailure(
            f"{code_namespace}_PROVIDER_CONFIGURATION",
            "CONFIGURATION",
            retryable=False,
            **detail,
        )
    if isinstance(exc, ValueError):
        suffix = "JSON_INVALID" if name == "JSONDecodeError" else "INVALID_RESULT"
        return StageFailure(f"{code_namespace}_{suffix}", "PROTOCOL", **detail)
    return StageFailure(
        f"{code_namespace}_UNEXPECTED_ERROR",
        "INTERNAL",
        retryable=False,
        **detail,
    )
