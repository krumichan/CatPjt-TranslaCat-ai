"""Request-local stage budgets, response diagnostics and bounded SAME-stage retries."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import random
import re
import time
from collections import Counter
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel

from app.ai.model_policy import get_task_model_policy
from app.ai.ports import StructuredGenerationResult, TextGenerationProvider
from app.features.language_learning.writing.review_diagnostics import (
    ReviewFailure, WritingProviderConfigurationError, WritingVerificationUnavailableError,
    classify_exception,
)

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


def safe_identifier(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", value):
        return value
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _token_count(value: Any) -> int | None:
    # Metadata is diagnostic, never a reason to crash or emit NaN/Infinity.
    return value if type(value) is int and value >= 0 else None


def emit_event(event: str, *, level: int = logging.INFO, **fields: Any) -> None:
    """One-line JSON also attached as LogRecord.writing_event for JSON handlers.

    Call sites supply allowlisted metadata only. Never pass exceptions, raw output,
    prompts, evidence quotes, validation input/ctx/msg, Profile or request bodies.
    """
    for name in ("request_id", "candidate_id", "model", "provider", "provider_route"):
        value = fields.get(name)
        if value is not None:
            fields[name] = safe_identifier(str(value))
    payload = {"schema_version": 1, "event": event, **fields}
    logger.log(level, "%s", json.dumps(payload, ensure_ascii=True, allow_nan=False, separators=(",", ":")),
               extra={"writing_event": payload})


@dataclass(frozen=True)
class ReviewContext:
    request_id: str
    candidate_id: str
    content_hash: str
    slot: int = 1  # AI request-local order, not BE global item order.
    generation_attempt: int = 1

    def fields(self) -> dict[str, Any]:
        return {"request_id": self.request_id, "candidate_id": self.candidate_id,
                "content_hash": self.content_hash, "slot": self.slot,
                "generation_attempt": self.generation_attempt}


@dataclass
class ReviewMetrics:
    calls: Counter[str] = field(default_factory=Counter)
    elapsed_ms: Counter[str] = field(default_factory=Counter)
    failures: Counter[str] = field(default_factory=Counter)
    retries: Counter[str] = field(default_factory=Counter)
    backoff_ms: float = 0.0
    active: int = 0
    peak_active: int = 0

    def summary(self) -> dict[str, Any]:
        return {"review_calls": dict(self.calls),
                "review_elapsed_ms": {task: round(value, 2) for task, value in self.elapsed_ms.items()},
                "review_failures": dict(self.failures), "review_retries": dict(self.retries),
                "review_backoff_ms": round(self.backoff_ms, 2), "peak_inflight_reviews": self.peak_active}


class ReviewRuntime:
    def __init__(self, provider: TextGenerationProvider, timeout_seconds: float, *,
                 deadline: float | None = None, retry_base_seconds: float = 0.25,
                 max_retry_delay_seconds: float = 5.0) -> None:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Review timeout must be positive and finite")
        if any(not math.isfinite(v) or v < 0 for v in (retry_base_seconds, max_retry_delay_seconds)):
            raise ValueError("Review backoff must be nonnegative and finite")
        self.provider = provider
        self.timeout_seconds = timeout_seconds
        self.deadline = deadline
        self.retry_base_seconds = retry_base_seconds
        self.max_retry_delay_seconds = max_retry_delay_seconds
        self.metrics = ReviewMetrics()

    def remaining_ms(self) -> float | None:
        return round(max(0.0, (self.deadline - time.monotonic()) * 1000), 2) if self.deadline is not None else None

    async def review(self, *, task: str, prompt: str, model: type[T], context: ReviewContext,
                     inspect: Callable[[T], ReviewFailure | None], max_attempts: int = 2,
                     required: bool = True, retry_semantic: bool = False,
                     schema: dict[str, Any] | None = None) -> T | None:
        if max_attempts not in {1, 2}:
            raise ValueError("Review stage permits one initial call and at most one retry")
        # Request-specific echo constraints; no target band or desired verdict is supplied.
        schema = deepcopy(schema) if schema is not None else model.model_json_schema(by_alias=True)
        for name, value in (("candidateId", context.candidate_id), ("contentHash", context.content_hash)):
            if name in schema.get("properties", {}):
                schema["properties"][name]["enum"] = [value]
        failure: ReviewFailure | None = None
        for attempt in range(1, max_attempts + 1):
            remaining = self.remaining_ms()
            if remaining is not None and remaining <= 0:
                raise TimeoutError("Writing request deadline expired before review")
            timeout = min(self.timeout_seconds, remaining / 1000) if remaining is not None else self.timeout_seconds
            started = time.monotonic()
            self.metrics.calls[task] += 1
            self.metrics.active += 1
            self.metrics.peak_active = max(self.metrics.peak_active, self.metrics.active)
            metadata: dict[str, Any] = {}
            result: T | None = None
            caught: Exception | None = None
            outcome = "ERROR"
            retry_scheduled = False
            delay = 0.0
            stop_reason: str | None = None
            common = {**context.fields(), "task": task, "tier": get_task_model_policy(task).tier.value,
                      "provider_route": getattr(self.provider, "provider_name", "unknown"),
                      "attempt": attempt, "max_attempts": max_attempts, "timeout_ms": round(timeout * 1000, 2)}
            emit_event("writing.verifier.started", level=logging.DEBUG, **common, remaining_budget_ms=remaining)
            try:
                call_metadata = getattr(self.provider, "call_with_metadata", None)
                supports = getattr(self.provider, "supports", None)
                if callable(call_metadata) and (not callable(supports) or supports("call_with_metadata")):
                    envelope = await asyncio.wait_for(call_metadata(
                        type_name=task, data=prompt, schema=deepcopy(schema)), timeout=timeout)
                    if not isinstance(envelope, StructuredGenerationResult):
                        failure = ReviewFailure("VERIFIER_RESPONSE_TYPE_INVALID", "PROTOCOL")
                        raw = None
                    else:
                        raw = envelope.data
                        metadata = {"provider": envelope.provider, "model": envelope.model,
                                    "input_tokens": _token_count(envelope.input_tokens),
                                    "output_tokens": _token_count(envelope.output_tokens)}
                        failure = None
                else:
                    raw = await asyncio.wait_for(self.provider.call(
                        type_name=task, data=prompt, schema=deepcopy(schema)), timeout=timeout)
                    failure = None
                if failure is None:
                    if raw is None or raw == {} or raw == "":
                        failure = ReviewFailure("VERIFIER_EMPTY_RESPONSE", "PROTOCOL")
                    elif not isinstance(raw, dict):
                        failure = ReviewFailure("VERIFIER_RESPONSE_TYPE_INVALID", "PROTOCOL")
                    else:
                        result = model.model_validate(raw)
                        failure = inspect(result)
                outcome = "VALID" if failure is None else ("UNCERTAIN" if failure.category == "SEMANTIC" else "INVALID")
            except asyncio.CancelledError:
                expired = self.remaining_ms() == 0
                failure = ReviewFailure("VERIFIER_DEADLINE_EXCEEDED" if expired else "VERIFIER_CANCELLED",
                                        "CANCELLED", retryable=False)
                outcome = "CANCELLED"
                raise
            except Exception as exc:
                failure = classify_exception(exc)
                caught = exc
                outcome = "UNAVAILABLE" if failure.category == "DEPENDENCY" else "INVALID"
            finally:
                elapsed = (time.monotonic() - started) * 1000
                self.metrics.active -= 1
                self.metrics.elapsed_ms[task] += elapsed
                if failure is not None:
                    self.metrics.failures[failure.code] += 1
                    remaining = self.remaining_ms()
                    delay = self._retry_delay(failure, attempt)
                    retry_scheduled = (
                        outcome != "CANCELLED" and failure.retryable and attempt < max_attempts
                        and (retry_semantic or failure.category != "SEMANTIC")
                        and delay <= self.max_retry_delay_seconds
                        and (remaining is None or remaining > delay * 1000 + 50)
                    )
                    if not retry_scheduled:
                        stop_reason = ("NON_RETRYABLE" if not failure.retryable else
                                       "ATTEMPT_LIMIT" if attempt >= max_attempts else
                                       "SEMANTIC_FINAL" if failure.category == "SEMANTIC" and not retry_semantic else
                                       "RETRY_AFTER_TOO_LONG" if delay > self.max_retry_delay_seconds else
                                       "REQUEST_BUDGET")
                emit_event("writing.verifier.finished", **common, **metadata,
                           outcome=outcome, duration_ms=round(elapsed, 2),
                           remaining_budget_ms=self.remaining_ms(), retry_scheduled=retry_scheduled,
                           retry_delay_ms=round(delay * 1000, 2) if retry_scheduled else 0,
                           retry_stop_reason=stop_reason,
                           verdict=getattr(result, "verdict", None),
                           confidence=getattr(result, "confidence", None),
                           estimated_band=getattr(result, "estimated_band", None),
                           difficulty_confidence=getattr(result, "difficulty_confidence", None),
                           difficulty_status=getattr(result, "difficulty_status", None),
                           confidence_used_for_acceptance=False,
                           **(failure.fields() if failure is not None else {
                               "failure_code": None, "failure_category": None, "retryable": False}))
            if failure is None:
                return result
            if failure.category == "CONFIGURATION":
                raise WritingProviderConfigurationError("Writing reviewer configuration failure") from caught
            if failure.category == "INTERNAL":
                assert caught is not None
                raise caught
            if retry_scheduled:
                self.metrics.retries[task] += 1
                if delay:
                    waiting = time.monotonic()
                    wait_outcome = "CANCELLED"
                    try:
                        await asyncio.sleep(delay)
                        wait_outcome = "COMPLETED"
                    finally:
                        waited_ms = (time.monotonic() - waiting) * 1000
                        self.metrics.backoff_ms += waited_ms
                        emit_event("writing.verifier.retry_wait", **common, outcome=wait_outcome,
                                   duration_ms=round(waited_ms, 2), remaining_budget_ms=self.remaining_ms())
                continue
            if required and failure.category != "SEMANTIC":
                emit_event("writing.verifier.exhausted", level=logging.WARNING, **common,
                           candidate_action="RAISE_REQUIRED_STAGE_FAILURE", **failure.fields())
                raise WritingVerificationUnavailableError(task, failure, attempt) from caught
            return None
        raise AssertionError("Unreachable review loop")

    def _retry_delay(self, failure: ReviewFailure, attempt: int) -> float:
        if failure.category != "DEPENDENCY":
            return 0.0
        jitter = self.retry_base_seconds * (2 ** (attempt - 1)) * random.uniform(1.0, 1.25)
        return max(jitter, failure.retry_after_seconds or 0.0)
