"""Bounded stage execution with caller-owned metadata, failures and diagnostics."""
from __future__ import annotations

import asyncio
import logging
import math
import random
import time
from collections import Counter
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Awaitable, TypeVar, cast

from pydantic import BaseModel

from app.ai.ports import StructuredGenerationResult, TextGenerationProvider
from app.features.language_learning.difficulty.budget import RequestBudget
from app.features.language_learning.difficulty.diagnostics import safe_token_count
from app.features.language_learning.difficulty.failures import (
    RequiredStageUnavailableError,
    StageConfigurationError,
    StageFailure,
)

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class StageEventNames:
    started: str = "difficulty.stage.started"
    finished: str = "difficulty.stage.finished"
    retry_wait: str = "difficulty.stage.retry_wait"
    exhausted: str = "difficulty.stage.exhausted"


@dataclass(frozen=True)
class StageContext:
    request_id: str
    candidate_id: str
    content_hash: str
    slot: int = 1
    generation_attempt: int = 1

    def fields(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "candidate_id": self.candidate_id,
            "content_hash": self.content_hash,
            "slot": self.slot,
            "generation_attempt": self.generation_attempt,
        }


@dataclass
class StageMetrics:
    calls: Counter[str] = field(default_factory=Counter)
    elapsed_ms: Counter[str] = field(default_factory=Counter)
    failures: Counter[str] = field(default_factory=Counter)
    retries: Counter[str] = field(default_factory=Counter)
    backoff_ms: float = 0.0
    active: int = 0
    peak_active: int = 0

    def summary(self) -> dict[str, Any]:
        return {
            "stage_calls": dict(self.calls),
            "stage_elapsed_ms": {
                task: round(value, 2) for task, value in self.elapsed_ms.items()
            },
            "stage_failures": dict(self.failures),
            "stage_retries": dict(self.retries),
            "stage_backoff_ms": round(self.backoff_ms, 2),
            "peak_active_stages": self.peak_active,
        }


def _noop_event(_event: str, *, level: int = logging.INFO, **_fields: Any) -> None:
    return None


class _ProviderTimeout(Exception):
    def __init__(self, error: TimeoutError) -> None:
        super().__init__()
        self.error = error


async def _isolate_provider_timeout(call: Awaitable[Any]) -> Any:
    try:
        return await call
    except TimeoutError as exc:
        raise _ProviderTimeout(exc) from exc


class BoundedStageRuntime:
    def __init__(
        self,
        provider: TextGenerationProvider,
        timeout_seconds: float,
        *,
        deadline: float | None = None,
        retry_base_seconds: float = 0.25,
        max_retry_delay_seconds: float = 5.0,
        failure_classifier: Callable[[Exception], StageFailure],
        event_emitter: Callable[..., None] = _noop_event,
        stage_metadata: Callable[[str], Mapping[str, Any]] | None = None,
        event_names: StageEventNames = StageEventNames(),
        failure_code_namespace: str = "STAGE",
        deadline_message: str = "Request deadline expired before stage execution",
        configuration_error_message: str = "Stage provider configuration failure",
        timeout_error_message: str = "Stage timeout must be positive and finite",
        backoff_error_message: str = "Stage backoff must be nonnegative and finite",
        attempts_error_message: str = "Stage max_attempts must be a positive integer",
        configuration_error_type: type[StageConfigurationError] = StageConfigurationError,
        unavailable_error_type: type[RequiredStageUnavailableError] = RequiredStageUnavailableError,
        failure_factory: Callable[..., StageFailure] = StageFailure,
        metrics_factory: Callable[[], StageMetrics] = StageMetrics,
        result_projection: Callable[[BaseModel | None], Mapping[str, Any]] | None = None,
        schema_binding: Callable[[dict[str, Any], StageContext], None] | None = None,
        attempts_validator: Callable[[int], bool] | None = None,
    ) -> None:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError(timeout_error_message)
        if any(
            not math.isfinite(value) or value < 0
            for value in (retry_base_seconds, max_retry_delay_seconds)
        ):
            raise ValueError(backoff_error_message)
        self.provider = provider
        self.timeout_seconds = timeout_seconds
        self.deadline = deadline
        self.budget = RequestBudget(deadline)
        self.retry_base_seconds = retry_base_seconds
        self.max_retry_delay_seconds = max_retry_delay_seconds
        self.failure_classifier = failure_classifier
        self.event_emitter = event_emitter
        self.stage_metadata = stage_metadata or (lambda _task: {})
        self.event_names = event_names
        self.failure_code_namespace = failure_code_namespace
        self.deadline_message = deadline_message
        self.configuration_error_message = configuration_error_message
        self.attempts_error_message = attempts_error_message
        self.configuration_error_type = configuration_error_type
        self.unavailable_error_type = unavailable_error_type
        self.failure_factory = failure_factory
        self.result_projection = result_projection or (lambda _result: {})
        self.schema_binding = schema_binding
        self.attempts_validator = attempts_validator
        self.metrics = metrics_factory()

    def remaining_ms(self) -> float | None:
        return self.budget.remaining_ms()

    async def review(
        self,
        *,
        task: str,
        prompt: str,
        model: type[T],
        context: StageContext,
        inspect: Callable[[T], StageFailure | None],
        max_attempts: int = 2,
        required: bool = True,
        retry_semantic: bool = False,
        schema: dict[str, Any] | None = None,
    ) -> T | None:
        valid_attempts = (
            self.attempts_validator(max_attempts)
            if self.attempts_validator is not None
            else type(max_attempts) is int and max_attempts > 0
        )
        if not valid_attempts:
            raise ValueError(self.attempts_error_message)
        schema = deepcopy(schema) if schema is not None else model.model_json_schema(by_alias=True)
        if self.schema_binding is not None:
            self.schema_binding(schema, context)
        failure: StageFailure | None = None
        for attempt in range(1, max_attempts + 1):
            remaining = self.remaining_ms()
            if remaining is not None and remaining <= 0:
                raise TimeoutError(self.deadline_message)
            timeout = (
                min(self.timeout_seconds, remaining / 1000)
                if remaining is not None
                else self.timeout_seconds
            )
            request_budget_limited = (
                remaining is not None and remaining / 1000 <= self.timeout_seconds
            )
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
            common = {
                **context.fields(),
                "task": task,
                **dict(self.stage_metadata(task)),
                "attempt": attempt,
                "max_attempts": max_attempts,
                "timeout_ms": round(timeout * 1000, 2),
            }
            self.event_emitter(
                self.event_names.started,
                level=logging.DEBUG,
                **common,
                remaining_budget_ms=remaining,
            )
            try:
                call_metadata = getattr(self.provider, "call_with_metadata", None)
                supports = getattr(self.provider, "supports", None)
                if callable(call_metadata) and (
                    not callable(supports) or supports("call_with_metadata")
                ):
                    metadata_call = cast(
                        Callable[..., Awaitable[StructuredGenerationResult]],
                        call_metadata,
                    )
                    envelope = await asyncio.wait_for(
                        _isolate_provider_timeout(
                            metadata_call(
                                type_name=task,
                                data=prompt,
                                schema=deepcopy(schema),
                            )
                        ),
                        timeout=timeout,
                    )
                    if not isinstance(envelope, StructuredGenerationResult):
                        failure = self.failure_factory(
                            self._failure_code("RESPONSE_TYPE_INVALID"), "PROTOCOL"
                        )
                        raw = None
                    else:
                        raw = envelope.data
                        metadata = {
                            "provider": envelope.provider,
                            "model": envelope.model,
                            "input_tokens": safe_token_count(envelope.input_tokens),
                            "output_tokens": safe_token_count(envelope.output_tokens),
                        }
                        failure = None
                else:
                    raw = await asyncio.wait_for(
                        _isolate_provider_timeout(
                            self.provider.call(
                                type_name=task,
                                data=prompt,
                                schema=deepcopy(schema),
                            )
                        ),
                        timeout=timeout,
                    )
                    failure = None
                if failure is None:
                    if raw is None or raw == {} or raw == "":
                        failure = self.failure_factory(
                            self._failure_code("EMPTY_RESPONSE"), "PROTOCOL"
                        )
                    elif not isinstance(raw, dict):
                        failure = self.failure_factory(
                            self._failure_code("RESPONSE_TYPE_INVALID"), "PROTOCOL"
                        )
                    else:
                        result = model.model_validate(raw)
                        failure = inspect(result)
                outcome = (
                    "VALID"
                    if failure is None
                    else "UNCERTAIN"
                    if failure.category == "SEMANTIC"
                    else "INVALID"
                )
            except asyncio.CancelledError:
                expired = self.remaining_ms() == 0
                failure = self.failure_factory(
                    self._failure_code("DEADLINE_EXCEEDED" if expired else "CANCELLED"),
                    "CANCELLED",
                    retryable=False,
                )
                outcome = "CANCELLED"
                raise
            except _ProviderTimeout as exc:
                failure = self.failure_classifier(exc.error)
                caught = exc.error
                outcome = (
                    "UNAVAILABLE" if failure.category == "DEPENDENCY" else "INVALID"
                )
            except TimeoutError as exc:
                budget_remaining = self.budget.remaining_seconds()
                if (
                    request_budget_limited
                    and budget_remaining is not None
                    and budget_remaining > 0
                    and isinstance(exc.__cause__, asyncio.CancelledError)
                ):
                    try:
                        await asyncio.sleep(
                            budget_remaining
                            + time.get_clock_info("monotonic").resolution
                        )
                    except asyncio.CancelledError:
                        expired = self.budget.remaining_seconds() == 0
                        failure = self.failure_factory(
                            self._failure_code(
                                "DEADLINE_EXCEEDED" if expired else "CANCELLED"
                            ),
                            "CANCELLED",
                            retryable=False,
                        )
                        outcome = "CANCELLED"
                        raise
                if (
                    request_budget_limited
                    and self.budget.remaining_seconds() == 0
                    and isinstance(exc.__cause__, asyncio.CancelledError)
                ):
                    failure = self.failure_factory(
                        self._failure_code("DEADLINE_EXCEEDED"),
                        "CANCELLED",
                        retryable=False,
                    )
                    outcome = "CANCELLED"
                    raise TimeoutError(self.deadline_message) from exc
                failure = self.failure_classifier(exc)
                caught = exc
                outcome = (
                    "UNAVAILABLE" if failure.category == "DEPENDENCY" else "INVALID"
                )
            except Exception as exc:
                failure = self.failure_classifier(exc)
                caught = exc
                outcome = "UNAVAILABLE" if failure.category == "DEPENDENCY" else "INVALID"
            finally:
                elapsed = (time.monotonic() - started) * 1000
                self.metrics.active -= 1
                cast(Any, self.metrics.elapsed_ms)[task] += elapsed
                if failure is not None:
                    self.metrics.failures[failure.code] += 1
                    remaining = self.remaining_ms()
                    delay = self._retry_delay(failure, attempt)
                    retry_scheduled = (
                        outcome != "CANCELLED"
                        and failure.retryable
                        and attempt < max_attempts
                        and (retry_semantic or failure.category != "SEMANTIC")
                        and delay <= self.max_retry_delay_seconds
                        and (remaining is None or remaining > delay * 1000 + 50)
                    )
                    if not retry_scheduled:
                        stop_reason = (
                            "NON_RETRYABLE"
                            if not failure.retryable
                            else "ATTEMPT_LIMIT"
                            if attempt >= max_attempts
                            else "SEMANTIC_FINAL"
                            if failure.category == "SEMANTIC" and not retry_semantic
                            else "RETRY_AFTER_TOO_LONG"
                            if delay > self.max_retry_delay_seconds
                            else "REQUEST_BUDGET"
                        )
                self.event_emitter(
                    self.event_names.finished,
                    **common,
                    **metadata,
                    outcome=outcome,
                    duration_ms=round(elapsed, 2),
                    remaining_budget_ms=self.remaining_ms(),
                    retry_scheduled=retry_scheduled,
                    retry_delay_ms=round(delay * 1000, 2) if retry_scheduled else 0,
                    retry_stop_reason=stop_reason,
                    **dict(self.result_projection(result)),
                    **(
                        failure.fields()
                        if failure is not None
                        else {
                            "failure_code": None,
                            "failure_category": None,
                            "retryable": False,
                        }
                    ),
                )
            if failure is None:
                return result
            if failure.category == "CONFIGURATION":
                raise self.configuration_error_type(
                    self.configuration_error_message
                ) from caught
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
                        self.event_emitter(
                            self.event_names.retry_wait,
                            **common,
                            outcome=wait_outcome,
                            duration_ms=round(waited_ms, 2),
                            remaining_budget_ms=self.remaining_ms(),
                        )
                continue
            if required and failure.category != "SEMANTIC":
                self.event_emitter(
                    self.event_names.exhausted,
                    level=logging.WARNING,
                    **common,
                    candidate_action="RAISE_REQUIRED_STAGE_FAILURE",
                    **failure.fields(),
                )
                raise self.unavailable_error_type(task, failure, attempt) from caught
            return None
        raise AssertionError("Unreachable review loop")

    def _retry_delay(self, failure: StageFailure, attempt: int) -> float:
        if failure.category != "DEPENDENCY":
            return 0.0
        jitter = self.retry_base_seconds * (2 ** (attempt - 1)) * random.uniform(1.0, 1.25)
        return max(jitter, failure.retry_after_seconds or 0.0)

    def _failure_code(self, suffix: str) -> str:
        return f"{self.failure_code_namespace}_{suffix}"
