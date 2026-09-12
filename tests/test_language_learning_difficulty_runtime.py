from __future__ import annotations

import asyncio
import copy
import logging
import time

import pytest
from pydantic import BaseModel

from app.ai.ports import StructuredGenerationResult
from app.features.language_learning.difficulty.failures import (
    RequiredStageUnavailableError,
    StageConfigurationError,
    StageFailure,
    classify_exception,
    retry_after_seconds,
)
from app.features.language_learning.difficulty.runtime import (
    BoundedStageRuntime,
    StageContext,
    StageEventNames,
)


class Payload(BaseModel):
    candidate_id: str
    content_hash: str
    verdict: str


VALID = {"candidate_id": "candidate", "content_hash": "a" * 64, "verdict": "PASS"}


class Provider:
    def __init__(self, values):
        self.values = list(values)
        self.calls = []

    async def call(self, type_name, data, schema=None):
        self.calls.append((type_name, data, copy.deepcopy(schema)))
        value = self.values.pop(0)
        if isinstance(value, BaseException):
            raise value
        if callable(value):
            return value(schema)
        return copy.deepcopy(value)

    async def call_with_image(self, *args, **kwargs):
        raise AssertionError("stage runtime never calls vision")


class StatusError(Exception):
    def __init__(self, status_code: int, headers: dict[str, object] | None = None):
        super().__init__("PRIVATE_PROVIDER_BODY")
        self.status_code = status_code
        self.headers = headers or {}


def runtime(provider, **kwargs):
    return BoundedStageRuntime(
        provider,
        kwargs.pop("timeout_seconds", 1),
        failure_classifier=lambda exc: classify_exception(exc, code_namespace="TEST"),
        **kwargs,
    )


def review_kwargs(**updates):
    values = {
        "task": "SERVICE_TASK",
        "prompt": "opaque prompt",
        "model": Payload,
        "context": StageContext("r", "candidate", "a" * 64),
        "inspect": lambda _value: None,
    }
    values.update(updates)
    return values


def collect_events():
    events = []

    def emit(event, *, level=logging.INFO, **fields):
        events.append({"event": event, "level": level, **fields})

    return events, emit


def test_runtime_retries_only_same_stage_and_never_mutates_original_schema():
    original = Payload.model_json_schema()
    expected = copy.deepcopy(original)

    def mutate_and_fail(schema):
        schema["PRIVATE_MUTATION"] = True
        raise ValueError("bad structured response")

    def succeed(schema):
        assert "PRIVATE_MUTATION" not in schema
        return VALID

    provider = Provider([mutate_and_fail, succeed])
    result = asyncio.run(runtime(provider).review(**review_kwargs(schema=original)))
    assert result is not None
    assert result.verdict == "PASS"
    assert len(provider.calls) == 2
    assert provider.calls[0][:2] == provider.calls[1][:2]
    assert provider.calls[0][2] == provider.calls[1][2] == expected
    assert provider.calls[0][2] is not provider.calls[1][2]
    assert original == expected


def test_common_runtime_supports_arbitrary_positive_bounded_attempts():
    provider = Provider(
        [
            ValueError("first malformed response"),
            ValueError("second malformed response"),
            VALID,
        ]
    )
    common = runtime(provider, retry_base_seconds=0)

    result = asyncio.run(common.review(**review_kwargs(max_attempts=3)))

    assert result is not None
    assert result.verdict == "PASS"
    assert len(provider.calls) == 3
    assert common.metrics.calls == {"SERVICE_TASK": 3}


def test_expired_budget_never_calls_provider():
    provider = Provider([])
    with pytest.raises(TimeoutError):
        asyncio.run(runtime(provider, deadline=time.monotonic() - 1).review(**review_kwargs()))
    assert not provider.calls


def test_required_dependency_failure_stays_distinct_from_content_rejection():
    provider = Provider([ConnectionError("private"), ConnectionError("private")])
    with pytest.raises(RequiredStageUnavailableError) as raised:
        asyncio.run(runtime(provider, retry_base_seconds=0).review(**review_kwargs()))
    assert raised.value.failure.category == "DEPENDENCY"
    assert raised.value.failure.code == "TEST_CONNECTION_ERROR"


def test_stage_timeout_cleans_up_provider_coroutine():
    entered = asyncio.Event()
    cleaned = asyncio.Event()

    class HangingProvider:
        async def call(self, type_name, data, schema=None):
            entered.set()
            try:
                await asyncio.Future()
            finally:
                cleaned.set()

    async def scenario():
        with pytest.raises(RequiredStageUnavailableError) as raised:
            await runtime(HangingProvider(), timeout_seconds=0.01).review(
                **review_kwargs(max_attempts=1),
            )
        assert raised.value.failure.code == "TEST_TIMEOUT"
        assert entered.is_set() and cleaned.is_set()

    asyncio.run(scenario())


def test_request_budget_limited_timeout_raises_deadline_and_cleans_up_provider():
    entered = asyncio.Event()
    cleaned = asyncio.Event()

    class HangingProvider:
        async def call(self, type_name, data, schema=None):
            entered.set()
            try:
                await asyncio.Future()
            finally:
                cleaned.set()

    async def scenario():
        with pytest.raises(TimeoutError, match="Request deadline expired"):
            await runtime(
                HangingProvider(),
                timeout_seconds=1,
                deadline=time.monotonic() + 0.02,
            ).review(**review_kwargs(max_attempts=1))
        assert entered.is_set() and cleaned.is_set()

    asyncio.run(scenario())


def test_provider_internal_wait_for_timeout_is_not_request_deadline():
    class ProviderInternalTimeout:
        def __init__(self):
            self.calls = 0

        async def call(self, type_name, data, schema=None):
            self.calls += 1
            await asyncio.wait_for(asyncio.Event().wait(), timeout=0.01)

    async def scenario():
        provider = ProviderInternalTimeout()
        common = runtime(
            provider,
            timeout_seconds=1,
            deadline=time.monotonic() + 0.2,
        )
        with pytest.raises(RequiredStageUnavailableError) as raised:
            await common.review(**review_kwargs(max_attempts=1))
        remaining = common.remaining_ms()
        assert raised.value.failure.code == "TEST_TIMEOUT"
        assert remaining is not None and remaining > 100
        assert provider.calls == 1

    asyncio.run(scenario())


def test_request_deadline_wins_when_stage_and_budget_timeouts_compete():
    class HangingProvider:
        def __init__(self, cleaned):
            self.cleaned = cleaned

        async def call(self, type_name, data, schema=None):
            try:
                await asyncio.Future()
            finally:
                self.cleaned.set()

    async def scenario():
        for _ in range(3):
            cleaned = asyncio.Event()
            with pytest.raises(TimeoutError, match="Request deadline expired"):
                await runtime(
                    HangingProvider(cleaned),
                    timeout_seconds=0.03,
                    deadline=time.monotonic() + 0.03,
                ).review(**review_kwargs(max_attempts=1))
            assert cleaned.is_set()

    asyncio.run(scenario())


def test_cancellation_propagates_without_a_result_or_orphan():
    entered = asyncio.Event()
    cleaned = asyncio.Event()

    class HangingProvider:
        async def call(self, type_name, data, schema=None):
            entered.set()
            try:
                await asyncio.Future()
            finally:
                cleaned.set()

    async def scenario():
        task = asyncio.create_task(runtime(HangingProvider()).review(**review_kwargs()))
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cleaned.is_set()
        assert not [
            other for other in asyncio.all_tasks()
            if other is not asyncio.current_task() and not other.done()
        ]

    asyncio.run(scenario())


def test_cancellation_during_retry_backoff_propagates_without_second_call():
    called = asyncio.Event()

    class FailingProvider:
        def __init__(self):
            self.calls = 0

        async def call(self, type_name, data, schema=None):
            self.calls += 1
            called.set()
            raise ConnectionError("private")

    async def scenario():
        provider = FailingProvider()
        task = asyncio.create_task(runtime(
            provider, retry_base_seconds=5, max_retry_delay_seconds=10,
        ).review(**review_kwargs()))
        await asyncio.wait_for(called.wait(), 1)
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert provider.calls == 1

    asyncio.run(scenario())


def test_retry_after_above_max_delay_does_not_retry():
    provider = Provider([StatusError(429, {"retry-after": "10"})])
    with pytest.raises(RequiredStageUnavailableError):
        asyncio.run(runtime(provider, retry_base_seconds=0, max_retry_delay_seconds=5).review(
            **review_kwargs(),
        ))
    assert len(provider.calls) == 1


def test_retry_after_keeps_existing_numeric_and_bytes_conversion():
    assert retry_after_seconds(StatusError(429, {"retry-after-ms": b"1000"})) == 1.0


def test_retry_after_above_remaining_request_budget_does_not_retry():
    provider = Provider([StatusError(429, {"retry-after": "1"})])
    with pytest.raises(RequiredStageUnavailableError):
        asyncio.run(runtime(
            provider, deadline=time.monotonic() + 0.1, retry_base_seconds=0,
        ).review(**review_kwargs()))
    assert len(provider.calls) == 1


def test_configuration_failure_and_nonretryable_failure_stop_immediately():
    configured = Provider([StatusError(401)])
    with pytest.raises(StageConfigurationError):
        asyncio.run(runtime(configured, retry_base_seconds=0).review(**review_kwargs()))
    assert len(configured.calls) == 1

    stopped = Provider([ConnectionError("private")])
    common = BoundedStageRuntime(
        stopped,
        1,
        failure_classifier=lambda _exc: StageFailure(
            "SERVICE_STOP", "DEPENDENCY", retryable=False,
        ),
    )
    with pytest.raises(RequiredStageUnavailableError):
        asyncio.run(common.review(**review_kwargs()))
    assert len(stopped.calls) == 1


def test_semantic_failure_is_not_retried_and_optional_failure_returns_none():
    semantic = Provider([VALID])
    result = asyncio.run(runtime(semantic).review(**review_kwargs(
        inspect=lambda _value: StageFailure("SERVICE_CONTENT", "SEMANTIC"),
    )))
    assert result is None and len(semantic.calls) == 1

    optional = Provider([ConnectionError("private")])
    result = asyncio.run(runtime(optional, retry_base_seconds=0).review(**review_kwargs(
        max_attempts=1, required=False,
    )))
    assert result is None and len(optional.calls) == 1


def test_call_with_metadata_uses_safe_tokens_and_call_only_provider_falls_back():
    events, emit = collect_events()

    class MetadataProvider:
        def __init__(self):
            self.metadata_calls = 0
            self.call_calls = 0

        def supports(self, capability):
            return capability == "call_with_metadata"

        async def call_with_metadata(self, **kwargs):
            self.metadata_calls += 1
            return StructuredGenerationResult(
                VALID, input_tokens=True, output_tokens=-1,
                provider="service-provider", model="service-model",
            )

        async def call(self, **kwargs):
            self.call_calls += 1
            raise AssertionError("metadata path should be preferred")

    provider = MetadataProvider()
    result = asyncio.run(runtime(provider, event_emitter=emit).review(**review_kwargs()))
    finished = next(event for event in events if event["event"] == "difficulty.stage.finished")
    assert result is not None
    assert result.verdict == "PASS"
    assert (provider.metadata_calls, provider.call_calls) == (1, 0)
    assert finished["provider"] == "service-provider"
    assert finished["model"] == "service-model"
    assert finished["input_tokens"] is None and finished["output_tokens"] is None

    fallback = Provider([VALID])
    fallback_result = asyncio.run(runtime(fallback).review(**review_kwargs()))
    assert fallback_result is not None and fallback_result.verdict == "PASS"
    assert len(fallback.calls) == 1


@pytest.mark.parametrize("value", [ConnectionError("PRIVATE_EXCEPTION"), "PRIVATE_RAW_OUTPUT"])
def test_diagnostics_never_receive_prompt_raw_output_or_exception_body(value):
    events, emit = collect_events()
    provider = Provider([value])
    with pytest.raises(RequiredStageUnavailableError):
        asyncio.run(runtime(provider, event_emitter=emit, retry_base_seconds=0).review(
            **review_kwargs(prompt="PRIVATE_PROMPT", max_attempts=1),
        ))
    rendered = repr(events)
    assert "PRIVATE_PROMPT" not in rendered
    assert "PRIVATE_RAW_OUTPUT" not in rendered
    assert "PRIVATE_EXCEPTION" not in rendered


def test_service_event_namespace_and_result_projection_are_caller_owned():
    events, emit = collect_events()
    names = StageEventNames(
        started="service.check.started",
        finished="service.check.finished",
        retry_wait="service.check.retry",
        exhausted="service.check.exhausted",
    )
    result = asyncio.run(runtime(
        Provider([VALID]),
        event_emitter=emit,
        event_names=names,
        result_projection=lambda value: {
            "service_outcome": getattr(value, "verdict", None),
        },
    ).review(**review_kwargs()))
    assert result is not None
    assert result.verdict == "PASS"
    assert [event["event"] for event in events] == [
        "service.check.started", "service.check.finished",
    ]
    assert events[-1]["service_outcome"] == "PASS"


def test_runtime_instances_do_not_share_metrics():
    first = runtime(Provider([VALID]))
    second = runtime(Provider([VALID]))

    async def scenario():
        await asyncio.gather(
            first.review(**review_kwargs()),
            second.review(**review_kwargs()),
        )

    asyncio.run(scenario())
    assert first.metrics is not second.metrics
    assert first.metrics.calls == second.metrics.calls == {"SERVICE_TASK": 1}


def test_active_and_peak_active_account_for_concurrent_stages():
    entered = 0
    both_entered = asyncio.Event()
    release = asyncio.Event()

    class ConcurrentProvider:
        async def call(self, type_name, data, schema=None):
            nonlocal entered
            entered += 1
            if entered == 2:
                both_entered.set()
            await release.wait()
            return VALID

    async def scenario():
        common = runtime(ConcurrentProvider())
        tasks = [asyncio.create_task(common.review(**review_kwargs())) for _ in range(2)]
        await asyncio.wait_for(both_entered.wait(), 1)
        assert common.metrics.active == common.metrics.peak_active == 2
        release.set()
        await asyncio.gather(*tasks)
        assert common.metrics.active == 0
        assert common.metrics.calls == {"SERVICE_TASK": 2}

    asyncio.run(scenario())
