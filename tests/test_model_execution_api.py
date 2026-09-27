import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import APIConnectionError, APITimeoutError

from app.ai.model_policy import AiModelTier
from app.ai.ports import StructuredGenerationResult
from app.ai.providers.openai.client import OpenAIService
from app.ai.providers.openai.response import OpenAIProviderResponseError
from app.ai.providers.openai.schema import (
    OpenAISchemaConfigurationError,
    build_explicit_text_config,
)
from app.api.dependencies import get_model_execution_provider
from app.core.config import settings


@pytest.mark.asyncio
async def test_explicit_execution_limits_provider_concurrency_without_hidden_retry():
    # 준비: 같은 event loop의 두 요청을 첫 Provider 호출에서 멈춘다.
    from app.main import app

    entered = asyncio.Event()
    release = asyncio.Event()

    class BlockingProvider(FakeProvider):
        async def execute_explicit(self, **kwargs):
            self.calls.append(kwargs)
            entered.set()
            await release.wait()
            return self.result

    provider = BlockingProvider()
    original_key = settings.SERVER_API_KEY
    original_limit = settings.AI_TEXT_PROVIDER_MAX_CONCURRENCY
    settings.SERVER_API_KEY = "synthetic-internal-key"
    settings.AI_TEXT_PROVIDER_MAX_CONCURRENCY = 1
    app.dependency_overrides[get_model_execution_provider] = lambda: provider

    try:
        # 실행: 두 번째 요청은 슬롯을 기다리고 첫 요청만 Provider에 진입한다.
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            headers = {"X-API-KEY": "synthetic-internal-key"}
            first = asyncio.create_task(
                client.post(
                    "/internal/v1/model/execute",
                    json=_request(remainingMilliseconds=5000),
                    headers=headers,
                )
            )
            await asyncio.wait_for(entered.wait(), timeout=2)
            second = asyncio.create_task(
                client.post(
                    "/internal/v1/model/execute",
                    json=_request(remainingMilliseconds=5000),
                    headers=headers,
                )
            )
            await asyncio.sleep(0.05)

            # 검증: 대기 중 추가 SDK 호출은 없고 해제 뒤 각 요청이 한 번씩 실행된다.
            assert len(provider.calls) == 1
            assert not second.done()
            release.set()
            responses = await asyncio.gather(first, second)
            assert [response.status_code for response in responses] == [200, 200]
            assert len(provider.calls) == 2
    finally:
        release.set()
        app.dependency_overrides.pop(get_model_execution_provider, None)
        settings.SERVER_API_KEY = original_key
        settings.AI_TEXT_PROVIDER_MAX_CONCURRENCY = original_limit


def _request(**overrides):
    value = {
        "traceId": "synthetic-trace-1",
        "instructions": "Synthetic system instruction",
        "messages": [{"role": "user", "content": "Synthetic input"}],
        "tier": "MINI",
        "reasoningEffort": "low",
        "verbosity": "low",
        "maxOutputTokens": 2048,
        "remainingMilliseconds": 1000,
        "maxProviderCalls": 1,
        "responseSchema": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
        "schemaName": "synthetic_answer",
        "strict": True,
        "taskName": "LANGUAGE_LEARNING_WRITING_TASK_VERIFICATION",
    }
    value.update(overrides)
    return value


class FakeProvider:
    def __init__(self, result=None, failure=None):
        self.calls = []
        self.result = result or StructuredGenerationResult(
            data={"answer": "synthetic"},
            input_tokens=12,
            output_tokens=3,
            provider="fake-sdk",
            model="synthetic-model",
        )
        self.failure = failure

    async def execute_explicit(self, **kwargs):
        self.calls.append(kwargs)
        if self.failure:
            raise self.failure
        return self.result


def _client(fake):
    with patch("app.core.config_logger.setup_logging"):
        from app.main import app

    original_key = settings.SERVER_API_KEY
    settings.SERVER_API_KEY = "synthetic-internal-key"
    app.dependency_overrides[get_model_execution_provider] = lambda: fake
    return TestClient(app), original_key


def _restore(original_key):
    from app.main import app

    settings.SERVER_API_KEY = original_key
    app.dependency_overrides.pop(get_model_execution_provider, None)


def test_internal_execution_auth_and_explicit_contract():
    fake = FakeProvider()
    client, original_key = _client(fake)
    try:
        assert client.post("/internal/v1/model/execute", json=_request()).status_code == 401
        response = client.post(
            "/internal/v1/model/execute",
            json=_request(),
            headers={"X-API-KEY": "synthetic-internal-key"},
        )
        assert response.status_code == 200
        assert response.json() == {
            "output": {"answer": "synthetic"},
            "inputTokens": 12,
            "outputTokens": 3,
            "provider": "fake-sdk",
            "model": "synthetic-model",
            "providerCalls": 1,
        }
        assert len(fake.calls) == 1
        assert fake.calls[0]["instructions"] == "Synthetic system instruction"
        assert fake.calls[0]["messages"] == [{"role": "user", "content": "Synthetic input"}]
        assert fake.calls[0]["tier"] == AiModelTier.MINI
        assert "task_name" not in fake.calls[0]
    finally:
        _restore(original_key)


def test_execution_preserves_safe_provider_retry_after():
    # 준비
    failure = RuntimeError("private diagnostic")
    failure.status_code = 429
    failure.response = SimpleNamespace(headers={"retry-after-ms": "1250"})
    client, original_key = _client(FakeProvider(failure=failure))
    try:
        # 실행
        response = client.post(
            "/internal/v1/model/execute",
            json=_request(),
            headers={"X-API-KEY": "synthetic-internal-key"},
        )

        # 검증
        assert response.status_code == 503
        assert response.json()["detail"]["retryAfterSeconds"] == 2
        assert response.json()["detail"]["failureKind"] == "HTTP_STATUS"
        assert response.json()["detail"]["providerStatus"] == 429
        assert "private diagnostic" not in response.text
    finally:
        _restore(original_key)


@pytest.mark.parametrize("seconds", [600, 86_400])
def test_execution_preserves_long_safe_provider_retry_after(seconds):
    # 준비: SDK가 하루 상한 안의 재시도 시각을 기술 메타데이터로 전달한다.
    failure = RuntimeError("private diagnostic")
    failure.status_code = 429
    failure.retry_after_seconds = seconds
    client, original_key = _client(FakeProvider(failure=failure))

    try:
        # 실행
        response = client.post(
            "/internal/v1/model/execute",
            json=_request(),
            headers={"X-API-KEY": "synthetic-internal-key"},
        )

        # 검증: 긴 값을 잘라내지 않고 응답 본문은 안전한 코드만 포함한다.
        assert response.status_code == 503
        assert response.json()["detail"]["retryAfterSeconds"] == seconds
        assert "private diagnostic" not in response.text
    finally:
        _restore(original_key)


def test_invalid_request_does_not_echo_prompt_or_increase_provider_calls():
    fake = FakeProvider()
    client, original_key = _client(fake)
    try:
        response = client.post(
            "/internal/v1/model/execute",
            json=_request(instructions="SENSITIVE_SYNTHETIC_MARKER", maxProviderCalls=2),
            headers={"X-API-KEY": "synthetic-internal-key"},
        )
        assert response.status_code == 422
        assert response.json()["detail"]["code"] == "EXECUTION_REQUEST_INVALID"
        assert "SENSITIVE_SYNTHETIC_MARKER" not in response.text
        assert fake.calls == []
    finally:
        _restore(original_key)


def test_deep_invalid_json_uses_fixed_protocol_error_without_provider_call():
    fake = FakeProvider()
    client, original_key = _client(fake)
    try:
        response = client.post(
            "/internal/v1/model/execute",
            content="[" * 1100 + "0" + "]" * 1100,
            headers={"X-API-KEY": "synthetic-internal-key", "Content-Type": "application/json"},
        )
        assert response.status_code == 422
        assert response.json()["detail"] == {
            "code": "EXECUTION_REQUEST_INVALID",
            "retryable": False,
        }
        assert fake.calls == []
    finally:
        _restore(original_key)


def test_provider_refusal_and_schema_failure_remain_distinct():
    refusal = FakeProvider(
        failure=OpenAIProviderResponseError(
            "provider refused",
            reason_code="REFUSAL",
            retryable=False,
        )
    )
    client, original_key = _client(refusal)
    try:
        response = client.post(
            "/internal/v1/model/execute",
            json=_request(),
            headers={"X-API-KEY": "synthetic-internal-key"},
        )
        assert response.status_code == 502
        assert response.json()["detail"] == {"code": "REFUSAL", "retryable": False}
        assert len(refusal.calls) == 1
    finally:
        _restore(original_key)

    schema_failure = FakeProvider(failure=OpenAISchemaConfigurationError("synthetic"))
    client, original_key = _client(schema_failure)
    try:
        response = client.post(
            "/internal/v1/model/execute",
            json=_request(),
            headers={"X-API-KEY": "synthetic-internal-key"},
        )
        assert response.status_code == 422
        assert response.json()["detail"]["code"] == "EXECUTION_SCHEMA_INVALID"
    finally:
        _restore(original_key)


def test_schema_adapter_preserves_supplied_provider_schema_and_strict_flag():
    schema = _request()["responseSchema"]
    config = build_explicit_text_config(
        schema=schema,
        schema_name_value="synthetic_answer",
        strict=True,
        verbosity="low",
    )
    assert config["format"]["schema"] == schema
    assert config["format"]["strict"] is True
    config["format"]["schema"]["properties"]["answer"]["type"] = "integer"
    assert schema["properties"]["answer"]["type"] == "string"


def test_sdk_transport_timeout_and_configuration_keep_technical_failure_classes():
    # 준비: 실제 SDK 예외 형태를 사용하되 네트워크 요청과 모델 호출은 발생시키지 않는다.
    request = httpx.Request("POST", "https://synthetic.invalid")
    configuration = type("SyntheticConfigurationError", (Exception,), {"status_code": 401})
    cases = [
        (APITimeoutError(request=request), 504, "PROVIDER_TIMEOUT", True, "SDK_TIMEOUT", None),
        (TimeoutError(), 504, "PROVIDER_TIMEOUT", True, "TIMEOUT", None),
        (
            APIConnectionError(request=request),
            503,
            "PROVIDER_UNAVAILABLE",
            True,
            "SDK_CONNECTION",
            None,
        ),
        (
            httpx.ConnectError("PRIVATE_SYNTHETIC_MARKER"),
            503,
            "PROVIDER_UNAVAILABLE",
            True,
            "CONNECTION",
            None,
        ),
        (
            configuration("PRIVATE_SYNTHETIC_MARKER"),
            502,
            "PROVIDER_CONFIGURATION_ERROR",
            False,
            "HTTP_STATUS",
            401,
        ),
    ]
    for failure, status, code, retryable, kind, provider_status in cases:
        fake = FakeProvider(failure=failure)
        client, original_key = _client(fake)
        try:
            # 실행: 범용 HTTP 경계가 기술 실패를 한 번만 분류한다.
            response = client.post(
                "/internal/v1/model/execute",
                json=_request(),
                headers={"X-API-KEY": "synthetic-internal-key"},
            )

            # 검증: 업무 재시도는 LL이 결정하며 예외 원문은 응답에 포함하지 않는다.
            assert response.status_code == status
            expected = {"code": code, "retryable": retryable, "failureKind": kind}
            if provider_status is not None:
                expected["providerStatus"] = provider_status
            assert response.json()["detail"] == expected
            assert len(fake.calls) == 1
            assert "PRIVATE_SYNTHETIC_MARKER" not in response.text
        finally:
            _restore(original_key)


def test_openai_sdk_adapter_uses_explicit_messages_profile_and_one_call():
    calls = []

    class FakeResponses:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                status="completed",
                output_text=json.dumps({"answer": "synthetic"}),
                output=[],
                model="synthetic-model",
                usage=SimpleNamespace(input_tokens=8, output_tokens=2),
            )

    class FakeClient:
        responses = FakeResponses()

        def with_options(self, **kwargs):
            assert 0 < kwargs["timeout"] <= settings.OPENAI_REQUEST_TIMEOUT_SECONDS
            return self

    provider = OpenAIService()
    provider._client = FakeClient()
    schema = _request()["responseSchema"]
    result = asyncio.run(
        provider.execute_explicit(
            instructions="Synthetic system instruction",
            messages=[{"role": "user", "content": "Synthetic input"}],
            tier=AiModelTier.MINI,
            reasoning_effort="low",
            verbosity="low",
            max_output_tokens=2048,
            remaining_milliseconds=1000,
            response_schema=schema,
            schema_name_value="synthetic_answer",
            strict=True,
        )
    )
    assert result.data == {"answer": "synthetic"}
    assert len(calls) == 1
    assert calls[0]["instructions"] == "Synthetic system instruction"
    assert calls[0]["input"] == [{"role": "user", "content": "Synthetic input"}]
    assert calls[0]["text"]["format"]["schema"] == schema
    assert calls[0]["store"] is False


def test_provider_signal_crosses_http_as_enum_without_original_message():
    # 준비: 상태 없는 SDK 오류도 원본 재시도 정책에서 구분하던 표식을 제공한다.
    for message, signal in (
        ("rate limit", "RATE_LIMIT"),
        ("deadline exceeded", "DEADLINE"),
        ("safety blocked", "SAFETY"),
    ):
        fake = FakeProvider(failure=RuntimeError(f"{message} PRIVATE_SYNTHETIC_MARKER"))
        client, original_key = _client(fake)
        try:
            # 실행: 기술 API는 모델 실행 한 번 뒤 고정된 진단 enum만 전달한다.
            response = client.post(
                "/internal/v1/model/execute",
                json=_request(),
                headers={"X-API-KEY": "synthetic-internal-key"},
            )

            # 검증: 원문은 응답에서 제외하며 업무별 채택·재시도는 결정하지 않는다.
            assert response.status_code == 502
            assert response.json()["detail"] == {
                "code": "PROVIDER_EXECUTION_FAILED",
                "retryable": False,
                "failureSignal": signal,
            }
            assert len(fake.calls) == 1
            assert "PRIVATE_SYNTHETIC_MARKER" not in response.text
        finally:
            _restore(original_key)
