"""소설의 언어 정책은 호출자가 소유하며 AI는 범용 요청을 그대로 실행한다."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.ai.model_policy import AiModelTier, get_model_name_for_tier
from app.ai.providers.openai.client import OpenAIService
from app.ai.providers.openai.execution_profiles import EXECUTION_PROFILES
from app.ai.providers.openai.profile_execution import ProfileExecutor
from app.ai.providers.openai.response import OpenAIProviderResponseError
from app.api.dependencies import get_model_execution_provider
from app.api.internal.model_profile_execution import get_profile_executor
from app.core.config import settings

ENDPOINTS = ("execute", "execute-profile", "execute-profile-stream")
OUTPUTS = {
    "en": {"s0": '"The book opened."', "s1": "[A quiet morning]"},
    "ko": {"s0": '"책이 펼쳐졌다."', "s1": "[조용한 아침]"},
}
SCHEMA = {
    "type": "object",
    "properties": {"s0": {"type": "string"}, "s1": {"type": "string"}},
    "required": ["s0", "s1"],
    "additionalProperties": False,
}


def command(endpoint, target):
    language = "English" if target == "en" else "Korean"
    body = {
        "traceId": f"novel-{target}-synthetic",
        "instructions": f"Translate Japanese into {language}. Return one string per source ID.",
        "messages": [{"role": "user", "content": json.dumps({
            "sourceLanguage": "ja", "targetLanguage": target,
            "segments": [{"id": "s0", "text": "本が開いた。"},
                         {"id": "s1", "text": "[静かな朝]"}],
        }, ensure_ascii=False)}],
        "maxOutputTokens": 4096,
        "remainingMilliseconds": 30000,
        "maxProviderCalls": 1,
        "responseSchema": SCHEMA,
        "schemaName": "novel_translation",
        "strict": True,
    }
    if endpoint == "execute":
        body.update(tier="LUNA", reasoningEffort="none", verbosity="low")
    else:
        body["profileId"] = "luna-6-none-v1"
    return body


class SyntheticStream:
    def __init__(self, response):
        self.response = response
        self.closed = False

    async def __aiter__(self):
        text = self.response.output_text
        for part in (text[:len(text) // 2], text[len(text) // 2:]):
            yield SimpleNamespace(type="response.output_text.delta", delta=part)
        yield SimpleNamespace(type="response.completed", response=self.response)

    async def close(self):
        self.closed = True


class SyntheticSdk:
    def __init__(self):
        self.outputs = []
        self.calls = []
        self.options = []
        self.streams = []
        self.failure = None
        self.responses = SimpleNamespace(
            create=AsyncMock(side_effect=self.create),
            with_raw_response=SimpleNamespace(create=AsyncMock(side_effect=self.create_raw)),
        )

    def with_options(self, **kwargs):
        self.options.append(kwargs)
        return self

    def response(self, kwargs):
        self.calls.append(kwargs)
        if self.failure is not None:
            raise self.failure
        return SimpleNamespace(
            status="completed", output=[],
            output_text=json.dumps(self.outputs.pop(0), ensure_ascii=False),
            model=kwargs["model"], service_tier="default",
            usage=SimpleNamespace(
                input_tokens=10, output_tokens=8,
                input_tokens_details=SimpleNamespace(cached_tokens=0),
                output_tokens_details=SimpleNamespace(reasoning_tokens=0),
            ),
        )

    async def create(self, **kwargs):
        response = self.response(kwargs)
        if kwargs.get("stream"):
            stream = SyntheticStream(response)
            self.streams.append(stream)
            return stream
        return response

    async def create_raw(self, **kwargs):
        response = self.response(kwargs)
        return SimpleNamespace(parse=lambda: response, headers={"x-request-id": "synthetic"})


@pytest.fixture
def harness(monkeypatch):
    # 준비: 실제 route·DTO·executor는 사용하고 SDK의 네트워크 진입만 합성 구현으로 바꾼다.
    with patch("app.core.config_logger.setup_logging"):
        from app.main import app
    monkeypatch.setattr(settings, "SERVER_API_KEY", "synthetic-legacy-key")
    monkeypatch.setattr(settings, "NOVEL_AUTH_ENVIRONMENT", "Development")
    monkeypatch.setattr(settings, "NOVEL_SERVER_API_KEY", SecretStr("synthetic-novel-key"))
    sdk = SyntheticSdk()
    service = OpenAIService()
    service._client = sdk
    app.dependency_overrides[get_model_execution_provider] = lambda: service
    app.dependency_overrides[get_profile_executor] = lambda: ProfileExecutor(service)
    client = TestClient(app, raise_server_exceptions=False)
    try:
        yield client, sdk
    finally:
        client.close()
        app.dependency_overrides.pop(get_model_execution_provider, None)
        app.dependency_overrides.pop(get_profile_executor, None)


def events(response):
    return [json.loads(line.removeprefix("data: ")) for line in response.text.splitlines()
            if line.startswith("data: ")]


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_novel_english_and_korean_requests_preserve_opaque_contract_without_cross_language_state(
    harness, endpoint,
):
    # 준비: 같은 실행기를 영어→한국어→영어로 재사용하여 이전 언어의 상태 유입을 찾는다.
    client, sdk = harness
    targets = ("en", "ko", "en")
    sdk.outputs.extend(OUTPUTS[target] for target in targets)

    for target in targets:
        # 실행
        body = command(endpoint, target)
        response = client.post(f"/internal/v1/model/{endpoint}", json=body,
                               headers={"X-API-KEY": "synthetic-novel-key"})

        # 검증: 문장 ID·따옴표·대괄호와 언어별 원문 결과를 그대로 제공한다.
        assert response.status_code == 200
        if endpoint.endswith("stream"):
            emitted = events(response)
            assert [item["type"] for item in emitted] == ["field", "field", "result"]
            assert [(item["key"], item["value"]) for item in emitted[:2]] == list(
                OUTPUTS[target].items()
            )
            result = emitted[-1]["result"]
        else:
            result = response.json()
        assert result["output"] == OUTPUTS[target]
        assert result["providerCalls"] == 1
        sent = sdk.calls[-1]
        assert sent["instructions"] == body["instructions"]
        assert sent["input"] == body["messages"]
        assert sent["text"]["format"]["schema"] == SCHEMA
        assert sent["max_output_tokens"] == 4096
        assert sent["reasoning"] == {"effort": "none"}
        assert sent["store"] is False
        expected_model = (get_model_name_for_tier(AiModelTier.LUNA) if endpoint == "execute"
                          else EXECUTION_PROFILES["luna-6-none-v1"].model)
        assert sent["model"] == expected_model

    assert len(sdk.calls) == len(targets)
    assert all(0 < option["timeout"] <= 30 for option in sdk.options)
    if endpoint != "execute":
        assert all(option["max_retries"] == 0 for option in sdk.options)
    assert all(stream.closed for stream in sdk.streams)


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_novel_english_refusal_remains_failure_without_hidden_retry(harness, endpoint):
    # 준비
    client, sdk = harness
    sdk.failure = OpenAIProviderResponseError(
        "Synthetic private provider body", reason_code="REFUSAL", retryable=False,
    )

    # 실행
    response = client.post(f"/internal/v1/model/{endpoint}", json=command(endpoint, "en"),
                           headers={"X-API-KEY": "synthetic-novel-key"})

    # 검증: SSE와 일반 응답의 기존 실패 전달 계약을 유지하고 새 호출을 만들지 않는다.
    if endpoint.endswith("stream"):
        assert response.status_code == 200
        assert events(response) == [{"type": "error", "code": "REFUSAL"}]
    else:
        assert response.status_code == 502
        assert response.json()["detail"]["code"] == "REFUSAL"
        assert response.json()["detail"]["retryable"] is False
    assert len(sdk.calls) == 1
    assert "Synthetic private provider body" not in response.text


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_novel_english_cannot_expand_provider_call_budget(harness, endpoint):
    # 준비
    client, sdk = harness
    body = command(endpoint, "en")
    body["maxProviderCalls"] = 2

    # 실행
    response = client.post(f"/internal/v1/model/{endpoint}", json=body,
                           headers={"X-API-KEY": "synthetic-novel-key"})

    # 검증: 언어 확장으로 기존 1회 제한을 우회하지 못한다.
    assert response.status_code == 422
    assert sdk.calls == []
