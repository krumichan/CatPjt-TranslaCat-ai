"""문서 전용 모델 선택과 기존 내부 실행 경계를 검증한다."""

import asyncio
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError

from app.ai.model_policy import AiModelTier, get_model_name_for_tier
from app.ai.providers.openai.client import OpenAIService
from app.api.dependencies import get_model_execution_provider
from app.core.config import Settings, settings
from app.main import app

_PINNED = "gpt-5.4-mini-2026-03-17"
_PATH = "/internal/v1/model/execute"
_SHARED = "synthetic-shared-be-key"
_NOVEL = "synthetic-novel-execution-key"
_CHAT = "synthetic-chat-execution-key"
_EXPECTED_ARGUMENTS = {
    "instructions", "messages", "tier", "reasoning_effort", "verbosity", "max_output_tokens",
    "remaining_milliseconds", "response_schema", "schema_name_value", "strict",
}


def _body(**overrides):
    """문서 BE와 같은 실행 식별자 및 엄격 스키마를 가진 합성 요청을 만든다."""
    return {
        "traceId": "synthetic-document:0",
        "taskName": "DOCUMENT_TRANSLATION_V1",
        "instructions": "Synthetic system instruction",
        "messages": [{"role": "user", "content": "Synthetic document content"}],
        "tier": "MINI",
        "reasoningEffort": "low",
        "verbosity": "low",
        "maxOutputTokens": 256,
        "remainingMilliseconds": 5000,
        "maxProviderCalls": 1,
        "responseSchema": {
            "type": "object", "properties": {"answer": {"type": "string"}},
            "required": ["answer"], "additionalProperties": False,
        },
        "schemaName": "document_translation_v1",
        "strict": True,
    } | overrides


@pytest.fixture
def execution(monkeypatch):
    """실제 내부 인증과 HTTP 계약을 통과시키되 SDK 호출만 대체한다."""
    monkeypatch.setattr(settings, "SERVER_API_KEY", _SHARED)
    monkeypatch.setattr(settings, "NOVEL_AUTH_ENVIRONMENT", "Development")
    monkeypatch.setattr(settings, "NOVEL_SERVER_API_KEY", SecretStr(_NOVEL))
    monkeypatch.setattr(settings, "CHAT_AUTH_MODE", "dedicated")
    monkeypatch.setattr(settings, "CHAT_AUTH_ENVIRONMENT", "Development")
    monkeypatch.setattr(settings, "CHAT_SERVER_API_KEY", SecretStr(_CHAT))
    calls = []

    class FakeProvider:
        async def execute_explicit(self, **options):
            """선택된 옵션과 실제 응답에 사용하는 모델을 검증 전에 기록한다."""
            calls.append(options)
            return SimpleNamespace(
                data={"answer": "synthetic"}, input_tokens=2, output_tokens=1,
                provider="openai",
                model=options.get("model_override", get_model_name_for_tier(options["tier"])),
            )

    monkeypatch.setitem(app.dependency_overrides, get_model_execution_provider, FakeProvider)
    yield TestClient(app), calls


@pytest.mark.parametrize(
    ("configured", "changes", "key", "uses_override"),
    [
        ("", {}, _SHARED, False),
        (_PINNED, {}, _SHARED, True),
        (_PINNED, {"taskName": "LL_WRITING_V1"}, _SHARED, False),
        (_PINNED, {"taskName": None}, _SHARED, False),
        (_PINNED, {"schemaName": "another_schema"}, _SHARED, False),
        (_PINNED, {"strict": False}, _SHARED, False),
        (_PINNED, {"tier": "LUNA", "reasoningEffort": "none"}, _SHARED, False),
        (_PINNED, {}, _CHAT, False),
        (_PINNED, {}, _NOVEL, False),
    ],
)
def test_model_override_requires_one_document_contract(
    monkeypatch, execution, configured, changes, key, uses_override,
):
    """선택된 문서와 다른 내부 호출의 SDK 인자 및 결과 모델을 비교한다."""
    monkeypatch.setattr(settings, "OPENAI_MODEL_DOCUMENT_TRANSLATION", configured)
    client, calls = execution
    request = _body(**changes)

    response = client.post(_PATH, headers={"X-API-KEY": key}, json=request)

    assert response.status_code == 200
    assert response.json()["providerCalls"] == 1
    default_model = get_model_name_for_tier(AiModelTier(request["tier"]))
    assert response.json()["model"] == (_PINNED if uses_override else default_model)
    expected = _EXPECTED_ARGUMENTS | ({"model_override"} if uses_override else set())
    assert len(calls) == 1 and set(calls[0]) == expected
    assert calls[0]["remaining_milliseconds"] == request["remainingMilliseconds"]
    assert calls[0]["response_schema"] == request["responseSchema"]


def test_model_name_is_not_a_request_field(monkeypatch, execution):
    """인증 실패와 임의 모델 인자는 기존 상태 코드로 거절한다."""
    monkeypatch.setattr(settings, "OPENAI_MODEL_DOCUMENT_TRANSLATION", _PINNED)
    client, calls = execution

    unauthorized = client.post(_PATH, headers={"X-API-KEY": "untrusted"}, json=_body())
    arbitrary_model = client.post(
        _PATH, headers={"X-API-KEY": _SHARED}, json=_body(model="untrusted-model"),
    )

    assert unauthorized.status_code == 401 and arbitrary_model.status_code == 422
    assert not calls


def test_only_verified_snapshot_can_be_configured():
    """공백 설정으로 기존 모델을 유지하고 검증하지 않은 모델 이름을 거절한다."""
    default = Settings(_env_file=None, OPENAI_MODEL_DOCUMENT_TRANSLATION="")
    pinned = Settings(_env_file=None, OPENAI_MODEL_DOCUMENT_TRANSLATION=_PINNED)

    assert default.OPENAI_MODEL_DOCUMENT_TRANSLATION == ""
    assert pinned.OPENAI_MODEL_DOCUMENT_TRANSLATION == _PINNED
    with pytest.raises(ValidationError):
        Settings(_env_file=None, OPENAI_MODEL_DOCUMENT_TRANSLATION="gpt-5.4-mini")


@pytest.mark.parametrize("override", [None, _PINNED])
def test_sdk_uses_one_call_and_same_timeout(monkeypatch, override):
    """동일한 입력과 timeout에서 선택 모델만 달라지는지 확인한다."""
    calls = []
    timeouts = []

    class FakeSdk:
        def with_options(self, *, timeout):
            """SDK로 들어가는 제한 시간을 기록한다."""
            timeouts.append(timeout)
            return self

        async def create(self, **options):
            """한 번 호출된 시점에 JSON 결과와 모델 ID를 반환한다."""
            calls.append(options)
            return SimpleNamespace(status="completed", output_text='{"answer":"synthetic"}',
                                   output=[], model=options["model"], usage=None)

    fake = FakeSdk()
    fake.responses = fake
    service = OpenAIService()
    service._client = fake
    monkeypatch.setattr(settings, "OPENAI_REQUEST_TIMEOUT_SECONDS", 1.0)
    body = _body()
    extra = {} if override is None else {"model_override": override}

    result = asyncio.run(service.execute_explicit(
        instructions=body["instructions"], messages=body["messages"], tier=AiModelTier.MINI,
        reasoning_effort="low", verbosity="low", max_output_tokens=body["maxOutputTokens"],
        remaining_milliseconds=body["remainingMilliseconds"],
        response_schema=body["responseSchema"],
        schema_name_value=body["schemaName"], strict=True, **extra,
    ))

    assert len(calls) == 1 and timeouts == [1.0]
    assert calls[0]["model"] == result.model == (override or settings.OPENAI_MODEL_MINI)
    assert calls[0]["text"]["format"]["schema"] == body["responseSchema"]
    assert calls[0]["input"] == body["messages"] and calls[0]["store"] is False
