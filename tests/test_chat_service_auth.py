from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.ai.ports import StructuredGenerationResult
from app.api.dependencies import get_model_execution_provider
from app.core.chat_auth import authorize_chat_api_key, validate_chat_auth_configuration
from app.core.config import Settings, settings

CHAT_KEY = "synthetic-chat-local-credential"
LEGACY_KEY = "synthetic-legacy-server-credential"
PRODUCTION_SHAPE_KEY = "mUJ3OvSnvtzmvN20zGhFHRbltoCsmGIWbaDrhIkC2uE="
EXECUTE_PATH = "/internal/v1/model/execute"


class FakeModelProvider:
    def __init__(self):
        self.calls = []

    async def execute_explicit(self, **kwargs):
        self.calls.append(kwargs)
        return StructuredGenerationResult(
            data={"answer": "synthetic"},
            input_tokens=1,
            output_tokens=1,
            provider="fake",
            model="synthetic",
        )


@pytest.fixture
def authenticated_app(monkeypatch):
    # 준비 — 실제 Provider와 lifespan warm-up을 건너뛰고 실행 port만 대체한다.
    with patch("app.core.config_logger.setup_logging"):
        from app.main import app

    monkeypatch.setattr(settings, "SERVER_API_KEY", LEGACY_KEY)
    monkeypatch.setattr(settings, "CHAT_AUTH_MODE", "dual")
    monkeypatch.setattr(settings, "CHAT_AUTH_ENVIRONMENT", "Development")
    monkeypatch.setattr(settings, "CHAT_SERVER_API_KEY", SecretStr(CHAT_KEY))
    fake = FakeModelProvider()
    original_overrides = app.dependency_overrides.copy()
    app.dependency_overrides[get_model_execution_provider] = lambda: fake
    client = TestClient(app)
    try:
        yield client, fake
    finally:
        client.close()
        app.dependency_overrides.clear()
        app.dependency_overrides.update(original_overrides)


def _payload():
    return {
        "traceId": "synthetic-chat",
        "instructions": "Synthetic chat-owned instruction",
        "messages": [{"role": "user", "content": "Synthetic untrusted input"}],
        "tier": "LUNA",
        "reasoningEffort": "none",
        "verbosity": "medium",
        "maxOutputTokens": 2048,
        "remainingMilliseconds": 20000,
        "maxProviderCalls": 1,
        "responseSchema": None,
        "schemaName": None,
        "strict": False,
        "taskName": "AI_CHAT_REPLY",
    }


@pytest.mark.parametrize("mode", ["dual", "dedicated"])
def test_dedicated_key_executes_without_task_recipe(authenticated_app, monkeypatch, mode):
    # 준비
    client, fake = authenticated_app
    monkeypatch.setattr(settings, "CHAT_AUTH_MODE", mode)

    # 실행
    response = client.post(EXECUTE_PATH, json=_payload(), headers={"X-API-KEY": CHAT_KEY})

    # 검증 — taskName은 메타데이터이며 provider 인자에 전달되지 않는다.
    assert response.status_code == 200
    assert response.json()["providerCalls"] == 1
    assert len(fake.calls) == 1
    assert fake.calls[0]["instructions"] == "Synthetic chat-owned instruction"
    assert "task_name" not in fake.calls[0]


@pytest.mark.parametrize("mode", ["legacy", "dual", "dedicated"])
def test_existing_ll_global_key_keeps_generic_execution(authenticated_app, monkeypatch, mode):
    # 준비
    client, fake = authenticated_app
    monkeypatch.setattr(settings, "CHAT_AUTH_MODE", mode)

    # 실행
    response = client.post(EXECUTE_PATH, json=_payload(), headers={"X-API-KEY": LEGACY_KEY})

    # 검증
    assert response.status_code == 200
    assert len(fake.calls) == 1


def test_legacy_mode_does_not_activate_chat_key(authenticated_app, monkeypatch):
    # 준비
    client, fake = authenticated_app
    monkeypatch.setattr(settings, "CHAT_AUTH_MODE", "legacy")

    # 실행
    response = client.post(EXECUTE_PATH, json=_payload(), headers={"X-API-KEY": CHAT_KEY})

    # 검증
    assert response.status_code == 401
    assert fake.calls == []


@pytest.mark.parametrize(
    "key",
    [None, "", "wrong-key", "revoked-chat-key", "opposite-direction-key", "other-environment-key"],
)
def test_missing_or_wrong_key_never_reaches_provider(authenticated_app, key):
    # 준비
    client, fake = authenticated_app
    headers = {} if key is None else {"X-API-KEY": key}

    # 실행
    response = client.post(EXECUTE_PATH, json=_payload(), headers=headers)

    # 검증
    assert response.status_code == 401
    assert response.json() == {
        "detail": {"code": "MODEL_EXECUTION_UNAUTHORIZED", "retryable": False}
    }
    assert fake.calls == []


@pytest.mark.parametrize(
    "method,path",
    [
        ("GET", EXECUTE_PATH),
        ("DELETE", EXECUTE_PATH),
        ("POST", EXECUTE_PATH + "/"),
        ("POST", "/internal/v1/speech/transcribe"),
        ("POST", "/api/v1/voice/translate"),
        ("GET", "/docs"),
    ],
)
def test_chat_key_cannot_expand_scope(authenticated_app, method, path):
    # 준비
    client, fake = authenticated_app

    # 실행
    response = client.request(
        method, path, json={}, headers={"X-API-KEY": CHAT_KEY}, follow_redirects=False
    )

    # 검증
    assert response.status_code == 401
    assert fake.calls == []


@pytest.mark.parametrize(
    "changes",
    [
        {"CHAT_AUTH_MODE": "invalid"},
        {"CHAT_AUTH_ENVIRONMENT": ""},
        {"CHAT_AUTH_ENVIRONMENT": "local"},
        {"CHAT_SERVER_API_KEY": ""},
        {"CHAT_SERVER_API_KEY": "bad\r\nkey"},
        {"CHAT_SERVER_API_KEY": "bad key"},
        {"CHAT_SERVER_API_KEY": LEGACY_KEY},
        {"CHAT_AUTH_ENVIRONMENT": "Production", "CHAT_SERVER_API_KEY": "replace-with-real-key"},
        {"CHAT_AUTH_ENVIRONMENT": "Production", "CHAT_SERVER_API_KEY": "${CHAT_KEY}"},
    ],
)
def test_invalid_active_settings_fail_without_secret_diagnostics(changes):
    # 준비
    values = dict(
        CHAT_AUTH_MODE="dual",
        CHAT_AUTH_ENVIRONMENT="Development",
        CHAT_SERVER_API_KEY=CHAT_KEY,
        SERVER_API_KEY=LEGACY_KEY,
    )
    values.update(changes)
    values["CHAT_SERVER_API_KEY"] = SecretStr(values["CHAT_SERVER_API_KEY"])
    configuration = SimpleNamespace(**values)

    # 실행
    with pytest.raises(ValueError) as failure:
        validate_chat_auth_configuration(configuration)

    # 검증
    assert CHAT_KEY not in str(failure.value)
    assert LEGACY_KEY not in str(failure.value)
    assert authorize_chat_api_key(configuration, "POST", EXECUTE_PATH, CHAT_KEY) is False


def test_settings_binding_hides_key_and_separates_environment():
    # 준비
    configuration = Settings(
        _env_file=None,
        CHAT_AUTH_MODE="dedicated",
        CHAT_AUTH_ENVIRONMENT="Production",
        CHAT_SERVER_API_KEY=PRODUCTION_SHAPE_KEY,
        SERVER_API_KEY=LEGACY_KEY,
    )

    # 실행
    validate_chat_auth_configuration(configuration)

    # 검증
    assert PRODUCTION_SHAPE_KEY not in repr(configuration)
    assert authorize_chat_api_key(configuration, "POST", EXECUTE_PATH, PRODUCTION_SHAPE_KEY) is True
    assert authorize_chat_api_key(configuration, "POST", EXECUTE_PATH, CHAT_KEY) is None


@pytest.mark.asyncio
async def test_invalid_configuration_stops_before_warmup(authenticated_app, monkeypatch):
    # 준비
    from app.main import app, lifespan

    monkeypatch.setattr(settings, "CHAT_AUTH_MODE", "dedicated")
    monkeypatch.setattr(settings, "CHAT_SERVER_API_KEY", SecretStr(""))

    # 실행 / 검증
    with patch("app.main.get_voice_translation_service") as provider:
        with pytest.raises(ValueError):
            async with lifespan(app):
                pytest.fail("Invalid startup must not reach the serving phase")
        provider.assert_not_called()
