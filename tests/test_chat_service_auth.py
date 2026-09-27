from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.ai.ports import StructuredGenerationResult
from app.api.dependencies import (
    get_chat_ai_reply_service,
    get_chat_translation_service,
    get_model_execution_provider,
)
from app.core.chat_auth import (
    authorize_chat_api_key,
    validate_chat_auth_configuration,
)
from app.core.config import Settings, settings
from app.schemas.chat_ai import ChatAiReplyResponse

CHAT_KEY = "synthetic-chat-local-credential"
LEGACY_KEY = "synthetic-legacy-server-credential"
PRODUCTION_SHAPE_KEY = "mUJ3OvSnvtzmvN20zGhFHRbltoCsmGIWbaDrhIkC2uE="
CHAT_PATHS = ("/api/v1/chat/translate", "/api/v1/chat/ai/reply")


class FakeChatServices:
    def __init__(self):
        self.calls = []

    async def translate(self, **kwargs):
        self.calls.append(("translate", kwargs))
        return "synthetic translation"

    async def generate_reply(self, request):
        self.calls.append(("reply", request.request_id))
        return ChatAiReplyResponse(
            request_id=request.request_id,
            should_respond=False,
            reply=None,
            language_code=None,
        )

    async def execute_explicit(self, **kwargs):
        self.calls.append(("model", kwargs))
        return StructuredGenerationResult(
            data={"answer": "synthetic"},
            input_tokens=1,
            output_tokens=1,
            provider="fake",
            model="synthetic",
        )


@pytest.fixture
def authenticated_app(monkeypatch):
    # 준비 — 실제 Provider와 lifespan warm-up을 실행하지 않고 서비스 경계만 대체한다.
    with patch("app.core.config_logger.setup_logging"):
        from app.main import app

    monkeypatch.setattr(settings, "SERVER_API_KEY", LEGACY_KEY)
    monkeypatch.setattr(settings, "CHAT_AUTH_MODE", "dual")
    monkeypatch.setattr(settings, "CHAT_AUTH_ENVIRONMENT", "Development")
    monkeypatch.setattr(settings, "CHAT_SERVER_API_KEY", SecretStr(CHAT_KEY))
    fake = FakeChatServices()
    original_overrides = app.dependency_overrides.copy()
    app.dependency_overrides[get_chat_translation_service] = lambda: fake
    app.dependency_overrides[get_chat_ai_reply_service] = lambda: fake
    app.dependency_overrides[get_model_execution_provider] = lambda: fake
    client = TestClient(app)
    try:
        yield client, fake
    finally:
        client.close()
        app.dependency_overrides.clear()
        app.dependency_overrides.update(original_overrides)


def _payload(path):
    if path.endswith("/translate"):
        return {"text": "synthetic text", "target_language_code": "en"}
    return {
        "requestId": "synthetic-request",
        "triggerType": "MENTION",
        "room": {"roomId": 1, "roomType": "GROUP", "name": None, "description": None},
        "aiMember": {
            "aiMemberId": 2,
            "nickname": "synthetic",
            "bio": None,
            "personaPrompt": None,
            "originalLanguageCode": "en",
        },
        "triggerMessage": {
            "messageId": 3,
            "senderId": "member-1",
            "senderName": "synthetic",
            "content": "synthetic",
            "createdAt": "2026-09-27T01:00:00",
        },
        "contextMessages": [],
    }


@pytest.mark.parametrize("mode", ["dual", "dedicated"])
@pytest.mark.parametrize("path", CHAT_PATHS)
def test_dedicated_key_authorizes_only_current_chat_contracts(
    authenticated_app, monkeypatch, mode, path
):
    # 준비
    client, fake = authenticated_app
    monkeypatch.setattr(settings, "CHAT_AUTH_MODE", mode)

    # 실행
    response = client.post(path, json=_payload(path), headers={"X-API-KEY": CHAT_KEY})

    # 검증
    assert response.status_code == 200
    assert len(fake.calls) == 1
    if path.endswith("/translate"):
        assert response.json() == {"translated_text": "synthetic translation"}
    else:
        assert response.json() == {
            "requestId": "synthetic-request",
            "shouldRespond": False,
            "reply": None,
            "languageCode": None,
        }


@pytest.mark.parametrize("mode,expected", [("legacy", 200), ("dual", 200), ("dedicated", 401)])
@pytest.mark.parametrize("path", CHAT_PATHS)
def test_legacy_sender_transition_is_explicit(authenticated_app, monkeypatch, mode, expected, path):
    # 준비
    client, fake = authenticated_app
    monkeypatch.setattr(settings, "CHAT_AUTH_MODE", mode)

    # 실행
    response = client.post(path, json=_payload(path), headers={"X-API-KEY": LEGACY_KEY})

    # 검증
    assert response.status_code == expected
    assert len(fake.calls) == (1 if expected == 200 else 0)


@pytest.mark.parametrize("path", CHAT_PATHS)
def test_legacy_default_does_not_activate_new_key(authenticated_app, monkeypatch, path):
    # 준비
    client, fake = authenticated_app
    monkeypatch.setattr(settings, "CHAT_AUTH_MODE", "legacy")

    # 실행
    response = client.post(path, json=_payload(path), headers={"X-API-KEY": CHAT_KEY})

    # 검증
    assert response.status_code == 401
    assert fake.calls == []


@pytest.mark.parametrize(
    "key",
    [None, "", "wrong-key", "revoked-chat-key", "opposite-direction-key", "other-environment-key"],
)
@pytest.mark.parametrize("path", CHAT_PATHS)
def test_wrong_missing_revoked_direction_or_environment_key_never_reaches_service(
    authenticated_app, key, path
):
    # 준비
    client, fake = authenticated_app
    headers = {} if key is None else {"X-API-KEY": key}

    # 실행
    response = client.post(path, json=_payload(path), headers=headers)

    # 검증 — API key에는 자체 만료 claim이 없다. 폐기 키는 현재 설정에서 제거된 값을 뜻한다.
    assert response.status_code == 401
    assert response.json() == {"detail": "인증되지 않은 요청입니다."}
    assert fake.calls == []


@pytest.mark.parametrize(
    "method,path",
    [
        ("GET", "/api/v1/chat/translate"),
        ("DELETE", "/api/v1/chat/ai/reply"),
        ("POST", "/api/v1/chat/translate/"),
        ("POST", "/api/v1/chat/future"),
        ("POST", "/internal/v1/model/execute"),
        ("POST", "/internal/v1/speech/transcribe"),
        ("POST", "/api/v1/voice/translate"),
        ("GET", "/docs"),
        ("GET", "/"),
    ],
)
def test_chat_key_cannot_expand_scope_by_path_or_method(authenticated_app, method, path):
    # 준비
    client, fake = authenticated_app

    # 실행
    response = client.request(
        method, path, json={}, headers={"X-API-KEY": CHAT_KEY}, follow_redirects=False
    )

    # 검증
    assert response.status_code == 401
    assert fake.calls == []
    if path == "/internal/v1/model/execute":
        assert response.json() == {
            "detail": {"code": "MODEL_EXECUTION_UNAUTHORIZED", "retryable": False}
        }
    elif path.startswith("/internal/v1/"):
        assert response.json()["detail"]["retryable"] is False
        assert response.json()["detail"]["stage"] == "AUTH"


@pytest.mark.parametrize("mode", ["legacy", "dual", "dedicated"])
def test_original_generic_executor_key_and_wire_are_preserved(authenticated_app, monkeypatch, mode):
    # 준비
    client, fake = authenticated_app
    monkeypatch.setattr(settings, "CHAT_AUTH_MODE", mode)
    payload = {
        "traceId": "synthetic",
        "instructions": "synthetic",
        "messages": [{"role": "user", "content": "synthetic"}],
        "tier": "MINI",
        "reasoningEffort": "low",
        "verbosity": "low",
        "maxOutputTokens": 32,
        "remainingMilliseconds": 1000,
        "maxProviderCalls": 1,
        "responseSchema": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
        "schemaName": "synthetic",
        "strict": True,
        "taskName": "synthetic",
    }

    # 실행
    response = client.post(
        "/internal/v1/model/execute", json=payload, headers={"X-API-KEY": LEGACY_KEY}
    )

    # 검증 — 원래 범용 executor는 그대로 대역 Provider를 호출한다. Chat prompt로 대체하지 않는다.
    assert response.status_code == 200
    assert response.json()["providerCalls"] == 1
    assert len(fake.calls) == 1
    assert fake.calls[0][0] == "model"


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
    # 준비 — 개인 .env 파일을 사용하지 않는 합성 설정 객체다.
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
    assert failure.value.__cause__ is None
    assert authorize_chat_api_key(configuration, "POST", CHAT_PATHS[0], CHAT_KEY) is False


def test_settings_binding_hides_key_and_requires_explicit_environment():
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

    # 검증 — 환경 분리는 배포별 별도 키 주입이다. 키 문자열에서 환경이나 entropy를 증명하지 않는다.
    assert PRODUCTION_SHAPE_KEY not in repr(configuration)
    assert (
        authorize_chat_api_key(configuration, "POST", CHAT_PATHS[0], PRODUCTION_SHAPE_KEY) is True
    )
    assert authorize_chat_api_key(configuration, "POST", CHAT_PATHS[0], CHAT_KEY) is False


@pytest.mark.asyncio
async def test_invalid_configuration_stops_before_warmup(authenticated_app, monkeypatch):
    # 준비
    from app.main import app, lifespan

    monkeypatch.setattr(settings, "CHAT_AUTH_MODE", "dedicated")
    monkeypatch.setattr(settings, "CHAT_SERVER_API_KEY", SecretStr(""))

    # 실행 / 검증 — lifespan의 첫 검사에서 거부되어 외부 warm-up까지 진행하지 않는다.
    with patch("app.main.get_voice_translation_service") as provider:
        with pytest.raises(ValueError):
            async with lifespan(app):
                pytest.fail("Invalid startup must not reach the serving phase")
        provider.assert_not_called()
