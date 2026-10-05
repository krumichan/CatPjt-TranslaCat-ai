from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.core.config import settings


@pytest.fixture
def novel_auth_client(monkeypatch):
    # 준비: 실제 모델 실행 전에 인증 경계만 검증한다.
    with patch("app.core.config_logger.setup_logging"):
        from app.main import app

    monkeypatch.setattr(settings, "SERVER_API_KEY", "synthetic-legacy-key")
    monkeypatch.setattr(settings, "NOVEL_AUTH_ENVIRONMENT", "Development", raising=False)
    monkeypatch.setattr(
        settings, "NOVEL_SERVER_API_KEY", SecretStr("synthetic-novel-key"), raising=False
    )
    client = TestClient(app, raise_server_exceptions=False)
    try:
        yield client
    finally:
        client.close()


@pytest.mark.parametrize("path", [
    "/internal/v1/model/execute", "/internal/v1/model/execute-profile",
    "/internal/v1/model/execute-profile-stream", "/internal/v1/speech/synthesize",
])
def test_novel_key_authorizes_only_supported_explicit_execution(novel_auth_client, path):
    # 실행: 잘못된 본문은 인증 성공 후 DTO 검사에서 거부되어야 한다.
    response = novel_auth_client.post(path, json={}, headers={"X-API-KEY": "synthetic-novel-key"})

    # 검증: 인증 실패와 입력 실패를 구별하고 실제 유료 호출은 만들지 않는다.
    assert response.status_code == 422


@pytest.mark.parametrize(
    "method,path",
    [
        ("GET", "/internal/v1/model/execute"),
        ("POST", "/internal/v1/model/execute/"),
        ("POST", "/internal/v1/speech/transcribe"),
        ("POST", "/api/v1/translate"),
        ("GET", "/docs"),
    ],
)
def test_novel_key_cannot_reach_other_operations(novel_auth_client, method, path):
    # 실행
    response = novel_auth_client.request(
        method, path, json={}, headers={"X-API-KEY": "synthetic-novel-key"}, follow_redirects=False
    )

    # 검증
    assert response.status_code == 401


@pytest.mark.parametrize("shared", ["synthetic-legacy-key", "synthetic-chat-key"])
def test_novel_key_must_not_reuse_existing_service_credentials(shared):
    from app.core.novel_auth import validate_novel_auth_configuration

    # 준비
    configuration = SimpleNamespace(
        NOVEL_AUTH_ENVIRONMENT="Development",
        NOVEL_SERVER_API_KEY=SecretStr(shared),
        SERVER_API_KEY="synthetic-legacy-key",
        CHAT_SERVER_API_KEY=SecretStr("synthetic-chat-key"),
    )

    # 실행 / 검증
    with pytest.raises(ValueError, match="independent"):
        validate_novel_auth_configuration(configuration)


def test_retired_novel_auth_flag_cannot_disable_dedicated_authentication():
    from app.core.novel_auth import authorize_novel_api_key

    # 준비: 과거 false 입력이 남아 있어도 전용 인증은 항상 작동해야 한다.
    configuration = SimpleNamespace(
        NOVEL_AUTH_ENABLED=False,
        NOVEL_AUTH_ENVIRONMENT="Development",
        NOVEL_SERVER_API_KEY=SecretStr("synthetic-novel-key"),
        SERVER_API_KEY="synthetic-legacy-key",
        CHAT_SERVER_API_KEY=SecretStr("synthetic-chat-key"),
    )

    # 실행 / 검증
    assert "NOVEL_AUTH_ENABLED" not in type(settings).model_fields
    assert authorize_novel_api_key(
        configuration, "POST", "/internal/v1/model/execute-profile", "synthetic-novel-key"
    ) is True
    assert authorize_novel_api_key(
        configuration, "POST", "/internal/v1/model/execute", "synthetic-legacy-key"
    ) is None


@pytest.mark.parametrize(
    "changes",
    [
        {"NOVEL_AUTH_ENVIRONMENT": ""},
        {"NOVEL_SERVER_API_KEY": ""},
        {"NOVEL_SERVER_API_KEY": "bad key"},
        {"NOVEL_SERVER_API_KEY": "bad\r\nkey"},
        {"NOVEL_SERVER_API_KEY": "nonascii-키"},
        {"NOVEL_AUTH_ENVIRONMENT": "Production", "NOVEL_SERVER_API_KEY": "A" * 31},
        {"NOVEL_AUTH_ENVIRONMENT": "Production", "NOVEL_SERVER_API_KEY": "synthetic-placeholder"},
    ],
)
def test_invalid_novel_configuration_fails_closed_without_secret_echo(changes):
    from app.core.novel_auth import authorize_novel_api_key, validate_novel_auth_configuration

    # 준비
    values = {
        "NOVEL_AUTH_ENVIRONMENT": "Development",
        "NOVEL_SERVER_API_KEY": "synthetic-novel-key",
        "SERVER_API_KEY": "synthetic-legacy-key",
        "CHAT_SERVER_API_KEY": SecretStr("synthetic-chat-key"),
        **changes,
    }
    values["NOVEL_SERVER_API_KEY"] = SecretStr(values["NOVEL_SERVER_API_KEY"])
    configuration = SimpleNamespace(**values)

    # 실행 / 검증
    with pytest.raises(ValueError) as error:
        validate_novel_auth_configuration(configuration)
    assert "synthetic-novel-key" not in str(error.value)
    assert (
        authorize_novel_api_key(configuration, "POST", "/internal/v1/model/execute", None) is False
    )


def test_novel_enabled_keeps_existing_ll_key(novel_auth_client):
    # 실행
    response = novel_auth_client.post(
        "/internal/v1/model/execute", json={}, headers={"X-API-KEY": "synthetic-legacy-key"}
    )

    # 검증: 공통 키는 종전대로 DTO 검증까지 도달한다.
    assert response.status_code == 422


@pytest.mark.parametrize("path", [
    "/internal/v1/model/execute", "/internal/v1/model/execute-profile",
    "/internal/v1/model/execute-profile-stream", "/internal/v1/speech/synthesize",
])
@pytest.mark.parametrize("key", [None, "incorrect-synthetic-key"])
def test_novel_enabled_rejects_missing_or_wrong_keys(novel_auth_client, path, key):
    # 준비: 유효한 요청 본문이나 provider 실행 전에 인증을 거절해야 한다.
    headers = {} if key is None else {"X-API-KEY": key}

    # 실행
    response = novel_auth_client.post(path, json={}, headers=headers)

    # 검증
    assert response.status_code == 401


@pytest.mark.parametrize("path", [
    "/internal/v1/model/execute-profile", "/internal/v1/model/execute-profile-stream",
])
def test_novel_profiles_reject_existing_common_service_key(novel_auth_client, path):
    # 실행: 공통키로 소설 전용 profile 권한을 얻을 수 없다.
    response = novel_auth_client.post(
        path, json={}, headers={"X-API-KEY": "synthetic-legacy-key"}
    )

    # 검증
    assert response.status_code == 401


def test_novel_auth_preserves_receipt_identity_common_key(novel_auth_client):
    # 준비: BE 영수증의 읽기 전용 기존 계약은 전용 Novel 권한과 별개다.
    path = "/api/v1/account-book/receipts/runtime-identity"

    # 실행
    common = novel_auth_client.get(path, headers={"X-API-KEY": "synthetic-legacy-key"})
    novel = novel_auth_client.get(path, headers={"X-API-KEY": "synthetic-novel-key"})

    # 검증: 실제 provider 호출 없이 기존 공통키만 허용한다.
    assert common.status_code == 200
    assert common.json()["provider_call_count"] == 0
    assert novel.status_code == 401


def test_production_novel_key_shape_enables_only_dedicated_scope():
    from app.core.novel_auth import authorize_novel_api_key, validate_novel_auth_configuration

    # 준비: 난수 생성 검사가 아닌 Production 형식 검사용 합성 문자열이다.
    configuration = SimpleNamespace(
        NOVEL_AUTH_ENVIRONMENT="Production",
        NOVEL_SERVER_API_KEY=SecretStr("A" * 43),
        SERVER_API_KEY="B" * 43,
        CHAT_SERVER_API_KEY=SecretStr("C" * 43),
    )

    # 실행 / 검증: 전용 scope만 허용하고 기존 공통 인증 결정은 그대로 위임한다.
    validate_novel_auth_configuration(configuration)
    assert (
        authorize_novel_api_key(
            configuration, "POST", "/internal/v1/model/execute-profile", "A" * 43
        ) is True
    )
    assert (
        authorize_novel_api_key(
            configuration, "GET", "/api/v1/account-book/receipts/runtime-identity", "A" * 43
        ) is False
    )
    assert (
        authorize_novel_api_key(
            configuration, "POST", "/internal/v1/model/execute", "B" * 43
        ) is None
    )
