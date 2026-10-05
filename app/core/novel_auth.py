"""소설 서비스의 범용 모델·음성 실행에만 유효한 별도 수신 자격증명."""

import secrets
from typing import Protocol

from pydantic import SecretStr

NOVEL_OPERATIONS = frozenset(
    {
        ("POST", "/internal/v1/model/execute"),
        ("POST", "/internal/v1/model/execute-profile"),
        ("POST", "/internal/v1/model/execute-profile-stream"),
        ("POST", "/internal/v1/speech/synthesize"),
    }
)


class NovelAuthSettings(Protocol):
    NOVEL_AUTH_ENVIRONMENT: str
    NOVEL_SERVER_API_KEY: SecretStr
    SERVER_API_KEY: str
    CHAT_SERVER_API_KEY: SecretStr


def validate_novel_auth_configuration(settings: NovelAuthSettings) -> None:
    # 소설 전용 인증은 항상 검증한다. 서비스 비활성 설정으로 필수 키 검사를 건너뛰지 않는다.
    if settings.NOVEL_AUTH_ENVIRONMENT not in {"Development", "Production"}:
        raise ValueError("NOVEL_AUTH_ENVIRONMENT must be Development or Production")
    key = settings.NOVEL_SERVER_API_KEY.get_secret_value()
    if not key or not key.isascii() or any(ord(value) <= 32 or ord(value) == 127 for value in key):
        raise ValueError("NOVEL_SERVER_API_KEY must be a nonempty opaque header credential")

    # 타 서비스 키 재사용은 경로 권한의 주체를 구분할 수 없으므로 설정 단계에서 거부한다.
    for existing in (settings.SERVER_API_KEY, settings.CHAT_SERVER_API_KEY.get_secret_value()):
        if existing and secrets.compare_digest(key.encode(), existing.encode()):
            raise ValueError("NOVEL_SERVER_API_KEY must be independent from existing service keys")
    compact = key.lower().replace("-", "").replace("_", "")
    if settings.NOVEL_AUTH_ENVIRONMENT == "Production" and (
        len(key) < 32
        or compact in {"changeme", "replaceme", "placeholder", "yourapikey"}
        or "${" in key
        or key.startswith(("<", "%"))
        or key.lower().startswith(("replace-with-", "synthetic-", "test-", "development-"))
    ):
        raise ValueError("NOVEL_SERVER_API_KEY must not be a production placeholder")


def authorize_novel_api_key(
    settings: NovelAuthSettings, method: str, path: str, provided: str | None
) -> bool | None:
    # 잘못된 자격증명 구성은 fail-closed이며 None만 기존 공통 인증으로 넘긴다.
    try:
        validate_novel_auth_configuration(settings)
    except ValueError:
        return False
    key = settings.NOVEL_SERVER_API_KEY.get_secret_value()
    if provided and secrets.compare_digest(provided.encode(), key.encode()):
        return (method, path) in NOVEL_OPERATIONS
    return None
