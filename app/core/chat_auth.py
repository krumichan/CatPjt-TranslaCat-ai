import secrets
from typing import Protocol

from pydantic import SecretStr

CHAT_OPERATIONS = frozenset(
    {
        ("POST", "/api/v1/chat/translate"),
        ("POST", "/api/v1/chat/ai/reply"),
    }
)


class ChatAuthSettings(Protocol):
    CHAT_AUTH_MODE: str
    CHAT_AUTH_ENVIRONMENT: str
    CHAT_SERVER_API_KEY: SecretStr
    SERVER_API_KEY: str


def validate_chat_auth_configuration(settings: ChatAuthSettings) -> None:
    # 기본 legacy는 기존 BE/LL 자격증명을 교체하지 않는다. 전용 수신은 명시적으로 켠다.
    if settings.CHAT_AUTH_MODE == "legacy":
        return
    if settings.CHAT_AUTH_MODE not in {"dual", "dedicated"}:
        raise ValueError("CHAT_AUTH_MODE must be legacy, dual, or dedicated")
    if settings.CHAT_AUTH_ENVIRONMENT not in {"Development", "Production"}:
        raise ValueError("CHAT_AUTH_ENVIRONMENT must be Development or Production")

    # 전용 키의 원문은 그대로 비교한다. 다른 환경 키나 전역 키를 fallback으로 사용하지 않는다.
    key = settings.CHAT_SERVER_API_KEY.get_secret_value()
    if not key or not key.isascii() or any(ord(value) <= 32 or ord(value) == 127 for value in key):
        raise ValueError("CHAT_SERVER_API_KEY must be a nonempty opaque header credential")
    if secrets.compare_digest(key.encode(), settings.SERVER_API_KEY.encode()):
        raise ValueError("CHAT_SERVER_API_KEY must be independent from SERVER_API_KEY")
    if settings.CHAT_AUTH_ENVIRONMENT == "Production" and _is_placeholder(key):
        raise ValueError("CHAT_SERVER_API_KEY must not be a production placeholder")


def authorize_chat_api_key(
    settings: ChatAuthSettings, method: str, path: str, provided: str | None
) -> bool | None:
    # 잘못된 활성 구성은 요청 시에도 거부한다. None만 기존 공통 인증 경로로 넘긴다.
    try:
        validate_chat_auth_configuration(settings)
    except ValueError:
        return False
    if settings.CHAT_AUTH_MODE == "legacy":
        return None

    key = settings.CHAT_SERVER_API_KEY.get_secret_value()
    if provided and secrets.compare_digest(provided.encode(), key.encode()):
        return (method, path) in CHAT_OPERATIONS

    # 전환 완료 모드에서는 옛 전역 키가 Chat 추론으로 진입하지 못한다.
    if settings.CHAT_AUTH_MODE == "dedicated" and (
        path == "/api/v1/chat" or path.startswith("/api/v1/chat/")
    ):
        return False
    return None


def _is_placeholder(key: str) -> bool:
    compact = key.lower().replace("-", "").replace("_", "")
    return (
        compact in {"changeme", "replaceme", "placeholder", "example", "yourapikey"}
        or key.startswith("<")
        and key.endswith(">")
        or "${" in key
        or key.startswith("%")
        and key.endswith("%")
        or key.lower().startswith(
            ("replace-with-", "replace_with_", "synthetic-", "test-", "development-")
        )
    )
