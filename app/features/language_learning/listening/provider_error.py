from __future__ import annotations

from app.features.language_learning.listening.errors import ListeningStageException
from app.schemas.language_learning_listening import ListeningErrorCode, ListeningStage


def map_provider_exception(
    exc: Exception,
    *,
    stage: ListeningStage,
    fallback_code: ListeningErrorCode,
    fallback_message: str,
) -> ListeningStageException:
    status = _read_status_code(exc)
    message = f"{type(exc).__name__} {exc}".lower()
    if status == 429 or any(
        marker in message
        for marker in (
            "rate limit",
            "ratelimit",
            "resource_exhausted",
            "too many requests",
        )
    ):
        return ListeningStageException(
            ListeningErrorCode.PROVIDER_RATE_LIMITED,
            stage,
            "AI Provider 요청 한도에 도달했습니다.",
            True,
        )
    if status in {408, 504} or "deadline exceeded" in message:
        return ListeningStageException(
            ListeningErrorCode.PROVIDER_TIMEOUT,
            stage,
            "AI Provider 응답 시간이 초과되었습니다.",
            True,
        )
    if status is not None and 500 <= status <= 599:
        return ListeningStageException(
            ListeningErrorCode.PROVIDER_UNAVAILABLE,
            stage,
            "AI Provider를 일시적으로 사용할 수 없습니다.",
            True,
        )
    if any(marker in message for marker in ("safety", "blocked", "unsafe")):
        return ListeningStageException(
            ListeningErrorCode.UNSAFE_CONTENT,
            stage,
            "안전 정책에 따라 요청을 처리할 수 없습니다.",
            False,
        )
    return ListeningStageException(
        fallback_code,
        stage,
        fallback_message,
        False,
    )


def _read_status_code(exc: Exception) -> int | None:
    for attribute in ("status_code", "status", "code"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None
