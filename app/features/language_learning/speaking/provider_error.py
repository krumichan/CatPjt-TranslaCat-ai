from __future__ import annotations

from app.features.language_learning.speaking.errors import SpeakingStageException
from app.schemas.language_learning_speaking import (
    SpeakingErrorCode,
    SpeakingStage,
)


def map_provider_exception(
    exc: Exception,
    *,
    stage: SpeakingStage,
    fallback_code: SpeakingErrorCode,
    fallback_message: str,
) -> SpeakingStageException:
    status_code = _read_status_code(exc)
    message = f"{type(exc).__name__} {exc}".lower()

    if status_code == 429 or any(
        marker in message
        for marker in (
            "rate limit",
            "ratelimit",
            "resource_exhausted",
            "resource exhausted",
            "too many requests",
        )
    ):
        return SpeakingStageException(
            code=SpeakingErrorCode.PROVIDER_RATE_LIMITED,
            stage=stage,
            message="AI Provider 요청 한도에 도달했습니다.",
            retryable=True,
        )

    if status_code in {408, 504} or "deadline exceeded" in message:
        return SpeakingStageException(
            code=SpeakingErrorCode.PROVIDER_TIMEOUT,
            stage=stage,
            message="AI Provider 응답 시간이 초과되었습니다.",
            retryable=True,
        )

    if any(
        marker in message
        for marker in (
            "safety",
            "blocked",
            "prohibited content",
            "unsafe",
        )
    ):
        return SpeakingStageException(
            code=SpeakingErrorCode.UNSAFE_TOPIC,
            stage=stage,
            message="안전 정책에 따라 해당 Speaking 요청을 처리할 수 없습니다.",
            retryable=False,
        )

    return SpeakingStageException(
        code=fallback_code,
        stage=stage,
        message=fallback_message,
        retryable=True,
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
