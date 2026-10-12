from __future__ import annotations

from openai import APIConnectionError, APITimeoutError

from app.schemas.voice_translation import VoiceError, VoiceErrorCode, VoiceStage


class VoicePipelineException(Exception):
    def __init__(
        self,
        *,
        code: VoiceErrorCode,
        stage: VoiceStage,
        message: str,
        retryable: bool,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.stage = stage
        self.public_message = message
        self.retryable = retryable

    def as_error(self) -> VoiceError:
        return VoiceError(
            code=self.code,
            stage=self.stage,
            message=self.public_message,
            retryable=self.retryable,
        )


def map_translation_provider_exception(exc: Exception) -> VoicePipelineException:
    # 공급자의 영구 실패 판정을 우선 보존하고 SDK 전송 오류를 구분한다.
    status_code = _read_status_code(exc)
    explicitly_permanent = getattr(exc, "retryable", None) is False
    if isinstance(exc, (TimeoutError, APITimeoutError)) or status_code in {408, 504}:
        return VoicePipelineException(
            code=VoiceErrorCode.TRANSLATION_TIMEOUT,
            stage=VoiceStage.TRANSLATION,
            message="번역 Provider 응답 시간이 초과되었습니다.",
            retryable=not explicitly_permanent,
        )

    retryable = not explicitly_permanent and (
        isinstance(exc, (ConnectionError, APIConnectionError))
        or status_code
        in {
            429,
            500,
            502,
            503,
        }
    )
    return VoicePipelineException(
        code=VoiceErrorCode.TRANSLATION_FAILED,
        stage=VoiceStage.TRANSLATION,
        message="확정 원문의 번역에 실패했습니다.",
        retryable=retryable,
    )


def _read_status_code(exc: Exception) -> int | None:
    for attribute in ("status_code", "status", "code"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)

    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    return status_code if isinstance(status_code, int) else None
