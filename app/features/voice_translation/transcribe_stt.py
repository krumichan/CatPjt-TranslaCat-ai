from __future__ import annotations

import asyncio
import io
import wave

from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError, AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.ai.providers.openai.http_client import create_openai_http_client
from app.core.config import settings
from app.features.voice_translation.errors import VoicePipelineException
from app.features.voice_translation.stt import VoiceSttResult
from app.schemas.voice_translation import SUPPORTED_VOICE_LANGUAGES, VoiceErrorCode, VoiceStage

_LANGUAGE_ALIASES = {"eng": "en", "kor": "ko", "jpn": "ja"}


class _TranscribeLanguage(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    code: str = Field(pattern=r"^[a-z]{2,3}(?:-[a-z]{2,3})?$", max_length=7)


class _TranscribeTranscript(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    text: str = Field(max_length=10_000)
    languages: list[_TranscribeLanguage] = Field(max_length=32)


class OpenAITranscribeVoiceSttProvider:
    """Opt-in file transcription with independent language detection for previews and finals."""

    # 기존 600ms 주기 파일 재인식은 끈다. native preview의 1회 언어 검사는 별도로 호출한다.
    supports_partial = False
    supports_stt_prefetch = True

    def __init__(
        self,
        *,
        client: AsyncOpenAI | None = None,
        api_key: str | None = None,
        max_concurrency: int = 4,
        max_preview_concurrency: int = 1,
    ) -> None:
        if not 1 <= max_concurrency <= 4 or not 1 <= max_preview_concurrency <= 2:
            raise ValueError("Voice STT requires 1-4 final slots and 1-2 preview slots")
        key = settings.OPENAI_API_KEY if api_key is None else api_key
        self._owns_client = client is None
        self._closed = False
        self._client = client or (
            AsyncOpenAI(
                api_key=key,
                timeout=settings.AI_VOICE_STT_FINAL_TIMEOUT_SECONDS,
                max_retries=0,
                http_client=create_openai_http_client(),
            )
            if key.strip()
            else None
        )
        # 중간 인식이 느려져도 최종 인식의 실행 자리를 점유하지 않는다.
        self._final_slots = asyncio.Semaphore(max_concurrency)
        self._preview_slots = asyncio.Semaphore(max_preview_concurrency)

    @property
    def ready(self) -> bool:
        return not self._closed and self._client is not None and not self._client.is_closed()

    @property
    def partial_available(self) -> bool:
        return self.ready

    @property
    def model_version(self) -> str:
        return "gpt-transcribe@openai"

    async def shutdown(self) -> None:
        # 주입받은 SDK는 호출자 소유로 남기고 대기 중인 요청도 새 외부 호출을 시작하지 않는다.
        self._closed = True
        if self._owns_client and self._client is not None:
            await self._client.close()

    async def transcribe_pcm(
        self,
        pcm_bytes: bytes,
        *,
        language: str | None,
        is_final: bool,
        initial_prompt: str | None = None,
    ) -> VoiceSttResult:
        # 수동 언어는 후속 필터다. 이전 원문이나 언어 힌트로 새 입력의 독립 감지를 덮지 않는다.
        del language, initial_prompt
        if len(pcm_bytes) % 2 or len(pcm_bytes) > settings.AI_STT_MAX_AUDIO_FILE_BYTES:
            raise VoicePipelineException(
                code=VoiceErrorCode.INVALID_AUDIO_FRAME,
                stage=VoiceStage.AUDIO,
                message="음성 PCM 형식 또는 크기가 올바르지 않습니다.",
                retryable=False,
            )
        if not pcm_bytes or not any(pcm_bytes):
            return self._result(text="", language=None)
        client = self._client
        if client is None or not self.ready:
            raise self._not_ready()

        # WAV는 메모리에만 보관하고 preview/final 각각의 기한에 자리 대기까지 포함한다.
        audio_file = io.BytesIO()
        with wave.open(audio_file, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16_000)
            wav.writeframes(pcm_bytes)
        timeout_seconds = (
            settings.AI_VOICE_STT_FINAL_TIMEOUT_SECONDS
            if is_final
            else settings.AI_VOICE_STT_PARTIAL_TIMEOUT_SECONDS
        )
        slots = self._final_slots if is_final else self._preview_slots
        try:
            async with asyncio.timeout(timeout_seconds):
                async with slots:
                    if not self.ready:
                        raise self._not_ready()
                    response = await client.with_options(max_retries=0).audio.transcriptions.create(
                        file=("voice.wav", audio_file.getvalue(), "audio/wav"),
                        model="gpt-transcribe",
                        response_format="json",
                        timeout=timeout_seconds,
                    )
        except (TimeoutError, APITimeoutError) as exc:
            raise VoicePipelineException(
                code=VoiceErrorCode.STT_TIMEOUT,
                stage=VoiceStage.STT,
                message="음성 인식 응답 시간이 초과되었습니다.",
                retryable=True,
            ) from exc
        except APIError as exc:
            raise VoicePipelineException(
                code=VoiceErrorCode.STT_FAILED,
                stage=VoiceStage.STT,
                message="음성 인식 공급자 요청에 실패했습니다.",
                retryable=isinstance(exc, APIConnectionError)
                or (
                    isinstance(exc, APIStatusError)
                    and (exc.status_code >= 500 or exc.status_code == 429)
                ),
            ) from exc

        # API가 반환하지 않는 점수는 만들지 않는다. 언어 필드 누락은 감지 실패와 구분한다.
        try:
            transcript = _TranscribeTranscript.model_validate(response.model_dump(warnings=False))
        except (ValidationError, ValueError, TypeError, AttributeError) as exc:
            raise VoicePipelineException(
                code=VoiceErrorCode.STT_FAILED,
                stage=VoiceStage.STT,
                message="음성 인식 응답 형식이 올바르지 않습니다.",
                retryable=False,
            ) from exc
        return self._result(
            text=transcript.text.strip(),
            language=_detected_language(transcript.languages),
        )

    def _result(self, *, text: str, language: str | None) -> VoiceSttResult:
        return VoiceSttResult(
            text=text,
            language=language,
            language_confidence=None,
            no_speech_probability=None,
            provider="openai",
            model="gpt-transcribe",
            model_version=self.model_version,
            trusted_language=bool(text) and language in SUPPORTED_VOICE_LANGUAGES,
        )

    @staticmethod
    def _not_ready() -> VoicePipelineException:
        return VoicePipelineException(
            code=VoiceErrorCode.MODEL_NOT_READY,
            stage=VoiceStage.RUNTIME,
            message="선택한 음성 인식 공급자가 준비되지 않았습니다.",
            retryable=False,
        )


def _detected_language(languages: list[_TranscribeLanguage]) -> str | None:
    # 비지원 언어가 섞였으면 이전 언어 잠금으로 우회하지 못하도록 명시적으로 보존한다.
    detected: set[str] = set()
    for language in languages:
        code = language.code.split("-")[0]
        code = _LANGUAGE_ALIASES.get(code, code)
        if code != "und":
            detected.add(code)
    unsupported = sorted(detected - SUPPORTED_VOICE_LANGUAGES)
    if unsupported:
        return unsupported[0]

    # 현재 계약은 발화별 언어 하나다. 혼합 언어의 순서를 우세 언어라는 근거로 사용하지 않는다.
    return next(iter(detected)) if len(detected) == 1 else None
