from __future__ import annotations

import asyncio
import io
import re
import wave

from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError, AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.ai.providers.openai.http_client import create_openai_http_client
from app.core.config import settings
from app.features.voice_translation.errors import VoicePipelineException
from app.features.voice_translation.stt import VoiceSttResult
from app.schemas.voice_translation import SUPPORTED_VOICE_LANGUAGES, VoiceErrorCode, VoiceStage

_LANGUAGE_NAMES = {"english": "en", "korean": "ko", "japanese": "ja"}


class _HostedSegment(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    text: str
    no_speech_prob: float = Field(ge=0, le=1, allow_inf_nan=False)


class _HostedTranscript(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    language: str = Field(min_length=1, max_length=64)
    text: str = Field(max_length=10_000)
    segments: list[_HostedSegment] = Field(max_length=512)


class OpenAIWhisperVoiceSttProvider:
    """Explicit opt-in final transcription; never an automatic local-model fallback."""

    supports_partial = False
    supports_stt_prefetch = True

    def __init__(
        self,
        *,
        client: AsyncOpenAI | None = None,
        api_key: str | None = None,
        max_concurrency: int = 4,
    ) -> None:
        if not 1 <= max_concurrency <= 4:
            raise ValueError("Hosted Voice STT concurrency must be between 1 and 4")
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
        self._semaphore = asyncio.Semaphore(max_concurrency)

    @property
    def ready(self) -> bool:
        return not self._closed and self._client is not None and not self._client.is_closed()

    @property
    def model_version(self) -> str:
        return "whisper-1@openai"

    async def shutdown(self) -> None:
        # 앱이 직접 만든 SDK만 닫는다. 테스트/외부 주입 클라이언트의 소유권은 보존한다.
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
        # 언어 선택은 뒤 단계의 필터로 유지한다. 언어/문맥을 강제로 주입하지 않는다.
        del language, initial_prompt
        if not is_final:
            return self._result(text="", language=None, no_speech_probability=1.0)
        if len(pcm_bytes) % 2 or len(pcm_bytes) > settings.AI_STT_MAX_AUDIO_FILE_BYTES:
            raise VoicePipelineException(
                code=VoiceErrorCode.INVALID_AUDIO_FRAME,
                stage=VoiceStage.AUDIO,
                message="음성 PCM 형식 또는 크기가 올바르지 않습니다.",
                retryable=False,
            )
        if not pcm_bytes or not any(pcm_bytes):
            return self._result(text="", language=None, no_speech_probability=1.0)
        client = self._client
        if client is None or not self.ready:
            raise VoicePipelineException(
                code=VoiceErrorCode.MODEL_NOT_READY,
                stage=VoiceStage.RUNTIME,
                message="선택한 음성 인식 공급자가 준비되지 않았습니다.",
                retryable=False,
            )

        # 파일은 메모리에만 만들며 대기 시간까지 기존 최종 STT 기한 안으로 제한한다.
        audio_file = io.BytesIO()
        with wave.open(audio_file, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16_000)
            wav.writeframes(pcm_bytes)
        try:
            async with asyncio.timeout(settings.AI_VOICE_STT_FINAL_TIMEOUT_SECONDS):
                async with self._semaphore:
                    response = await client.with_options(max_retries=0).audio.transcriptions.create(
                        file=("voice.wav", audio_file.getvalue(), "audio/wav"),
                        model="whisper-1",
                        response_format="verbose_json",
                        timestamp_granularities=["segment"],
                        temperature=0,
                        timeout=settings.AI_VOICE_STT_FINAL_TIMEOUT_SECONDS,
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

        # 명시적 언어와 유효한 발화 구간만 사용하며 없는 confidence를 만들어 내지 않는다.
        try:
            # 잘못된 타입의 공급자 응답이 직렬화 경고로 원문을 출력하지 않게 한다.
            transcript = _HostedTranscript.model_validate(response.model_dump(warnings=False))
            if transcript.text.strip() and (
                not transcript.segments
                or not any(item.text.strip() for item in transcript.segments)
            ):
                raise ValueError("Missing transcription speech segments")
        except (ValidationError, ValueError, TypeError, AttributeError) as exc:
            raise VoicePipelineException(
                code=VoiceErrorCode.STT_FAILED,
                stage=VoiceStage.STT,
                message="음성 인식 응답 형식이 올바르지 않습니다.",
                retryable=False,
            ) from exc

        probabilities = [segment.no_speech_prob for segment in transcript.segments]
        no_speech_probability = sum(probabilities) / len(probabilities) if probabilities else 1.0
        text = transcript.text.strip()
        if no_speech_probability >= settings.AI_VOICE_STT_NO_SPEECH_PROBABILITY_THRESHOLD:
            text = ""
        detected_language = _normalize_hosted_language(transcript.language)
        return self._result(
            text=text,
            language=detected_language,
            no_speech_probability=no_speech_probability,
            trusted_language=bool(text) and detected_language in SUPPORTED_VOICE_LANGUAGES,
        )

    def _result(
        self,
        *,
        text: str,
        language: str | None,
        no_speech_probability: float,
        trusted_language: bool = False,
    ) -> VoiceSttResult:
        return VoiceSttResult(
            text=text,
            language=language,
            language_confidence=None,
            no_speech_probability=no_speech_probability,
            provider="openai",
            model="whisper-1",
            model_version=self.model_version,
            trusted_language=trusted_language,
        )


def _normalize_hosted_language(language: str) -> str | None:
    raw = language.strip().lower()
    if raw in {"", "und", "unknown"}:
        return None
    if raw in _LANGUAGE_NAMES:
        return _LANGUAGE_NAMES[raw]
    # 알려진 비지원 코드/이름도 이전 언어로 번역하지 않게 명시적 거절 대상으로 남긴다.
    return raw if re.fullmatch(r"[a-z]{2,3}", raw) else "unsupported"
