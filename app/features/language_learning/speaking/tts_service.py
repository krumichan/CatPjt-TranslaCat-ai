from __future__ import annotations

import asyncio
import hashlib
import time

from app.ai.ports import SpeechSynthesisProvider
from app.core.config import settings
from app.features.language_learning.speaking.audio_store import TemporaryTtsAudioStore
from app.features.language_learning.speaking.errors import SpeakingStageException
from app.features.language_learning.speaking.policy import SPEAKING_TTS_VERSION
from app.features.language_learning.speaking.provider_error import map_provider_exception
from app.features.language_learning.speaking.retry import run_with_stage_retry
from app.schemas.language_learning_speaking import (
    AssistantAudio,
    SpeakingErrorCode,
    SpeakingStage,
    SpeakingUsage,
    StageUsage,
    TtsRequest,
    TtsResponse,
)


class SpeakingTtsService:
    def __init__(
        self,
        provider: SpeechSynthesisProvider,
        audio_store: TemporaryTtsAudioStore,
        *,
        timeout_seconds: float | None = None,
        automatic_retries: int | None = None,
    ) -> None:
        self.provider = provider
        self.audio_store = audio_store
        self.timeout_seconds = timeout_seconds or settings.AI_SPEAKING_TTS_TIMEOUT_SECONDS
        self.automatic_retries = (
            settings.AI_SPEAKING_AUTOMATIC_RETRY_LIMIT
            if automatic_retries is None
            else automatic_retries
        )

    async def synthesize(self, request: TtsRequest) -> TtsResponse:
        if request.manual_retry_attempt > settings.AI_SPEAKING_MANUAL_RETRY_LIMIT:
            raise SpeakingStageException(
                code=SpeakingErrorCode.MANUAL_RETRY_LIMIT_EXCEEDED,
                stage=SpeakingStage.TTS,
                message="TTS 수동 재시도 가능 횟수를 초과했습니다.",
                retryable=False,
            )

        cache_key = self._cache_key(request)
        existing_reference = f"tts-{hashlib.sha256(cache_key.encode()).hexdigest()[:32]}"
        existing = self.audio_store.get(existing_reference)
        if existing is not None:
            return TtsResponse(
                request_id=request.request_id,
                session_id=request.session_id,
                audio=AssistantAudio(
                    audio_reference=existing.reference,
                    content_type=existing.content_type,
                    voice=request.voice,
                    cache_key=cache_key,
                    status="READY",
                ),
                usage=SpeakingUsage(
                    tts=StageUsage(
                        latency_ms=0,
                        tts_characters=len(request.text),
                        prompt_version=SPEAKING_TTS_VERSION,
                    )
                ),
            )

        started = time.perf_counter()

        async def operation():
            try:
                return await asyncio.wait_for(
                    self.provider.synthesize_speech(
                        text=request.text,
                        voice=request.voice,
                        language=request.learning_language,
                        speed=request.playback_speed,
                    ),
                    timeout=self.timeout_seconds,
                )
            except (TimeoutError, asyncio.TimeoutError) as exc:
                raise SpeakingStageException(
                    code=SpeakingErrorCode.PROVIDER_TIMEOUT,
                    stage=SpeakingStage.TTS,
                    message="TTS Provider 응답 시간이 초과되었습니다.",
                    retryable=True,
                ) from exc
            except SpeakingStageException:
                raise
            except Exception as exc:
                raise map_provider_exception(
                    exc,
                    stage=SpeakingStage.TTS,
                    fallback_code=SpeakingErrorCode.TTS_FAILED,
                    fallback_message="AI 응답 음성 생성에 실패했습니다.",
                ) from exc

        max_retries = min(
            self.automatic_retries,
            request.automatic_retry_limit,
        )
        result = await run_with_stage_retry(
            operation,
            max_retries=max_retries,
        )
        stored = self.audio_store.put(result.audio_bytes, cache_key=cache_key)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return TtsResponse(
            request_id=request.request_id,
            session_id=request.session_id,
            audio=AssistantAudio(
                audio_reference=stored.reference,
                content_type=stored.content_type,
                voice=request.voice,
                cache_key=cache_key,
                duration_seconds=result.duration_seconds,
                status="READY",
            ),
            usage=SpeakingUsage(
                tts=StageUsage(
                    latency_ms=elapsed_ms,
                    tts_characters=len(request.text),
                    tts_audio_seconds=result.duration_seconds or 0,
                    provider=result.provider,
                    model=result.model,
                    prompt_version=SPEAKING_TTS_VERSION,
                )
            ),
        )

    @staticmethod
    def _cache_key(request: TtsRequest) -> str:
        return "|".join(
            [
                request.learning_language,
                request.voice,
                request.playback_speed,
                request.text,
                SPEAKING_TTS_VERSION,
            ]
        )
