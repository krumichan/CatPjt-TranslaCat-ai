from __future__ import annotations

import asyncio
import hashlib
import time

from app.ai.ports import SpeechSynthesisProvider
from app.core.config import settings
from app.features.language_learning.listening.audio_store import (
    StoredListeningAudio,
    TemporaryListeningAudioStore,
)
from app.features.language_learning.listening.errors import ListeningStageException
from app.features.language_learning.listening.normalization import normalize_text
from app.features.language_learning.listening.policy import LISTENING_TTS_VERSION
from app.features.language_learning.listening.provider_error import (
    map_provider_exception,
)
from app.features.language_learning.listening.retry import run_with_stage_retry
from app.schemas.language_learning_listening import (
    ListeningAudio,
    ListeningErrorCode,
    ListeningStage,
    ListeningTtsRequest,
    ListeningTtsResponse,
    ListeningUsage,
    StageUsage,
)


class ListeningTtsService:
    def __init__(
        self,
        provider: SpeechSynthesisProvider,
        audio_store: TemporaryListeningAudioStore,
        *,
        timeout_seconds: float | None = None,
        automatic_retries: int | None = None,
    ) -> None:
        self.provider = provider
        self.audio_store = audio_store
        self.timeout_seconds = (
            timeout_seconds or settings.AI_LISTENING_TTS_TIMEOUT_SECONDS
        )
        self.automatic_retries = (
            settings.AI_LISTENING_AUTOMATIC_RETRY_LIMIT
            if automatic_retries is None
            else min(automatic_retries, 2)
        )

    async def synthesize(self, request: ListeningTtsRequest) -> ListeningTtsResponse:
        self._validate_request(request)
        text_hash = hashlib.sha256(request.source_text.encode("utf-8")).hexdigest()
        cache_key = self._cache_key(request, text_hash)
        reference = self.audio_store.reference_for(cache_key)
        existing = self.audio_store.get(reference)
        if existing is not None:
            return self._ready_response(
                request,
                existing,
                text_hash=text_hash,
                cache_key=cache_key,
                latency_ms=0,
            )

        started = time.perf_counter()

        async def operation():
            try:
                return await asyncio.wait_for(
                    self.provider.synthesize_speech(
                        text=request.source_text,
                        voice=request.voice.voice_key,
                        language=request.voice.locale,
                        speed="NORMAL",
                    ),
                    timeout=self.timeout_seconds,
                )
            except (TimeoutError, asyncio.TimeoutError) as exc:
                raise ListeningStageException(
                    ListeningErrorCode.PROVIDER_TIMEOUT,
                    ListeningStage.TTS,
                    "Reference TTS Provider 응답 시간이 초과되었습니다.",
                    True,
                ) from exc
            except ListeningStageException:
                raise
            except Exception as exc:
                raise map_provider_exception(
                    exc,
                    stage=ListeningStage.TTS,
                    fallback_code=ListeningErrorCode.TTS_FAILED,
                    fallback_message="Reference TTS 생성에 실패했습니다.",
                ) from exc

        try:
            result = await run_with_stage_retry(
                operation,
                max_retries=min(self.automatic_retries, request.automatic_retry_limit),
            )
        except ListeningStageException as exc:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            return ListeningTtsResponse(
                request_id=request.request_id,
                item_id=request.item_id,
                status="FAILED",
                source_text=request.source_text,
                content_hash=request.content_hash,
                generation_version=request.generation_version,
                error=exc.to_schema(),
                usage=ListeningUsage(
                    tts=StageUsage(
                        latency_ms=elapsed_ms,
                        tts_characters=len(request.source_text),
                        prompt_version=LISTENING_TTS_VERSION,
                    )
                ),
            )

        duration = result.duration_seconds or 0.0
        stored = self.audio_store.put(
            result.audio_bytes,
            cache_key=cache_key,
            content_type=result.content_type,
            duration_seconds=duration,
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return self._ready_response(
            request,
            stored,
            text_hash=text_hash,
            cache_key=cache_key,
            latency_ms=elapsed_ms,
            provider=result.provider,
            model=result.model,
        )

    def _ready_response(
        self,
        request: ListeningTtsRequest,
        stored: StoredListeningAudio,
        *,
        text_hash: str,
        cache_key: str,
        latency_ms: int,
        provider: str | None = None,
        model: str | None = None,
    ) -> ListeningTtsResponse:
        return ListeningTtsResponse(
            request_id=request.request_id,
            item_id=request.item_id,
            status="READY",
            source_text=request.source_text,
            content_hash=request.content_hash,
            generation_version=request.generation_version,
            audio=ListeningAudio(
                audio_reference=stored.reference,
                duration_ms=int(stored.duration_seconds * 1000 + 0.5),
                format=stored.content_type,
                sample_rate=24000,
                channels=1,
                voice=request.voice,
                text_hash=text_hash,
                checksum=stored.checksum,
                cache_key=cache_key,
                tts_version=LISTENING_TTS_VERSION,
            ),
            usage=ListeningUsage(
                tts=StageUsage(
                    latency_ms=latency_ms,
                    tts_characters=len(request.source_text),
                    tts_audio_seconds=stored.duration_seconds,
                    provider=provider,
                    model=model,
                    prompt_version=LISTENING_TTS_VERSION,
                )
            ),
        )

    @staticmethod
    def _cache_key(request: ListeningTtsRequest, text_hash: str) -> str:
        return "|".join(
            [
                request.voice.locale,
                request.voice.voice_key,
                request.voice.version,
                text_hash,
                LISTENING_TTS_VERSION,
            ]
        )

    @staticmethod
    def _validate_request(request: ListeningTtsRequest) -> None:
        if request.manual_retry_attempt > settings.AI_LISTENING_MANUAL_RETRY_LIMIT:
            raise ListeningStageException(
                ListeningErrorCode.MANUAL_RETRY_LIMIT_EXCEEDED,
                ListeningStage.TTS,
                "TTS 수동 재시도 가능 횟수를 초과했습니다.",
                False,
            )
        normalized = normalize_text(request.source_text, request.learning_language)
        expected_content_hash = hashlib.sha256(
            normalized.text.encode("utf-8")
        ).hexdigest()
        if request.content_hash != expected_content_hash:
            raise ListeningStageException(
                ListeningErrorCode.INVALID_REQUEST,
                ListeningStage.TTS,
                "sourceText와 contentHash가 일치하지 않습니다.",
                False,
            )
