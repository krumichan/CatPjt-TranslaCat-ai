from __future__ import annotations

import asyncio
import math
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.core.config import settings
from app.features.language_learning.speaking.audio_processor import NormalizedAudio
from app.features.language_learning.speaking.errors import SpeakingStageException
from app.features.language_learning.speaking.idempotency import InMemoryIdempotencyStore
from app.features.language_learning.speaking.policy import STT_HINT_VERSION
from app.features.language_learning.speaking.provider_error import map_provider_exception
from app.features.language_learning.speaking.retry import run_with_stage_retry
from app.schemas.language_learning_speaking import (
    SpeakingErrorCode,
    SpeakingStage,
    SpeakingUsage,
    StageUsage,
    SttAnalysisMetadata,
    SttResponse,
    SttSegment,
    TranscriptResult,
)


@dataclass(frozen=True)
class SttProviderSegment:
    start_seconds: float
    end_seconds: float
    text: str
    avg_logprob: float = -1.0


@dataclass(frozen=True)
class SttProviderResult:
    text: str
    language: str
    language_probability: float
    segments: list[SttProviderSegment]
    provider: str
    model: str
    model_version: str | None = None


class SpeakingSttProvider(Protocol):
    async def transcribe(
        self,
        wav_bytes: bytes,
        *,
        language: str,
        phrase_hints: list[str] | None = None,
    ) -> SttProviderResult:
        ...


class SpeakingSttService:
    def __init__(
        self,
        provider: SpeakingSttProvider,
        *,
        timeout_seconds: float | None = None,
        automatic_retries: int | None = None,
        idempotency_store: InMemoryIdempotencyStore[SttResponse] | None = None,
    ) -> None:
        self.provider = provider
        self.timeout_seconds = timeout_seconds or settings.AI_SPEAKING_STT_TIMEOUT_SECONDS
        self.automatic_retries = (
            settings.AI_SPEAKING_AUTOMATIC_RETRY_LIMIT
            if automatic_retries is None
            else automatic_retries
        )
        self.idempotency_store = idempotency_store or InMemoryIdempotencyStore()

    async def transcribe(
        self,
        *,
        request_id: str,
        session_id: str,
        turn_index: int,
        learning_language: str,
        normalized_audio: NormalizedAudio,
        phrase_hints: list[str] | None = None,
        idempotency_key: str | None = None,
        automatic_retry_limit: int | None = None,
        manual_retry_attempt: int = 0,
    ) -> SttResponse:
        if manual_retry_attempt > settings.AI_SPEAKING_MANUAL_RETRY_LIMIT:
            raise SpeakingStageException(
                code=SpeakingErrorCode.MANUAL_RETRY_LIMIT_EXCEEDED,
                stage=SpeakingStage.STT,
                message="STT 수동 재시도 가능 횟수를 초과했습니다.",
                retryable=False,
            )

        key = idempotency_key or request_id
        response, _ = await self.idempotency_store.execute(
            key,
            lambda: self._transcribe_once(
                request_id=request_id,
                session_id=session_id,
                turn_index=turn_index,
                learning_language=learning_language,
                normalized_audio=normalized_audio,
                phrase_hints=phrase_hints,
                automatic_retry_limit=automatic_retry_limit,
            ),
        )
        return response.model_copy(deep=True)

    async def _transcribe_once(
        self,
        *,
        request_id: str,
        session_id: str,
        turn_index: int,
        learning_language: str,
        normalized_audio: NormalizedAudio,
        phrase_hints: list[str] | None,
        automatic_retry_limit: int | None,
    ) -> SttResponse:
        started = time.perf_counter()

        async def operation() -> SttProviderResult:
            try:
                return await asyncio.wait_for(
                    self.provider.transcribe(
                        normalized_audio.wav_bytes,
                        language=learning_language,
                        phrase_hints=phrase_hints,
                    ),
                    timeout=self.timeout_seconds,
                )
            except (TimeoutError, asyncio.TimeoutError) as exc:
                raise SpeakingStageException(
                    code=SpeakingErrorCode.PROVIDER_TIMEOUT,
                    stage=SpeakingStage.STT,
                    message="STT Provider 응답 시간이 초과되었습니다.",
                    retryable=True,
                ) from exc
            except SpeakingStageException:
                raise
            except Exception as exc:
                raise map_provider_exception(
                    exc,
                    stage=SpeakingStage.STT,
                    fallback_code=SpeakingErrorCode.STT_FAILED,
                    fallback_message="음성 전사에 실패했습니다.",
                ) from exc

        max_retries = self.automatic_retries
        if automatic_retry_limit is not None:
            max_retries = min(max_retries, automatic_retry_limit)

        result = await run_with_stage_retry(
            operation,
            max_retries=max_retries,
        )
        text = result.text.strip()
        if not text:
            raise SpeakingStageException(
                code=SpeakingErrorCode.INVALID_AUDIO,
                stage=SpeakingStage.STT,
                message="전사 가능한 발화가 확인되지 않았습니다.",
                retryable=False,
            )

        segments = [
            SttSegment(
                start_ms=max(0, int(segment.start_seconds * 1000)),
                end_ms=max(0, int(segment.end_seconds * 1000)),
                text=segment.text.strip(),
                confidence=self._segment_confidence(segment.avg_logprob),
            )
            for segment in result.segments
            if segment.text.strip()
        ]
        confidence_values = [segment.confidence for segment in segments]
        confidence = (
            sum(confidence_values) / len(confidence_values)
            if confidence_values
            else result.language_probability
        )
        confidence = max(0.0, min(1.0, confidence))
        low_confidence_threshold = settings.AI_SPEAKING_STT_LOW_CONFIDENCE_THRESHOLD
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        return SttResponse(
            request_id=request_id,
            session_id=session_id,
            turn_index=turn_index,
            transcript=TranscriptResult(
                text=text,
                language=result.language or learning_language,
                confidence=round(confidence, 4),
                is_low_confidence=confidence < low_confidence_threshold,
                segments=segments,
                metadata=SttAnalysisMetadata(
                    provider=result.provider,
                    model=result.model,
                    model_version=result.model_version,
                    detected_language=result.language or None,
                    requested_language=learning_language,
                    low_confidence_threshold=low_confidence_threshold,
                    audio_duration=normalized_audio.duration_seconds,
                    audio_quality_signals=normalized_audio.quality,
                    normalization_version=normalized_audio.normalization_version,
                    stt_hint_version=STT_HINT_VERSION,
                ),
            ),
            usage=SpeakingUsage(
                stt=StageUsage(
                    latency_ms=elapsed_ms,
                    audio_seconds=normalized_audio.duration_seconds,
                    provider=result.provider,
                    model=result.model,
                )
            ),
        )

    @staticmethod
    def _segment_confidence(avg_logprob: float) -> float:
        return max(0.0, min(1.0, math.exp(avg_logprob)))


class FasterWhisperSpeakingSttProvider:
    def __init__(self) -> None:
        self._model = None

    def _get_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                settings.AI_SPEAKING_STT_MODEL_NAME,
                device=settings.AI_SPEAKING_STT_DEVICE,
                compute_type=settings.AI_SPEAKING_STT_COMPUTE_TYPE,
                cpu_threads=1,
                num_workers=1,
            )
        return self._model

    async def transcribe(
        self,
        wav_bytes: bytes,
        *,
        language: str,
        phrase_hints: list[str] | None = None,
    ) -> SttProviderResult:
        def run() -> SttProviderResult:
            model = self._get_model()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(wav_bytes)
                path = Path(temp_file.name)
            try:
                raw_segments, info = model.transcribe(
                    str(path),
                    beam_size=1,
                    language=language,
                    initial_prompt=(
                        ", ".join(phrase_hints[:20]) if phrase_hints else None
                    ),
                    vad_filter=True,
                    vad_parameters={"min_silence_duration_ms": 500},
                )
                segments = [
                    SttProviderSegment(
                        start_seconds=float(segment.start),
                        end_seconds=float(segment.end),
                        text=segment.text,
                        avg_logprob=float(getattr(segment, "avg_logprob", -1.0)),
                    )
                    for segment in raw_segments
                ]
                return SttProviderResult(
                    text="".join(segment.text for segment in segments).strip(),
                    language=str(getattr(info, "language", language) or language),
                    language_probability=float(
                        getattr(info, "language_probability", 0.0) or 0.0
                    ),
                    segments=segments,
                    provider="faster-whisper",
                    model=settings.AI_SPEAKING_STT_MODEL_NAME,
                    model_version=None,
                )
            finally:
                path.unlink(missing_ok=True)

        return await asyncio.to_thread(run)
