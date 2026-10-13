from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from uuid import uuid4

from app.ai.model_policy import get_model_name_for_task
from app.ai.ports import VoiceTranslationGenerationResult, VoiceTranslationProvider
from app.common.idempotency import InMemoryIdempotencyStore
from app.core.config import settings
from app.features.voice_translation.errors import (
    VoicePipelineException,
    map_translation_provider_exception,
)
from app.schemas.voice_translation import (
    VoiceErrorCode,
    VoiceModelMetadata,
    VoiceReadingToken,
    VoiceStage,
    VoiceTranslationRetryRequest,
    VoiceTranslationRetryResponse,
    VoiceWarning,
)

_HTML_PATTERN = re.compile(r"<\s*/?\s*[a-zA-Z!][^>]*>")
_KANJI_PATTERN = re.compile(r"[一-龯々〆ヵヶ]")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VoiceTranslationResult:
    translated_text: str
    translation_skipped: bool
    source_reading_tokens: list[VoiceReadingToken]
    warnings: list[VoiceWarning]
    latency_ms: int
    model: VoiceModelMetadata


class VoiceTranslationService:
    def __init__(
        self,
        provider: VoiceTranslationProvider,
        *,
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
        max_concurrency: int | None = None,
        total_timeout_seconds: float | None = None,
        idempotency_store: InMemoryIdempotencyStore[VoiceTranslationRetryResponse] | None = None,
    ) -> None:
        self.provider = provider
        self.timeout_seconds = (
            settings.AI_VOICE_TRANSLATION_TIMEOUT_SECONDS
            if timeout_seconds is None
            else timeout_seconds
        )
        self.max_retries = (
            settings.AI_VOICE_TRANSLATION_MAX_RETRIES if max_retries is None else max_retries
        )
        self.max_concurrency = (
            settings.AI_VOICE_TRANSLATION_MAX_CONCURRENCY
            if max_concurrency is None
            else max_concurrency
        )
        self.total_timeout_seconds = (
            settings.AI_VOICE_TRANSLATION_TOTAL_TIMEOUT_SECONDS
            if total_timeout_seconds is None
            else total_timeout_seconds
        )
        if self.max_concurrency < 1:
            raise ValueError("Voice translation concurrency must be positive")
        self._translation_slots = asyncio.Semaphore(self.max_concurrency)
        self.idempotency_store = idempotency_store or InMemoryIdempotencyStore(
            ttl_seconds=settings.AI_VOICE_TRANSLATION_IDEMPOTENCY_TTL_SECONDS,
            max_entries=settings.AI_VOICE_TRANSLATION_IDEMPOTENCY_MAX_ENTRIES,
        )

    @property
    def ready(self) -> bool:
        return bool(getattr(self.provider, "ready", True))

    @property
    def model_name(self) -> str:
        resolver = getattr(self.provider, "model_name_for", None)
        if callable(resolver):
            return str(resolver("VOICE_TRANSLATION"))
        return get_model_name_for_task("VOICE_TRANSLATION")

    async def warm_up(self) -> None:
        warm_up = getattr(self.provider, "warm_up", None)
        if warm_up is not None:
            await warm_up()

    async def translate(
        self,
        *,
        source_text: str,
        source_language: str,
        target_language: str,
        previous_source_text: str | None = None,
    ) -> VoiceTranslationResult:
        started = time.perf_counter()
        if source_language == target_language:
            return VoiceTranslationResult(
                translated_text=source_text,
                translation_skipped=True,
                source_reading_tokens=[],
                warnings=[],
                latency_ms=0,
                model=self._model_metadata(self.model_name),
            )

        # 슬롯 대기와 재시도를 합친 전체 기한은 유지하고, 원문 없이 호출별 지연을 추적한다.
        operation_id = uuid4().hex
        outcome = "cancelled"
        failure_code: str | None = None
        try:
            async with asyncio.timeout(self.total_timeout_seconds):
                provider_result = await self._translate_with_retries(
                    source_text=source_text,
                    source_language=source_language,
                    target_language=target_language,
                    previous_source_text=previous_source_text,
                    operation_id=operation_id,
                )
            outcome = "provider_completed"
        except TimeoutError as exc:
            mapped = map_translation_provider_exception(exc)
            outcome = "timeout"
            failure_code = mapped.code.value
            raise mapped from exc
        except VoicePipelineException as exc:
            outcome = "failed"
            failure_code = exc.code.value
            raise
        finally:
            logger.info(
                "Voice translation finished operation=%s total_ms=%s outcome=%s code=%s",
                operation_id,
                max(0, int((time.perf_counter() - started) * 1000)),
                outcome,
                failure_code,
            )

        translated_text = provider_result.translated_text.strip()
        self._validate_plain_text(translated_text)
        reading_tokens, warnings = self._validate_reading_tokens(
            source_text=source_text,
            source_language=source_language,
            tokens=[
                VoiceReadingToken(surface=token.surface, reading=token.reading)
                for token in provider_result.source_reading_tokens
            ],
        )
        latency_ms = max(0, int((time.perf_counter() - started) * 1000))
        return VoiceTranslationResult(
            translated_text=translated_text,
            translation_skipped=False,
            source_reading_tokens=reading_tokens,
            warnings=warnings,
            latency_ms=latency_ms,
            model=self._model_metadata(provider_result.model),
        )

    async def _translate_with_retries(
        self,
        *,
        source_text: str,
        source_language: str,
        target_language: str,
        previous_source_text: str | None,
        operation_id: str,
    ) -> VoiceTranslationGenerationResult:
        queued_at = time.perf_counter()
        async with self._translation_slots:
            logger.info(
                "Voice translation admitted operation=%s slot_wait_ms=%s",
                operation_id,
                max(0, int((time.perf_counter() - queued_at) * 1000)),
            )
            provider_result = None
            last_error: VoicePipelineException | None = None
            for attempt in range(self.max_retries + 1):
                attempt_started = time.perf_counter()
                attempt_outcome = "cancelled"
                failure_code: str | None = None
                try:
                    provider_result = await asyncio.wait_for(
                        self.provider.translate_voice_utterance(
                            source_text=source_text,
                            source_language=source_language,
                            target_language=target_language,
                            previous_source_text=previous_source_text,
                        ),
                        timeout=self.timeout_seconds,
                    )
                    attempt_outcome = "completed"
                    break
                except TimeoutError as exc:
                    # 응답 대기를 소비한 요청을 취소한 뒤 같은 작업을 즉시 다시 시작하지 않는다.
                    # 빠른 공급자 오류의 재시도만 아래 경로에서 남은 전체 기한 안에 허용한다.
                    mapped = map_translation_provider_exception(exc)
                    attempt_outcome = "timeout"
                    failure_code = mapped.code.value
                    raise mapped from exc
                except VoicePipelineException as exc:
                    mapped = exc
                    attempt_outcome = "provider_failed"
                    failure_code = mapped.code.value
                except Exception as exc:
                    mapped = map_translation_provider_exception(exc)
                    attempt_outcome = "provider_failed"
                    failure_code = mapped.code.value
                finally:
                    logger.info(
                        "Voice translation attempt operation=%s attempt=%s "
                        "duration_ms=%s outcome=%s code=%s",
                        operation_id,
                        attempt + 1,
                        max(0, int((time.perf_counter() - attempt_started) * 1000)),
                        attempt_outcome,
                        failure_code,
                    )

                last_error = mapped
                if not mapped.retryable or attempt >= self.max_retries:
                    raise mapped
                await asyncio.sleep(min(0.05 * (2**attempt), 0.2))

        if provider_result is None:
            assert last_error is not None
            raise last_error

        return provider_result

    async def retry(
        self,
        request: VoiceTranslationRetryRequest,
    ) -> VoiceTranslationRetryResponse:
        async def operation() -> VoiceTranslationRetryResponse:
            result = await self.translate(
                source_text=request.source_text,
                source_language=request.source_language,
                target_language=request.target_language,
            )
            return VoiceTranslationRetryResponse(
                request_id=request.request_id,
                session_id=request.session_id,
                segment_id=request.segment_id,
                source_text=request.source_text,
                source_language=request.source_language,
                target_language=request.target_language,
                translated_text=result.translated_text,
                translation_skipped=result.translation_skipped,
                source_reading_tokens=result.source_reading_tokens,
                warnings=result.warnings,
                latency_ms=result.latency_ms,
                model=result.model,
            )

        result, cached = await self.idempotency_store.execute(
            request.request_id,
            operation,
        )
        if cached and (
            result.session_id != request.session_id
            or result.segment_id != request.segment_id
            or result.source_text != request.source_text
            or result.source_language != request.source_language
            or result.target_language != request.target_language
        ):
            raise VoicePipelineException(
                code=VoiceErrorCode.INVALID_EVENT_SCHEMA,
                stage=VoiceStage.TRANSLATION,
                message="동일 requestId에 서로 다른 Translation Retry Payload가 전달되었습니다.",
                retryable=False,
            )
        return result.model_copy(deep=True)

    @staticmethod
    def _validate_plain_text(text: str) -> None:
        if not text:
            raise VoicePipelineException(
                code=VoiceErrorCode.TRANSLATION_FAILED,
                stage=VoiceStage.TRANSLATION,
                message="번역 결과가 비어 있습니다.",
                retryable=False,
            )
        if "```" in text or _HTML_PATTERN.search(text):
            raise VoicePipelineException(
                code=VoiceErrorCode.TRANSLATION_FAILED,
                stage=VoiceStage.TRANSLATION,
                message="번역 결과가 허용되지 않는 Markup을 포함합니다.",
                retryable=False,
            )

    @staticmethod
    def _validate_reading_tokens(
        *,
        source_text: str,
        source_language: str,
        tokens: list[VoiceReadingToken],
    ) -> tuple[list[VoiceReadingToken], list[VoiceWarning]]:
        if source_language != "ja":
            return [], []
        if (
            tokens
            and "".join(token.surface for token in tokens) == source_text
            and all(
                "```" not in token.surface
                and "```" not in token.reading
                and not _HTML_PATTERN.search(token.surface)
                and not _HTML_PATTERN.search(token.reading)
                and not _KANJI_PATTERN.search(token.reading)
                for token in tokens
            )
        ):
            return tokens, []

        return [], [
            VoiceWarning(
                code=VoiceErrorCode.READING_WARNING,
                stage=VoiceStage.READING,
                message="일본어 읽기 Token의 원문 정합성 검증에 실패했습니다.",
            )
        ]

    @staticmethod
    def _model_metadata(translation_model: str) -> VoiceModelMetadata:
        return VoiceModelMetadata(
            stt_version="not-applicable",
            translation_version=translation_model,
            prompt_version=settings.AI_VOICE_PROMPT_VERSION,
        )
