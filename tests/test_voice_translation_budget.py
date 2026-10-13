from __future__ import annotations

import asyncio
import logging
import time

import pytest

from app.ai.ports import VoiceTranslationGenerationResult
from app.core.config import settings
from app.features.voice_translation.errors import VoicePipelineException
from app.features.voice_translation.translation import VoiceTranslationService
from app.schemas.voice_translation import VoiceErrorCode


class DelayedProvider:
    def __init__(self, delay: float = 0, failure: Exception | None = None) -> None:
        self.delay = delay
        self.failure = failure
        self.calls = 0
        self.cancelled = False

    async def translate_voice_utterance(
        self, **kwargs: str | None
    ) -> VoiceTranslationGenerationResult:
        del kwargs
        self.calls += 1
        if self.failure is not None and self.calls == 1:
            raise self.failure
        try:
            await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return VoiceTranslationGenerationResult(
            translated_text="합성 테스트 번역",
            source_reading_tokens=[],
            provider="fake",
            model="synthetic-model",
        )


async def translate(service: VoiceTranslationService):
    return await service.translate(
        source_text="Private synthetic source marker",
        source_language="en",
        target_language="ko",
        previous_source_text="Private synthetic context marker",
    )


@pytest.mark.asyncio
async def test_normal_response_longer_than_former_attempt_budget_uses_one_call() -> None:
    # 준비: 운영 기본 기한의 1/20로 축소해 과거 3초와 전체 6초 사이의 정상 응답을 재현한다.
    attempt_default = type(settings).model_fields["AI_VOICE_TRANSLATION_TIMEOUT_SECONDS"].default
    total_default = type(settings).model_fields[
        "AI_VOICE_TRANSLATION_TOTAL_TIMEOUT_SECONDS"
    ].default
    assert attempt_default == 6.0
    assert total_default == 6.0
    provider = DelayedProvider(delay=0.2)
    service = VoiceTranslationService(
        provider,
        timeout_seconds=attempt_default / 20,
        total_timeout_seconds=total_default / 20,
        max_retries=1,
    )

    # 실행
    result = await translate(service)

    # 검증: 기존 시도 제한 0.15초를 넘겨도 정상 응답을 버리거나 중복 호출하지 않는다.
    assert result.translated_text == "합성 테스트 번역"
    assert provider.calls == 1
    assert not provider.cancelled


@pytest.mark.asyncio
async def test_explicit_short_attempt_timeout_is_preserved_without_duplicate_call() -> None:
    # 준비: 호출자가 명시한 짧은 제한은 기본 정책 변경 이후에도 적용한다.
    provider = DelayedProvider(delay=0.2)
    service = VoiceTranslationService(
        provider, timeout_seconds=0.01, total_timeout_seconds=0.5, max_retries=1
    )

    # 실행
    with pytest.raises(VoicePipelineException) as caught:
        await translate(service)

    # 검증: 취소된 느린 요청을 다시 시작하지 않고 원문 재시도 가능 오류로 반환한다.
    assert caught.value.code == VoiceErrorCode.TRANSLATION_TIMEOUT
    assert caught.value.retryable
    assert provider.calls == 1
    assert provider.cancelled


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        type("SyntheticRateLimit", (RuntimeError,), {"status_code": 429})(
            "private provider marker"
        ),
        ConnectionError("private provider marker"),
    ],
)
async def test_early_retryable_failure_retries_once_within_total_budget(error: Exception) -> None:
    # 준비: 첫 요청이 시간을 소비하지 않고 명시적인 일시 오류로 거절된다.
    provider = DelayedProvider(delay=0.01, failure=error)
    service = VoiceTranslationService(
        provider, timeout_seconds=0.3, total_timeout_seconds=0.3, max_retries=1
    )

    # 실행
    result = await translate(service)

    # 검증: 빠른 실패의 제한된 재시도는 유지된다.
    assert result.translated_text == "합성 테스트 번역"
    assert provider.calls == 2


@pytest.mark.asyncio
async def test_slot_wait_and_provider_response_share_the_same_total_deadline() -> None:
    # 준비: 먼저 점유한 슬롯의 대기 시간이 provider 호출 기한에 다시 더해지면 안 된다.
    provider = DelayedProvider(delay=0.1)
    service = VoiceTranslationService(
        provider, timeout_seconds=0.3, total_timeout_seconds=0.08, max_retries=1, max_concurrency=1
    )
    await service._translation_slots.acquire()
    operation = asyncio.create_task(translate(service))
    started = time.perf_counter()

    # 실행: 전체 기한 중 일부를 슬롯 대기에 소비한 뒤 공급자 응답도 지연된다.
    await asyncio.sleep(0.04)
    service._translation_slots.release()
    with pytest.raises(VoicePipelineException) as caught:
        await asyncio.wait_for(operation, timeout=0.5)

    # 검증: provider에 새로운 0.3초 예산을 주지 않고 원래 전체 제한으로 취소한다.
    assert caught.value.code == VoiceErrorCode.TRANSLATION_TIMEOUT
    assert provider.calls == 1
    assert provider.cancelled
    assert time.perf_counter() - started < 0.3


@pytest.mark.asyncio
async def test_attempt_and_queue_diagnostics_never_contain_source_or_provider_message(
    caplog,
) -> None:
    # 준비: 실패 이유는 코드로만 기록하고 원문/문맥/공급자 메시지는 내보내지 않는다.
    error = type("SyntheticUnavailable", (RuntimeError,), {"status_code": 503})(
        "private provider marker"
    )
    provider = DelayedProvider(failure=error)
    service = VoiceTranslationService(provider, max_retries=1)
    caplog.set_level(logging.INFO, logger="app.features.voice_translation.translation")
    diagnostic_logger = logging.getLogger("app.features.voice_translation.translation")
    diagnostic_logger.addHandler(caplog.handler)

    # 실행
    try:
        await translate(service)
    finally:
        diagnostic_logger.removeHandler(caplog.handler)

    # 검증: 시도별 시간과 슬롯/전체 대기 근거를 원문 없이 비교할 수 있다.
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "slot_wait_ms=" in messages
    assert "attempt=1" in messages and "attempt=2" in messages
    assert "duration_ms=" in messages and "total_ms=" in messages
    assert "operation=" in messages
    assert "Private synthetic" not in messages
    assert "private provider marker" not in messages
    assert "합성 테스트 번역" not in messages
