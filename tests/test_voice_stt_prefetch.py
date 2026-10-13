from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.core.config import settings
from app.features.voice_translation.speech_detector import VoiceSpeechEvidenceGuard
from app.features.voice_translation.stream import (
    VoiceChannelStreamContext,
    VoiceStreamApplicationService,
)
from app.features.voice_translation.stt import VoiceSttResult
from app.features.voice_translation.translation import VoiceTranslationService
from app.schemas.voice_translation import (
    VoiceErrorCode,
    VoiceEventBase,
    VoiceNoSpeechEvent,
    VoicePipelineCompletedEvent,
    VoicePipelineFailedEvent,
    VoiceStreamClosedEvent,
    VoiceStreamOpen,
)
from tests.test_voice_translation import FakeTranslationProvider, pcm_frame, stream_open_payload


class OrderedStt:
    ready = True
    model_version = "synthetic-hosted-stt"
    supports_partial = False

    def __init__(
        self, *, prefetch: bool = True, hold_second: bool = False, ignore_cancel: bool = False
    ) -> None:
        self.supports_stt_prefetch = prefetch
        self.hold_second = hold_second
        self.ignore_cancel = ignore_cancel
        self.calls: list[int] = []
        self.second_started = asyncio.Event()
        self.second_cancelled = asyncio.Event()
        self.release_second = asyncio.Event()

    async def transcribe_pcm(self, pcm_bytes: bytes, **kwargs: Any) -> VoiceSttResult:
        del pcm_bytes
        assert kwargs["is_final"] and kwargs["language"] is None
        sequence = len(self.calls) + 1
        self.calls.append(sequence)
        if sequence == 2:
            self.second_started.set()
            if self.hold_second:
                try:
                    await self.release_second.wait()
                except asyncio.CancelledError:
                    self.second_cancelled.set()
                    if not self.ignore_cancel:
                        raise
                    await self.release_second.wait()
        return VoiceSttResult(
            text=f"Synthetic sentence {sequence}.",
            language="en",
            language_confidence=None if self.supports_stt_prefetch else 0.99,
            no_speech_probability=0.01,
            provider="fake",
            model="fake",
            trusted_language=self.supports_stt_prefetch,
        )


class HeldFirstTranslation(FakeTranslationProvider):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.sources: list[str] = []

    async def translate_voice_utterance(self, **kwargs: Any):
        self.sources.append(kwargs["source_text"])
        if len(self.sources) == 1:
            self.started.set()
            await self.release.wait()
        return await super().translate_voice_utterance(**kwargs)


async def make_context(
    stt: OrderedStt,
    provider: HeldFirstTranslation,
    guard: VoiceSpeechEvidenceGuard | None = None,
) -> VoiceChannelStreamContext:
    return await VoiceStreamApplicationService(
        stt_provider=stt,
        translation_service=VoiceTranslationService(provider, max_retries=0),
        speech_evidence_guard=guard,
    ).open_stream(VoiceStreamOpen.model_validate(stream_open_payload()))


async def feed_utterance(context: VoiceChannelStreamContext) -> None:
    for frame in [pcm_frame(4000)] * 3 + [pcm_frame(0)] * 3:
        await context.feed_audio(frame)
        await asyncio.sleep(0)
    await context._input_queue.join()


async def finish(context: VoiceChannelStreamContext) -> list[VoiceEventBase]:
    await asyncio.wait_for(context.close("TEST"), 1)
    events = []
    while True:
        event = await asyncio.wait_for(context.next_event(), 1)
        events.append(event)
        if isinstance(event, VoiceStreamClosedEvent):
            return events


@pytest.mark.asyncio
async def test_hosted_stt_prepares_only_one_ahead_and_keeps_ordered_translation_context():
    # 준비: 첫 번역을 멈춘 상태에서 두 번째와 세 번째 발화를 수락한다.
    stt = OrderedStt()
    translation = HeldFirstTranslation()
    context = await make_context(stt, translation)
    try:
        await feed_utterance(context)
        await asyncio.wait_for(translation.started.wait(), 1)
        await feed_utterance(context)
        await feed_utterance(context)

        # 실행: 두 번째 STT는 번역과 겹치지만 세 번째를 미리 요청하지 않는다.
        await asyncio.wait_for(stt.second_started.wait(), 1)
        assert stt.calls == [1, 2]
        assert translation.sources == ["Synthetic sentence 1."]
        translation.release.set()
        events = await finish(context)

        # 검증: STT를 중복 호출하지 않고 완성/번역 문맥은 항상 발화 순서다.
        completed = [event for event in events if isinstance(event, VoicePipelineCompletedEvent)]
        assert [event.utterance_sequence for event in completed] == [1, 2, 3]
        assert stt.calls == [1, 2, 3]
        assert translation.previous_source_texts == [
            None,
            "Synthetic sentence 1.",
            "Synthetic sentence 2.",
        ]
        assert context._prefetched_job is None
    finally:
        translation.release.set()
        await context.abort()


@pytest.mark.asyncio
async def test_native_provider_keeps_serial_contextual_stt():
    # 준비: native 공급자는 새 선행 준비 경로를 명시하지 않는다.
    stt = OrderedStt(prefetch=False)
    translation = HeldFirstTranslation()
    context = await make_context(stt, translation)
    try:
        await feed_utterance(context)
        await asyncio.wait_for(translation.started.wait(), 1)
        await feed_utterance(context)

        # 실행 / 검증: 첫 번역이 끝나기 전에는 다음 native 추론이 시작되지 않는다.
        await asyncio.sleep(0)
        assert stt.calls == [1]
        assert context._prefetched_job is None
        translation.release.set()
        events = await finish(context)
        completed = [event for event in events if isinstance(event, VoicePipelineCompletedEvent)]
        assert len(completed) == 2
    finally:
        translation.release.set()
        await context.abort()


@pytest.mark.asyncio
async def test_prefetch_keeps_original_job_deadline_and_does_not_repeat_timeout(
    monkeypatch: pytest.MonkeyPatch,
):
    # 준비: STT 자체 제한보다 짧은 전체 발화 기한을 설정한다.
    monkeypatch.setattr(settings, "AI_VOICE_PIPELINE_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(settings, "AI_VOICE_STT_FINAL_TIMEOUT_SECONDS", 1.0)
    stt = OrderedStt(hold_second=True)
    translation = HeldFirstTranslation()
    context = await make_context(stt, translation)
    try:
        await feed_utterance(context)
        await asyncio.wait_for(translation.started.wait(), 1)
        await feed_utterance(context)
        await asyncio.wait_for(stt.second_started.wait(), 1)

        # 실행: 선행 STT가 멈춰도 원래 수락 시점의 기한까지만 기다린다.
        translation.release.set()
        events = await finish(context)

        # 검증: 두 번째 발화는 한 번만 실패하며 재시도/새 STT 호출이 없다.
        failures = [event for event in events if isinstance(event, VoicePipelineFailedEvent)]
        assert len(failures) == 1
        assert failures[0].utterance_sequence == 2
        assert failures[0].error.code == VoiceErrorCode.STT_TIMEOUT
        assert stt.calls == [1, 2]
        assert stt.second_cancelled.is_set()
    finally:
        translation.release.set()
        stt.release_second.set()
        await context.abort()


@pytest.mark.asyncio
async def test_abort_cancels_prefetch_and_ignores_cancellation_resistant_late_result():
    # 준비: 취소 응답이 늦은 공급자도 종료된 발화 소유권을 되살리지 못해야 한다.
    stt = OrderedStt(hold_second=True, ignore_cancel=True)
    translation = HeldFirstTranslation()
    context = await make_context(stt, translation)
    try:
        await feed_utterance(context)
        await asyncio.wait_for(translation.started.wait(), 1)
        await feed_utterance(context)
        await asyncio.wait_for(stt.second_started.wait(), 1)
        assert context._prefetched_job is not None
        preparation = context._prefetched_job.preparation_task
        assert preparation is not None

        # 실행: abort는 늦은 응답을 기다리지 않고 종료 소유권을 먼저 제거한다.
        await asyncio.wait_for(context.abort(), 0.2)
        await asyncio.wait_for(stt.second_cancelled.wait(), 1)
        event_count = context._event_queue.qsize()
        stt.release_second.set()
        await asyncio.wait_for(asyncio.gather(preparation, return_exceptions=True), 1)

        # 검증: 늦은 STT 결과는 자막/번역을 만들지 않고 관련 task가 정리된다.
        assert context._event_queue.qsize() == event_count
        assert context._pending_final_jobs == {}
        assert context._prefetched_job is None
        assert len(translation.sources) == 1
    finally:
        translation.release.set()
        stt.release_second.set()
        await context.abort()


@pytest.mark.asyncio
async def test_prefetched_noise_does_not_call_stt_or_skip_to_third_job():
    # 준비: 두 번째 발화 후보만 잡음으로 분류한다.
    class SecondIsNoise:
        ready = True
        model_version = "synthetic-noise-guard"

        def __init__(self) -> None:
            self.calls = 0
            self.second_checked = asyncio.Event()

        async def has_speech(self, pcm_bytes: bytes) -> bool:
            del pcm_bytes
            self.calls += 1
            if self.calls == 2:
                self.second_checked.set()
                return False
            return True

    guard = SecondIsNoise()
    stt = OrderedStt()
    translation = HeldFirstTranslation()
    context = await make_context(stt, translation, guard)
    try:
        await feed_utterance(context)
        await asyncio.wait_for(translation.started.wait(), 1)
        await feed_utterance(context)
        await feed_utterance(context)

        # 실행 / 검증: 잡음 후보의 VAD까지만 수행하며 다음 다음 음성을 미리 보내지 않는다.
        await asyncio.wait_for(guard.second_checked.wait(), 1)
        assert stt.calls == [1]
        translation.release.set()
        events = await finish(context)
        no_speech = [event for event in events if isinstance(event, VoiceNoSpeechEvent)]
        completed = [event for event in events if isinstance(event, VoicePipelineCompletedEvent)]
        assert [event.utterance_sequence for event in no_speech] == [2]
        assert [event.utterance_sequence for event in completed] == [1, 3]
        assert stt.calls == [1, 2]
        # 확인된 잡음/무음 후보만 사이에 있으면 가까운 같은 언어 원문은 번역 문맥으로 유지한다.
        assert translation.previous_source_texts == [None, "Synthetic sentence 1."]
    finally:
        translation.release.set()
        await context.abort()
