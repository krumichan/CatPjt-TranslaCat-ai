from __future__ import annotations

import asyncio

import pytest

from app.features.voice_translation.errors import VoicePipelineException
from app.features.voice_translation.stream import VoiceStreamApplicationService
from app.features.voice_translation.translation import VoiceTranslationService
from app.schemas.voice_translation import VoiceErrorCode, VoiceStage, VoiceStreamOpen
from tests.test_voice_translation import (
    FakeTranslationProvider,
    FakeVoiceSttProvider,
    pcm_frame,
    stream_open_payload,
)


class WaitingStt(FakeVoiceSttProvider):
    supports_partial = False

    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def transcribe_pcm(self, pcm_bytes: bytes, **kwargs):
        self.started.set()
        await self.release.wait()
        return await super().transcribe_pcm(pcm_bytes, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("control", ["flush", "close"])
@pytest.mark.parametrize("termination", ["failure", "abort"])
async def test_termination_resolves_control_already_dequeued_by_worker(
    control: str, termination: str, monkeypatch: pytest.MonkeyPatch
):
    # 준비: 실제 확정 대기열이 진행 중인 동안 worker가 제어 명령을 꺼내도록 한다.
    stt = WaitingStt()
    service = VoiceStreamApplicationService(
        stt_provider=stt,
        translation_service=VoiceTranslationService(FakeTranslationProvider()),
    )
    context = await service.open_stream(VoiceStreamOpen.model_validate(stream_open_payload()))
    for frame in [pcm_frame(4000)] * 3 + [pcm_frame(0)] * 3:
        await context.feed_audio(frame)
        await asyncio.sleep(0)
    await context._input_queue.join()
    await asyncio.wait_for(stt.started.wait(), 1)
    control_started = asyncio.Event()
    original_drain = context._drain_final_jobs

    async def observe_drain() -> None:
        control_started.set()
        await original_drain()

    monkeypatch.setattr(context, "_drain_final_jobs", observe_drain)
    pending_control = asyncio.create_task(getattr(context, control)("TEST"))
    await asyncio.wait_for(control_started.wait(), 1)

    try:
        # 실행: 큐에서 이미 제거된 제어 명령을 기다리는 중에 종료 경로를 실행한다.
        if termination == "failure":
            await context.fail_and_close(
                VoicePipelineException(
                    code=VoiceErrorCode.BACKPRESSURE,
                    stage=VoiceStage.STREAM,
                    message="Synthetic test overload",
                    retryable=True,
                )
            )
        else:
            await context.abort()

        # 검증: 재호출도 취소 예외나 미완료 Future에 걸리지 않고 종료 결과를 관찰한다.
        await asyncio.wait_for(asyncio.shield(pending_control), 0.2)
        await asyncio.wait_for(getattr(context, control)("SECOND"), 0.2)
        assert context.closed
        assert context._worker_task is not None and context._worker_task.done()
    finally:
        stt.release.set()
        pending_control.cancel()
        await asyncio.gather(pending_control, return_exceptions=True)
        await service.release_stream(context)
