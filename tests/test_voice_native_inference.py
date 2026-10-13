from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

import pytest

from app.core.config import settings
from app.features.speech_to_text.runtime import (
    FasterWhisperRuntime,
    InferencePriority,
    SpeechRuntimeQueueFull,
    WhisperRuntimeResult,
)
from app.features.voice_translation.speech_detector import SileroSpeechEvidenceGuard


class BlockingRuntime(FasterWhisperRuntime):
    def __init__(self) -> None:
        super().__init__(max_concurrency=1, run_warm_up_inference=False)
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls: list[str] = []

    def _load_and_warm_model(self) -> object:
        return object()

    def _transcribe_sync(self, audio: Any, options: dict[str, Any]) -> WhisperRuntimeResult:
        del options
        self.calls.append(audio)
        if audio == "partial":
            self.started.set()
            if not self.release.wait(2):
                raise RuntimeError("test release timed out")
        return WhisperRuntimeResult(
            text="synthetic private words",
            language="en",
            language_probability=1.0,
            duration_seconds=1.0,
            segments=[],
            provider="fake",
            model="fake",
            model_version="fake",
        )


class BlockingGuard(SileroSpeechEvidenceGuard):
    def __init__(self, *, fail_first: bool = False) -> None:
        super().__init__(enabled=True)
        self._ready = True
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls: list[bytes] = []
        self.fail_first = fail_first
        self._has_speech_sync = self._blocked_speech

    def _blocked_speech(self, pcm_bytes: bytes) -> bool:
        self.calls.append(pcm_bytes)
        if pcm_bytes == b"first":
            self.started.set()
            if not self.release.wait(2):
                raise RuntimeError("test release timed out")
            if self.fail_first:
                raise ValueError("synthetic private error")
        return True


async def wait_started(event: threading.Event) -> None:
    async with asyncio.timeout(1):
        while not event.is_set():
            await asyncio.sleep(0.001)


@pytest.mark.asyncio
async def test_cancelled_partial_keeps_native_slot_and_final_priority(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
):
    # 준비: 실제 모델 없이 취소 이후에도 실행 중인 native 추론을 만든다.
    runtime = BlockingRuntime()
    await runtime.warm_up()
    # 앱 로거의 전파 설정과 무관하게 이 테스트 안에서만 기록을 직접 수집한다.
    logger = logging.getLogger("app.features.speech_to_text.runtime")
    monkeypatch.setattr(logger, "handlers", [*logger.handlers, caplog.handler])
    caplog.set_level(logging.INFO, logger=logger.name)
    partial = asyncio.create_task(
        runtime.transcribe("partial", options={}, priority=InferencePriority.PARTIAL)
    )
    await wait_started(runtime.started)

    try:
        # 실행: 호출자를 취소한 뒤 추가 임시 요청과 확정 요청을 보낸다.
        partial.cancel()
        with pytest.raises(asyncio.CancelledError):
            await partial
        assert runtime.active_inferences == 1
        assert not runtime.partial_available
        with pytest.raises(SpeechRuntimeQueueFull):
            await runtime.transcribe(
                "extra partial", options={}, priority=InferencePriority.PARTIAL
            )
        final = asyncio.create_task(
            runtime.transcribe("final", options={}, priority=InferencePriority.FINAL)
        )
        await asyncio.sleep(0)
        assert not final.done()
        runtime.release.set()
        await asyncio.wait_for(final, 1)

        # 검증: native 완료 후에만 확정이 시작되고 임시 추론은 잠시 억제된다.
        assert runtime.calls == ["partial", "final"]
        assert runtime.active_inferences == 0
        assert not runtime.partial_available
        assert "priority=PARTIAL queue_wait_ms=" in caplog.text
        assert "inference_ms=" in caplog.text
        assert "cancelled=True" in caplog.text
        assert "synthetic private words" not in caplog.text
    finally:
        runtime.release.set()
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_partial_admission_recovers_after_bounded_slow_inference_cooldown(
    monkeypatch: pytest.MonkeyPatch,
):
    # 준비: 실제 시간을 오래 기다리지 않고 갱신 주기를 짧게 잡는다.
    monkeypatch.setattr(settings, "AI_VOICE_PARTIAL_INTERVAL_MS", 1)
    runtime = BlockingRuntime()
    await runtime.warm_up()
    partial = asyncio.create_task(
        runtime.transcribe("partial", options={}, priority=InferencePriority.PARTIAL)
    )
    await wait_started(runtime.started)
    try:
        # 실행: 갱신 주기보다 느린 임시 추론은 완료 후 같은 시간만큼 쉬게 한다.
        await asyncio.sleep(0.01)
        runtime.release.set()
        await partial
        assert not runtime.partial_available
        async with asyncio.timeout(1):
            while not runtime.partial_available:
                await asyncio.sleep(0.001)

        # 검증: 무기한 임시 자막을 끄지 않고 쉬는 시간이 끝나면 다시 허용한다.
        await runtime.transcribe("recovered", options={}, priority=InferencePriority.PARTIAL)
        assert runtime.calls == ["partial", "recovered"]
    finally:
        runtime.release.set()
        await runtime.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_first", [False, True])
async def test_cancelled_vad_retains_native_slot_until_result_or_error(fail_first: bool):
    # 준비: 취소 뒤 성공/예외 양쪽에서 실행 자리가 보존되는지 확인한다.
    guard = BlockingGuard(fail_first=fail_first)
    first = asyncio.create_task(guard.has_speech(b"first"))
    await wait_started(guard.started)
    try:
        # 실행: 첫 호출자는 바로 취소되지만 두 번째 native 작업은 대기해야 한다.
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        second = asyncio.create_task(guard.has_speech(b"second"))
        await asyncio.sleep(0.01)
        assert not second.done()
        assert guard.calls == [b"first"]
        guard.release.set()
        assert await asyncio.wait_for(second, 1)

        # 검증: 첫 native 작업이 끝난 뒤에만 다음 작업을 시작하고 참조를 회수한다.
        assert guard.calls == [b"first", b"second"]
        assert not guard._native_tasks
    finally:
        guard.release.set()
        await guard.shutdown()


@pytest.mark.asyncio
async def test_vad_waiter_timeout_does_not_release_running_native_slot():
    # 준비: 첫 native 작업을 유지한 상태에서 대기 중인 요청을 timeout시킨다.
    guard = BlockingGuard()
    first = asyncio.create_task(guard.has_speech(b"first"))
    await wait_started(guard.started)
    try:
        # 실행
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(guard.has_speech(b"timed out waiter"), 0.01)
        third = asyncio.create_task(guard.has_speech(b"third"))
        await asyncio.sleep(0.01)
        assert guard.calls == [b"first"]
        guard.release.set()
        await asyncio.wait_for(asyncio.gather(first, third), 1)

        # 검증: 대기 취소가 실행 자리 수를 늘리거나 취소된 파형을 추론하지 않는다.
        assert guard.calls == [b"first", b"third"]
    finally:
        guard.release.set()
        await guard.shutdown()
