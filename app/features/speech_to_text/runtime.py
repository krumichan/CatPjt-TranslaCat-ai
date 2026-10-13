from __future__ import annotations

import asyncio
import itertools
import logging
import time
from dataclasses import dataclass
from enum import IntEnum
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)


class InferencePriority(IntEnum):
    FINAL = 0
    STANDARD = 5
    PARTIAL = 10


class SpeechRuntimeNotReady(RuntimeError):
    pass


class SpeechRuntimeQueueFull(RuntimeError):
    pass


class SpeechRuntimeClosed(RuntimeError):
    pass


@dataclass(frozen=True)
class WhisperRuntimeSegment:
    start_seconds: float
    end_seconds: float
    text: str
    avg_logprob: float
    no_speech_probability: float | None


@dataclass(frozen=True)
class WhisperRuntimeResult:
    text: str
    language: str | None
    language_probability: float | None
    duration_seconds: float | None
    segments: list[WhisperRuntimeSegment]
    provider: str
    model: str
    model_version: str | None


@dataclass
class _InferenceRequest:
    audio: Any
    options: dict[str, Any]
    future: asyncio.Future[WhisperRuntimeResult]
    priority: InferencePriority
    queued_at: float
    native_started_at: float | None = None
    native_finished_at: float | None = None


class FasterWhisperRuntime:
    """One warm, bounded, priority-aware Faster-Whisper runtime per process."""

    def __init__(
        self,
        *,
        model_name: str | None = None,
        model_revision: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
        cpu_threads: int | None = None,
        num_workers: int | None = None,
        max_concurrency: int | None = None,
        queue_capacity: int | None = None,
        run_warm_up_inference: bool | None = None,
    ) -> None:
        self.model_name = model_name or settings.AI_VOICE_STT_MODEL_NAME
        configured_revision = (
            settings.AI_VOICE_STT_MODEL_REVISION if model_revision is None else model_revision
        )
        self.model_revision = configured_revision.strip() if configured_revision else None
        self.device = device or settings.AI_VOICE_STT_DEVICE
        self.compute_type = compute_type or settings.AI_VOICE_STT_COMPUTE_TYPE
        self.cpu_threads = cpu_threads or settings.AI_VOICE_STT_CPU_THREADS
        self.num_workers = num_workers or settings.AI_VOICE_STT_NUM_WORKERS
        self.max_concurrency = max_concurrency or settings.AI_VOICE_STT_MAX_CONCURRENCY
        self.queue_capacity = queue_capacity or settings.AI_VOICE_STT_QUEUE_CAPACITY
        self.run_warm_up_inference = (
            settings.AI_VOICE_STT_RUN_WARM_UP_INFERENCE
            if run_warm_up_inference is None
            else run_warm_up_inference
        )
        if (
            min(
                self.cpu_threads,
                self.num_workers,
                self.max_concurrency,
                self.queue_capacity,
            )
            < 1
        ):
            raise ValueError("Speech runtime limits must be positive")

        self._model: Any | None = None
        self._ready = False
        self._accepting = False
        self._closed = False
        self._warm_up_lock = asyncio.Lock()
        self._queue: asyncio.PriorityQueue[tuple[int, int, _InferenceRequest]] = (
            asyncio.PriorityQueue(maxsize=self.queue_capacity)
        )
        self._sequence = itertools.count()
        self._workers: list[asyncio.Task[None]] = []
        self._active_inferences = 0
        self._partial_not_before = 0.0

    @property
    def ready(self) -> bool:
        return self._ready and self._accepting and not self._closed

    @property
    def queued_requests(self) -> int:
        return self._queue.qsize()

    @property
    def active_inferences(self) -> int:
        return self._active_inferences

    @property
    def partial_available(self) -> bool:
        # 호출자 취소와 무관하게 native 추론이 점유한 worker와 대기 중인 확정을 우선한다.
        return (
            self.ready
            and self._active_inferences < self.max_concurrency
            and self._queue.empty()
            and time.perf_counter() >= self._partial_not_before
        )

    @property
    def model_version(self) -> str:
        try:
            runtime_version = version("faster-whisper")
        except PackageNotFoundError:
            runtime_version = "unknown"
        model_identifier = self.model_name
        if self.model_revision:
            model_identifier = f"{model_identifier}@{self.model_revision}"
        return f"{model_identifier}@faster-whisper-{runtime_version}"

    async def warm_up(self) -> None:
        async with self._warm_up_lock:
            if self.ready:
                return
            if self._closed:
                raise SpeechRuntimeClosed("Speech runtime is closed")

            logger.info(
                "Loading shared STT runtime model=%s device=%s compute=%s",
                self.model_name,
                self.device,
                self.compute_type,
            )
            self._model = await asyncio.to_thread(self._load_and_warm_model)
            self._ready = True
            self._accepting = True
            self._workers = [
                asyncio.create_task(
                    self._worker(index),
                    name=f"voice-stt-worker-{index}",
                )
                for index in range(self.max_concurrency)
            ]
            logger.info("Shared STT runtime is ready")

    async def transcribe(
        self,
        audio: Any,
        *,
        options: dict[str, Any],
        priority: InferencePriority = InferencePriority.STANDARD,
    ) -> WhisperRuntimeResult:
        if self._closed:
            raise SpeechRuntimeClosed("Speech runtime is closed")
        if not self.ready:
            raise SpeechRuntimeNotReady("Speech runtime is not ready")

        # 임시 자막은 빈 추론 자리에서만 계산한다. 취소된 native 작업 뒤로 쌓지 않는다.
        if priority == InferencePriority.PARTIAL and not self.partial_available:
            raise SpeechRuntimeQueueFull("Partial STT deferred while inference capacity is busy")

        # 확정 음성이 들어올 수 있도록 일반 요청도 마지막 대기 자리를 사용하지 않는다.
        if priority != InferencePriority.FINAL and self._queue.qsize() >= max(
            0, self.queue_capacity - 1
        ):
            raise SpeechRuntimeQueueFull("Non-final STT queue capacity exceeded")

        loop = asyncio.get_running_loop()
        future: asyncio.Future[WhisperRuntimeResult] = loop.create_future()
        request = _InferenceRequest(
            audio=audio,
            options=options,
            future=future,
            priority=priority,
            queued_at=time.perf_counter(),
        )
        try:
            self._queue.put_nowait((int(priority), next(self._sequence), request))
        except asyncio.QueueFull as exc:
            raise SpeechRuntimeQueueFull("STT queue capacity exceeded") from exc

        try:
            return await future
        except asyncio.CancelledError:
            future.cancel()
            raise

    async def shutdown(self, *, grace_seconds: float | None = None) -> None:
        if self._closed:
            return
        self._accepting = False
        self._ready = False
        timeout = (
            settings.AI_VOICE_SHUTDOWN_GRACE_SECONDS if grace_seconds is None else grace_seconds
        )
        try:
            await asyncio.wait_for(self._queue.join(), timeout=max(0.1, timeout))
        except TimeoutError:
            logger.warning("STT runtime shutdown grace period elapsed")
        finally:
            self._closed = True
            for worker in self._workers:
                worker.cancel()
            if self._workers:
                await asyncio.gather(*self._workers, return_exceptions=True)
            self._workers.clear()
            self._fail_queued_requests()
            self._model = None

    def _load_and_warm_model(self) -> Any:
        from faster_whisper import WhisperModel

        model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=self.cpu_threads,
            num_workers=self.num_workers,
            revision=self.model_revision,
        )
        if self.run_warm_up_inference:
            import numpy as np

            segments, _ = model.transcribe(
                np.zeros(4000, dtype=np.float32),
                language="en",
                beam_size=1,
                vad_filter=False,
                condition_on_previous_text=False,
            )
            list(segments)
        return model

    async def _worker(self, index: int) -> None:
        del index
        while True:
            _, _, request = await self._queue.get()
            inference_started = False
            error_type: str | None = None
            try:
                if request.future.cancelled():
                    continue
                self._active_inferences += 1
                inference_started = True
                result = await asyncio.to_thread(self._run_inference, request)
                if not request.future.done():
                    request.future.set_result(result)
            except asyncio.CancelledError:
                if not request.future.done():
                    request.future.cancel()
                raise
            except Exception as exc:
                error_type = type(exc).__name__
                if not request.future.done():
                    request.future.set_exception(exc)
            finally:
                if inference_started:
                    self._active_inferences -= 1
                    self._record_inference(request, error_type=error_type)
                self._queue.task_done()

    def _run_inference(self, request: _InferenceRequest) -> WhisperRuntimeResult:
        # 실제 native 실행 시간을 분리해 executor/추론 대기와 계산 병목을 구별한다.
        request.native_started_at = time.perf_counter()
        try:
            return self._transcribe_sync(request.audio, request.options)
        finally:
            request.native_finished_at = time.perf_counter()

    def _record_inference(self, request: _InferenceRequest, *, error_type: str | None) -> None:
        started = request.native_started_at
        finished = request.native_finished_at
        if started is None or finished is None:
            return
        inference_seconds = max(0.0, finished - started)

        # 갱신 주기보다 느린 임시 인식은 계산 시간만큼 쉬어 확정 요청에 여유를 준다.
        # 상한은 기존 최종 STT 기한이며 요청 timeout 자체를 늘리지 않는다.
        if request.priority == InferencePriority.PARTIAL and (
            request.future.cancelled()
            or inference_seconds > settings.AI_VOICE_PARTIAL_INTERVAL_MS / 1000
        ):
            cooldown = min(
                settings.AI_VOICE_STT_FINAL_TIMEOUT_SECONDS,
                max(settings.AI_VOICE_PARTIAL_INTERVAL_MS / 1000, inference_seconds),
            )
            self._partial_not_before = max(self._partial_not_before, finished + cooldown)

        # 음성·원문·공급자 메시지 없이 대기/추론 시간과 실패 종류만 남긴다.
        logger.info(
            "Shared STT inference priority=%s queue_wait_ms=%s inference_ms=%s "
            "cancelled=%s error_type=%s",
            request.priority.name,
            max(0, int((started - request.queued_at) * 1000)),
            int(inference_seconds * 1000),
            request.future.cancelled(),
            error_type or "none",
        )

    def _transcribe_sync(
        self,
        audio: Any,
        options: dict[str, Any],
    ) -> WhisperRuntimeResult:
        if self._model is None:
            raise SpeechRuntimeNotReady("Speech model is not loaded")
        raw_segments, info = self._model.transcribe(audio, **options)
        segments = [
            WhisperRuntimeSegment(
                start_seconds=float(segment.start),
                end_seconds=float(segment.end),
                text=str(segment.text),
                avg_logprob=float(getattr(segment, "avg_logprob", -1.0)),
                no_speech_probability=_optional_float(getattr(segment, "no_speech_prob", None)),
            )
            for segment in raw_segments
        ]
        return WhisperRuntimeResult(
            text="".join(segment.text for segment in segments).strip(),
            language=_optional_string(getattr(info, "language", None)),
            language_probability=_optional_float(getattr(info, "language_probability", None)),
            duration_seconds=_optional_float(getattr(info, "duration", None)),
            segments=segments,
            provider="faster-whisper",
            model=self.model_name,
            model_version=self.model_version,
        )

    def _fail_queued_requests(self) -> None:
        while True:
            try:
                _, _, request = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if not request.future.done():
                request.future.set_exception(SpeechRuntimeClosed("Speech runtime was shut down"))
            self._queue.task_done()


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
