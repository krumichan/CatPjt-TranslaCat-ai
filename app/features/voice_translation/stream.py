from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import deque
from dataclasses import dataclass, replace
from typing import TypeAlias
from uuid import uuid4

from app.core.config import settings
from app.features.voice_translation.audio import (
    AudioFrameValidator,
    PcmFrame,
    VadNoSpeech,
    VadPartial,
    VadProcessResult,
    VadUtterance,
    VoiceActivityDetector,
    normalize_voice_pcm,
)
from app.features.voice_translation.errors import VoicePipelineException
from app.features.voice_translation.language import (
    LanguageStateManager,
    VoiceLanguageObservation,
    normalize_confidence,
    normalize_voice_language,
)
from app.features.voice_translation.realtime_preview import (
    VoicePreviewJob,
    VoicePreviewProvider,
    VoicePreviewUpdate,
)
from app.features.voice_translation.speech_detector import (
    PassThroughSpeechEvidenceGuard,
    SileroSpeechEvidenceGuard,
    VoiceSpeechEvidenceGuard,
)
from app.features.voice_translation.stable_prefix import StablePrefixAssembler
from app.features.voice_translation.stt import VoiceSttProvider, VoiceSttResult
from app.features.voice_translation.translation import VoiceTranslationService
from app.schemas.voice_translation import (
    SUPPORTED_VOICE_LANGUAGES,
    VoiceBackpressureEvent,
    VoiceErrorCode,
    VoiceEventBase,
    VoiceLatency,
    VoiceModelMetadata,
    VoiceNoSpeechEvent,
    VoicePipelineCompletedEvent,
    VoicePipelineFailedEvent,
    VoiceSpeechStartedEvent,
    VoiceStage,
    VoiceStreamClosedEvent,
    VoiceStreamOpen,
    VoiceStreamReadyEvent,
    VoiceTranscriptFinalEvent,
    VoiceTranscriptPartialEvent,
)

logger = logging.getLogger(__name__)
_MAX_TRANSLATION_CONTEXT_GAP_MS = 5000
_MAX_TRANSLATION_CONTEXT_SILENCE_SEGMENTS = 256


def _is_repetitive_transcript(text: str) -> bool:
    words = re.findall(r"[^\W_]+(?:['’][^\W_]+)*", text.casefold())
    return len(words) >= 8 and len(set(words)) / len(words) <= 0.40


@dataclass(frozen=True)
class _UtteranceIdentity:
    key: str
    sequence: int


@dataclass(frozen=True)
class _AudioCommand:
    frame: PcmFrame


@dataclass(frozen=True)
class _FlushCommand:
    reason: str
    future: asyncio.Future[None]


@dataclass(frozen=True)
class _CloseCommand:
    reason: str
    future: asyncio.Future[None]


@dataclass
class _PreparedUtterance:
    stt_result: VoiceSttResult | None = None
    stt_finalize_ms: int = 0
    speech_evidence_ms: int = 0
    no_speech: bool = False
    error: VoicePipelineException | None = None


@dataclass
class _FinalizationJob:
    identity: _UtteranceIdentity
    utterance: VadUtterance
    partial_inference_ms: int | None
    queued_at: float
    deadline: float
    stage: VoiceStage = VoiceStage.STT
    source_text: str | None = None
    detected_language: str | None = None
    language_confidence: float | None = None
    preparation_task: asyncio.Task[_PreparedUtterance] | None = None


_StreamCommand: TypeAlias = _AudioCommand | _FlushCommand | _CloseCommand


class VoiceChannelStreamContext:
    """Owns all mutable state for one Session/Channel WebSocket."""

    def __init__(
        self,
        stream_open: VoiceStreamOpen,
        *,
        stt_provider: VoiceSttProvider,
        translation_service: VoiceTranslationService,
        speech_evidence_guard: VoiceSpeechEvidenceGuard,
        preview_provider: VoicePreviewProvider | None = None,
        preview_language_provider: VoiceSttProvider | None = None,
        preview_translation_service: VoiceTranslationService | None = None,
        preview_slots: asyncio.Semaphore | None = None,
    ) -> None:
        self.stream_open = stream_open
        self.stt_provider = stt_provider
        self.translation_service = translation_service
        self.speech_evidence_guard = speech_evidence_guard
        self._preview_provider = preview_provider
        self._preview_language_provider = preview_language_provider
        self._preview_translation_service = preview_translation_service
        self._preview_slots = preview_slots or asyncio.Semaphore(1)
        self._preview_jobs: dict[str, VoicePreviewJob] = {}

        frame_duration_ms = stream_open.audio_format.frame_duration_ms
        input_capacity = max(
            1,
            settings.AI_VOICE_MAX_BUFFERED_AUDIO_MS // frame_duration_ms,
        )
        self._input_capacity = input_capacity
        self._input_queue: asyncio.Queue[_StreamCommand] = asyncio.Queue(maxsize=input_capacity)
        self._event_queue: asyncio.Queue[VoiceEventBase] = asyncio.Queue(maxsize=64)
        self._final_queue: asyncio.Queue[_FinalizationJob] = asyncio.Queue(
            maxsize=settings.AI_VOICE_FINAL_QUEUE_CAPACITY,
        )
        self._pending_final_jobs: dict[str, _FinalizationJob] = {}
        self._pending_final_audio_ms = 0
        self._active_final_job: _FinalizationJob | None = None
        self._prefetched_job: _FinalizationJob | None = None
        self._validator = AudioFrameValidator(
            stream_open.audio_format,
            minimum_frame_duration_ms=settings.AI_VOICE_MIN_FRAME_DURATION_MS,
            maximum_frame_duration_ms=settings.AI_VOICE_MAX_FRAME_DURATION_MS,
            speech_rms_threshold=settings.AI_VOICE_VAD_RMS_THRESHOLD,
            speech_candidate_rms_threshold=(
                settings.AI_VOICE_VAD_QUIET_RMS_THRESHOLD
                if isinstance(speech_evidence_guard, SileroSpeechEvidenceGuard)
                and speech_evidence_guard.enabled
                else None
            ),
        )
        policy = stream_open.policy
        self._vad = VoiceActivityDetector(
            endpointing_silence_ms=policy.endpointing_silence_ms,
            minimum_utterance_ms=policy.min_utterance_duration_ms,
            maximum_utterance_ms=policy.max_utterance_duration_ms,
            start_evidence_ms=settings.AI_VOICE_VAD_START_EVIDENCE_MS,
            partial_interval_ms=settings.AI_VOICE_PARTIAL_INTERVAL_MS,
            pre_roll_ms=settings.AI_VOICE_VAD_PRE_ROLL_MS,
            post_roll_ms=settings.AI_VOICE_VAD_POST_ROLL_MS,
            force_split_overlap_ms=settings.AI_VOICE_FORCE_SPLIT_OVERLAP_MS,
        )
        self._language = LanguageStateManager(
            lock_confidence=policy.language_lock_confidence,
            switch_confidence=policy.language_switch_confidence,
            switch_consecutive_count=policy.language_switch_consecutive_count,
            manual_language=stream_open.manual_source_language,
            initial_locked_language=stream_open.last_locked_language,
            minimum_detection_confidence=settings.AI_VOICE_LANGUAGE_MIN_CONFIDENCE,
        )
        self._last_completed_source_text: str | None = None
        self._last_completed_source_language: str | None = None
        self._last_completed_sequence: int | None = None
        self._last_completed_end_offset_ms: int | None = None
        self._context_no_speech_sequences: set[int] = set()

        self._worker_task: asyncio.Task[None] | None = None
        self._final_worker_task: asyncio.Task[None] | None = None
        self._partial_task: asyncio.Task[None] | None = None
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._current_identity: _UtteranceIdentity | None = None
        self._current_assembler: StablePrefixAssembler | None = None
        self._utterance_sequence = 0
        self._finalized_utterance_keys: set[str] = set()
        self._finalized_utterance_order: deque[str] = deque()
        self._last_partial_inference_ms: int | None = None
        self._accepting_audio = True
        self._flushed = False
        self._closed = False
        self._backpressure_announced = False
        self._flush_future: asyncio.Future[None] | None = None
        self._close_future: asyncio.Future[None] | None = None
        self._termination_lock = asyncio.Lock()
        self._termination_task: asyncio.Task[None] | None = None
        self._pipeline_failure_emitted = False

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def buffered_audio_ms(self) -> int:
        return self._input_queue.qsize() * self.stream_open.audio_format.frame_duration_ms

    async def start(self) -> None:
        if self._worker_task is not None:
            return
        self._worker_task = asyncio.create_task(
            self._run(),
            name=(f"voice-stream-{self.stream_open.session_id}-{self.stream_open.channel.value}"),
        )
        self._final_worker_task = asyncio.create_task(
            self._run_finalizations(),
            name=f"voice-final-{self.stream_open.session_id}-{self.stream_open.channel.value}",
        )
        self._final_worker_task.add_done_callback(self._inference_task_done)
        await self._emit(
            VoiceStreamReadyEvent(
                event_id=self._event_id(),
                session_id=self.stream_open.session_id,
                channel=self.stream_open.channel,
                request_id=self.stream_open.request_id,
                model_ready=self.stt_provider.ready,
            )
        )

    async def next_event(self) -> VoiceEventBase:
        # 전송 큐가 가득 차 종료 이벤트를 넣지 못한 경우에도 소비자는 종료를 확인한다.
        if self._closed and self._event_queue.empty():
            return VoiceStreamClosedEvent(
                event_id=self._event_id(),
                session_id=self.stream_open.session_id,
                channel=self.stream_open.channel,
                reason="PIPELINE_ERROR",
            )
        return await self._event_queue.get()

    async def feed_audio(self, data: bytes) -> bool:
        if not self._accepting_audio or self._closed:
            raise VoicePipelineException(
                code=VoiceErrorCode.INVALID_AUDIO_FRAME,
                stage=VoiceStage.STREAM,
                message="Flush 또는 Close 이후에는 Audio를 받을 수 없습니다.",
                retryable=False,
            )

        frame = self._validator.validate(data)
        try:
            self._input_queue.put_nowait(_AudioCommand(frame=frame))
            if self._input_queue.qsize() <= self._input_capacity // 2:
                self._backpressure_announced = False
            return True
        except asyncio.QueueFull as exc:
            if not self._backpressure_announced:
                self._backpressure_announced = True
                await self._emit(
                    VoiceBackpressureEvent(
                        event_id=self._event_id(),
                        session_id=self.stream_open.session_id,
                        channel=self.stream_open.channel,
                        utterance_key=(
                            self._current_identity.key
                            if self._current_identity is not None
                            else None
                        ),
                        utterance_sequence=(
                            self._current_identity.sequence
                            if self._current_identity is not None
                            else None
                        ),
                        error=VoicePipelineException(
                            code=VoiceErrorCode.BACKPRESSURE,
                            stage=VoiceStage.STREAM,
                            message="Channel Audio Buffer가 가득 찼습니다.",
                            retryable=True,
                        ).as_error(),
                        buffered_audio_ms=self.buffered_audio_ms,
                        retry_after_ms=settings.AI_VOICE_BACKPRESSURE_RETRY_AFTER_MS,
                    )
                )
            # 수신 거절을 숨기고 계속 진행하면 음성과 타임라인이 서로 어긋난다.
            self._accepting_audio = False
            raise VoicePipelineException(
                code=VoiceErrorCode.BACKPRESSURE,
                stage=VoiceStage.STREAM,
                message="음성 입력 처리 용량을 초과하여 스트림을 종료합니다. 다시 시작해 주세요.",
                retryable=True,
            ) from exc

    async def flush(self, reason: str) -> None:
        if self._flush_future is not None:
            await asyncio.shield(self._flush_future)
            return
        if self._closed:
            return
        self._accepting_audio = False
        self._flushed = True
        loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = loop.create_future()
        self._flush_future = future
        await self._input_queue.put(_FlushCommand(reason=reason, future=future))
        await asyncio.shield(future)

    async def close(self, reason: str) -> None:
        if self._close_future is not None:
            await asyncio.shield(self._close_future)
            return
        if self._closed:
            return
        self._accepting_audio = False
        loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = loop.create_future()
        self._close_future = future
        await self._input_queue.put(_CloseCommand(reason=reason, future=future))
        await asyncio.shield(future)
        if self._worker_task is not None:
            await asyncio.gather(self._worker_task, return_exceptions=True)

    async def abort(self) -> None:
        if self._closed:
            return
        self._accepting_audio = False
        self._closed = True
        if self._worker_task is not None:
            self._worker_task.cancel()
            await asyncio.gather(self._worker_task, return_exceptions=True)
        await self._cancel_background_tasks()
        await self._cancel_final_jobs()
        self._vad.clear()
        self._drain_input_queue()

    async def emit_failure(
        self,
        error: VoicePipelineException,
        *,
        source_text: str | None = None,
        detected_language: str | None = None,
        language_confidence: float | None = None,
        identity: _UtteranceIdentity | None = None,
    ) -> None:
        identity = identity or self._current_identity
        # 실패한 구간을 건너 이전 성공 원문의 의미를 다음 발화에 이어 붙이지 않는다.
        self._clear_translation_context()
        if identity is not None:
            self._context_no_speech_sequences.discard(identity.sequence)
        # 원문/공급자 오류 메시지를 남기지 않고 실패 단계와 발화를 추적한다.
        logger.warning(
            "Voice pipeline failed session=%s channel=%s code=%s stage=%s "
            "retryable=%s utterance_sequence=%s",
            self.stream_open.session_id,
            self.stream_open.channel.value,
            error.code.value,
            error.stage.value,
            error.retryable,
            identity.sequence if identity else None,
        )
        await self._emit(
            VoicePipelineFailedEvent(
                event_id=self._event_id(),
                session_id=self.stream_open.session_id,
                channel=self.stream_open.channel,
                utterance_key=identity.key if identity else None,
                utterance_sequence=identity.sequence if identity else None,
                source_text=source_text,
                detected_language=detected_language,
                language_confidence=language_confidence,
                locked_language=self._language.locked_language,
                target_language=self.stream_open.target_language,
                error=error.as_error(),
            )
        )

    async def _run(self) -> None:
        try:
            while True:
                command = await self._input_queue.get()
                try:
                    should_close = await self._process_command(command)
                    if should_close:
                        return
                except Exception as exc:
                    self._fail_control_command(command, exc)
                    raise
                finally:
                    self._input_queue.task_done()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Exception messages and tracebacks can contain provider payloads.
            # Keep Voice logs metadata-only to protect transcripts.
            logger.error(
                "Voice stream worker failed session=%s channel=%s error_type=%s",
                self.stream_open.session_id,
                self.stream_open.channel.value,
                type(exc).__name__,
            )
            await self.fail_and_close(
                exc
                if isinstance(exc, VoicePipelineException)
                else VoicePipelineException(
                    code=VoiceErrorCode.INTERNAL_ERROR,
                    stage=VoiceStage.STREAM,
                    message="Voice Stream 처리 중 내부 오류가 발생했습니다.",
                    retryable=True,
                )
            )

    async def fail_and_close(self, error: VoicePipelineException) -> None:
        # worker와 WebSocket 경계가 같은 실패를 받아도 한 번만 알린다.
        # 실패한 worker에 CLOSE를 다시 넣으면 정리 중 취소된 Future를 기다리게 된다.
        self._accepting_audio = False
        if not self._pipeline_failure_emitted and not self._closed:
            self._pipeline_failure_emitted = True
            try:
                await self.emit_failure(error)
            except VoicePipelineException:
                pass
        await self._terminate_stream()

    async def _terminate_stream(self) -> None:
        async with self._termination_lock:
            if self._closed:
                return
            self._accepting_audio = False
            try:
                if (
                    self._worker_task is not asyncio.current_task()
                    and self._worker_task is not None
                ):
                    self._worker_task.cancel()
                    await asyncio.gather(self._worker_task, return_exceptions=True)
                try:
                    await self._cancel_final_jobs(
                        VoicePipelineException(
                            code=VoiceErrorCode.BACKPRESSURE,
                            stage=VoiceStage.STREAM,
                            message="음성 스트림이 종료되어 대기 중인 발화를 완료하지 못했습니다.",
                            retryable=True,
                        )
                    )
                except VoicePipelineException:
                    # 전송 자체가 포화여도 상태/대기 중인 제어 명령은 반드시 정리한다.
                    pass
                await self._cancel_background_tasks()
                self._vad.clear()
                self._drain_input_queue()
                try:
                    await self._finish_close("PIPELINE_ERROR")
                except VoicePipelineException:
                    pass
            finally:
                self._closed = True

    def _inference_task_done(self, task: asyncio.Task[None]) -> None:
        if task.cancelled() or task.exception() is None or self._closed:
            return
        if self._termination_task is None:
            self._termination_task = asyncio.create_task(self._terminate_stream())
            self._termination_task.add_done_callback(self._consume_task_result)

    async def _process_command(self, command: _StreamCommand) -> bool:
        if isinstance(command, _AudioCommand):
            await self._handle_vad_result(self._vad.process(command.frame))
            return False
        if isinstance(command, _FlushCommand):
            await self._handle_vad_result(self._vad.flush())
            await self._drain_final_jobs()
            if not command.future.done():
                command.future.set_result(None)
            return False

        if not self._flushed:
            await self._handle_vad_result(self._vad.flush())
        await self._drain_final_jobs()
        await self._finish_close(command.reason)
        if not command.future.done():
            command.future.set_result(None)
        return True

    @staticmethod
    def _fail_control_command(command: _StreamCommand, exc: Exception) -> None:
        if isinstance(command, (_FlushCommand, _CloseCommand)):
            if not command.future.done():
                command.future.set_exception(exc)

    async def _handle_vad_result(self, result: VadProcessResult) -> None:
        if result.speech_started_at_offset_ms is not None:
            await self._start_utterance(result.speech_started_at_offset_ms)

        if result.utterance is None and result.no_speech is None and result.partial:
            self._schedule_partial(result.partial)

        if result.utterance is not None:
            self._queue_finalization(result.utterance)
        elif result.no_speech is not None:
            await self._emit_no_speech(result.no_speech)

    async def _start_utterance(self, started_at_offset_ms: int) -> None:
        self._utterance_sequence += 1
        identity = _UtteranceIdentity(
            key=f"{self.stream_open.channel.value}-{self._utterance_sequence}",
            sequence=self._utterance_sequence,
        )
        self._current_identity = identity
        self._current_assembler = StablePrefixAssembler()
        await self._emit(
            VoiceSpeechStartedEvent(
                event_id=self._event_id(),
                session_id=self.stream_open.session_id,
                channel=self.stream_open.channel,
                utterance_key=identity.key,
                utterance_sequence=identity.sequence,
                started_at_offset_ms=started_at_offset_ms,
            )
        )

    def _schedule_partial(self, partial: VadPartial) -> None:
        identity = self._current_identity
        assembler = self._current_assembler
        if identity is None or assembler is None:
            return
        if self._preview_provider is not None and self._preview_language_provider is not None:
            self._offer_realtime_preview(partial, identity)
            return
        if self._pending_final_jobs:
            return
        if not getattr(self.stt_provider, "supports_partial", True):
            return
        # 취소된 native 추론도 실제로 끝날 때까지 자리를 차지하므로 공급자의 여유를 확인한다.
        if not getattr(self.stt_provider, "partial_available", True):
            return
        if self._partial_task is not None and not self._partial_task.done():
            return

        task = asyncio.create_task(
            self._run_partial(partial, identity, assembler),
            name=f"voice-partial-{identity.key}",
        )
        self._partial_task = task
        self._background_tasks.add(task)
        task.add_done_callback(self._partial_done)

    def _offer_realtime_preview(self, partial: VadPartial, identity: _UtteranceIdentity) -> None:
        existing = self._preview_jobs.get(identity.key)
        if existing is not None:
            existing.offer(partial)
            return
        if (
            partial.ended_at_offset_ms - partial.started_at_offset_ms
            < settings.AI_VOICE_REALTIME_PREVIEW_MIN_AUDIO_MS
            or len(self._preview_jobs) >= 2
            or len(self._pending_final_jobs) >= 2
            or self._preview_provider is None
            or self._preview_language_provider is None
            or self._preview_translation_service is None
        ):
            return

        async def emit_preview(update: VoicePreviewUpdate) -> None:
            # 최종 원문 이벤트가 소유권을 제거한다. 큐의 여유도 확정 결과에 남긴다.
            if (
                self._closed
                or identity.key not in self._preview_jobs
                or self._event_queue.qsize() >= 48
            ):
                return
            await self._emit(VoiceTranscriptPartialEvent(
                event_id=self._event_id(),
                session_id=self.stream_open.session_id,
                channel=self.stream_open.channel,
                utterance_key=identity.key,
                utterance_sequence=identity.sequence,
                revision=update.revision,
                source_text=update.source_text,
                translated_text=update.translated_text,
                target_language=self.stream_open.target_language,
                detected_language=update.detected_language,
                started_at_offset_ms=update.started_at_offset_ms,
                ended_at_offset_ms=update.ended_at_offset_ms,
            ))

        job = VoicePreviewJob(
            partial,
            provider=self._preview_provider,
            language_provider=self._preview_language_provider,
            translation_service=self._preview_translation_service,
            speech_guard=self.speech_evidence_guard,
            target_language=self.stream_open.target_language,
            manual_language=self.stream_open.manual_source_language,
            emit=emit_preview,
            slots=self._preview_slots,
        )
        self._preview_jobs[identity.key] = job
        self._background_tasks.add(job.task)
        job.task.add_done_callback(self._partial_done)

    async def _run_partial(
        self,
        partial: VadPartial,
        identity: _UtteranceIdentity,
        assembler: StablePrefixAssembler,
    ) -> None:
        started = time.perf_counter()
        try:
            normalized_pcm = normalize_voice_pcm(partial.pcm_bytes)
            # 잡음/음악은 임시 자막에서도 제외하고 확정 작업에는 추론 자리를 양보한다.
            has_speech = await asyncio.wait_for(
                self.speech_evidence_guard.has_speech(normalized_pcm),
                timeout=settings.AI_VOICE_VAD_SILERO_TIMEOUT_SECONDS,
            )
            if (
                not has_speech
                or identity.key in self._finalized_utterance_keys
                or self._pending_final_jobs
            ):
                return
            result = await asyncio.wait_for(
                self.stt_provider.transcribe_pcm(
                    normalized_pcm,
                    language=None,
                    is_final=False,
                    initial_prompt=assembler.stable_prefix or None,
                ),
                timeout=settings.AI_VOICE_STT_PARTIAL_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            code = (
                exc.code.value
                if isinstance(exc, VoicePipelineException)
                else VoiceErrorCode.STT_TIMEOUT.value
                if isinstance(exc, (TimeoutError, asyncio.TimeoutError))
                else VoiceErrorCode.STT_FAILED.value
            )
            logger.warning(
                "Voice partial STT skipped session=%s channel=%s code=%s",
                self.stream_open.session_id,
                self.stream_open.channel.value,
                code,
            )
            return

        partial_inference_ms = max(
            0,
            int((time.perf_counter() - started) * 1000),
        )
        if (
            self._closed
            or identity.key in self._finalized_utterance_keys
            or self._current_identity != identity
        ):
            return
        self._last_partial_inference_ms = partial_inference_ms

        # 표시하지 않을 언어의 임시 결과는 다음 추론의 힌트에도 섞지 않는다.
        detected_language = normalize_voice_language(result.language)
        if detected_language is None:
            return
        if self.stream_open.manual_source_language not in (None, detected_language):
            return
        if self.stream_open.manual_source_language is not None:
            confidence = normalize_confidence(result.language_confidence)
            if confidence is None or confidence < settings.AI_VOICE_LANGUAGE_MIN_CONFIDENCE:
                return
        update = assembler.update(result.text)
        if not update.changed or not update.text:
            return
        if len(update.text) > 10_000:
            logger.warning(
                "Voice partial STT skipped session=%s channel=%s code=%s",
                self.stream_open.session_id,
                self.stream_open.channel.value,
                VoiceErrorCode.INVALID_EVENT_SCHEMA.value,
            )
            return
        if identity.key in self._finalized_utterance_keys:
            return

        await self._emit(
            VoiceTranscriptPartialEvent(
                event_id=self._event_id(),
                session_id=self.stream_open.session_id,
                channel=self.stream_open.channel,
                utterance_key=identity.key,
                utterance_sequence=identity.sequence,
                revision=update.revision,
                source_text=update.text,
                detected_language=detected_language,
                language_confidence=(
                    normalize_confidence(result.language_confidence)
                    if detected_language is not None
                    else None
                ),
                started_at_offset_ms=partial.started_at_offset_ms,
                ended_at_offset_ms=partial.ended_at_offset_ms,
            )
        )
        logger.info(
            "Voice partial emitted session=%s channel=%s sequence=%s revision=%s "
            "partial_inference_ms=%s first_partial_ms=%s",
            self.stream_open.session_id,
            self.stream_open.channel.value,
            identity.sequence,
            update.revision,
            partial_inference_ms,
            (
                partial.ended_at_offset_ms - partial.started_at_offset_ms + partial_inference_ms
                if update.revision == 1
                else None
            ),
        )

    def _queue_finalization(self, utterance: VadUtterance) -> None:
        identity = self._current_identity
        if identity is None:
            return
        preview = self._preview_jobs.get(identity.key)
        if preview is not None:
            preview.offer(VadPartial(
                pcm_bytes=utterance.pcm_bytes,
                started_at_offset_ms=utterance.started_at_offset_ms,
                ended_at_offset_ms=utterance.ended_at_offset_ms,
            ), final=True)

        # 확정 추론과 오디오 소비를 분리하되 저장하는 PCM과 발화 수를 함께 제한한다.
        audio_ms = len(utterance.pcm_bytes) // 32
        if (
            len(self._pending_final_jobs) >= settings.AI_VOICE_FINAL_QUEUE_CAPACITY
            or self._pending_final_audio_ms + audio_ms
            > settings.AI_VOICE_MAX_PENDING_FINAL_AUDIO_MS
        ):
            raise VoicePipelineException(
                code=VoiceErrorCode.BACKPRESSURE,
                stage=VoiceStage.STREAM,
                message="확정 음성 처리 용량을 초과하여 스트림을 종료합니다. 다시 시작해 주세요.",
                retryable=True,
            )
        self._mark_finalized(identity.key)
        if self._partial_task is not None and not self._partial_task.done():
            self._partial_task.cancel()
        self._partial_task = None

        queued_at = asyncio.get_running_loop().time()
        job = _FinalizationJob(
            identity=identity,
            utterance=utterance,
            partial_inference_ms=self._last_partial_inference_ms,
            queued_at=queued_at,
            deadline=queued_at + settings.AI_VOICE_PIPELINE_TIMEOUT_SECONDS,
        )
        self._pending_final_jobs[identity.key] = job
        self._pending_final_audio_ms += audio_ms
        self._final_queue.put_nowait(job)
        self._clear_current_utterance()
        self._schedule_stt_prefetch()

    def _schedule_stt_prefetch(self) -> None:
        # 외부 STT만 바로 다음 한 발화를 준비한다. 언어 상태/번역 문맥은 건드리지 않는다.
        active = self._active_final_job
        if (
            not getattr(self.stt_provider, "supports_stt_prefetch", False)
            or active is None
            or active.stage != VoiceStage.TRANSLATION
            or self._prefetched_job is not None
            or self._closed
        ):
            return
        following = next(
            (
                job for job in self._pending_final_jobs.values()
                if job.identity.sequence > active.identity.sequence
            ),
            None,
        )
        if following is None:
            return
        self._prefetched_job = following
        following.preparation_task = asyncio.create_task(
            self._prepare_prefetched_utterance(following),
            name=f"voice-stt-prefetch-{following.identity.key}",
        )
        following.preparation_task.add_done_callback(self._consume_preparation_result)

    async def _prepare_prefetched_utterance(self, job: _FinalizationJob) -> _PreparedUtterance:
        # 미리 시작해도 원래 수락 시점의 전체 기한과 STT 기한을 그대로 적용한다.
        async with asyncio.timeout_at(job.deadline):
            self._check_final_job_ownership(job)
            return await self._prepare_utterance(job)

    @staticmethod
    def _consume_preparation_result(task: asyncio.Task[_PreparedUtterance]) -> None:
        if not task.cancelled():
            task.exception()

    async def _run_finalizations(self) -> None:
        while True:
            job = await self._final_queue.get()
            self._active_final_job = job
            if self._prefetched_job is job:
                self._prefetched_job = None
            try:
                # 같은 채널의 확정 순서와 언어 상태는 하나의 worker에서 보존한다.
                if asyncio.get_running_loop().time() >= job.deadline:
                    raise TimeoutError
                async with asyncio.timeout_at(job.deadline):
                    await self._finalize_utterance(job)
            except TimeoutError:
                await self._emit_final_job_failure(
                    job,
                    VoicePipelineException(
                        code=VoiceErrorCode.STT_TIMEOUT,
                        stage=VoiceStage.STREAM,
                        message="대기 시간을 포함한 음성 처리 기한을 초과했습니다.",
                        retryable=True,
                    ),
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                await self._emit_final_job_failure(
                    job,
                    VoicePipelineException(
                        code=VoiceErrorCode.INTERNAL_ERROR,
                        stage=VoiceStage.STREAM,
                        message="확정 음성 처리 중 오류가 발생했습니다.",
                        retryable=True,
                    ),
                )
            finally:
                if job.preparation_task is not None and not job.preparation_task.done():
                    job.preparation_task.cancel()
                if self._pending_final_jobs.pop(job.identity.key, None) is not None:
                    self._pending_final_audio_ms -= len(job.utterance.pcm_bytes) // 32
                self._active_final_job = None
                self._final_queue.task_done()

    async def _drain_final_jobs(self) -> None:
        # flush/close는 남은 확정을 최대 20초만 기다리며 늦은 결과를 취소한다.
        try:
            await asyncio.wait_for(
                self._final_queue.join(),
                timeout=settings.AI_VOICE_DRAIN_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            await self._cancel_final_jobs(
                VoicePipelineException(
                    code=VoiceErrorCode.STT_TIMEOUT,
                    stage=VoiceStage.STREAM,
                    message="음성 종료 처리 기한을 초과하여 남은 발화를 중단했습니다.",
                    retryable=True,
                )
            )

    async def _cancel_final_jobs(self, error: VoicePipelineException | None = None) -> None:
        pending = list(self._pending_final_jobs.values())
        # 공급자의 취소 응답을 기다리며 종료 기한을 다시 늘리지 않는다.
        # 소유권을 먼저 제거하므로 뒤늦게 돌아온 결과도 상태를 변경할 수 없다.
        self._pending_final_jobs.clear()
        self._pending_final_audio_ms = 0
        self._prefetched_job = None
        # 선행 STT도 같은 소유권에 속한다. 종료 후 늦은 결과가 이벤트를 만들지 못한다.
        for job in pending:
            if job.preparation_task is not None and not job.preparation_task.done():
                job.preparation_task.cancel()
        if self._final_worker_task is not None:
            worker = self._final_worker_task
            self._final_worker_task = None
            worker.cancel()
            worker.add_done_callback(self._consume_task_result)
            await asyncio.sleep(0)
        while not self._final_queue.empty():
            self._final_queue.get_nowait()
            self._final_queue.task_done()

        # 이미 수락한 발화마다 실패를 알려 DB/화면에 처리 중 상태를 남기지 않는다.
        if error is not None:
            for job in pending:
                await self._emit_final_job_failure(job, error)

    async def _emit_final_job_failure(
        self,
        job: _FinalizationJob,
        error: VoicePipelineException,
    ) -> None:
        # STT가 이미 끝난 발화는 종료/전체 기한 오류에도 원문과 번역 실패 단계를 보존한다.
        code = error.code
        if code == VoiceErrorCode.STT_TIMEOUT and job.stage == VoiceStage.TRANSLATION:
            code = VoiceErrorCode.TRANSLATION_TIMEOUT
        await self.emit_failure(
            VoicePipelineException(
                code=code,
                stage=job.stage,
                message=error.public_message,
                retryable=error.retryable,
            ),
            source_text=job.source_text,
            detected_language=job.detected_language,
            language_confidence=job.language_confidence,
            identity=job.identity,
        )

    async def _transcribe_final_with_context(self, pcm_bytes: bytes) -> VoiceSttResult:
        # 자동 감지를 먼저 유지해 실제 언어 전환과 비지원 언어를 확인한다.
        deadline = asyncio.get_running_loop().time() + settings.AI_VOICE_STT_FINAL_TIMEOUT_SECONDS
        original = await self.stt_provider.transcribe_pcm(
            pcm_bytes,
            language=None,
            is_final=True,
            initial_prompt=None,
        )

        # 영어 등으로 잠긴 대화에서 저확신 비지원 언어와 반복 원문이 함께 나올 때만 재인식한다.
        # 수동 언어 선택과 자연스러운 비지원 언어 발화에는 이 문맥 보정을 적용하지 않는다.
        detected = (original.language or "").strip().lower().split("-")[0]
        confidence = normalize_confidence(original.language_confidence)
        locked = self._language.locked_language
        if (
            self.stream_open.manual_source_language is not None
            or locked is None
            or detected in SUPPORTED_VOICE_LANGUAGES
            or detected in ("", "und")
            or confidence is None
            or confidence >= 0.65
            or not _is_repetitive_transcript(original.text)
        ):
            return original

        # 두 번째 추론은 첫 번째와 같은 전체 STT 기한 안에서만 수행한다.
        remaining = deadline - asyncio.get_running_loop().time() - 0.25
        if remaining < 0.5:
            return original
        try:
            candidate = await asyncio.wait_for(
                self.stt_provider.transcribe_pcm(
                    pcm_bytes,
                    language=locked,
                    is_final=True,
                    initial_prompt=None,
                ),
                timeout=min(3.0, remaining),
            )
        except Exception:
            logger.info(
                "Voice contextual STT retry unavailable session=%s channel=%s original_language=%s",
                self.stream_open.session_id,
                self.stream_open.channel.value,
                detected,
            )
            return original

        # 강제 언어의 confidence는 독립적인 언어 판정이 아니므로 공개 결과에 사용하지 않는다.
        words = re.findall(r"[^\W_]+(?:['’][^\W_]+)*", candidate.text.casefold())
        accepted = (
            normalize_voice_language(candidate.language) == locked
            and len(words) >= 5
            and len(set(words)) / len(words) > 0.50
            and (candidate.no_speech_probability is None or candidate.no_speech_probability < 0.50)
        )
        logger.info(
            "Voice contextual STT retry session=%s channel=%s original_language=%s "
            "original_confidence=%.2f locked_language=%s accepted=%s",
            self.stream_open.session_id,
            self.stream_open.channel.value,
            detected,
            confidence,
            locked,
            accepted,
        )
        if not accepted:
            return original
        return replace(candidate, language_confidence=None)

    async def _prepare_utterance(self, job: _FinalizationJob) -> _PreparedUtterance:
        utterance = job.utterance

        # 음량 보정은 한 번만 수행해 VAD 확인과 STT가 같은 파형을 사용한다.
        normalized_pcm = normalize_voice_pcm(utterance.pcm_bytes)
        job.stage = VoiceStage.VAD
        try:
            has_speech, speech_evidence_ms = await self._confirm_speech_evidence(normalized_pcm)
        except VoicePipelineException as exc:
            self._check_final_job_ownership(job)
            return _PreparedUtterance(error=exc)
        self._check_final_job_ownership(job)
        if not has_speech:
            return _PreparedUtterance(no_speech=True)

        stt_started = time.perf_counter()
        job.stage = VoiceStage.STT
        try:
            stt_result = await asyncio.wait_for(
                self.stt_provider.transcribe_pcm(
                    normalized_pcm, language=None, is_final=True, initial_prompt=None
                )
                if getattr(self.stt_provider, "supports_stt_prefetch", False)
                else self._transcribe_final_with_context(normalized_pcm),
                timeout=settings.AI_VOICE_STT_FINAL_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            self._check_final_job_ownership(job)
            return _PreparedUtterance(
                error=VoicePipelineException(
                    code=VoiceErrorCode.STT_TIMEOUT,
                    stage=VoiceStage.STT,
                    message="Final STT 응답 시간이 초과되었습니다.",
                    retryable=True,
                ),
            )
        except VoicePipelineException as exc:
            self._check_final_job_ownership(job)
            return _PreparedUtterance(error=exc)
        except Exception:
            self._check_final_job_ownership(job)
            return _PreparedUtterance(
                error=VoicePipelineException(
                    code=VoiceErrorCode.STT_FAILED,
                    stage=VoiceStage.STT,
                    message="Final STT 처리에 실패했습니다.",
                    retryable=True,
                ),
            )

        stt_finalize_ms = max(0, int((time.perf_counter() - stt_started) * 1000))
        self._check_final_job_ownership(job)
        source_text = stt_result.text.strip()
        if not source_text:
            return _PreparedUtterance(no_speech=True)
        if len(source_text) > 10_000:
            return _PreparedUtterance(
                error=VoicePipelineException(
                    code=VoiceErrorCode.STT_FAILED,
                    stage=VoiceStage.STT,
                    message="Final STT 결과가 허용 길이를 초과했습니다.",
                    retryable=False,
                ),
            )
        return _PreparedUtterance(
            stt_result=stt_result,
            stt_finalize_ms=stt_finalize_ms,
            speech_evidence_ms=speech_evidence_ms,
        )

    async def _finalize_utterance(self, job: _FinalizationJob) -> None:
        identity = job.identity
        utterance = job.utterance

        # 준비 결과의 성공/무음/오류는 모두 순서대로 소비해 발화 이벤트 순서를 유지한다.
        prepared = (
            await job.preparation_task
            if job.preparation_task is not None
            else await self._prepare_utterance(job)
        )
        self._check_final_job_ownership(job)
        if prepared.error is not None:
            await self.emit_failure(prepared.error, identity=identity)
            return
        if prepared.no_speech:
            await self._emit_no_speech(
                VadNoSpeech(
                    started_at_offset_ms=utterance.started_at_offset_ms,
                    ended_at_offset_ms=utterance.ended_at_offset_ms,
                ),
                identity=identity,
                clear_current=False,
            )
            return
        stt_result = prepared.stt_result
        if stt_result is None:
            raise RuntimeError("Missing prepared STT result")
        source_text = stt_result.text.strip()
        stt_finalize_ms = prepared.stt_finalize_ms
        speech_evidence_ms = prepared.speech_evidence_ms

        # 원문은 보존하되 비지원/선택 불일치 언어를 번역하거나 잠금에 반영하지 않는다.
        job.source_text = source_text
        job.stage = VoiceStage.LANGUAGE
        language_error: VoicePipelineException | None = None
        try:
            observation = self._language.observe(
                stt_result.language,
                stt_result.language_confidence,
                trusted_language=stt_result.trusted_language,
            )
        except VoicePipelineException as exc:
            language_error = exc
            detected = normalize_voice_language(stt_result.language)
            observation = VoiceLanguageObservation(
                detected_language=detected,
                confidence=normalize_confidence(stt_result.language_confidence)
                if detected
                else None,
                locked_language=self._language.locked_language,
            )
        job.detected_language = observation.detected_language
        job.language_confidence = observation.confidence
        final_event = VoiceTranscriptFinalEvent(
            event_id=self._event_id(),
            session_id=self.stream_open.session_id,
            channel=self.stream_open.channel,
            utterance_key=identity.key,
            utterance_sequence=identity.sequence,
            source_text=source_text,
            detected_language=observation.detected_language,
            language_confidence=observation.confidence,
            locked_language=observation.locked_language,
            started_at_offset_ms=utterance.started_at_offset_ms,
            ended_at_offset_ms=utterance.ended_at_offset_ms,
            speech_duration_ms=utterance.speech_duration_ms,
            no_speech_probability=stt_result.no_speech_probability,
        )
        await self._emit(final_event)

        try:
            if language_error is not None:
                raise language_error
            job.stage = VoiceStage.TRANSLATION
            source_language = self._language.resolve_translation_source(
                detected_language=observation.detected_language,
                confidence=observation.confidence,
                trusted_language=stt_result.trusted_language,
            )
            # 확인된 무음만 건너 같은 언어의 가까운 성공 원문을 제공한다.
            # 미확인/실패 구간은 문맥을 연결하는 근거로 사용하지 않는다.
            gap_ms = (
                max(0, utterance.started_at_offset_ms - self._last_completed_end_offset_ms)
                if self._last_completed_end_offset_ms is not None
                else None
            )
            previous_source_text = (
                self._last_completed_source_text
                if self._last_completed_source_language == source_language
                and observation.detected_language == source_language
                and self._has_only_silence_context_gap(identity.sequence)
                and gap_ms is not None
                and gap_ms <= _MAX_TRANSLATION_CONTEXT_GAP_MS
                else None
            )
            self._schedule_stt_prefetch()
            translation = await self.translation_service.translate(
                source_text=source_text,
                source_language=source_language,
                target_language=self.stream_open.target_language,
                previous_source_text=previous_source_text,
            )
        except VoicePipelineException as exc:
            self._check_final_job_ownership(job)
            await self.emit_failure(
                exc,
                source_text=source_text,
                detected_language=observation.detected_language,
                language_confidence=observation.confidence,
                identity=identity,
            )
            return
        except Exception:
            self._check_final_job_ownership(job)
            await self.emit_failure(
                VoicePipelineException(
                    code=VoiceErrorCode.TRANSLATION_FAILED,
                    stage=VoiceStage.TRANSLATION,
                    message="확정 원문의 번역에 실패했습니다.",
                    retryable=False,
                ),
                source_text=source_text,
                detected_language=observation.detected_language,
                language_confidence=observation.confidence,
                identity=identity,
            )
            return

        self._check_final_job_ownership(job)
        total_after_speech_ms = utterance.endpointing_ms + max(
            0, int((asyncio.get_running_loop().time() - job.queued_at) * 1000)
        )
        await self._emit(
            VoicePipelineCompletedEvent(
                event_id=self._event_id(),
                session_id=self.stream_open.session_id,
                channel=self.stream_open.channel,
                utterance_key=identity.key,
                utterance_sequence=identity.sequence,
                source_text=source_text,
                detected_language=observation.detected_language,
                language_confidence=observation.confidence,
                locked_language=observation.locked_language,
                target_language=self.stream_open.target_language,
                translated_text=translation.translated_text,
                translation_skipped=translation.translation_skipped,
                source_reading_tokens=translation.source_reading_tokens,
                warnings=translation.warnings,
                started_at_offset_ms=utterance.started_at_offset_ms,
                ended_at_offset_ms=utterance.ended_at_offset_ms,
                speech_duration_ms=utterance.speech_duration_ms,
                no_speech_probability=stt_result.no_speech_probability,
                latency=VoiceLatency(
                    endpointing_ms=utterance.endpointing_ms,
                    speech_evidence_ms=speech_evidence_ms,
                    partial_inference_ms=job.partial_inference_ms,
                    stt_finalize_ms=stt_finalize_ms,
                    translation_ms=translation.latency_ms,
                    ai_total_after_speech_ms=total_after_speech_ms,
                ),
                model=VoiceModelMetadata(
                    stt_version=(
                        stt_result.model_version
                        or stt_result.model
                        or self.stt_provider.model_version
                    ),
                    translation_version=translation.model.translation_version,
                    prompt_version=translation.model.prompt_version,
                ),
            )
        )
        # 과거 잠금으로 번역한 미확정 언어는 다음 문장의 문맥 근거로 재사용하지 않는다.
        if observation.detected_language == source_language:
            self._last_completed_source_text = source_text
            self._last_completed_source_language = source_language
            self._last_completed_sequence = identity.sequence
            self._last_completed_end_offset_ms = utterance.ended_at_offset_ms
        else:
            self._clear_translation_context()
        # VAD 무음은 앞선 최종 번역보다 먼저 올 수 있어 미래 sequence의 판정은 보존한다.
        self._context_no_speech_sequences = {
            sequence
            for sequence in self._context_no_speech_sequences
            if sequence > identity.sequence
        }
        logger.info(
            "Voice utterance completed session=%s channel=%s sequence=%s "
            "duration_ms=%s detected_language=%s total_after_speech_ms=%s "
            "rms=%.4f clipping_ratio=%.4f vad_version=%s schema_version=%s",
            self.stream_open.session_id,
            self.stream_open.channel.value,
            identity.sequence,
            utterance.speech_duration_ms,
            observation.detected_language or "und",
            total_after_speech_ms,
            utterance.rms,
            utterance.clipping_ratio,
            settings.AI_VOICE_VAD_VERSION,
            settings.AI_VOICE_SCHEMA_VERSION,
        )

    def _has_only_silence_context_gap(self, sequence: int) -> bool:
        previous = self._last_completed_sequence
        if (
            previous is None
            or not 0 < sequence - previous <= len(self._context_no_speech_sequences) + 1
        ):
            return False
        return all(
            skipped in self._context_no_speech_sequences
            for skipped in range(previous + 1, sequence)
        )

    def _clear_translation_context(self) -> None:
        self._last_completed_source_text = None
        self._last_completed_source_language = None
        self._last_completed_sequence = None
        self._last_completed_end_offset_ms = None

    def _check_final_job_ownership(self, job: _FinalizationJob) -> None:
        if job.identity.key not in self._pending_final_jobs:
            raise asyncio.CancelledError
        if asyncio.get_running_loop().time() >= job.deadline:
            raise TimeoutError

    @staticmethod
    def _consume_task_result(task: asyncio.Task[None]) -> None:
        if not task.cancelled():
            task.exception()

    async def _confirm_speech_evidence(
        self,
        pcm_bytes: bytes,
    ) -> tuple[bool, int]:
        started = time.perf_counter()
        try:
            has_speech = await asyncio.wait_for(
                self.speech_evidence_guard.has_speech(pcm_bytes),
                timeout=settings.AI_VOICE_VAD_SILERO_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            error = VoicePipelineException(
                code=VoiceErrorCode.INTERNAL_ERROR,
                stage=VoiceStage.VAD,
                message="Speech Evidence 확인 시간이 초과되었습니다.",
                retryable=True,
            )
        except Exception:
            error = VoicePipelineException(
                code=VoiceErrorCode.INTERNAL_ERROR,
                stage=VoiceStage.VAD,
                message="Speech Evidence 확인에 실패했습니다.",
                retryable=True,
            )
        else:
            latency_ms = max(0, int((time.perf_counter() - started) * 1000))
            return has_speech, latency_ms

        raise error

    async def _emit_no_speech(
        self,
        outcome: VadNoSpeech,
        *,
        identity: _UtteranceIdentity | None = None,
        clear_current: bool = True,
    ) -> None:
        identity = identity or self._current_identity
        if identity is not None:
            self._mark_finalized(identity.key)
            self._context_no_speech_sequences.add(identity.sequence)
            if len(self._context_no_speech_sequences) > _MAX_TRANSLATION_CONTEXT_SILENCE_SEGMENTS:
                self._context_no_speech_sequences.remove(min(self._context_no_speech_sequences))
        await self._emit(
            VoiceNoSpeechEvent(
                event_id=self._event_id(),
                session_id=self.stream_open.session_id,
                channel=self.stream_open.channel,
                utterance_key=identity.key if identity else None,
                utterance_sequence=identity.sequence if identity else None,
                started_at_offset_ms=outcome.started_at_offset_ms,
                ended_at_offset_ms=outcome.ended_at_offset_ms,
                speech_duration_ms=0,
            )
        )
        if clear_current:
            self._clear_current_utterance()

    async def _finish_close(self, reason: str) -> None:
        self._accepting_audio = False
        await self._cancel_background_tasks()
        await self._cancel_final_jobs()
        self._vad.clear()
        await self._emit(
            VoiceStreamClosedEvent(
                event_id=self._event_id(),
                session_id=self.stream_open.session_id,
                channel=self.stream_open.channel,
                reason=reason,
            )
        )
        self._closed = True

    async def _cancel_background_tasks(self) -> None:
        for preview in self._preview_jobs.values():
            preview.cancel()
        self._preview_jobs.clear()
        # partial 공급자가 취소 응답을 늦추더라도 스트림 종료는 지연하지 않는다.
        tasks = [task for task in self._background_tasks if not task.done()]
        for task in tasks:
            task.cancel()
            task.add_done_callback(self._consume_task_result)
        if tasks:
            await asyncio.sleep(0)
        self._background_tasks.clear()
        self._partial_task = None

    def _partial_done(self, task: asyncio.Task[None]) -> None:
        self._background_tasks.discard(task)
        if self._partial_task is task:
            self._partial_task = None
        self._inference_task_done(task)

    def _clear_current_utterance(self) -> None:
        self._current_identity = None
        self._current_assembler = None
        self._last_partial_inference_ms = None

    async def _emit(self, event: VoiceEventBase) -> None:
        # 원문 확정/실패/무음은 모두 임시 번역보다 우선한다.
        if isinstance(event, (
            VoiceTranscriptFinalEvent, VoicePipelineCompletedEvent,
            VoicePipelineFailedEvent, VoiceNoSpeechEvent,
        )):
            key = getattr(event, "utterance_key", None)
            preview = self._preview_jobs.pop(key, None) if isinstance(key, str) else None
            if preview is not None:
                preview.cancel()
        try:
            self._event_queue.put_nowait(event)
        except asyncio.QueueFull as exc:
            self._accepting_audio = False
            raise VoicePipelineException(
                code=VoiceErrorCode.BACKPRESSURE,
                stage=VoiceStage.STREAM,
                message="Voice Event Queue가 가득 찼습니다.",
                retryable=True,
            ) from exc

    def _mark_finalized(self, key: str) -> None:
        if key in self._finalized_utterance_keys:
            return
        self._finalized_utterance_keys.add(key)
        self._finalized_utterance_order.append(key)
        while len(self._finalized_utterance_order) > 256:
            expired = self._finalized_utterance_order.popleft()
            self._finalized_utterance_keys.discard(expired)

    def _drain_input_queue(self) -> None:
        while True:
            try:
                command = self._input_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if isinstance(command, (_FlushCommand, _CloseCommand)):
                if not command.future.done():
                    command.future.set_result(None)
            self._input_queue.task_done()

        # worker가 이미 꺼낸 제어 명령은 큐에 없지만 호출자는 같은 Future를 기다린다.
        # 종료 오류는 이벤트로 전달하고 제어 대기는 종료 관찰 후 정상적으로 해제한다.
        for future in (self._flush_future, self._close_future):
            if future is not None and not future.done():
                future.set_result(None)

    @staticmethod
    def _event_id() -> str:
        return str(uuid4())


class VoiceStreamApplicationService:
    def __init__(
        self,
        *,
        stt_provider: VoiceSttProvider,
        translation_service: VoiceTranslationService,
        speech_evidence_guard: VoiceSpeechEvidenceGuard | None = None,
        preview_provider: VoicePreviewProvider | None = None,
        preview_language_provider: VoiceSttProvider | None = None,
        preview_translation_service: VoiceTranslationService | None = None,
    ) -> None:
        self.stt_provider = stt_provider
        self.translation_service = translation_service
        self.speech_evidence_guard = speech_evidence_guard or PassThroughSpeechEvidenceGuard()
        self.preview_provider = preview_provider
        self.preview_language_provider = preview_language_provider
        self.preview_translation_service = preview_translation_service
        self._preview_slots = asyncio.Semaphore(settings.AI_VOICE_REALTIME_PREVIEW_MAX_CONCURRENCY)
        self._streams: dict[str, VoiceChannelStreamContext] = {}
        self._accepting_streams = True
        self._lock = asyncio.Lock()

    @property
    def ready(self) -> bool:
        return (
            self._accepting_streams
            and self.stt_provider.ready
            and self.translation_service.ready
            and self.speech_evidence_guard.ready
        )

    @property
    def accepting_streams(self) -> bool:
        return self._accepting_streams and len(self._streams) < settings.AI_VOICE_MAX_ACTIVE_STREAMS

    @property
    def active_stream_count(self) -> int:
        return len(self._streams)

    async def open_stream(
        self,
        stream_open: VoiceStreamOpen,
    ) -> VoiceChannelStreamContext:
        if not self.ready:
            raise VoicePipelineException(
                code=VoiceErrorCode.MODEL_NOT_READY,
                stage=VoiceStage.RUNTIME,
                message="Voice STT Runtime이 준비되지 않았습니다.",
                retryable=True,
            )
        key = self._stream_key(stream_open)
        async with self._lock:
            if not self.ready:
                raise VoicePipelineException(
                    code=VoiceErrorCode.MODEL_NOT_READY,
                    stage=VoiceStage.RUNTIME,
                    message="Voice STT Runtime이 준비되지 않았습니다.",
                    retryable=True,
                )
            existing = self._streams.get(key)
            if existing is not None and not existing.closed:
                raise VoicePipelineException(
                    code=VoiceErrorCode.INVALID_STREAM_OPEN,
                    stage=VoiceStage.STREAM,
                    message="동일 Session과 Channel의 Stream이 이미 열려 있습니다.",
                    retryable=False,
                )
            if existing is None and len(self._streams) >= settings.AI_VOICE_MAX_ACTIVE_STREAMS:
                raise VoicePipelineException(
                    code=VoiceErrorCode.BACKPRESSURE,
                    stage=VoiceStage.STREAM,
                    message="Voice Stream 동시 연결 상한에 도달했습니다.",
                    retryable=True,
                )
            context = VoiceChannelStreamContext(
                stream_open,
                stt_provider=self.stt_provider,
                translation_service=self.translation_service,
                speech_evidence_guard=self.speech_evidence_guard,
                preview_provider=self.preview_provider,
                preview_language_provider=self.preview_language_provider,
                preview_translation_service=self.preview_translation_service,
                preview_slots=self._preview_slots,
            )
            self._streams[key] = context
        await context.start()
        return context

    async def release_stream(self, context: VoiceChannelStreamContext) -> None:
        if not context.closed:
            await context.abort()
        key = self._stream_key(context.stream_open)
        async with self._lock:
            if self._streams.get(key) is context:
                self._streams.pop(key, None)

    async def shutdown(self) -> None:
        async with self._lock:
            self._accepting_streams = False
            streams = list(self._streams.values())
        if streams:
            await asyncio.gather(
                *(self._close_for_shutdown(stream) for stream in streams),
                return_exceptions=True,
            )
        async with self._lock:
            self._streams.clear()
        # 별도로 생성한 언어 확인 클라이언트만 닫고 주 STT의 소유권은 보존한다.
        if self.preview_language_provider is not self.stt_provider:
            shutdown = getattr(self.preview_language_provider, "shutdown", None)
            if shutdown is not None:
                await shutdown()

    async def _close_for_shutdown(self, stream: VoiceChannelStreamContext) -> None:
        try:
            await asyncio.wait_for(
                stream.close("SERVER_SHUTDOWN"),
                timeout=settings.AI_VOICE_SHUTDOWN_GRACE_SECONDS,
            )
        except Exception:
            await stream.abort()

    @staticmethod
    def _stream_key(stream_open: VoiceStreamOpen) -> str:
        return f"{stream_open.session_id}:{stream_open.channel.value}"
