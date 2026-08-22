from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
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
)
from app.features.voice_translation.errors import VoicePipelineException
from app.features.voice_translation.language import (
    LanguageStateManager,
    normalize_confidence,
    normalize_voice_language,
)
from app.features.voice_translation.stable_prefix import StablePrefixAssembler
from app.features.voice_translation.speech_detector import (
    PassThroughSpeechEvidenceGuard,
    VoiceSpeechEvidenceGuard,
)
from app.features.voice_translation.stt import VoiceSttProvider
from app.features.voice_translation.translation import VoiceTranslationService
from app.schemas.voice_translation import (
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
    ) -> None:
        self.stream_open = stream_open
        self.stt_provider = stt_provider
        self.translation_service = translation_service
        self.speech_evidence_guard = speech_evidence_guard

        frame_duration_ms = stream_open.audio_format.frame_duration_ms
        input_capacity = max(
            1,
            settings.AI_VOICE_MAX_BUFFERED_AUDIO_MS // frame_duration_ms,
        )
        self._input_capacity = input_capacity
        self._input_queue: asyncio.Queue[_StreamCommand] = asyncio.Queue(
            maxsize=input_capacity
        )
        self._event_queue: asyncio.Queue[VoiceEventBase] = asyncio.Queue(maxsize=64)
        self._validator = AudioFrameValidator(
            stream_open.audio_format,
            minimum_frame_duration_ms=settings.AI_VOICE_MIN_FRAME_DURATION_MS,
            maximum_frame_duration_ms=settings.AI_VOICE_MAX_FRAME_DURATION_MS,
            speech_rms_threshold=settings.AI_VOICE_VAD_RMS_THRESHOLD,
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

        self._worker_task: asyncio.Task[None] | None = None
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

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def buffered_audio_ms(self) -> int:
        return (
            self._input_queue.qsize() * self.stream_open.audio_format.frame_duration_ms
        )

    async def start(self) -> None:
        if self._worker_task is not None:
            return
        self._worker_task = asyncio.create_task(
            self._run(),
            name=(
                "voice-stream-"
                f"{self.stream_open.session_id}-{self.stream_open.channel.value}"
            ),
        )
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
        except asyncio.QueueFull:
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
            return False

    async def flush(self, reason: str) -> None:
        if self._closed or self._flushed:
            return
        self._accepting_audio = False
        self._flushed = True
        loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = loop.create_future()
        await self._input_queue.put(_FlushCommand(reason=reason, future=future))
        await future

    async def close(self, reason: str) -> None:
        if self._closed:
            return
        self._accepting_audio = False
        loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = loop.create_future()
        await self._input_queue.put(_CloseCommand(reason=reason, future=future))
        await future
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
        self._vad.clear()
        self._drain_input_queue()

    async def emit_failure(
        self,
        error: VoicePipelineException,
        *,
        source_text: str | None = None,
        detected_language: str | None = None,
        language_confidence: float | None = None,
    ) -> None:
        identity = self._current_identity
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
            try:
                await self.emit_failure(
                    VoicePipelineException(
                        code=VoiceErrorCode.INTERNAL_ERROR,
                        stage=VoiceStage.STREAM,
                        message="Voice Stream 처리 중 내부 오류가 발생했습니다.",
                        retryable=True,
                    )
                )
            except Exception:
                pass
            self._accepting_audio = False
            self._closed = True
            await self._cancel_background_tasks()
            self._vad.clear()
            self._drain_input_queue()

    async def _process_command(self, command: _StreamCommand) -> bool:
        if isinstance(command, _AudioCommand):
            await self._handle_vad_result(self._vad.process(command.frame))
            return False
        if isinstance(command, _FlushCommand):
            await self._handle_vad_result(self._vad.flush())
            if not command.future.done():
                command.future.set_result(None)
            return False

        if not self._flushed:
            await self._handle_vad_result(self._vad.flush())
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
            await self._finalize_utterance(result.utterance)
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
        if self._partial_task is not None and not self._partial_task.done():
            return

        task = asyncio.create_task(
            self._run_partial(partial, identity, assembler),
            name=f"voice-partial-{identity.key}",
        )
        self._partial_task = task
        self._background_tasks.add(task)
        task.add_done_callback(self._partial_done)

    async def _run_partial(
        self,
        partial: VadPartial,
        identity: _UtteranceIdentity,
        assembler: StablePrefixAssembler,
    ) -> None:
        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                self.stt_provider.transcribe_pcm(
                    partial.pcm_bytes,
                    language=self.stream_open.manual_source_language,
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
            identity.key in self._finalized_utterance_keys
            or self._current_identity != identity
        ):
            return
        self._last_partial_inference_ms = partial_inference_ms
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

        detected_language = normalize_voice_language(result.language)

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
                partial.ended_at_offset_ms
                - partial.started_at_offset_ms
                + partial_inference_ms
                if update.revision == 1
                else None
            ),
        )

    async def _finalize_utterance(self, utterance: VadUtterance) -> None:
        identity = self._current_identity
        if identity is None:
            return
        self._mark_finalized(identity.key)

        speech_evidence = await self._confirm_speech_evidence(utterance.pcm_bytes)
        if speech_evidence is None:
            return
        has_speech, speech_evidence_ms = speech_evidence
        if not has_speech:
            await self._emit_no_speech(
                VadNoSpeech(
                    started_at_offset_ms=utterance.started_at_offset_ms,
                    ended_at_offset_ms=utterance.ended_at_offset_ms,
                )
            )
            return

        stt_started = time.perf_counter()
        try:
            stt_result = await asyncio.wait_for(
                self.stt_provider.transcribe_pcm(
                    utterance.pcm_bytes,
                    language=self.stream_open.manual_source_language,
                    is_final=True,
                    initial_prompt=None,
                ),
                timeout=settings.AI_VOICE_STT_FINAL_TIMEOUT_SECONDS,
            )
        except (TimeoutError, asyncio.TimeoutError):
            await self.emit_failure(
                VoicePipelineException(
                    code=VoiceErrorCode.STT_TIMEOUT,
                    stage=VoiceStage.STT,
                    message="Final STT 응답 시간이 초과되었습니다.",
                    retryable=True,
                )
            )
            self._clear_current_utterance()
            return
        except VoicePipelineException as exc:
            await self.emit_failure(exc)
            self._clear_current_utterance()
            return
        except Exception:
            await self.emit_failure(
                VoicePipelineException(
                    code=VoiceErrorCode.STT_FAILED,
                    stage=VoiceStage.STT,
                    message="Final STT 처리에 실패했습니다.",
                    retryable=True,
                )
            )
            self._clear_current_utterance()
            return

        stt_finalize_ms = max(0, int((time.perf_counter() - stt_started) * 1000))
        source_text = stt_result.text.strip()
        if not source_text:
            await self._emit_no_speech(
                VadNoSpeech(
                    started_at_offset_ms=utterance.started_at_offset_ms,
                    ended_at_offset_ms=utterance.ended_at_offset_ms,
                )
            )
            return
        if len(source_text) > 10_000:
            await self.emit_failure(
                VoicePipelineException(
                    code=VoiceErrorCode.STT_FAILED,
                    stage=VoiceStage.STT,
                    message="Final STT 결과가 허용 길이를 초과했습니다.",
                    retryable=False,
                )
            )
            self._clear_current_utterance()
            return

        observation = self._language.observe(
            stt_result.language,
            stt_result.language_confidence,
        )
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
            source_language = self._language.resolve_translation_source(
                detected_language=observation.detected_language,
                confidence=observation.confidence,
            )
            translation = await self.translation_service.translate(
                source_text=source_text,
                source_language=source_language,
                target_language=self.stream_open.target_language,
            )
        except VoicePipelineException as exc:
            await self.emit_failure(
                exc,
                source_text=source_text,
                detected_language=observation.detected_language,
                language_confidence=observation.confidence,
            )
            self._clear_current_utterance()
            return
        except Exception:
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
            )
            self._clear_current_utterance()
            return

        total_after_speech_ms = (
            utterance.endpointing_ms
            + speech_evidence_ms
            + stt_finalize_ms
            + translation.latency_ms
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
                    partial_inference_ms=self._last_partial_inference_ms,
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
        self._clear_current_utterance()

    async def _confirm_speech_evidence(
        self,
        pcm_bytes: bytes,
    ) -> tuple[bool, int] | None:
        started = time.perf_counter()
        try:
            has_speech = await asyncio.wait_for(
                self.speech_evidence_guard.has_speech(pcm_bytes),
                timeout=settings.AI_VOICE_VAD_SILERO_TIMEOUT_SECONDS,
            )
        except (TimeoutError, asyncio.TimeoutError):
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

        await self.emit_failure(error)
        self._clear_current_utterance()
        return None

    async def _emit_no_speech(self, outcome: VadNoSpeech) -> None:
        identity = self._current_identity
        if identity is not None:
            self._mark_finalized(identity.key)
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
        self._clear_current_utterance()

    async def _finish_close(self, reason: str) -> None:
        self._accepting_audio = False
        await self._cancel_background_tasks()
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
        tasks = [task for task in self._background_tasks if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._background_tasks.clear()
        self._partial_task = None

    def _partial_done(self, task: asyncio.Task[None]) -> None:
        self._background_tasks.discard(task)
        if self._partial_task is task:
            self._partial_task = None

    def _clear_current_utterance(self) -> None:
        self._current_identity = None
        self._current_assembler = None
        self._last_partial_inference_ms = None

    async def _emit(self, event: VoiceEventBase) -> None:
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
                return
            if isinstance(command, (_FlushCommand, _CloseCommand)):
                if not command.future.done():
                    command.future.cancel()
            self._input_queue.task_done()

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
    ) -> None:
        self.stt_provider = stt_provider
        self.translation_service = translation_service
        self.speech_evidence_guard = (
            speech_evidence_guard or PassThroughSpeechEvidenceGuard()
        )
        self._streams: dict[str, VoiceChannelStreamContext] = {}
        self._accepting_streams = True
        self._lock = asyncio.Lock()

    @property
    def ready(self) -> bool:
        return (
            settings.AI_VOICE_ENABLED
            and self._accepting_streams
            and self.stt_provider.ready
            and self.translation_service.ready
            and self.speech_evidence_guard.ready
        )

    @property
    def accepting_streams(self) -> bool:
        return (
            settings.AI_VOICE_ENABLED
            and self._accepting_streams
            and len(self._streams) < settings.AI_VOICE_MAX_ACTIVE_STREAMS
        )

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
            if (
                existing is None
                and len(self._streams) >= settings.AI_VOICE_MAX_ACTIVE_STREAMS
            ):
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
