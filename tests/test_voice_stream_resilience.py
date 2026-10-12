from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

import httpx
from openai import APIConnectionError, APITimeoutError

from app.ai.providers.openai.response import OpenAIProviderResponseError
from app.core.config import settings
from app.features.voice_translation.errors import VoicePipelineException
from app.features.voice_translation.speech_detector import SileroSpeechEvidenceGuard
from app.features.voice_translation.stream import VoiceStreamApplicationService
from app.features.voice_translation.stt import VoiceSttResult
from app.features.voice_translation.translation import VoiceTranslationService
from app.schemas.voice_translation import VoiceErrorCode, VoiceStage, VoiceStreamOpen
from tests.test_voice_translation import (
    FakeTranslationProvider,
    FakeVoiceSttProvider,
    RejectSpeechEvidenceGuard,
    pcm_frame,
    stream_open_payload,
)


class ControlledTranslation(FakeTranslationProvider):
    def __init__(self):
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def translate_voice_utterance(self, **kwargs):
        self.started.set()
        await self.release.wait()
        return await super().translate_voice_utterance(**kwargs)


class DetectedStt(FakeVoiceSttProvider):
    def __init__(self, language="en", confidence=0.99):
        super().__init__()
        self.language = language
        self.confidence = confidence
        self.languages_requested = []

    async def transcribe_pcm(self, pcm_bytes, **kwargs):
        self.calls.append(kwargs["is_final"])
        self.languages_requested.append(kwargs["language"])
        return VoiceSttResult(
            text="Synthetic public test sentence.",
            language=self.language,
            language_confidence=self.confidence,
            no_speech_probability=0.01,
            provider="test",
            model="test",
        )


class VoiceStreamResilienceTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.contexts = []

    async def asyncTearDown(self):
        for context in self.contexts:
            await context.abort()

    async def create_context(self, *, stt=None, provider=None, guard=None, **fields):
        context = await VoiceStreamApplicationService(
            stt_provider=stt or FakeVoiceSttProvider(),
            translation_service=VoiceTranslationService(provider or FakeTranslationProvider()),
            speech_evidence_guard=guard,
        ).open_stream(VoiceStreamOpen.model_validate(stream_open_payload(**fields)))
        self.contexts.append(context)
        return context

    async def feed(self, context, frames):
        for frame in frames:
            await context.feed_audio(frame)
            await asyncio.sleep(0)
        await context._input_queue.join()

    async def utterance(self, context):
        await self.feed(context, [pcm_frame(4000)] * 3 + [pcm_frame(0)] * 3)

    async def closed_events(self, context):
        await asyncio.wait_for(context.close("TEST"), timeout=2)
        events = []
        while True:
            event = await asyncio.wait_for(context.next_event(), timeout=1)
            events.append(event)
            if event.type == "STREAM_CLOSED":
                return events

    async def test_pipeline_failure_logs_metadata_without_transcript_or_error_payload(self):
        # 준비: 외부 오류 메시지와 원문이 진단 로그로 새지 않아야 한다.
        context = await self.create_context()
        await self.feed(context, [pcm_frame(4000)])
        source_text = "Synthetic transcript must stay out of operational logs."
        provider_message = "Synthetic provider payload must stay out of operational logs."
        error = VoicePipelineException(
            code=VoiceErrorCode.TRANSLATION_TIMEOUT,
            stage=VoiceStage.TRANSLATION,
            message=provider_message,
            retryable=True,
        )

        # 실행
        with self.assertLogs("app.features.voice_translation.stream", level="WARNING") as logs:
            await context.emit_failure(error, source_text=source_text)

        # 검증: 실패 코드와 발화만 기록하고 민감할 수 있는 두 문자열은 제외한다.
        logged = "\n".join(logs.output)
        for expected in (
            f"session={context.stream_open.session_id}",
            f"channel={context.stream_open.channel.value}",
            "code=VOICE_AI_TRANSLATION_TIMEOUT",
            "stage=TRANSLATION",
            "retryable=True",
            "utterance_sequence=1",
        ):
            self.assertIn(expected, logged)
        self.assertNotIn(source_text, logged)
        self.assertNotIn(provider_message, logged)

    async def test_noise_guard_blocks_partial_as_well_as_final_stt(self):
        # 준비: partial 주기를 넘기는 잡음을 음성 확인 단계에서 거절한다.
        stt = FakeVoiceSttProvider()
        provider = FakeTranslationProvider()
        context = await self.create_context(
            stt=stt, provider=provider, guard=RejectSpeechEvidenceGuard()
        )

        # 실행
        await self.feed(context, [pcm_frame(4000)] * 8)
        await asyncio.sleep(0.01)
        events = await self.closed_events(context)

        # 검증
        self.assertEqual(stt.calls, [])
        self.assertEqual(provider.calls, 0)
        self.assertNotIn("TRANSCRIPT_PARTIAL", [event.type for event in events])
        self.assertIn("NO_SPEECH", [event.type for event in events])

    async def test_audio_consumption_continues_while_translation_waits(self):
        # 준비: 첫 발화 번역을 보류한다.
        provider = ControlledTranslation()
        context = await self.create_context(provider=provider)
        await self.utterance(context)
        await asyncio.wait_for(provider.started.wait(), timeout=1)

        # 실행: 기존 입력 버퍼보다 긴 4초 음성을 연속으로 보낸다.
        await self.feed(context, [pcm_frame(4000)] * 40)
        self.assertEqual(context._vad.current_offset_ms, 4600)
        self.assertEqual(context.buffered_audio_ms, 0)
        provider.release.set()
        events = await self.closed_events(context)

        # 검증: 두 발화 모두 순서대로 완료되고 입력 손실 경고가 없다.
        completed = [event for event in events if event.type == "VOICE_PIPELINE_COMPLETED"]
        self.assertEqual([event.utterance_sequence for event in completed], [1, 2])
        self.assertNotIn("BACKPRESSURE", [event.type for event in events])

    async def test_pending_final_capacity_causes_terminal_failure_for_all_accepted_jobs(self):
        # 준비: 처리 중인 발화를 포함해 확정 용량을 두 개로 제한한다.
        provider = ControlledTranslation()
        with patch.object(settings, "AI_VOICE_FINAL_QUEUE_CAPACITY", 2):
            context = await self.create_context(provider=provider)
            await self.utterance(context)
            await asyncio.wait_for(provider.started.wait(), timeout=1)

            # 실행: 세 번째 발화가 용량을 넘는다.
            await self.utterance(context)
            await self.utterance(context)
            await asyncio.wait_for(context._worker_task, timeout=1)

        # 검증: 연결을 종료하고 모든 미완료 발화를 명시적으로 실패시킨다.
        events = list(context._event_queue._queue)
        failures = [event for event in events if event.type == "VOICE_PIPELINE_FAILED"]
        self.assertEqual(sorted(event.utterance_sequence for event in failures), [1, 2, 3])
        self.assertTrue(all(event.error.code == VoiceErrorCode.BACKPRESSURE for event in failures))
        self.assertEqual(events[-1].type, "STREAM_CLOSED")
        self.assertTrue(context.closed)

    async def test_close_timeout_fails_pending_jobs_and_rejects_late_completion(self):
        # 준비: 짧은 종료 기한을 두고 번역 응답을 보류한다.
        provider = ControlledTranslation()
        context = await self.create_context(provider=provider)
        await self.utterance(context)
        await asyncio.wait_for(provider.started.wait(), timeout=1)
        await self.utterance(context)

        # 실행
        with patch.object(settings, "AI_VOICE_DRAIN_TIMEOUT_SECONDS", 0.03):
            events = await self.closed_events(context)
        provider.release.set()
        await asyncio.sleep(0.02)

        # 검증: 최종 원문은 보존하고 두 발화를 실패시킨 뒤 닫힌다.
        self.assertIn("TRANSCRIPT_FINAL", [event.type for event in events])
        failures = [event for event in events if event.type == "VOICE_PIPELINE_FAILED"]
        self.assertEqual(sorted(event.utterance_sequence for event in failures), [1, 2])
        first_failure = next(event for event in failures if event.utterance_sequence == 1)
        self.assertEqual(first_failure.source_text, "오늘 회의를 시작합니다")
        self.assertEqual(first_failure.error.stage, VoiceStage.TRANSLATION)
        self.assertEqual(first_failure.error.code, VoiceErrorCode.TRANSLATION_TIMEOUT)
        self.assertNotIn("VOICE_PIPELINE_COMPLETED", [event.type for event in events])
        self.assertTrue(context._event_queue.empty())

    async def test_expired_queued_job_does_not_infer_or_mutate_language(self):
        # 준비: 첫 STT를 보류해 뒤의 발화까지 전체 기한을 넘긴다.
        class SlowStt(DetectedStt):
            async def transcribe_pcm(self, pcm_bytes, **kwargs):
                self.calls.append(kwargs["is_final"])
                await asyncio.sleep(1)
                raise AssertionError("Cancelled STT must not return")

        stt = SlowStt(language="ja")
        with patch.object(settings, "AI_VOICE_PIPELINE_TIMEOUT_SECONDS", 0.05):
            context = await self.create_context(stt=stt, lastLockedLanguage="ko")
            await self.utterance(context)
            with patch.object(settings, "AI_VOICE_PIPELINE_TIMEOUT_SECONDS", 0.01):
                await self.utterance(context)

        # 실행
        events = await self.closed_events(context)

        # 검증: 이미 만료한 대기 발화는 새 추론을 시작하지 않는다.
        self.assertEqual(stt.calls, [True])
        self.assertEqual(context._language.locked_language, "ko")
        self.assertEqual(sum(event.type == "VOICE_PIPELINE_FAILED" for event in events), 2)

    async def test_concurrent_flush_and_close_do_not_duplicate_or_hang(self):
        # 준비
        context = await self.create_context()
        await self.feed(context, [pcm_frame(4000)] * 4)

        # 실행
        await asyncio.wait_for(
            asyncio.gather(
                context.flush("TEST"),
                context.flush("TEST"),
                context.close("TEST"),
                context.close("TEST"),
            ),
            timeout=2,
        )

        # 검증
        events = list(context._event_queue._queue)
        self.assertEqual(sum(event.type == "VOICE_PIPELINE_COMPLETED" for event in events), 1)
        self.assertEqual(sum(event.type == "STREAM_CLOSED" for event in events), 1)

    async def test_close_deadline_does_not_wait_for_provider_cancellation_acknowledgement(self):
        # 준비: 취소 후에도 응답을 돌려주는 공급자를 흉내 낸다.
        class SlowCancellationProvider(ControlledTranslation):
            def __init__(self):
                super().__init__()
                self.returned = asyncio.Event()

            async def translate_voice_utterance(self, **kwargs):
                self.started.set()
                try:
                    await self.release.wait()
                except asyncio.CancelledError:
                    await self.release.wait()
                self.returned.set()
                return await FakeTranslationProvider.translate_voice_utterance(self, **kwargs)

        provider = SlowCancellationProvider()
        context = await self.create_context(provider=provider)
        await self.utterance(context)
        await asyncio.wait_for(provider.started.wait(), timeout=1)

        # 실행: 공급자가 아직 응답하지 않아도 종료 완료를 먼저 반환한다.
        with patch.object(settings, "AI_VOICE_DRAIN_TIMEOUT_SECONDS", 0.02):
            await asyncio.wait_for(context.close("TEST"), timeout=0.2)
        before = list(context._event_queue._queue)
        provider.release.set()
        await asyncio.wait_for(provider.returned.wait(), timeout=1)
        await asyncio.sleep(0.02)

        # 검증: 늦은 결과는 완료 이벤트와 언어 상태를 추가하지 않는다.
        self.assertEqual(list(context._event_queue._queue), before)
        self.assertEqual(sum(event.type == "VOICE_PIPELINE_FAILED" for event in before), 1)
        self.assertEqual(before[-1].type, "STREAM_CLOSED")

    async def test_final_audio_budget_includes_the_active_job(self):
        # 준비: 한 발화 PCM은 400ms이며 활성 작업도 예산에 포함한다.
        provider = ControlledTranslation()
        with patch.object(settings, "AI_VOICE_MAX_PENDING_FINAL_AUDIO_MS", 500):
            context = await self.create_context(provider=provider)
            await self.utterance(context)
            await asyncio.wait_for(provider.started.wait(), timeout=1)

            # 실행
            await self.utterance(context)
            await asyncio.wait_for(context._worker_task, timeout=1)

        # 검증
        events = list(context._event_queue._queue)
        self.assertTrue(context.closed)
        self.assertEqual(sum(event.type == "VOICE_PIPELINE_FAILED" for event in events), 2)
        self.assertEqual(context._pending_final_audio_ms, 0)

    async def test_pipeline_deadline_after_stt_preserves_source_and_translation_failure(self):
        # 준비: 원문 확정 후 번역 중 전체 기한을 넘긴다.
        provider = ControlledTranslation()
        with patch.object(settings, "AI_VOICE_PIPELINE_TIMEOUT_SECONDS", 0.03):
            context = await self.create_context(provider=provider)
            await self.utterance(context)

        # 실행
        await asyncio.wait_for(provider.started.wait(), timeout=1)
        events = await self.closed_events(context)

        # 검증: BE가 확정 원문을 남기고 번역 실패로 저장할 수 있는 계약을 유지한다.
        failure = next(event for event in events if event.type == "VOICE_PIPELINE_FAILED")
        self.assertEqual(failure.source_text, "오늘 회의를 시작합니다")
        self.assertEqual(failure.detected_language, "ko")
        self.assertEqual(failure.error.stage, VoiceStage.TRANSLATION)
        self.assertEqual(failure.error.code, VoiceErrorCode.TRANSLATION_TIMEOUT)

    async def test_late_provider_errors_do_not_emit_after_stream_closed(self):
        errors = [
            VoicePipelineException(
                code=VoiceErrorCode.TRANSLATION_FAILED,
                stage=VoiceStage.TRANSLATION,
                message="Synthetic failure",
                retryable=False,
            ),
            RuntimeError("Synthetic failure"),
            TimeoutError(),
        ]
        for failure in errors:
            # 준비: 취소 후 일반/도메인/timeout 오류로 돌아오는 공급자.
            class LateFailureProvider(ControlledTranslation):
                def __init__(self, error):
                    super().__init__()
                    self.failure = error
                    self.returned = asyncio.Event()

                async def translate_voice_utterance(self, **kwargs):
                    self.started.set()
                    try:
                        await self.release.wait()
                    except asyncio.CancelledError:
                        await self.release.wait()
                    self.returned.set()
                    raise self.failure

            provider = LateFailureProvider(failure)
            context = await self.create_context(provider=provider)
            await self.utterance(context)
            await asyncio.wait_for(provider.started.wait(), timeout=1)

            # 실행
            with patch.object(settings, "AI_VOICE_DRAIN_TIMEOUT_SECONDS", 0.02):
                await asyncio.wait_for(context.close("TEST"), timeout=0.2)
            events = list(context._event_queue._queue)
            provider.release.set()
            await asyncio.wait_for(provider.returned.wait(), timeout=1)
            await asyncio.sleep(0.02)

            # 검증: 종료 이후 오류도 성공과 동일하게 소유권 검사로 버린다.
            self.assertEqual(list(context._event_queue._queue), events)
            self.assertEqual(events[-1].type, "STREAM_CLOSED")
            self.assertEqual(sum(event.type == "VOICE_PIPELINE_FAILED" for event in events), 1)

    async def test_full_event_queue_still_closes_and_resolves_control_waiters(self):
        # 준비: 전송 큐가 막힌 채 번역 하나와 다음 음성이 대기하는 상황.
        provider = ControlledTranslation()
        context = await self.create_context(provider=provider)
        await self.utterance(context)
        await asyncio.wait_for(provider.started.wait(), timeout=1)
        retained = context._event_queue.get_nowait()
        context._event_queue = asyncio.Queue(maxsize=1)
        context._event_queue.put_nowait(retained)

        # 실행: 다음 발화 시작 이벤트를 넣지 못해도 종료 처리가 중단되면 안 된다.
        await context.feed_audio(pcm_frame(4000))
        await asyncio.wait_for(
            asyncio.gather(context._worker_task, context.close("TEST"), return_exceptions=True),
            timeout=1,
        )

        # 검증: 포화된 큐의 마지막 이벤트를 소비하면 별도 종료 상태를 확인할 수 있다.
        self.assertTrue(context.closed)
        self.assertEqual(context._pending_final_jobs, {})
        self.assertEqual(context._pending_final_audio_ms, 0)
        await context.next_event()
        self.assertEqual((await context.next_event()).type, "STREAM_CLOSED")

    async def test_final_worker_event_overflow_closes_idle_audio_worker(self):
        # 준비: 오디오 입력이 멈춘 뒤 확정 STT가 도착하도록 제어한다.
        class ControlledStt(FakeVoiceSttProvider):
            def __init__(self):
                super().__init__()
                self.started = asyncio.Event()
                self.release = asyncio.Event()

            async def transcribe_pcm(self, pcm_bytes, **kwargs):
                self.started.set()
                await self.release.wait()
                return await super().transcribe_pcm(pcm_bytes, **kwargs)

        stt = ControlledStt()
        context = await self.create_context(stt=stt)
        await self.utterance(context)
        await asyncio.wait_for(stt.started.wait(), timeout=1)
        retained = context._event_queue.get_nowait()
        context._event_queue = asyncio.Queue(maxsize=1)
        context._event_queue.put_nowait(retained)

        # 실행
        stt.release.set()

        async def wait_closed():
            while not context.closed:
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_closed(), timeout=1)

        # 검증: 추가 입력/클라이언트 close가 없어도 두 worker가 종료된다.
        self.assertTrue(context._worker_task.done())
        self.assertEqual(context._pending_final_jobs, {})
        await asyncio.wait_for(context.close("TEST"), timeout=0.1)

    async def test_partial_event_overflow_closes_idle_audio_worker(self):
        # 준비: 중간 STT 결과가 돌아오기 전에 이벤트 큐를 채운다.
        class ControlledStt(FakeVoiceSttProvider):
            def __init__(self):
                super().__init__()
                self.started = asyncio.Event()
                self.release = asyncio.Event()

            async def transcribe_pcm(self, pcm_bytes, **kwargs):
                self.started.set()
                await self.release.wait()
                return await super().transcribe_pcm(pcm_bytes, **kwargs)

        stt = ControlledStt()
        context = await self.create_context(stt=stt)
        await self.feed(context, [pcm_frame(4000)] * 8)
        await asyncio.wait_for(stt.started.wait(), timeout=1)
        retained = context._event_queue.get_nowait()
        context._event_queue = asyncio.Queue(maxsize=1)
        context._event_queue.put_nowait(retained)

        # 실행
        stt.release.set()

        async def wait_closed():
            while not context.closed:
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_closed(), timeout=1)

        # 검증: 임시 자막 전송 실패도 같은 종료 경로로 정리한다.
        self.assertTrue(context._worker_task.done())
        self.assertEqual(context._background_tasks, set())
        await asyncio.wait_for(context.close("TEST"), timeout=0.1)

    async def test_guard_and_stt_receive_identical_normalized_audio(self):
        # 준비: DC가 아닌 작은 파형으로 실제 보정이 수행되는 입력을 사용한다.
        class RecordingGuard:
            ready = True
            model_version = "recording"

            def __init__(self):
                self.inputs = []

            async def has_speech(self, pcm_bytes):
                self.inputs.append(pcm_bytes)
                return True

        class RecordingStt(FakeVoiceSttProvider):
            def __init__(self):
                super().__init__()
                self.inputs = []

            async def transcribe_pcm(self, pcm_bytes, **kwargs):
                self.inputs.append(pcm_bytes)
                return await super().transcribe_pcm(pcm_bytes, **kwargs)

        guard = RecordingGuard()
        stt = RecordingStt()
        context = await self.create_context(stt=stt, guard=guard)
        alternating = (pcm_frame(1000)[:2] + pcm_frame(-1000)[:2]) * 800

        # 실행
        await self.feed(context, [alternating] * 8)
        await asyncio.sleep(0.02)
        await self.closed_events(context)

        # 검증: partial과 final 모두 별도의 중복 보정 없이 같은 입력을 사용한다.
        self.assertEqual(len(stt.inputs), 2)
        self.assertEqual(guard.inputs, stt.inputs)
        self.assertNotEqual(stt.inputs[0][:3200], alternating)

    async def test_known_unsupported_language_never_uses_locked_language(self):
        for target in ("ja", "ko"):
            # 준비
            provider = FakeTranslationProvider()
            context = await self.create_context(
                stt=DetectedStt("fr"),
                provider=provider,
                lastLockedLanguage="ko",
                targetLanguage=target,
            )

            # 실행
            await self.utterance(context)
            events = await self.closed_events(context)

            # 검증: 원문은 남기지만 오번역/같은 언어 생략 성공은 발생하지 않는다.
            failure = next(event for event in events if event.type == "VOICE_PIPELINE_FAILED")
            self.assertEqual(failure.error.code, VoiceErrorCode.UNSUPPORTED_LANGUAGE)
            self.assertEqual(failure.source_text, "Synthetic public test sentence.")
            self.assertEqual(context._language.locked_language, "ko")
            self.assertEqual(provider.calls, 0)

    async def test_manual_language_uses_detection_and_blocks_mismatch(self):
        # 준비
        stt = DetectedStt("ja")
        provider = FakeTranslationProvider()
        context = await self.create_context(
            stt=stt, provider=provider, sourceLanguageMode="MANUAL", manualSourceLanguage="en"
        )

        # 실행
        await self.utterance(context)
        events = await self.closed_events(context)

        # 검증
        self.assertEqual(stt.languages_requested, [None])
        self.assertEqual(provider.calls, 0)
        failure = next(event for event in events if event.type == "VOICE_PIPELINE_FAILED")
        self.assertEqual(failure.error.code, VoiceErrorCode.LANGUAGE_MISMATCH)
        self.assertEqual(failure.detected_language, "ja")

    async def test_manual_language_requires_reliable_matching_observation(self):
        for language, confidence, expected in (
            ("en", 0.99, None),
            ("en", 0.2, VoiceErrorCode.LANGUAGE_UNDETERMINED),
            (None, None, VoiceErrorCode.LANGUAGE_UNDETERMINED),
        ):
            # 준비
            context = await self.create_context(
                stt=DetectedStt(language, confidence),
                sourceLanguageMode="MANUAL",
                manualSourceLanguage="en",
            )

            # 실행
            await self.utterance(context)
            events = await self.closed_events(context)

            # 검증
            if expected is None:
                self.assertIn("VOICE_PIPELINE_COMPLETED", [event.type for event in events])
            else:
                failure = next(event for event in events if event.type == "VOICE_PIPELINE_FAILED")
                self.assertEqual(failure.error.code, expected)

    async def test_quiet_threshold_requires_enabled_silero_guard(self):
        # 준비: 모델 호출 없이 enabled/disabled 연결 기준만 확인한다.
        class ReadySilero(SileroSpeechEvidenceGuard):
            def __init__(self):
                super().__init__(enabled=True)
                self._ready = True

        # 실행
        passthrough = await self.create_context()
        disabled = await self.create_context(guard=SileroSpeechEvidenceGuard(enabled=False))
        enabled = await self.create_context(guard=ReadySilero())

        # 검증: disabled 모델과 pass-through에는 낮은 후보 기준을 적용하지 않는다.
        for context in (passthrough, disabled):
            self.assertEqual(
                context._validator.speech_candidate_rms_threshold,
                settings.AI_VOICE_VAD_RMS_THRESHOLD,
            )
        self.assertEqual(
            enabled._validator.speech_candidate_rms_threshold,
            settings.AI_VOICE_VAD_QUIET_RMS_THRESHOLD,
        )


class VoiceProviderContractTest(unittest.IsolatedAsyncioTestCase):
    async def test_sdk_connection_and_timeout_are_retryable_but_permanent_errors_are_not(self):
        request = httpx.Request("POST", "https://example.invalid/test")
        cases = [
            (APIConnectionError(request=request), 2, VoiceErrorCode.TRANSLATION_FAILED, True),
            (APITimeoutError(request=request), 2, VoiceErrorCode.TRANSLATION_TIMEOUT, True),
            (
                OpenAIProviderResponseError("test", reason_code="REFUSAL", retryable=False),
                1,
                VoiceErrorCode.TRANSLATION_FAILED,
                False,
            ),
            (
                OpenAIProviderResponseError(
                    "test", reason_code="OUTPUT_TOKEN_LIMIT", retryable=False
                ),
                1,
                VoiceErrorCode.TRANSLATION_FAILED,
                False,
            ),
        ]
        for error, calls, code, retryable in cases:
            # 준비
            class ErrorProvider(FakeTranslationProvider):
                def __init__(self, failure):
                    super().__init__()
                    self.failure = failure

                async def translate_voice_utterance(self, **kwargs):
                    self.calls += 1
                    raise self.failure

            provider = ErrorProvider(error)
            service = VoiceTranslationService(provider)

            # 실행
            with self.assertRaises(VoicePipelineException) as caught:
                await service.translate(
                    source_text="Hello.", source_language="en", target_language="ko"
                )

            # 검증
            self.assertEqual(provider.calls, calls)
            self.assertEqual(caught.exception.code, code)
            self.assertEqual(caught.exception.retryable, retryable)

    async def test_total_translation_deadline_includes_semaphore_wait(self):
        # 준비: 사용 가능한 공급자 슬롯을 선점한다.
        provider = FakeTranslationProvider()
        service = VoiceTranslationService(provider, max_concurrency=1, total_timeout_seconds=0.02)
        await service._translation_slots.acquire()

        # 실행
        with self.assertRaises(VoicePipelineException) as caught:
            await service.translate(
                source_text="Hello.", source_language="en", target_language="ko"
            )
        service._translation_slots.release()

        # 검증
        self.assertEqual(caught.exception.code, VoiceErrorCode.TRANSLATION_TIMEOUT)
        self.assertEqual(provider.calls, 0)

    async def test_same_language_metadata_uses_actual_provider_model(self):
        # 준비
        class NamedProvider(FakeTranslationProvider):
            def model_name_for(self, task):
                return "actual-voice-model"

        service = VoiceTranslationService(NamedProvider())

        # 실행
        result = await service.translate(
            source_text="Hello.", source_language="en", target_language="en"
        )

        # 검증
        self.assertEqual(result.model.translation_version, "actual-voice-model")
