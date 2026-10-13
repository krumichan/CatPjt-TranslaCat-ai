from __future__ import annotations

import asyncio

import pytest

from app.features.voice_translation.audio import VadNoSpeech, VadUtterance
from app.features.voice_translation.errors import VoicePipelineException
from app.features.voice_translation.speech_detector import PassThroughSpeechEvidenceGuard
from app.features.voice_translation.stream import (
    VoiceChannelStreamContext,
    _FinalizationJob,
    _PreparedUtterance,
    _UtteranceIdentity,
)
from app.features.voice_translation.stt import VoiceSttResult
from app.features.voice_translation.translation import VoiceTranslationService
from app.schemas.voice_translation import VoiceErrorCode, VoiceStage, VoiceStreamOpen
from tests.test_voice_translation import (
    FakeTranslationProvider,
    FakeVoiceSttProvider,
    stream_open_payload,
)


def make_context(
    *, manual: bool = False
) -> tuple[VoiceChannelStreamContext, FakeTranslationProvider]:
    provider = FakeTranslationProvider()
    context = VoiceChannelStreamContext(
        VoiceStreamOpen.model_validate(
            stream_open_payload(
                sourceLanguageMode="MANUAL" if manual else "AUTO",
                manualSourceLanguage="en" if manual else None,
                targetLanguage="ko",
            )
        ),
        stt_provider=FakeVoiceSttProvider(),
        translation_service=VoiceTranslationService(provider, max_retries=0),
        speech_evidence_guard=PassThroughSpeechEvidenceGuard(),
    )
    return context, provider


def identity(sequence: int) -> _UtteranceIdentity:
    return _UtteranceIdentity(key=f"SELF-{sequence}", sequence=sequence)


async def speech(
    context: VoiceChannelStreamContext,
    sequence: int,
    text: str,
    *,
    start: int,
    end: int,
    language: str | None = "en",
) -> None:
    # 준비: 음성 인식은 고정 결과로 주입하고 실제 final 처리/언어 필터/번역 문맥 경계를 실행한다.
    async def prepare() -> _PreparedUtterance:
        return _PreparedUtterance(
            stt_result=VoiceSttResult(
                text,
                language,
                0.99 if language else None,
                0.01,
                "fake",
                "fake-stt",
            )
        )

    now = asyncio.get_running_loop().time()
    job = _FinalizationJob(
        identity=identity(sequence),
        utterance=VadUtterance(b"", start, end, end - start, 0, False, 0.03, 0.1, 0),
        partial_inference_ms=None,
        queued_at=now,
        deadline=now + 5,
        preparation_task=asyncio.create_task(prepare()),
    )
    context._pending_final_jobs[job.identity.key] = job
    try:
        await context._finalize_utterance(job)
    finally:
        context._pending_final_jobs.pop(job.identity.key, None)


async def silence(context: VoiceChannelStreamContext, sequence: int, start: int, end: int) -> None:
    await context._emit_no_speech(
        VadNoSpeech(start, end),
        identity=identity(sequence),
        clear_current=False,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("silence_first", [False, True])
async def test_nearby_context_crosses_only_confirmed_silence_even_if_vad_arrives_first(
    silence_first: bool,
):
    # 준비: 운영에서 완료 #3과 #5 사이 #4가 NO_SPEECH로 판정된 900ms 간격을 재현한다.
    context, provider = make_context()
    if silence_first:
        await silence(context, 4, 1000, 1800)
    await speech(context, 3, "We should keep the", start=0, end=1000)
    if not silence_first:
        await silence(context, 4, 1000, 1800)

    # 실행
    await speech(context, 5, "original deadline.", start=1900, end=2900)

    # 검증: 원문은 합치거나 수정하지 않고 직전 원문만 번역 참고 문맥에 제공한다.
    assert provider.previous_source_texts == [None, "We should keep the"]
    completed = []
    while not context._event_queue.empty():
        event = context._event_queue.get_nowait()
        if event.type == "VOICE_PIPELINE_COMPLETED":
            completed.append(event)
    assert [(event.utterance_sequence, event.source_text) for event in completed] == [
        (3, "We should keep the"),
        (5, "original deadline."),
    ]
    assert not context._context_no_speech_sequences


@pytest.mark.asyncio
async def test_unobserved_skipped_sequence_is_a_context_barrier():
    # 준비: #2의 판정이 없는 상태는 무음이라고 추정하지 않는다.
    context, provider = make_context()
    await speech(context, 1, "Previous source.", start=0, end=1000)

    # 실행
    await speech(context, 3, "New source.", start=1900, end=2900)

    # 검증
    assert provider.previous_source_texts == [None, None]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code,stage",
    [
        (VoiceErrorCode.STT_FAILED, VoiceStage.STT),
        (VoiceErrorCode.UNSUPPORTED_LANGUAGE, VoiceStage.LANGUAGE),
        (VoiceErrorCode.LANGUAGE_MISMATCH, VoiceStage.LANGUAGE),
        (VoiceErrorCode.LANGUAGE_UNDETERMINED, VoiceStage.LANGUAGE),
        (VoiceErrorCode.TRANSLATION_FAILED, VoiceStage.TRANSLATION),
    ],
)
async def test_failed_utterance_resets_context_even_when_followed_by_silence(
    code: VoiceErrorCode,
    stage: VoiceStage,
):
    # 준비
    context, provider = make_context()
    await speech(context, 1, "Previous source.", start=0, end=1000)

    # 실행: 실패 뒤의 확인된 무음이 실패 장벽을 없애면 안 된다.
    await context.emit_failure(
        VoicePipelineException(code=code, stage=stage, message="synthetic", retryable=False),
        identity=identity(2),
    )
    assert context._last_completed_source_text is None
    await silence(context, 3, 1200, 1600)
    await speech(context, 4, "New source.", start=1900, end=2900)

    # 검증
    assert provider.previous_source_texts == [None, None]


@pytest.mark.asyncio
async def test_failure_that_arrives_before_prior_completion_still_blocks_context():
    # 준비: 오디오 수신 오류가 이전 문장 번역 완료보다 앞서 도착할 수 있다.
    context, provider = make_context()
    await context.emit_failure(
        VoicePipelineException(
            code=VoiceErrorCode.STT_FAILED,
            stage=VoiceStage.STT,
            message="synthetic",
            retryable=False,
        ),
        identity=identity(2),
    )
    await speech(context, 1, "Previous source.", start=0, end=1000)
    await silence(context, 3, 1200, 1600)

    # 실행
    await speech(context, 4, "New source.", start=1900, end=2900)

    # 검증
    assert provider.previous_source_texts == [None, None]


@pytest.mark.asyncio
@pytest.mark.parametrize("manual,rejected_language", [(False, "fr"), (True, "ja")])
async def test_actual_language_rejection_does_not_leak_context_to_next_utterance(
    manual: bool,
    rejected_language: str,
):
    # 준비
    context, provider = make_context(manual=manual)
    await speech(context, 1, "Previous source.", start=0, end=1000)

    # 실행: 언어 필터의 실제 실패 경로 뒤에 같은 언어가 다시 들어온다.
    await speech(context, 2, "Rejected source.", start=1200, end=1700, language=rejected_language)
    await speech(context, 3, "New source.", start=1900, end=2900)

    # 검증
    assert provider.previous_source_texts == [None, None]


@pytest.mark.asyncio
async def test_unknown_language_fallback_is_not_evidence_for_reusing_context():
    # 준비: 기존 자동 언어 잠금에 의한 번역 동작 자체는 유지한다.
    context, provider = make_context()
    await speech(context, 1, "Previous source.", start=0, end=1000)

    # 실행: 감지 언어 없는 구간은 앞뒤 문맥을 이어 붙일 근거가 아니다.
    await speech(context, 2, "Unknown source.", start=1200, end=1700, language=None)
    await speech(context, 3, "New source.", start=1900, end=2900)

    # 검증
    assert provider.calls == 3
    assert provider.previous_source_texts == [None, None, None]


@pytest.mark.asyncio
@pytest.mark.parametrize("next_start,next_language", [(6001, "en"), (1900, "ja")])
async def test_silence_does_not_extend_time_bound_or_override_language_change(
    next_start: int,
    next_language: str,
):
    # 준비
    context, provider = make_context()
    await speech(context, 1, "Previous source.", start=0, end=1000)
    await silence(context, 2, 1000, next_start)

    # 실행
    await speech(
        context, 3, "New source.", start=next_start, end=next_start + 1000, language=next_language
    )

    # 검증
    assert provider.previous_source_texts == [None, None]
