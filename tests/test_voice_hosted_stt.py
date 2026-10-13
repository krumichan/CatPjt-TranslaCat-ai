from __future__ import annotations

import asyncio
import io
import json
import wave
from collections.abc import AsyncIterator, Callable, Coroutine
from typing import Any

import httpx2 as httpx
import pytest
import pytest_asyncio
from openai import AsyncOpenAI

from app.core.config import Settings, settings
from app.features.voice_translation.errors import VoicePipelineException
from app.features.voice_translation.hosted_stt import OpenAIWhisperVoiceSttProvider
from app.features.voice_translation.language import LanguageStateManager
from app.schemas.voice_translation import VoiceErrorCode

PCM = b"\x01\x00" * 1600


@pytest_asyncio.fixture
async def clients() -> AsyncIterator[list[AsyncOpenAI]]:
    # 준비 / 정리: 합성 키와 메모리 HTTP transport만 사용한다.
    created: list[AsyncOpenAI] = []
    yield created
    for client in created:
        await client.close()


def provider_for(
    clients: list[AsyncOpenAI],
    handler: Callable[[httpx.Request], httpx.Response]
    | Callable[[httpx.Request], Coroutine[None, None, httpx.Response]],
    *,
    max_concurrency: int = 4,
) -> OpenAIWhisperVoiceSttProvider:
    client = AsyncOpenAI(
        api_key="synthetic-test-key",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        max_retries=2,
    )
    clients.append(client)
    return OpenAIWhisperVoiceSttProvider(client=client, max_concurrency=max_concurrency)


def transcript(language: str = "english", probability: float = 0.1) -> dict[str, Any]:
    return {
        "text": "Synthetic spoken sentence.",
        "language": language,
        "duration": 0.1,
        "segments": [
            {
                "id": 0,
                "seek": 0,
                "start": 0,
                "end": 0.1,
                "text": "Synthetic spoken sentence.",
                "tokens": [1],
                "temperature": 0,
                "avg_logprob": -0.1,
                "compression_ratio": 1,
                "no_speech_prob": probability,
            }
        ],
    }


def language_state(*, manual: str | None = None) -> LanguageStateManager:
    return LanguageStateManager(
        lock_confidence=0.8,
        switch_confidence=0.85,
        switch_consecutive_count=3,
        manual_language=manual,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider_language,expected", [("english", "en"), ("korean", "ko"), ("japanese", "ja")]
)
async def test_detected_language_has_no_invented_confidence_or_forced_language(
    clients: list[AsyncOpenAI], provider_language: str, expected: str
):
    # 준비: 응답 언어는 공급자가 독립적으로 감지한 값이다.
    captured: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request.content)
        return httpx.Response(200, json=transcript(provider_language))

    provider = provider_for(clients, handler)

    # 실행: 수동 언어/이전 문맥 인자를 줘도 외부 요청에는 강제로 넣지 않는다.
    result = await provider.transcribe_pcm(
        PCM, language="ko", is_final=True, initial_prompt="must not bias language"
    )
    state = language_state()
    observation = state.observe(
        result.language, result.language_confidence, trusted_language=result.trusted_language
    )

    # 검증: 외부 호출은 하나이며 원본 PCM이 16kHz mono WAV로만 포장된다.
    assert len(captured) == 1
    request_body = captured[0]
    assert b'name="language"' not in request_body
    assert b'name="prompt"' not in request_body
    assert b"whisper-1" in request_body
    assert b"verbose_json" in request_body
    wav_start = request_body.index(b"RIFF")
    with wave.open(io.BytesIO(request_body[wav_start:]), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16000)
        assert wav.readframes(wav.getnframes()) == PCM
    assert result.language == expected
    assert result.language_confidence is None
    assert result.trusted_language
    assert observation.locked_language is None
    assert (
        state.resolve_translation_source(
            detected_language=result.language,
            confidence=None,
            trusted_language=result.trusted_language,
        )
        == expected
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("language", ["fr", "french", "arbitrary unsupported language"])
async def test_unsupported_language_still_rejects_after_previous_lock(
    clients: list[AsyncOpenAI], language: str
):
    # 준비: 이전 영어 잠금이 있는 상태에 비지원 언어를 반환한다.
    provider = provider_for(clients, lambda _: httpx.Response(200, json=transcript(language)))
    state = language_state()
    state.observe("en", 0.99)

    # 실행
    result = await provider.transcribe_pcm(PCM, language=None, is_final=True)

    # 검증: 이전 영어 잠금으로 덮어쓰지 않는다.
    assert not result.trusted_language
    with pytest.raises(VoicePipelineException) as error:
        state.observe(result.language, None, trusted_language=result.trusted_language)
    assert error.value.code == VoiceErrorCode.UNSUPPORTED_LANGUAGE


@pytest.mark.parametrize("manual,detected", [("ko", "en"), ("ja", "ko"), ("en", "ja")])
def test_manual_language_filter_preserves_mismatch_without_numeric_confidence(
    manual: str, detected: str
):
    # 준비
    state = language_state(manual=manual)

    # 실행 / 검증: 공급자 감지는 수동 선택 일치 여부를 검사하며 잘못된 언어는 거절한다.
    with pytest.raises(VoicePipelineException) as error:
        state.observe(detected, None, trusted_language=True)
    assert error.value.code == VoiceErrorCode.LANGUAGE_MISMATCH
    assert (
        state.resolve_translation_source(
            detected_language=manual, confidence=None, trusted_language=True
        )
        == manual
    )
    with pytest.raises(VoicePipelineException) as error:
        state.resolve_translation_source(detected_language=manual, confidence=None)
    assert error.value.code == VoiceErrorCode.LANGUAGE_UNDETERMINED


@pytest.mark.asyncio
async def test_mean_no_speech_probability_suppresses_text_and_language_trust(
    clients: list[AsyncOpenAI],
):
    # 준비: 여러 구간의 평균 확률이 무음 기준에 도달한다.
    payload = transcript(probability=0.9)
    payload["segments"].append({**payload["segments"][0], "no_speech_prob": 0.7})
    provider = provider_for(clients, lambda _: httpx.Response(200, json=payload))

    # 실행
    result = await provider.transcribe_pcm(PCM, language=None, is_final=True)

    # 검증
    assert result.text == ""
    assert result.no_speech_probability == pytest.approx(0.8)
    assert not result.trusted_language


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "malformation",
    [
        "missing_segments",
        "empty_segments",
        "missing_probability",
        "invalid_probability",
        "invalid_language",
        "invalid_text",
        "empty_segment_text",
    ],
)
async def test_malformed_response_fails_without_trusting_language(
    clients: list[AsyncOpenAI], malformation: str
):
    # 준비: 겉으로는 문자열이 있지만 발화/언어 증거가 잘못된 공급자 응답이다.
    payload = transcript()
    if malformation == "missing_segments":
        payload.pop("segments")
    elif malformation == "empty_segments":
        payload["segments"] = []
    elif malformation == "missing_probability":
        payload["segments"][0].pop("no_speech_prob")
    elif malformation == "invalid_probability":
        payload["segments"][0]["no_speech_prob"] = 1.1
    elif malformation == "invalid_language":
        payload["language"] = None
    elif malformation == "invalid_text":
        payload["text"] = 123
    else:
        payload["segments"][0]["text"] = " "
    provider = provider_for(clients, lambda _: httpx.Response(200, json=payload))

    # 실행 / 검증
    with pytest.raises(VoicePipelineException) as error:
        await provider.transcribe_pcm(PCM, language=None, is_final=True)
    assert error.value.code == VoiceErrorCode.STT_FAILED
    assert not error.value.retryable


@pytest.mark.asyncio
async def test_partial_and_digital_silence_never_upload_audio(clients: list[AsyncOpenAI]):
    # 준비: 업로드가 실행되면 테스트를 실패시킨다.
    def handler(_: httpx.Request) -> httpx.Response:
        raise AssertionError("Unexpected external transcription")

    provider = provider_for(clients, handler)

    # 실행 / 검증
    partial = await provider.transcribe_pcm(PCM, language=None, is_final=False)
    silence = await provider.transcribe_pcm(bytes(3200), language=None, is_final=True)
    assert partial.text == silence.text == ""
    assert not provider.supports_partial


@pytest.mark.asyncio
async def test_deadline_includes_slot_wait_and_releases_request_capacity(
    clients: list[AsyncOpenAI], monkeypatch: pytest.MonkeyPatch
):
    # 준비: HTTP가 멈추면 호출과 자리 대기 모두 같은 짧은 기한을 적용한다.
    monkeypatch.setattr(settings, "AI_VOICE_STT_FINAL_TIMEOUT_SECONDS", 0.05)
    release = asyncio.Event()
    calls = 0

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        await release.wait()
        return httpx.Response(200, json=transcript())

    provider = provider_for(clients, handler, max_concurrency=1)

    # 실행
    results = await asyncio.gather(
        provider.transcribe_pcm(PCM, language=None, is_final=True),
        provider.transcribe_pcm(PCM, language=None, is_final=True),
        return_exceptions=True,
    )

    # 검증: 둘 다 timeout이며 취소가 끝나면 후속 요청을 받을 수 있다.
    assert all(
        isinstance(result, VoicePipelineException) and result.code == VoiceErrorCode.STT_TIMEOUT
        for result in results
    )
    assert calls <= 2
    release.set()
    recovered = await provider.transcribe_pcm(PCM, language=None, is_final=True)
    assert recovered.text


@pytest.mark.asyncio
async def test_sdk_timeout_has_no_retry_or_local_fallback(clients: list[AsyncOpenAI]):
    # 준비: 주입한 SDK 기본 재시도가 켜져 있어도 공급자 요청은 재시도하지 않는다.
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("synthetic timeout", request=request)

    provider = provider_for(clients, handler)

    # 실행 / 검증
    with pytest.raises(VoicePipelineException) as error:
        await provider.transcribe_pcm(PCM, language=None, is_final=True)
    assert error.value.code == VoiceErrorCode.STT_TIMEOUT
    assert calls == 1


@pytest.mark.asyncio
async def test_hosted_http_concurrency_is_bounded_at_four(clients: list[AsyncOpenAI]):
    # 준비: 다섯 요청을 동시에 보내 실제 HTTP 동시 실행 수를 관찰한다.
    release = asyncio.Event()
    four_started = asyncio.Event()
    active = 0
    peak = 0
    calls = 0

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal active, peak, calls
        active += 1
        peak = max(peak, active)
        calls += 1
        if active == 4:
            four_started.set()
        try:
            await release.wait()
            return httpx.Response(200, json=transcript())
        finally:
            active -= 1

    provider = provider_for(clients, handler)
    tasks = [
        asyncio.create_task(provider.transcribe_pcm(PCM, language=None, is_final=True))
        for _ in range(5)
    ]
    try:
        # 실행 / 검증: 네 요청만 외부에 도달하며 이후 요청은 자리가 나면 처리한다.
        await asyncio.wait_for(four_started.wait(), 1)
        assert peak == calls == 4
        release.set()
        results = await asyncio.wait_for(asyncio.gather(*tasks), 1)
        assert len(results) == 5
        assert calls == 5
        assert peak == 4
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)


def test_external_provider_is_opt_in_and_missing_key_is_not_ready():
    # 준비 / 실행: 로컬 설정 파일을 읽지 않고 기본값 자체를 확인한다.
    configuration = Settings.model_construct()
    provider = OpenAIWhisperVoiceSttProvider(api_key="")

    # 검증
    assert configuration.AI_VOICE_STT_PROVIDER == "faster-whisper"
    assert not provider.ready
    with pytest.raises(ValueError):
        OpenAIWhisperVoiceSttProvider(api_key="", max_concurrency=5)


@pytest.mark.asyncio
async def test_provider_http_failure_is_redacted_and_not_retried(clients: list[AsyncOpenAI]):
    # 준비: 공급자 응답 원문을 사용자 오류에 포함하지 않는다.
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500, content=json.dumps({"error": "private provider detail"}))

    provider = provider_for(clients, handler)

    # 실행 / 검증
    with pytest.raises(VoicePipelineException) as error:
        await provider.transcribe_pcm(PCM, language=None, is_final=True)
    assert error.value.code == VoiceErrorCode.STT_FAILED
    assert "private provider detail" not in error.value.public_message
    assert calls == 1


@pytest.mark.asyncio
async def test_shutdown_closes_owned_client_and_preserves_borrowed_client(
    clients: list[AsyncOpenAI],
):
    # 준비: 앱 생성 SDK와 외부에서 주입받은 SDK의 소유권을 구분한다.
    owned = OpenAIWhisperVoiceSttProvider(api_key="synthetic-test-key")
    borrowed = provider_for(clients, lambda _: httpx.Response(200, json=transcript()))
    assert owned.ready and borrowed.ready

    # 실행
    await owned.shutdown()
    await borrowed.shutdown()

    # 검증: 양쪽 모두 새 요청을 거절하지만 외부 주입 SDK는 호출자가 닫는다.
    assert not owned.ready and not borrowed.ready
    assert owned._client is not None and owned._client.is_closed()
    assert not clients[-1].is_closed()
    with pytest.raises(VoicePipelineException) as error:
        await borrowed.transcribe_pcm(PCM, language=None, is_final=True)
    assert error.value.code == VoiceErrorCode.MODEL_NOT_READY


@pytest.mark.asyncio
async def test_malformed_pcm_never_uploads_even_when_all_bytes_are_zero(
    clients: list[AsyncOpenAI],
):
    # 준비
    def handler(_: httpx.Request) -> httpx.Response:
        raise AssertionError("Invalid PCM was uploaded")

    provider = provider_for(clients, handler)

    # 실행 / 검증: 16bit 샘플의 일부만 들어온 무음은 유효한 입력으로 취급하지 않는다.
    with pytest.raises(VoicePipelineException) as error:
        await provider.transcribe_pcm(b"\x00", language=None, is_final=True)
    assert error.value.code == VoiceErrorCode.INVALID_AUDIO_FRAME


def test_trusted_provider_flag_does_not_override_an_actual_low_confidence_score():
    # 준비
    state = language_state(manual="ko")

    # 실행 / 검증: 신뢰도 없는 공급자를 위한 표시가 실제 낮은 점수를 우회하지 않는다.
    with pytest.raises(VoicePipelineException) as error:
        state.observe("ko", 0.1, trusted_language=True)
    assert error.value.code == VoiceErrorCode.LANGUAGE_UNDETERMINED
