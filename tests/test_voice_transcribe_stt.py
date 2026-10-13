from __future__ import annotations

import asyncio
import io
import wave
from collections.abc import AsyncIterator, Callable, Coroutine
from typing import Any

import httpx2 as httpx
import pytest
import pytest_asyncio
from openai import AsyncOpenAI

from app.core.config import settings
from app.features.voice_translation.errors import VoicePipelineException
from app.features.voice_translation.language import LanguageStateManager
from app.features.voice_translation.transcribe_stt import OpenAITranscribeVoiceSttProvider
from app.schemas.voice_translation import VoiceErrorCode

PCM = b"\x01\x00" * 1600


@pytest_asyncio.fixture
async def clients() -> AsyncIterator[list[AsyncOpenAI]]:
    # 준비 / 정리: 실제 자격 증명과 네트워크를 사용하지 않는다.
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
) -> OpenAITranscribeVoiceSttProvider:
    client = AsyncOpenAI(
        api_key="synthetic-test-key",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        max_retries=2,
    )
    clients.append(client)
    return OpenAITranscribeVoiceSttProvider(client=client, max_concurrency=max_concurrency)


def transcript(*languages: str) -> dict[str, Any]:
    return {
        "text": "Synthetic spoken sentence.",
        "languages": [{"code": item} for item in languages],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("is_final", [True, False])
@pytest.mark.parametrize(
    "language,expected", [("en", "en"), ("ko", "ko"), ("ja", "ja"), ("jpn", "ja")]
)
async def test_final_and_preview_detect_language_without_forcing_or_invented_scores(
    clients: list[AsyncOpenAI], language: str, expected: str, is_final: bool
):
    # 준비
    captured: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request.content)
        return httpx.Response(200, json=transcript(language))

    provider = provider_for(clients, handler)

    # 실행: 수동 언어 및 이전 문맥은 외부 인식의 언어를 강제하지 않는다.
    result = await provider.transcribe_pcm(
        PCM, language="en", is_final=is_final, initial_prompt="Previous sentence."
    )

    # 검증: 오디오는 메모리 WAV에만 포장되며 final/preview 모두 동일한 감지 계약이다.
    assert result.language == expected
    assert result.language_confidence is None
    assert result.no_speech_probability is None
    assert result.trusted_language
    assert result.model == "gpt-transcribe"
    assert not provider.supports_partial
    assert provider.partial_available
    assert len(captured) == 1
    body = captured[0]
    assert b"gpt-transcribe" in body and b'name="response_format"\r\n\r\njson' in body
    assert b'name="language"' not in body and b'name="languages"' not in body
    assert b'name="prompt"' not in body
    with wave.open(io.BytesIO(body[body.index(b"RIFF") :]), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16000)
        assert wav.readframes(wav.getnframes()) == PCM


@pytest.mark.asyncio
@pytest.mark.parametrize("languages", [[], ["und"], ["en", "ja"]])
async def test_unknown_or_mixed_supported_language_does_not_claim_one_language(
    clients: list[AsyncOpenAI], languages: list[str]
):
    # 준비
    provider = provider_for(clients, lambda _: httpx.Response(200, json=transcript(*languages)))

    # 실행
    result = await provider.transcribe_pcm(PCM, language="en", is_final=False)

    # 검증
    assert result.text
    assert result.language is None and not result.trusted_language


@pytest.mark.asyncio
@pytest.mark.parametrize("languages", [["fr"], ["en", "fr"], ["zh-cn"]])
async def test_unsupported_language_cannot_become_previously_locked_language(
    clients: list[AsyncOpenAI], languages: list[str]
):
    # 준비: 이전 영어 잠금보다 새 비지원 언어 감지를 우선한다.
    provider = provider_for(clients, lambda _: httpx.Response(200, json=transcript(*languages)))
    state = LanguageStateManager(
        lock_confidence=0.8,
        switch_confidence=0.85,
        switch_consecutive_count=3,
        initial_locked_language="en",
    )

    # 실행
    result = await provider.transcribe_pcm(PCM, language=None, is_final=True)

    # 검증
    assert not result.trusted_language
    with pytest.raises(VoicePipelineException) as error:
        state.observe(result.language, None, trusted_language=result.trusted_language)
    assert error.value.code == VoiceErrorCode.UNSUPPORTED_LANGUAGE


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"text": "missing language field"},
        {"text": 123, "languages": [{"code": "en"}]},
        {"text": "test", "languages": None},
        {"text": "test", "languages": ["en"]},
        {"text": "test", "languages": [{"code": 123}]},
        {"text": "test", "languages": [{"code": "english"}]},
        {"text": "test", "languages": [{"code": ""}]},
        {"text": "x" * 10_001, "languages": []},
        {"text": "test", "languages": [{"code": "en"}] * 33},
    ],
)
async def test_invalid_provider_payload_is_nonretryable_and_redacted(
    clients: list[AsyncOpenAI], payload: dict[str, Any]
):
    # 준비
    provider = provider_for(clients, lambda _: httpx.Response(200, json=payload))

    # 실행 / 검증
    with pytest.raises(VoicePipelineException) as error:
        await provider.transcribe_pcm(PCM, language=None, is_final=True)
    assert error.value.code == VoiceErrorCode.STT_FAILED
    assert not error.value.retryable
    assert "missing language" not in error.value.public_message


@pytest.mark.asyncio
async def test_empty_response_and_digital_silence_do_not_create_probability(
    clients: list[AsyncOpenAI],
):
    # 준비
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"text": "  ", "languages": []})

    provider = provider_for(clients, handler)

    # 실행
    empty = await provider.transcribe_pcm(PCM, language=None, is_final=True)
    silent = await provider.transcribe_pcm(bytes(3200), language=None, is_final=False)

    # 검증: 무음 PCM은 업로드하지 않으며 점수 부재를 실제 확률로 바꾸지 않는다.
    assert calls == 1
    for result in [empty, silent]:
        assert result.text == "" and not result.trusted_language
        assert result.no_speech_probability is None


@pytest.mark.asyncio
async def test_invalid_pcm_is_rejected_before_silence_shortcut(clients: list[AsyncOpenAI]):
    # 준비
    def handler(_: httpx.Request) -> httpx.Response:
        raise AssertionError("Invalid PCM must not upload")

    provider = provider_for(clients, handler)

    # 실행 / 검증
    with pytest.raises(VoicePipelineException) as error:
        await provider.transcribe_pcm(b"\x00", language=None, is_final=True)
    assert error.value.code == VoiceErrorCode.INVALID_AUDIO_FRAME


@pytest.mark.asyncio
async def test_blocked_preview_does_not_occupy_final_slot(clients: list[AsyncOpenAI]):
    # 준비: 하나의 preview가 멈춘 동안 최종 인식을 시작한다.
    release = asyncio.Event()
    preview_started = asyncio.Event()
    calls = 0

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            preview_started.set()
            await release.wait()
        return httpx.Response(200, json=transcript("en"))

    provider = provider_for(clients, handler, max_concurrency=1)
    preview = asyncio.create_task(provider.transcribe_pcm(PCM, language=None, is_final=False))
    try:
        await asyncio.wait_for(preview_started.wait(), 1)

        # 실행 / 검증: preview 완료를 기다리지 않고 final이 반환된다.
        final = await asyncio.wait_for(
            provider.transcribe_pcm(PCM, language=None, is_final=True), 1
        )
        assert final.text and calls == 2
        assert not preview.done()
    finally:
        release.set()
        await preview


@pytest.mark.asyncio
@pytest.mark.parametrize("is_final", [True, False])
async def test_deadline_includes_slot_wait_and_capacity_recovers(
    clients: list[AsyncOpenAI], monkeypatch: pytest.MonkeyPatch, is_final: bool
):
    # 준비: 실행 요청과 자리 대기 요청 모두 기한 안에 반환해야 한다.
    monkeypatch.setattr(settings, "AI_VOICE_STT_FINAL_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(settings, "AI_VOICE_STT_PARTIAL_TIMEOUT_SECONDS", 0.05)
    release = asyncio.Event()

    async def handler(_: httpx.Request) -> httpx.Response:
        await release.wait()
        return httpx.Response(200, json=transcript("en"))

    provider = provider_for(clients, handler, max_concurrency=1)

    # 실행
    results = await asyncio.gather(
        *(provider.transcribe_pcm(PCM, language=None, is_final=is_final) for _ in range(2)),
        return_exceptions=True,
    )

    # 검증: 취소 후 같은 슬롯을 후속 호출이 다시 사용할 수 있다.
    assert all(
        isinstance(result, VoicePipelineException) and result.code == VoiceErrorCode.STT_TIMEOUT
        for result in results
    )
    release.set()
    assert (await provider.transcribe_pcm(PCM, language=None, is_final=is_final)).text


@pytest.mark.asyncio
@pytest.mark.parametrize("status,retryable", [(400, False), (401, False), (429, True), (500, True)])
async def test_http_failure_is_redacted_and_never_retried(
    clients: list[AsyncOpenAI], status: int, retryable: bool
):
    # 준비
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status, json={"error": {"message": "Private provider details"}})

    provider = provider_for(clients, handler)

    # 실행 / 검증
    with pytest.raises(VoicePipelineException) as error:
        await provider.transcribe_pcm(PCM, language=None, is_final=True)
    assert error.value.code == VoiceErrorCode.STT_FAILED
    assert error.value.retryable is retryable
    assert "Private" not in error.value.public_message
    assert calls == 1


@pytest.mark.asyncio
async def test_sdk_timeout_preserves_timeout_code_without_retries(clients: list[AsyncOpenAI]):
    # 준비
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("synthetic timeout", request=request)

    provider = provider_for(clients, handler)

    # 실행 / 검증
    with pytest.raises(VoicePipelineException) as error:
        await provider.transcribe_pcm(PCM, language=None, is_final=False)
    assert error.value.code == VoiceErrorCode.STT_TIMEOUT
    assert calls == 1


@pytest.mark.asyncio
async def test_shutdown_blocks_queued_upload_and_preserves_borrowed_client(
    clients: list[AsyncOpenAI],
):
    # 준비: 한 요청 실행 중 다른 요청은 같은 preview 슬롯을 기다린다.
    release = asyncio.Event()
    started = asyncio.Event()
    calls = 0

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return httpx.Response(200, json=transcript("en"))

    provider = provider_for(clients, handler)
    active = asyncio.create_task(provider.transcribe_pcm(PCM, language=None, is_final=False))
    await asyncio.wait_for(started.wait(), 1)
    waiting = asyncio.create_task(provider.transcribe_pcm(PCM, language=None, is_final=False))
    await asyncio.sleep(0)

    # 실행
    await provider.shutdown()
    release.set()
    results = await asyncio.gather(active, waiting, return_exceptions=True)

    # 검증: 대기 요청은 종료 후 새 업로드를 하지 않는다. 주입 SDK는 호출자가 닫는다.
    assert calls == 1 and not provider.ready and not provider.partial_available
    assert not clients[-1].is_closed()
    assert isinstance(results[1], VoicePipelineException)
    assert results[1].code == VoiceErrorCode.MODEL_NOT_READY


@pytest.mark.asyncio
async def test_owned_client_is_closed_and_missing_key_is_not_ready():
    # 준비
    owned = OpenAITranscribeVoiceSttProvider(api_key="synthetic-test-key")
    missing = OpenAITranscribeVoiceSttProvider(api_key="")

    # 실행
    await owned.shutdown()

    # 검증
    assert not owned.ready and not missing.ready
    assert owned._client is not None and owned._client.is_closed()
    with pytest.raises(ValueError):
        OpenAITranscribeVoiceSttProvider(api_key="", max_concurrency=5)
    with pytest.raises(ValueError):
        OpenAITranscribeVoiceSttProvider(api_key="", max_preview_concurrency=3)
