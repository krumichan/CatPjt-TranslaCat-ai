"""No downloaded model: verify the existing fallback's application boundary only."""
from __future__ import annotations

import asyncio
from typing import cast

import pytest

from app.features.language_learning.speaking.stt_service import FasterWhisperSpeakingSttProvider
from app.features.speech_to_text import FasterWhisperRuntime, WhisperRuntimeResult
from app.features.speech_to_text.runtime import SpeechRuntimeNotReady
from tests.test_language_learning_speaking import make_wav


class Guard:
    enabled = True
    ready = True
    model_version = "explicit-fake"

    def __init__(self, speech=False):
        self.speech = speech
        self.calls = 0

    async def has_speech(self, pcm_bytes: bytes) -> bool:
        assert pcm_bytes and not pcm_bytes.startswith(b"RIFF")
        self.calls += 1
        return self.speech


class Runtime:
    ready = True

    def __init__(self, initial=""):
        self.initial = initial
        self.calls = []

    async def transcribe(self, audio, *, options, priority):
        self.calls.append(options)
        return WhisperRuntimeResult(self.initial if len(self.calls) == 1 else "Synthetic fallback",
                                    "ja", .9, 2.0, [], "fake", "fake", "fake")


def provider(runtime, guard, **kwargs):
    return FasterWhisperSpeakingSttProvider(cast(FasterWhisperRuntime, runtime), speech_guard=guard, **kwargs)


@pytest.mark.asyncio
async def test_empty_vad_noise_does_not_reach_unfiltered_whisper() -> None:
    runtime, guard = Runtime(), Guard(False)
    result = await provider(runtime, guard).transcribe(make_wav(2), language="ja")
    assert result.text == ""
    assert len(runtime.calls) == guard.calls == 1
    assert runtime.calls[0]["vad_filter"] is True


@pytest.mark.asyncio
async def test_confirmed_quiet_speech_keeps_exactly_one_existing_fallback() -> None:
    runtime, guard = Runtime(), Guard(True)
    result = await provider(runtime, guard, beam_size=5).transcribe(make_wav(2), language="ja")
    assert result.text == "Synthetic fallback"
    assert [call["vad_filter"] for call in runtime.calls] == [True, False]
    assert [call["beam_size"] for call in runtime.calls] == [5, 5]
    assert guard.calls == 1


@pytest.mark.asyncio
async def test_normal_vad_transcript_has_no_extra_guard_or_stt_call() -> None:
    runtime, guard = Runtime("Synthetic recognized speech"), Guard(False)
    result = await provider(runtime, guard).transcribe(make_wav(2), language="ja")
    assert result.text == "Synthetic recognized speech"
    assert len(runtime.calls) == 1 and guard.calls == 0


@pytest.mark.asyncio
async def test_disabled_pass_through_is_not_speech_evidence() -> None:
    runtime, guard = Runtime(), Guard(True)
    guard.enabled = False
    result = await provider(runtime, guard).transcribe(make_wav(2), language="ja")
    assert result.text == "" and len(runtime.calls) == 1 and guard.calls == 0


@pytest.mark.asyncio
async def test_unavailable_guard_is_infrastructure_not_no_speech() -> None:
    runtime, guard = Runtime(), Guard(True)
    guard.ready = False
    with pytest.raises(SpeechRuntimeNotReady):
        await provider(runtime, guard).transcribe(make_wav(2), language="ja")
    assert len(runtime.calls) == 1 and guard.calls == 0


@pytest.mark.asyncio
async def test_cancelled_guard_never_starts_fallback() -> None:
    class CancelledGuard(Guard):
        async def has_speech(self, pcm_bytes):
            raise asyncio.CancelledError
    runtime = Runtime()
    with pytest.raises(asyncio.CancelledError):
        await provider(runtime, CancelledGuard()).transcribe(make_wav(2), language="ja")
    assert len(runtime.calls) == 1


@pytest.mark.parametrize("beam", [0, 6, True])
def test_beam_configuration_is_bounded(beam) -> None:
    with pytest.raises(ValueError):
        provider(Runtime(), Guard(), beam_size=beam)
