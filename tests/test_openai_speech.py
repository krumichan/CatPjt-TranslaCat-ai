from __future__ import annotations

import io
import wave
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.ai.provider_factory import create_speech_synthesis_provider, create_text_generation_provider
from app.ai.providers.openai.speech import (
    OpenAISpeechService, normalize_complete_speech_wav, validate_speech_wav,
)
from app.core.config import settings
from app.schemas.language_learning_level_test import LevelTestReferenceAudioUpload


def _wav(seconds: float = 1.0) -> bytes:
    data = io.BytesIO()
    with wave.open(data, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\x00\x00" * int(seconds * 16_000))
    return data.getvalue()


@pytest.mark.asyncio
async def test_factory_uses_openai_speech_without_gemini_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "GOOGLE_API_KEY", "")
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "local-test-only")
    provider = create_speech_synthesis_provider()
    assert isinstance(provider, OpenAISpeechService)
    assert provider.ready
    create = AsyncMock(return_value=SimpleNamespace(aread=AsyncMock(return_value=_wav())))
    provider._client = SimpleNamespace(audio=SimpleNamespace(speech=SimpleNamespace(create=create)))  # type: ignore[assignment]
    result = await provider.synthesize_speech(text="明日は会議です。", voice="marin", language="ja", speed="NORMAL")
    assert result.provider == "openai"
    assert result.duration_seconds == 1
    assert create.call_args.kwargs["response_format"] == "wav"
    assert create.call_args.kwargs["voice"] == "marin"
    assert create.call_args.kwargs["model"] == settings.OPENAI_SPEECH_MODEL


@pytest.mark.asyncio
async def test_gemini_voice_rejected_before_any_speech_call() -> None:
    provider = OpenAISpeechService()
    create = AsyncMock()
    provider._client = SimpleNamespace(audio=SimpleNamespace(speech=SimpleNamespace(create=create)))  # type: ignore[assignment]
    with pytest.raises(ValueError, match="voice"):
        await provider.synthesize_speech(text="こんにちは", voice="Kore", language="ja", speed="NORMAL")
    create.assert_not_called()


def test_text_factory_refuses_new_gemini_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "AI_TEXT_PROVIDER", "gemini")
    with pytest.raises(ValueError, match="disabled"):
        create_text_generation_provider()


def test_wav_full_frame_decode_rejects_truncation() -> None:
    good = _wav(0.5)
    assert validate_speech_wav(good) == 0.5
    with pytest.raises(ValueError):
        validate_speech_wav(good[:-8])
    with pytest.raises(ValueError):
        validate_speech_wav(b"<html>upstream error</html>")


def test_complete_streaming_wav_with_unknown_header_lengths_uses_actual_frames() -> None:
    # Speech transports may publish RIFF/data lengths as 0xFFFFFFFF before
    # streaming but still return a complete HTTP body; this is not a truncated
    # PCM frame. A normally declared truncated body remains rejected above.
    audio = bytearray(_wav(1.25))
    audio[4:8] = b"\xff" * 4
    audio[40:44] = b"\xff" * 4
    assert validate_speech_wav(bytes(audio)) == 1.25
    normalized, duration = normalize_complete_speech_wav(bytes(audio))
    assert duration == 1.25
    assert int.from_bytes(normalized[4:8], "little") == len(normalized) - 8
    assert int.from_bytes(normalized[40:44], "little") == len(normalized) - 44
    assert normalized[44:] == audio[44:]


def test_new_level_test_audio_defaults_to_resolved_openai_voice() -> None:
    upload = LevelTestReferenceAudioUpload(
        upload_url="https://example.invalid/qa-object",
        object_key="qa-reference-audio",
    )
    assert upload.voice == "marin"
    assert upload.content_type == "audio/wav"
