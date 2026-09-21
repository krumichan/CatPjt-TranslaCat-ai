import asyncio
import hashlib
import io
import wave
from dataclasses import replace

import pytest

from app.ai.ports import SpeechSynthesisResult
from app.features.language_learning.listening.audio_store import TemporaryListeningAudioStore
from app.features.language_learning.listening.duration import (
    decode_reference_audio, duration_demand, generation_duration_guidance,
    minimum_short_correction_characters,
)
from app.features.language_learning.listening.errors import ListeningStageException
from app.features.language_learning.listening.normalization import normalize_text
from app.features.language_learning.listening.tts_service import ListeningTtsService
from app.schemas.language_learning_listening import ListeningSetGenerationRequest, ListeningTtsRequest


def wav(seconds, *, rate=16000, channels=1):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(b"\x02\x00" * round(seconds * rate) * channels)
    return buffer.getvalue()


def request(demand=None):
    text = "音声の長さを実際のフレームで確認します。"
    return ListeningTtsRequest.model_validate({
        "requestId": "duration-test", "idempotencyKey": "duration-key", "itemId": 1,
        "sourceText": text, "contentHash": hashlib.sha256(normalize_text(text, "ja").text.encode()).hexdigest(),
        "generationVersion": "generation", "learningLanguage": "ja",
        "voice": {"locale": "ja", "voiceKey": "Kore"}, "durationDemand": demand,
    })


class Provider:
    def __init__(self, audio, metadata_seconds=99):
        self.audio = audio
        self.metadata_seconds = metadata_seconds
        self.calls = 0

    async def synthesize_speech(self, **kwargs):
        self.calls += 1
        assert kwargs["speed"] == "NORMAL"
        return SpeechSynthesisResult(
            audio_bytes=self.audio, content_type="audio/wav", provider="fake", model="fake",
            duration_seconds=self.metadata_seconds,
        )


def service(tmp_path, audio):
    store = TemporaryListeningAudioStore()
    store.base_dir = tmp_path
    provider = Provider(audio)
    return ListeningTtsService(provider, store, automatic_retries=2), provider, store


@pytest.mark.parametrize("low,high,seconds,code", [
    (5, 12, 4.999, "AUDIO_TOO_SHORT"), (5, 12, 5, None), (5, 12, 12, None),
    (8, 20, 7.999, "AUDIO_TOO_SHORT"), (8, 20, 8, None), (8, 20, 20.001, "AUDIO_TOO_LONG"),
    (15, 30, 15, None), (15, 30, 30, None), (15, 30, 30.001, "AUDIO_TOO_LONG"),
])
def test_actual_frames_gate_not_provider_estimate(tmp_path, low, high, seconds, code):
    svc, provider, _ = service(tmp_path, wav(seconds))
    result = asyncio.run(svc.synthesize(request({"minSeconds": low, "maxSeconds": high})))
    assert provider.calls == 1  # content failure never becomes TTS infrastructure retry
    if code:
        assert result.status == "FAILED" and result.audio is None
        assert result.error is not None
        assert result.error.code.value == code and not result.error.retryable
        assert result.error.details["measuredSeconds"] == pytest.approx(seconds)
    else:
        assert result.status == "READY"
        assert result.audio is not None
        assert result.audio.duration_ms == round(seconds * 1000)
        assert result.audio.sample_rate == 16000 and result.audio.channels == 1
        assert result.audio.duration_validated


def test_legacy_cache_is_rechecked_against_current_demand(tmp_path):
    svc, provider, _ = service(tmp_path, wav(6))
    legacy = asyncio.run(svc.synthesize(request()))
    assert legacy.audio is not None
    assert legacy.status == "READY" and not legacy.audio.duration_validated
    current = asyncio.run(svc.synthesize(request({"minSeconds": 8, "maxSeconds": 20})))
    assert current.error is not None
    assert current.status == "FAILED" and current.error.code.value == "AUDIO_TOO_SHORT"
    assert provider.calls == 1


@pytest.mark.parametrize("corrupt_metadata", [False, True])
def test_cached_bytes_and_metadata_cannot_disagree(tmp_path, corrupt_metadata):
    svc, provider, store = service(tmp_path, wav(9))
    initial = asyncio.run(svc.synthesize(request()))
    assert initial.audio is not None
    reference = initial.audio.audio_reference
    stored = store.get(reference)
    assert stored is not None
    if corrupt_metadata:
        store._items[reference] = replace(stored, duration_seconds=19)
    else:
        stored.path.write_bytes(wav(10))
    result = asyncio.run(svc.synthesize(request()))
    assert result.error is not None
    assert result.status == "FAILED" and result.error.code.value == "AUDIO_DECODE_FAILED"
    assert provider.calls == 1


@pytest.mark.parametrize("audio", [b"RIFFfake", wav(5)[:-4]], ids=["invalid-header", "truncated-pcm"])
def test_incomplete_decode_never_ready(tmp_path, audio):
    svc, provider, _ = service(tmp_path, audio)
    response = asyncio.run(svc.synthesize(request()))
    assert response.error is not None
    assert response.status == "FAILED" and response.error.code.value == "AUDIO_DECODE_FAILED"
    assert provider.calls == 1


def generation(difficulty="MY_LEVEL", **updates):
    data = {
        "requestId": "generation", "idempotencyKey": "generation-key",
        "userContext": {"originLanguage": "ko", "learningLanguage": "ja"},
        "setContext": {"learningDate": "2026-09-19", "topic": {"id": 1, "title": "Daily"}, "difficulty": difficulty},
        "referenceVoice": {"locale": "ja", "voiceKey": "Kore"},
    }
    data.update(updates)
    return ListeningSetGenerationRequest.model_validate(data)


def test_effective_intersection_and_lane_specific_guidance():
    req = generation(constraints={"audioSecondsMin": 10, "audioSecondsMax": 25})
    demand = duration_demand(req)
    assert (demand.min_seconds, demand.max_seconds) == (10, 20)
    guidance = generation_duration_guidance(req)
    assert guidance["interiorTargetSeconds"] == 15
    assert guidance["empiricalPlanningReference"]["sampleCount"] == 30
    other = req.model_copy(update={"reference_voice": None})
    assert "empiricalPlanningReference" not in generation_duration_guidance(other)
    with pytest.raises(ListeningStageException):
        duration_demand(generation("EASY", constraints={"audioSecondsMin": 15}))
    assert decode_reference_audio(wav(1, channels=2), "audio/wav").channels == 2


def test_openai_marin_guidance_uses_only_its_measured_samples():
    req = generation(referenceVoice={"locale": "ja", "voiceKey": "marin"})
    guidance = generation_duration_guidance(req)
    reference = guidance["empiricalPlanningReference"]
    assert (reference["provider"], reference["voice"], reference["sampleCount"]) == (
        "openai", "marin", 3,
    )
    assert reference["observedCharactersPerSecond"]["mean"] != 5.894
    assert guidance["estimatedAudioSecondsIsDiagnosticOnly"] is True
    cedar = generation(referenceVoice={"locale": "ja", "voiceKey": "cedar"})
    assert "empiricalPlanningReference" not in generation_duration_guidance(cedar)


def test_measured_short_correction_requires_real_expansion_before_tts():
    previous = "環境を守るために、使わない電気は消しています。"
    req = generation("EASY", referenceVoice={"locale": "ja", "voiceKey": "marin"},
                     durationCorrection={"previousSourceText": previous,
                                         "previousMeasuredSeconds": 4.45})
    floor = minimum_short_correction_characters(req)
    assert floor == 26
    guidance = generation_duration_guidance(req)
    assert guidance["durationCorrectionPlanning"]["minimumSourceCharacters"] == 26
    assert guidance["durationCorrectionPlanning"]["notAudioAcceptance"] is True
    assert (duration_demand(req).min_seconds, duration_demand(req).max_seconds) == (5, 12)
    assert minimum_short_correction_characters(generation("EASY")) is None
