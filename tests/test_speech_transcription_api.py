import base64
import io
import wave
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_speaking_speech_runtime, get_speech_runtime
from app.api.internal.speech_transcription import get_speech_evidence_runtime
from app.core.config import settings
from app.features.speech_execution.audio import AudioDecoder
from app.features.speech_to_text.runtime import WhisperRuntimeResult, WhisperRuntimeSegment


def synthetic_wave(seconds=1, silent=False):
    values = (
        np.zeros(int(22050 * seconds))
        if silent
        else np.sin(np.arange(int(22050 * seconds)) * 0.04) * 0.1
    )
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(22050)
        writer.writeframes((values * 32767).astype("<i2").tobytes())
    return output.getvalue()


class SyntheticRuntime:
    ready = True

    def __init__(self):
        self.calls = []

    async def transcribe(self, audio, **kwargs):
        self.calls.append(kwargs)
        return WhisperRuntimeResult(
            "Synthetic transcript",
            "en",
            0.9,
            1,
            [WhisperRuntimeSegment(0, 1, "Synthetic transcript", -0.2, 0.01)],
            "test",
            "test",
            "synthetic",
        )

    async def has_speech(self, pcm):
        self.calls.append({"pcm_bytes": len(pcm)})
        return bool(pcm)


@pytest.fixture
def transcription_client():
    with patch("app.core.config_logger.setup_logging"):
        from app.main import app
    original = settings.SERVER_API_KEY
    settings.SERVER_API_KEY = "synthetic-stt-key"
    runtime = SyntheticRuntime()
    app.dependency_overrides[get_speech_runtime] = lambda: runtime
    app.dependency_overrides[get_speaking_speech_runtime] = lambda: runtime
    app.dependency_overrides[get_speech_evidence_runtime] = lambda: runtime
    try:
        yield TestClient(app), runtime
    finally:
        settings.SERVER_API_KEY = original
        app.dependency_overrides.pop(get_speech_runtime, None)
        app.dependency_overrides.pop(get_speaking_speech_runtime, None)
        app.dependency_overrides.pop(get_speech_evidence_runtime, None)


def command(**overrides):
    return {
        "requestId": "synthetic-stt",
        "audioBase64": base64.b64encode(synthetic_wave()).decode(),
        "remainingMilliseconds": 1000,
        **overrides,
    }


def test_decoder_preserves_existing_pcm_conversion_and_measurements():
    # 준비: 퇴역 전 원본 Speaking 정규화의 합성 PCM 해시와 측정값을 읽는다.
    import hashlib
    import json
    from pathlib import Path

    expected = json.loads(
        (Path(__file__).parent / "fixtures/speech-normalization-speaking-original.json")
        .read_text(encoding="utf-8")
    )

    # 실행
    current = AudioDecoder().decode(synthetic_wave())

    # 검증: 업무 소스를 제거한 뒤에도 PCM과 수치가 기존 값과 일치해야 한다.
    assert hashlib.sha256(current.wav_bytes).hexdigest() == expected["wavSha256"]
    assert round(current.duration_seconds, 3) == expected["durationSeconds"]
    assert round(current.rms, 6) == expected["rms"]
    assert round(current.peak, 6) == expected["peak"]
    assert round(current.silence_ratio, 6) == expected["silenceRatio"]


def test_normalizer_preserves_raw_precision_before_learning_thresholds():
    # 준비: 소수 여섯째 자리 반올림 전에는 기존 무음 경계보다 작은 PCM이다.
    import struct
    output = io.BytesIO()
    sample = int(0.0029996 * 2147483648)
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(4)
        writer.setframerate(16000)
        writer.writeframes(struct.pack("<i", sample) * 16000)

    # 실행
    decoded = AudioDecoder().decode(output.getvalue())

    # 검증: 반올림으로 무음을 합격시키지 않고 LL이 원시값으로 판정할 수 있다.
    assert decoded.rms < 0.003
    assert round(decoded.rms, 6) == 0.003


def test_normalizer_leaves_silence_decision_to_ll(transcription_client):
    # 준비
    client, runtime = transcription_client
    payload = command(audioBase64=base64.b64encode(synthetic_wave(silent=True)).decode())

    # 실행
    response = client.post(
        "/internal/v1/speech/normalize", json=payload, headers={"X-API-KEY": "synthetic-stt-key"}
    )

    # 검증
    assert response.status_code == 200
    assert response.json()["rms"] == 0
    assert response.json()["silenceRatio"] == 1
    assert runtime.calls == []


def test_transcription_uses_only_caller_options(transcription_client):
    # 준비
    client, runtime = transcription_client
    payload = command(
        runtime="shared",
        language=None,
        initialPrompt="Synthetic hint",
        beamSize=1,
        vadFilter=True,
        minSilenceDurationMs=500,
        conditionOnPreviousText=False,
        maxProviderCalls=1,
    )

    # 실행
    unauthorized = client.post("/internal/v1/speech/transcribe", json=payload)
    response = client.post(
        "/internal/v1/speech/transcribe", json=payload, headers={"X-API-KEY": "synthetic-stt-key"}
    )

    # 검증
    assert unauthorized.status_code == 401
    assert response.status_code == 200
    assert response.json()["providerCalls"] == 1
    assert len(runtime.calls) == 1
    assert runtime.calls[0]["options"] == {
        "beam_size": 1,
        "language": None,
        "initial_prompt": "Synthetic hint",
        "vad_filter": True,
        "vad_parameters": {"min_silence_duration_ms": 500},
        "condition_on_previous_text": False,
    }


def test_evidence_executes_guard_without_hidden_stt_fallback(transcription_client):
    # 준비
    client, runtime = transcription_client
    audio = AudioDecoder().decode(synthetic_wave())
    payload = command(audioBase64=base64.b64encode(audio.wav_bytes).decode())

    # 실행
    response = client.post(
        "/internal/v1/speech/evidence", json=payload, headers={"X-API-KEY": "synthetic-stt-key"}
    )

    # 검증
    assert response.status_code == 200
    assert response.json() == {"hasSpeech": True}
    assert runtime.calls == [{"pcm_bytes": 32000}]


@pytest.mark.parametrize(
    "override",
    [
        {"audioBase64": "invalid"},
        {"sourceText": "hidden grading input"},
        {"remainingMilliseconds": 0},
    ],
)
def test_normalization_rejects_invalid_contract_safely(transcription_client, override):
    # 준비
    client, runtime = transcription_client

    # 실행
    response = client.post(
        "/internal/v1/speech/normalize",
        json=command(**override),
        headers={"X-API-KEY": "synthetic-stt-key"},
    )

    # 검증
    assert response.status_code == 422
    assert "hidden grading" not in response.text
    assert runtime.calls == []


@pytest.mark.parametrize(
    "failure,status,code,retryable",
    [
        (
            RuntimeError("resource exhausted sensitive transcript"),
            429,
            "PROVIDER_RATE_LIMITED",
            True,
        ),
        (TimeoutError("sensitive transcript"), 504, "PROVIDER_TIMEOUT", True),
        (ValueError("sensitive transcript"), 400, "STT_EXECUTION_FAILED", False),
        (RuntimeError("unsafe sensitive transcript"), 422, "PROVIDER_REFUSAL", False),
        (RuntimeError("sensitive transcript"), 503, "STT_EXECUTION_FAILED", True),
    ],
)
def test_transcription_preserves_technical_failure_classification(
    transcription_client, failure, status, code, retryable
):
    # 준비
    client, runtime = transcription_client

    async def failed(*args, **kwargs):
        raise failure

    runtime.transcribe = failed

    # 실행
    response = client.post(
        "/internal/v1/speech/transcribe",
        json=command(
            runtime="accurate",
            language="en",
            initialPrompt=None,
            beamSize=1,
            vadFilter=True,
            minSilenceDurationMs=500,
            conditionOnPreviousText=False,
            maxProviderCalls=1,
        ),
        headers={"X-API-KEY": "synthetic-stt-key"},
    )

    # 검증
    assert response.status_code == status
    expected = {"code": code, "retryable": retryable}
    if isinstance(failure, TimeoutError):
        expected["failureKind"] = "TIMEOUT"
    elif isinstance(failure, ValueError):
        expected["failureKind"] = "VALUE_ERROR"
    if code == "PROVIDER_RATE_LIMITED":
        expected["failureSignal"] = "RATE_LIMIT"
    elif code == "PROVIDER_REFUSAL":
        expected["failureSignal"] = "SAFETY"
    assert response.json()["detail"] == expected
    assert "sensitive transcript" not in response.text
