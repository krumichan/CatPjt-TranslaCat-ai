import asyncio
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import APITimeoutError

from app.ai.ports import SpeechSynthesisResult
from app.api.dependencies import get_speech_execution_provider
from app.core.config import settings


class SyntheticSpeechProvider:
    def __init__(self, failure=None, delay=0):
        self.calls = []
        self.failure = failure
        self.delay = delay

    async def synthesize_speech(self, **kwargs):
        self.calls.append(kwargs)
        await asyncio.sleep(self.delay)
        if self.failure:
            raise self.failure
        return SpeechSynthesisResult(b"synthetic-audio", "audio/wav", "test", "test", 1.0)


@pytest.fixture
def speech_client():
    with patch("app.core.config_logger.setup_logging"):
        from app.main import app

    original_key = settings.SERVER_API_KEY
    settings.SERVER_API_KEY = "synthetic-speech-key"
    provider = SyntheticSpeechProvider()
    app.dependency_overrides[get_speech_execution_provider] = lambda: provider
    try:
        yield TestClient(app), provider
    finally:
        settings.SERVER_API_KEY = original_key
        app.dependency_overrides.pop(get_speech_execution_provider, None)


def command(**overrides):
    return {
        "requestId": "synthetic-tts-1", "text": "Synthetic speech.",
        "voice": "marin", "language": "en", "speed": "NORMAL",
        "remainingMilliseconds": 1000, "maxProviderCalls": 1, **overrides,
    }


def test_speech_auth_and_explicit_single_execution(speech_client):
    # 준비
    client, provider = speech_client
    path = "/internal/v1/speech/synthesize"

    # 실행
    unauthorized = client.post(path, json=command())
    response = client.post(path, json=command(), headers={"X-API-KEY": "synthetic-speech-key"})

    # 검증: 업무 이름에 따른 숨은 발화 변환 없이 한 번만 실행한다.
    assert unauthorized.status_code == 401
    assert response.status_code == 200
    assert response.json()["audioBase64"] == "c3ludGhldGljLWF1ZGlv"
    assert response.json()["providerCalls"] == 1
    assert provider.calls == [{
        "text": "Synthetic speech.", "voice": "marin", "language": "en", "speed": "NORMAL",
    }]


@pytest.mark.parametrize("overrides", [
    {"maxProviderCalls": 2}, {"remainingMilliseconds": 0},
    {"taskName": "hidden-policy"}, {"url": "http://localhost/private"},
])
def test_speech_rejects_invalid_contract_without_echo(speech_client, overrides):
    # 준비
    client, provider = speech_client

    # 실행
    response = client.post("/internal/v1/speech/synthesize", json=command(**overrides),
                           headers={"X-API-KEY": "synthetic-speech-key"})

    # 검증
    assert response.status_code == 422
    assert response.json() == {"detail": {"code": "SPEECH_REQUEST_INVALID", "retryable": False}}
    assert "Synthetic speech" not in response.text
    assert provider.calls == []


def test_speech_timeout_does_not_retry(speech_client):
    # 준비
    client, provider = speech_client
    provider.delay = 0.1

    # 실행
    response = client.post("/internal/v1/speech/synthesize", json=command(remainingMilliseconds=1),
                           headers={"X-API-KEY": "synthetic-speech-key"})

    # 검증
    assert response.status_code == 504
    assert len(provider.calls) == 1


def test_speech_provider_error_is_safe(speech_client):
    # 준비
    client, provider = speech_client
    provider.failure = RuntimeError("synthetic-sensitive-provider-body")

    # 실행
    response = client.post("/internal/v1/speech/synthesize", json=command(),
                           headers={"X-API-KEY": "synthetic-speech-key"})

    # 검증
    assert response.status_code == 502
    assert response.json()["detail"]["retryable"] is False
    assert "synthetic-sensitive" not in response.text
    assert len(provider.calls) == 1


@pytest.mark.parametrize("status", [400, 401, 403, 408, 429, 500, 504])
def test_speech_preserves_provider_http_origin(speech_client, status):
    # 준비: 외부 요청 없이 SDK 상태만 있는 예외를 만든다.
    client, provider = speech_client
    provider.failure = RuntimeError("PRIVATE_SYNTHETIC_BODY")
    provider.failure.status_code = status

    # 실행: 범용 TTS는 한 번 실행한 기술 오류만 전달한다.
    response = client.post("/internal/v1/speech/synthesize", json=command(),
                           headers={"X-API-KEY": "synthetic-speech-key"})

    # 검증: LL이 원본 업무 정책을 복원할 수 있고 본문은 노출되지 않는다.
    assert response.json()["detail"]["providerStatus"] == status
    assert response.json()["detail"]["failureKind"] == "HTTP_STATUS"
    assert "PRIVATE_SYNTHETIC_BODY" not in response.text
    assert len(provider.calls) == 1


def test_speech_preserves_sdk_timeout_origin(speech_client):
    # 준비
    client, provider = speech_client
    provider.failure = APITimeoutError(request=httpx.Request("POST", "https://synthetic.invalid"))

    # 실행
    response = client.post("/internal/v1/speech/synthesize", json=command(),
                           headers={"X-API-KEY": "synthetic-speech-key"})

    # 검증: 기존 기술 코드와 SDK 출처를 함께 보존한다.
    assert response.json()["detail"]["failureKind"] == "SDK_TIMEOUT"
    assert len(provider.calls) == 1
