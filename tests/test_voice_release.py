import io
import json
from unittest.mock import patch

import pytest

from scripts.check_voice_release import verify_voice_idle, verify_voice_release


@pytest.mark.parametrize("state,accepted", [
    ({"activeStreams": 0}, True),
    ({"activeStreams": 1}, False),
    ({"activeStreams": -1}, False),
    ({"activeStreams": False}, False),
    ({"activeStreams": "0"}, False),
    ({}, False),
    ([], False),
])
def test_pre_stop_check_requires_verified_zero_streams(monkeypatch, state, accepted):
    # 준비: 이전 CPU 이미지의 복구도 가능하되 진행 중인 연결은 보호한다.
    monkeypatch.setenv("SERVER_API_KEY", "synthetic-service-key")
    monkeypatch.setenv("AI_VOICE_STT_PROVIDER", "faster-whisper")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with patch(
        "scripts.check_voice_release.urlopen", return_value=io.BytesIO(json.dumps(state).encode())
    ) as request:
        # 실행 / 검증
        if accepted:
            verify_voice_idle()
        else:
            with pytest.raises(RuntimeError):
                verify_voice_idle()
        assert request.call_args.args[0].get_header("X-api-key") == "synthetic-service-key"


def test_pre_stop_check_does_not_treat_unavailable_readiness_as_idle(monkeypatch):
    # 준비
    monkeypatch.setenv("SERVER_API_KEY", "synthetic-service-key")
    with patch("scripts.check_voice_release.urlopen", side_effect=TimeoutError):
        # 실행 / 검증
        with pytest.raises(TimeoutError):
            verify_voice_idle()


@pytest.fixture(autouse=True)
def production_preview(monkeypatch):
    monkeypatch.setenv("AI_VOICE_REALTIME_PREVIEW_ENABLED", "true")


@pytest.mark.parametrize("enabled", ["", "false"])
def test_release_rejects_missing_preview_without_http(monkeypatch, enabled):
    # 준비
    monkeypatch.setenv("AI_VOICE_STT_PROVIDER", "openai")
    monkeypatch.setenv("AI_VOICE_REALTIME_PREVIEW_ENABLED", enabled)
    with patch("scripts.check_voice_release.urlopen") as request:
        # 실행 / 검증
        with pytest.raises(RuntimeError):
            verify_voice_release()
        request.assert_not_called()


@pytest.mark.parametrize("provider", ["", "faster-whisper", "openai-transcribe"])
def test_release_rejects_unapproved_provider_without_http(monkeypatch, provider):
    # 준비
    monkeypatch.setenv("AI_VOICE_STT_PROVIDER", provider)
    with patch("scripts.check_voice_release.urlopen") as request:
        # 실행 / 검증
        with pytest.raises(RuntimeError):
            verify_voice_release()
        request.assert_not_called()


@pytest.mark.parametrize(
    "state,accepted",
    [
        ({"ready": True, "acceptingStreams": True, "sttModel": "whisper-1@openai"}, True),
        ({"ready": True, "acceptingStreams": True, "sttModel": "faster-whisper"}, False),
        ({"ready": False, "acceptingStreams": True, "sttModel": "whisper-1@openai"}, False),
        ({"ready": True, "acceptingStreams": False, "sttModel": "whisper-1@openai"}, False),
        ({"ready": "true", "acceptingStreams": True, "sttModel": "whisper-1@openai"}, False),
        ([], False),
    ],
)
def test_release_verifies_effective_authenticated_readiness(monkeypatch, state, accepted):
    # 준비
    monkeypatch.setenv("AI_VOICE_STT_PROVIDER", "openai")
    monkeypatch.setenv("SERVER_API_KEY", "synthetic-service-key")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-provider-key")
    with patch(
        "scripts.check_voice_release.urlopen", return_value=io.BytesIO(json.dumps(state).encode())
    ) as request:
        # 실행 / 검증
        if accepted:
            verify_voice_release()
        else:
            with pytest.raises(RuntimeError):
                verify_voice_release()
        args, kwargs = request.call_args
        assert args[0].full_url == "http://127.0.0.1:8000/internal/v1/voice/readiness"
        assert args[0].get_header("X-api-key") == "synthetic-service-key"
        assert kwargs["timeout"] == 3


@pytest.mark.parametrize("missing", ["SERVER_API_KEY", "OPENAI_API_KEY"])
def test_release_missing_credentials_does_not_send_request(monkeypatch, missing):
    # 준비
    for name in ("SERVER_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(name, "synthetic")
    monkeypatch.setenv("AI_VOICE_STT_PROVIDER", "openai")
    monkeypatch.delenv(missing)
    with patch("scripts.check_voice_release.urlopen") as request:
        # 실행 / 검증
        with pytest.raises(RuntimeError):
            verify_voice_release()
        request.assert_not_called()
