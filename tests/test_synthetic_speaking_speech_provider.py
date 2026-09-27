import json

import pytest

from scripts import synthetic_speech_provider as fixture


@pytest.fixture
def synthetic(monkeypatch, tmp_path):
    # 준비: 제어 파일과 통계는 테스트별 임시 디렉터리로 격리한다.
    monkeypatch.setenv("TRANSLACAT_TEST_MODEL_EXECUTION", "1")
    monkeypatch.setattr(fixture, "ROOT", tmp_path)
    directory = tmp_path
    return fixture.SyntheticSpeechProvider(), directory


@pytest.mark.asyncio
async def test_explicit_control_changes_only_provider_output_and_counts(synthetic):
    # 준비
    provider, directory = synthetic
    (directory / "speaking-control.json").write_text(
        json.dumps({"speechEnabled": True, "practiceMode": "READ_ALOUD"}), encoding="utf-8"
    )

    # 실행
    transcript = await provider.transcribe(b"synthetic")
    audio = await provider.synthesize_speech(
        text="Synthetic text", voice="marin", language="en", speed="NORMAL"
    )

    # 검증: 통계에는 단계와 숫자만 남기고 합성 PCM은 실제 음성 HTTP로 전송할 수 있다.
    assert transcript.text == "I went to the park yesterday."
    assert audio.audio_bytes[:4] == b"RIFF"
    stats = json.loads((directory / "speaking-speech-stats.json").read_text())
    assert stats == {"STT": 1, "TTS": 1}


@pytest.mark.asyncio
async def test_stage_failure_does_not_change_other_speech_stage(synthetic):
    # 준비
    provider, directory = synthetic
    (directory / "speaking-control.json").write_text(
        json.dumps({"speechEnabled": True, "scenario": "tts_failure", "stage": "TTS"}),
        encoding="utf-8",
    )

    # 실행
    transcript = await provider.transcribe(b"synthetic")
    with pytest.raises(RuntimeError) as failure:
        await provider.synthesize_speech(
            text="Synthetic text", voice="marin", language="en", speed="NORMAL"
        )

    # 검증
    assert transcript.text.startswith("Yesterday")
    assert failure.value.status_code == 503


@pytest.mark.asyncio
async def test_disabled_control_preserves_other_feature_fixture(synthetic):
    # 준비
    provider, directory = synthetic
    (directory / "speaking-control.json").write_text('{"scenario":"stt_failure"}', encoding="utf-8")

    # 실행
    transcript = await provider.transcribe(b"synthetic")

    # 검증
    assert transcript.text == "Synthetic transcript"
    assert not (directory / "speaking-speech-stats.json").exists()
