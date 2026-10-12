from __future__ import annotations

import math

import numpy as np

from app.features.voice_translation.audio import AudioFrameValidator, normalize_voice_pcm
from app.schemas.voice_translation import VoiceAudioFormat


def validator(*, candidate_threshold: float | None = None) -> AudioFrameValidator:
    return AudioFrameValidator(
        VoiceAudioFormat(frame_duration_ms=100),
        minimum_frame_duration_ms=20,
        maximum_frame_duration_ms=200,
        speech_rms_threshold=0.012,
        speech_candidate_rms_threshold=candidate_threshold,
    )


def tone(amplitude: int, *, offset: int = 0) -> bytes:
    values = [
        round(offset + amplitude * math.sin(index * 2 * math.pi / 40))
        for index in range(1600)
    ]
    return np.array(values, dtype="<i2").tobytes()


def test_quiet_candidate_requires_explicit_guarded_path():
    # 준비: 종전 기준보다 낮지만 보완 기준보다 높은 파형을 사용한다.
    pcm = tone(100)

    # 실행
    previous = validator().validate(pcm)
    guarded = validator(candidate_threshold=0.0003).validate(pcm)

    # 검증: PCM과 측정값을 바꾸지 않고 후보 분류만 확장한다.
    assert not previous.is_speech
    assert guarded.is_speech
    assert guarded.data == pcm
    assert guarded.rms == previous.rms
    assert guarded.peak == previous.peak


def test_silence_and_subfloor_input_do_not_start_quiet_candidates():
    # 준비
    guarded = validator(candidate_threshold=0.0003)

    # 실행 / 검증
    assert not guarded.validate(bytes(3200)).is_speech
    assert not guarded.validate(tone(10)).is_speech


def test_normalization_bounds_gain_and_preserves_duration():
    # 준비
    pcm = tone(100)
    original = np.frombuffer(pcm, dtype="<i2").astype(np.float32)

    # 실행
    adjusted_bytes = normalize_voice_pcm(pcm)
    adjusted = np.frombuffer(adjusted_bytes, dtype="<i2").astype(np.float32)

    # 검증: 실제 증폭은 하되 12배 및 PCM 범위를 넘지 않는다.
    assert len(adjusted_bytes) == len(pcm)
    assert np.max(np.abs(adjusted)) == 12 * np.max(np.abs(original))
    assert np.max(np.abs(adjusted)) < 32767


def test_normalization_keeps_digital_silence_and_removes_dc():
    # 준비
    silence = bytes(3200)
    dc = np.full(1600, 100, dtype="<i2").tobytes()

    # 실행 / 검증
    assert normalize_voice_pcm(silence) == silence
    assert normalize_voice_pcm(dc) == silence
    assert normalize_voice_pcm(b"") == b""


def test_near_silence_is_not_amplified():
    # 준비
    pcm = tone(2)

    # 실행 / 검증
    assert normalize_voice_pcm(pcm) == pcm


def test_normalization_does_not_amplify_loud_audio_or_clip_impulses():
    # 준비: 큰 음성은 유지하고 작은 평균 음량 속 순간 피크는 포화되지 않게 한다.
    loud = tone(10000)
    impulse = np.zeros(1600, dtype="<i2")
    impulse[200] = 20000
    impulse[201] = -20000

    # 실행
    normalized_loud = normalize_voice_pcm(loud)
    normalized_impulse = np.frombuffer(normalize_voice_pcm(impulse.tobytes()), dtype="<i2")

    # 검증
    assert normalized_loud == loud
    assert np.max(np.abs(normalized_impulse.astype(np.int32))) <= round(32768 * 0.9)
    assert not np.any(normalized_impulse == 32767)
    assert not np.any(normalized_impulse == -32768)
