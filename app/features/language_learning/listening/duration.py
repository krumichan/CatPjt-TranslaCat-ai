from __future__ import annotations

import io
import math
import wave
from dataclasses import dataclass

from app.features.language_learning.listening.errors import ListeningStageException
from app.features.language_learning.listening.policy import DIFFICULTY_DURATION_RANGES
from app.schemas.language_learning_listening import (
    ListeningDurationDemand,
    ListeningErrorCode,
    ListeningSetGenerationRequest,
    ListeningStage,
)


LISTENING_DURATION_POLICY_VERSION = "listening-audio-duration-v1"


def minimum_short_correction_characters(request: ListeningSetGenerationRequest) -> int | None:
    """A pre-TTS planning floor, never the actual WAV duration acceptance gate.

    A bounded correction of measured short audio must at least grow enough to
    cross the existing minimum at the *previously observed* speech rate. This
    rejects mere paraphrases such as the observed 23 -> 24 character repair.
    TTS still measures and decides the new audio independently.
    """
    correction = request.duration_correction
    if correction is None or correction.previous_measured_seconds <= 0:
        return None
    minimum = duration_demand(request).min_seconds
    if correction.previous_measured_seconds >= minimum:
        return None
    # Count the text as submitted to TTS. Answer normalization removes Japanese
    # punctuation, but that punctuation is present in the measured speech input.
    previous_chars = len(correction.previous_source_text.strip())
    return max(
        previous_chars + 1,
        math.ceil(previous_chars * minimum / correction.previous_measured_seconds),
    )


def duration_demand(request: ListeningSetGenerationRequest) -> ListeningDurationDemand:
    low, high = DIFFICULTY_DURATION_RANGES[request.set_context.difficulty.value]
    low = max(low, request.constraints.audio_seconds_min or low)
    high = min(high, request.constraints.audio_seconds_max or high)
    if low > high:
        raise ListeningStageException(
            ListeningErrorCode.INVALID_REQUEST, ListeningStage.GENERATION,
            "요청 Duration 범위와 Difficulty 정책 범위가 겹치지 않습니다.", False,
        )
    return ListeningDurationDemand(
        min_seconds=low, max_seconds=high,
        policy_version=LISTENING_DURATION_POLICY_VERSION,
    )


def generation_duration_guidance(request: ListeningSetGenerationRequest) -> dict:
    demand = duration_demand(request)
    target = (demand.min_seconds + demand.max_seconds) / 2
    guidance: dict = {
        "durationDemand": demand.model_dump(mode="json", by_alias=True),
        "interiorTargetSeconds": target,
        "estimatedAudioSecondsIsDiagnosticOnly": True,
        "acceptanceAuthority": "decoded NORMAL audio frames, not text length or estimated seconds",
    }
    correction_floor = minimum_short_correction_characters(request)
    if correction_floor is not None:
        guidance["durationCorrectionPlanning"] = {
            "direction": "EXPAND_MEANINGFUL_CONTENT",
            "minimumSourceCharacters": correction_floor,
            "basis": "previous measured NORMAL audio rate and unchanged minimum duration",
            "notAudioAcceptance": True,
        }
    voice = request.reference_voice
    if (request.user_context.learning_language.split("-")[0].lower() == "ja"
            and voice is not None and voice.voice_key == "marin"):
        # Three fixed Japanese/OpenAI/marin NORMAL samples (2026-09-20):
        # 28/5.50, 29/6.35, and 36/5.30 characters/second. This is a small
        # planning sample, not a prediction or an audio acceptance threshold.
        observed = (28 / 5.5, 29 / 6.35, 36 / 5.3)
        mean = sum(observed) / len(observed)
        guidance["empiricalPlanningReference"] = {
            "language": "ja", "provider": "openai", "voice": "marin",
            "model": "gpt-4o-mini-tts-2025-12-15", "playbackSpeed": "NORMAL",
            "sampleCount": len(observed),
            "observedCharactersPerSecond": {
                "min": round(min(observed), 3), "max": round(max(observed), 3),
                "mean": round(mean, 3),
            },
            "approximateTargetCharacters": round(target * mean),
            "scope": "small planning sample only; decoded NORMAL WAV remains the duration authority",
        }
    elif (request.user_context.learning_language.split("-")[0].lower() == "ja"
            and voice is not None and voice.voice_key == "Kore"):
        # 30 synthetic QA clips (2026-09-19), ja/Gemini/Kore/NORMAL:
        # observed 4.813..6.665 Unicode characters/sec, mean 5.894.
        # This is planning evidence for this lane only, never a duration validator.
        guidance["empiricalPlanningReference"] = {
            "language": "ja", "provider": "gemini", "voice": "Kore",
            "playbackSpeed": "NORMAL", "sampleCount": 30,
            "observedCharactersPerSecond": {"min": 4.813, "max": 6.665, "mean": 5.894},
            "approximateTargetCharacters": round(target * 5.894),
            "scope": "planning reference only; other languages/providers/voices are not calibrated",
        }
    return guidance


@dataclass(frozen=True)
class DecodedReferenceAudio:
    duration_seconds: float
    sample_rate: int
    channels: int
    frame_count: int


def decode_reference_audio(audio_bytes: bytes, content_type: str) -> DecodedReferenceAudio:
    """Decode the complete PCM WAV supplied by the current reference TTS provider."""
    try:
        if content_type.split(";", 1)[0].lower() not in {
            "audio/wav", "audio/x-wav", "audio/vnd.wave", "wav", "wave",
        }:
            raise ValueError("unsupported reference codec")
        with wave.open(io.BytesIO(audio_bytes), "rb") as audio:
            channels, width, rate, frames = (
                audio.getnchannels(), audio.getsampwidth(),
                audio.getframerate(), audio.getnframes(),
            )
            if (audio.getcomptype() != "NONE" or channels not in {1, 2}
                    or width not in {1, 2, 3, 4} or not 8000 <= rate <= 96000
                    or frames <= 0):
                raise ValueError("invalid reference PCM format")
            if len(audio.readframes(frames)) != frames * channels * width:
                raise ValueError("truncated reference PCM frames")
        return DecodedReferenceAudio(frames / rate, rate, channels, frames)
    except (ValueError, EOFError, wave.Error) as exc:
        raise ListeningStageException(
            ListeningErrorCode.AUDIO_DECODE_FAILED, ListeningStage.TTS,
            "Reference 음성을 PCM WAV로 완전히 해독할 수 없습니다.", False,
        ) from exc


def validate_actual_duration(
    decoded: DecodedReferenceAudio, demand: ListeningDurationDemand | None,
) -> None:
    # Legacy callers receive format-ready audio, explicitly NOT duration-validated.
    if demand is None:
        return
    seconds = decoded.duration_seconds
    if demand.min_seconds <= seconds <= demand.max_seconds:
        return
    code = (ListeningErrorCode.AUDIO_TOO_SHORT if seconds < demand.min_seconds
            else ListeningErrorCode.AUDIO_TOO_LONG)
    raise ListeningStageException(
        code, ListeningStage.TTS, "실측 Reference 음성 길이가 학습 범위를 벗어났습니다.",
        False, {"measuredSeconds": seconds, "minSeconds": demand.min_seconds,
                "maxSeconds": demand.max_seconds, "durationPolicyVersion": demand.policy_version},
    )
