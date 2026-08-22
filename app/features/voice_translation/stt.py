from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.core.config import settings
from app.features.speech_to_text import (
    FasterWhisperRuntime,
    InferencePriority,
    SpeechRuntimeClosed,
    SpeechRuntimeNotReady,
    SpeechRuntimeQueueFull,
)
from app.features.voice_translation.errors import VoicePipelineException
from app.schemas.voice_translation import VoiceErrorCode, VoiceStage


@dataclass(frozen=True)
class VoiceSttResult:
    text: str
    language: str | None
    language_confidence: float | None
    no_speech_probability: float | None
    provider: str
    model: str
    model_version: str | None = None


class VoiceSttProvider(Protocol):
    @property
    def ready(self) -> bool: ...

    @property
    def model_version(self) -> str: ...

    async def transcribe_pcm(
        self,
        pcm_bytes: bytes,
        *,
        language: str | None,
        is_final: bool,
        initial_prompt: str | None = None,
    ) -> VoiceSttResult: ...


class FasterWhisperVoiceSttProvider:
    def __init__(self, runtime: FasterWhisperRuntime) -> None:
        self.runtime = runtime

    @property
    def ready(self) -> bool:
        return self.runtime.ready

    @property
    def model_version(self) -> str:
        return self.runtime.model_version

    async def transcribe_pcm(
        self,
        pcm_bytes: bytes,
        *,
        language: str | None,
        is_final: bool,
        initial_prompt: str | None = None,
    ) -> VoiceSttResult:
        import numpy as np

        audio = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32) / 32768.0
        options = {
            "language": language,
            "initial_prompt": initial_prompt or None,
            "beam_size": 1,
            "temperature": 0,
            "vad_filter": False,
            "condition_on_previous_text": False,
            "word_timestamps": False,
        }
        try:
            result = await self.runtime.transcribe(
                audio,
                options=options,
                priority=(
                    InferencePriority.FINAL if is_final else InferencePriority.PARTIAL
                ),
            )
        except SpeechRuntimeQueueFull as exc:
            raise VoicePipelineException(
                code=VoiceErrorCode.BACKPRESSURE,
                stage=VoiceStage.STT,
                message="STT 추론 Queue가 가득 찼습니다.",
                retryable=True,
            ) from exc
        except (SpeechRuntimeNotReady, SpeechRuntimeClosed) as exc:
            raise VoicePipelineException(
                code=VoiceErrorCode.MODEL_NOT_READY,
                stage=VoiceStage.RUNTIME,
                message="STT Model이 요청을 받을 준비가 되지 않았습니다.",
                retryable=True,
            ) from exc

        no_speech_values = [
            segment.no_speech_probability
            for segment in result.segments
            if segment.no_speech_probability is not None
        ]
        no_speech_probability = (
            sum(no_speech_values) / len(no_speech_values) if no_speech_values else None
        )
        if no_speech_probability is not None:
            no_speech_probability = max(0.0, min(1.0, no_speech_probability))
        text = result.text.strip()
        if (
            no_speech_probability is not None
            and no_speech_probability
            >= settings.AI_VOICE_STT_NO_SPEECH_PROBABILITY_THRESHOLD
        ):
            text = ""

        return VoiceSttResult(
            text=text,
            language=result.language,
            language_confidence=result.language_probability,
            no_speech_probability=no_speech_probability,
            provider=result.provider,
            model=result.model,
            model_version=result.model_version,
        )
