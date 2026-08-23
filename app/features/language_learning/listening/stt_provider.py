from __future__ import annotations

import io

from app.core.config import settings
from app.features.language_learning.speaking.stt_service import (
    SttProviderResult,
    SttProviderSegment,
)
from app.features.speech_to_text import FasterWhisperRuntime, InferencePriority


class FasterWhisperListeningSttProvider:
    """Detects the spoken language instead of forcing the expected locale."""

    def __init__(self, runtime: FasterWhisperRuntime | None = None) -> None:
        self.runtime = runtime or FasterWhisperRuntime(
            model_name=settings.AI_LISTENING_STT_MODEL_NAME,
            model_revision="",
            device=settings.AI_LISTENING_STT_DEVICE,
            compute_type=settings.AI_LISTENING_STT_COMPUTE_TYPE,
        )

    async def transcribe(
        self,
        wav_bytes: bytes,
        *,
        language: str,
        phrase_hints: list[str] | None = None,
    ) -> SttProviderResult:
        if not self.runtime.ready:
            await self.runtime.warm_up()
        result = await self.runtime.transcribe(
            io.BytesIO(wav_bytes),
            options={
                "beam_size": 1,
                "language": None,
                "initial_prompt": ", ".join(phrase_hints[:20])
                if phrase_hints
                else None,
                "vad_filter": True,
                "vad_parameters": {"min_silence_duration_ms": 500},
                "condition_on_previous_text": False,
            },
            priority=InferencePriority.STANDARD,
        )
        return SttProviderResult(
            text=result.text,
            language=result.language or language,
            language_probability=result.language_probability or 0.0,
            segments=[
                SttProviderSegment(
                    start_seconds=segment.start_seconds,
                    end_seconds=segment.end_seconds,
                    text=segment.text,
                    avg_logprob=segment.avg_logprob,
                )
                for segment in result.segments
            ],
            provider=result.provider,
            model=result.model,
            model_version=result.model_version,
        )
