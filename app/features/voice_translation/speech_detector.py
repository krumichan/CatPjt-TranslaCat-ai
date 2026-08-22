from __future__ import annotations

import asyncio
from typing import Protocol

from app.core.config import settings


class VoiceSpeechEvidenceGuard(Protocol):
    @property
    def ready(self) -> bool: ...

    @property
    def model_version(self) -> str: ...

    async def has_speech(self, pcm_bytes: bytes) -> bool: ...


class PassThroughSpeechEvidenceGuard:
    ready = True
    model_version = "disabled"

    async def has_speech(self, pcm_bytes: bytes) -> bool:
        return bool(pcm_bytes)


class SileroSpeechEvidenceGuard:
    """Warm Silero confirmation guard for music/noise hallucination reduction."""

    def __init__(self) -> None:
        self.enabled = settings.AI_VOICE_VAD_SILERO_GUARD_ENABLED
        self._ready = not self.enabled
        self._warm_up_lock = asyncio.Lock()
        self._inference_semaphore = asyncio.Semaphore(1)

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def model_version(self) -> str:
        return "silero-vad:faster-whisper"

    async def warm_up(self) -> None:
        if not self.enabled or self._ready:
            return
        async with self._warm_up_lock:
            if self._ready:
                return
            await asyncio.to_thread(self._warm_up_sync)
            self._ready = True

    async def has_speech(self, pcm_bytes: bytes) -> bool:
        if not self.enabled:
            return bool(pcm_bytes)
        if not self._ready:
            raise RuntimeError("Silero speech evidence guard is not ready")
        async with self._inference_semaphore:
            return await asyncio.to_thread(self._has_speech_sync, pcm_bytes)

    async def shutdown(self) -> None:
        self._ready = not self.enabled

    @staticmethod
    def _warm_up_sync() -> None:
        import numpy as np
        from faster_whisper.vad import get_vad_model

        get_vad_model()(np.zeros(512, dtype=np.float32))

    @staticmethod
    def _has_speech_sync(pcm_bytes: bytes) -> bool:
        import numpy as np
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        audio = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32) / 32768.0
        timestamps = get_speech_timestamps(
            audio,
            VadOptions(
                threshold=settings.AI_VOICE_VAD_SILERO_THRESHOLD,
                min_speech_duration_ms=(settings.AI_VOICE_VAD_SILERO_MIN_SPEECH_MS),
                min_silence_duration_ms=100,
                speech_pad_ms=0,
            ),
            sampling_rate=16_000,
        )
        return bool(timestamps)
