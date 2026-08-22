"""Shared speech-to-text model runtime and adapters."""

from app.features.speech_to_text.runtime import (
    FasterWhisperRuntime,
    InferencePriority,
    SpeechRuntimeClosed,
    SpeechRuntimeNotReady,
    SpeechRuntimeQueueFull,
    WhisperRuntimeResult,
    WhisperRuntimeSegment,
)

__all__ = [
    "FasterWhisperRuntime",
    "InferencePriority",
    "SpeechRuntimeClosed",
    "SpeechRuntimeNotReady",
    "SpeechRuntimeQueueFull",
    "WhisperRuntimeResult",
    "WhisperRuntimeSegment",
]
