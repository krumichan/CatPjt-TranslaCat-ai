from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class StructuredGenerationResult:
    data: Any
    input_tokens: int = 0
    output_tokens: int = 0
    provider: str = "unknown"
    model: str = "unknown"


@dataclass(frozen=True)
class SpeechSynthesisResult:
    audio_bytes: bytes
    content_type: str
    provider: str
    model: str
    duration_seconds: float | None = None


@dataclass(frozen=True)
class VoiceReadingGenerationToken:
    surface: str
    reading: str


@dataclass(frozen=True)
class VoiceTranslationGenerationResult:
    translated_text: str
    source_reading_tokens: list[VoiceReadingGenerationToken]
    input_tokens: int = 0
    output_tokens: int = 0
    provider: str = "unknown"
    model: str = "unknown"


class TextGenerationProvider(Protocol):
    async def call(
        self,
        type_name: str,
        data: str,
        schema: dict | None = None,
    ) -> Any: ...

    async def call_with_image(
        self,
        type_name: str,
        prompt: str,
        image_bytes: bytes,
        mime_type: str,
        schema: dict | None = None,
    ) -> Any: ...


class StructuredTextGenerationProvider(Protocol):
    async def call_with_metadata(
        self,
        type_name: str,
        data: str,
        schema: dict | None = None,
    ) -> StructuredGenerationResult: ...


class SpeechSynthesisProvider(Protocol):
    async def synthesize_speech(
        self,
        *,
        text: str,
        voice: str,
        language: str,
        speed: str,
    ) -> SpeechSynthesisResult: ...


class ChatTranslationProvider(Protocol):
    async def translate_chat_message(
        self,
        text: str,
        target_language_code: str,
        source_language_code: str | None = None,
    ) -> str: ...


class VoiceTranslationProvider(Protocol):
    async def translate_voice_utterance(
        self,
        *,
        source_text: str,
        source_language: str,
        target_language: str,
    ) -> VoiceTranslationGenerationResult: ...
