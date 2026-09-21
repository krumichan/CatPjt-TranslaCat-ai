"""OpenAI Speech adapter for new audio; existing stored Gemini files remain untouched."""
from __future__ import annotations

import io
import wave

from openai import AsyncOpenAI

from app.ai.ports import SpeechSynthesisResult
from app.core.config import settings


OPENAI_SPEECH_POLICY_VERSION = "openai-speech-v1"
OPENAI_SPEECH_INSTRUCTIONS = (
    "Read the supplied text exactly in its original language. Do not translate, "
    "summarize, add, repeat, or omit words. Speak naturally and clearly."
)
_SUPPORTED_VOICES = frozenset({"marin", "cedar"})


class SpeechAudioDecodeError(ValueError):
    """Safe, typed invalid-audio diagnostic without response body or user text."""

    diagnostic_code = "OPENAI_SPEECH_WAV_INVALID"

    def __init__(self, *, byte_count: int, header_magic: str) -> None:
        self.byte_count = byte_count
        self.header_magic = header_magic
        super().__init__(self.diagnostic_code)


def validate_speech_wav(audio_bytes: bytes) -> float:
    """Measure a complete, uncompressed WAV; never trust a provider duration hint."""
    try:
        if len(audio_bytes) < 44 or audio_bytes[:4] != b"RIFF" or audio_bytes[8:12] != b"WAVE":
            raise ValueError("missing RIFF/WAVE header")
        riff_size = int.from_bytes(audio_bytes[4:8], "little")
        if riff_size != 0xFFFFFFFF and len(audio_bytes) < riff_size + 8:
            raise ValueError("truncated declared RIFF body")
        offset = 12
        data_size = None
        data_offset = None
        while offset + 8 <= len(audio_bytes):
            chunk_size = int.from_bytes(audio_bytes[offset + 4:offset + 8], "little")
            if audio_bytes[offset:offset + 4] == b"data":
                data_size, data_offset = chunk_size, offset + 8
                break
            if chunk_size == 0xFFFFFFFF:
                raise ValueError("unknown non-audio chunk length")
            offset += 8 + chunk_size + chunk_size % 2
        if data_size is None or data_offset is None:
            raise ValueError("missing WAV data chunk")
        with wave.open(io.BytesIO(audio_bytes), "rb") as source:
            channels = source.getnchannels()
            width = source.getsampwidth()
            rate = source.getframerate()
            frames = source.getnframes()
            if (source.getcomptype() != "NONE" or channels not in {1, 2}
                    or width not in {1, 2, 3, 4} or not 8000 <= rate <= 96000
                    or frames <= 0):
                raise ValueError("invalid PCM WAV format")
            decoded = source.readframes(frames)
            frame_bytes = channels * width
            if len(decoded) % frame_bytes:
                raise ValueError("incomplete PCM frame")
            if data_size == 0xFFFFFFFF:
                # OpenAI can return a fully downloaded streaming WAV whose RIFF
                # and data lengths remain sentinel values. The HTTP body is
                # complete, so measure its actual aligned PCM frames.
                if len(decoded) != len(audio_bytes) - data_offset:
                    raise ValueError("streaming WAV body mismatch")
                frames = len(decoded) // frame_bytes
            elif len(decoded) != frames * frame_bytes or len(decoded) != data_size:
                raise ValueError("truncated declared PCM data")
            return frames / rate
    except (EOFError, ValueError, wave.Error) as exc:
        raise SpeechAudioDecodeError(
            byte_count=len(audio_bytes), header_magic=audio_bytes[:4].hex(),
        ) from exc


def normalize_complete_speech_wav(audio_bytes: bytes) -> tuple[bytes, float]:
    """Replace streaming length sentinels with observed PCM length, not duration."""
    duration = validate_speech_wav(audio_bytes)
    offset = 12
    while offset + 8 <= len(audio_bytes):
        size = int.from_bytes(audio_bytes[offset + 4:offset + 8], "little")
        if audio_bytes[offset:offset + 4] == b"data":
            break
        offset += 8 + size + size % 2
    if (audio_bytes[4:8] != b"\xff" * 4
            and audio_bytes[offset + 4:offset + 8] != b"\xff" * 4):
        return audio_bytes, duration
    observed_size = len(audio_bytes) - offset - 8
    if len(audio_bytes) - 8 >= 0xFFFFFFFF or observed_size >= 0xFFFFFFFF:
        raise SpeechAudioDecodeError(byte_count=len(audio_bytes), header_magic="oversized")
    normalized = bytearray(audio_bytes)
    normalized[4:8] = (len(audio_bytes) - 8).to_bytes(4, "little")
    normalized[offset + 4:offset + 8] = observed_size.to_bytes(4, "little")
    return bytes(normalized), validate_speech_wav(bytes(normalized))


class OpenAISpeechService:
    provider_name = "openai"

    def __init__(self) -> None:
        self._client: AsyncOpenAI | None = None

    @property
    def ready(self) -> bool:
        return bool(settings.OPENAI_API_KEY.strip())

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=settings.OPENAI_API_KEY,
                timeout=settings.OPENAI_SPEECH_TIMEOUT_SECONDS,
                max_retries=0,
            )
        return self._client

    async def warm_up(self) -> None:
        if self.ready:
            _ = self.client

    async def shutdown(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def synthesize_speech(
        self, *, text: str, voice: str, language: str, speed: str,
    ) -> SpeechSynthesisResult:
        # In particular, never send a Gemini voice name to OpenAI or disguise
        # a marin recording as Kore in downstream metadata.
        if voice not in _SUPPORTED_VOICES:
            raise ValueError("Unsupported OpenAI speech voice")
        if not text.strip() or len(text) > 4096:
            raise ValueError("Speech input must contain 1..4096 characters")
        # UTF-8 bytes conservatively bound ordinary input tokenization. Reject
        # rather than silently truncate a long Level Test reference passage.
        if len((text + OPENAI_SPEECH_INSTRUCTIONS).encode("utf-8")) > 2000:
            raise ValueError("Speech input exceeds conservative 2000-token limit")
        if speed not in {"NORMAL", "SLOW"}:
            raise ValueError("Unsupported speech playback speed")
        if not language.strip():
            raise ValueError("Speech language is required")
        response = await self.client.audio.speech.create(
            model=settings.OPENAI_SPEECH_MODEL,
            voice=voice,
            input=text,
            instructions=OPENAI_SPEECH_INSTRUCTIONS,
            response_format="wav",
            speed=1.0 if speed == "NORMAL" else 0.8,
        )
        audio_bytes = await response.aread()
        audio_bytes, duration_seconds = normalize_complete_speech_wav(audio_bytes)
        return SpeechSynthesisResult(
            audio_bytes=audio_bytes,
            content_type="audio/wav",
            provider="openai",
            model=settings.OPENAI_SPEECH_MODEL,
            duration_seconds=duration_seconds,
        )
