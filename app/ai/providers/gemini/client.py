import base64
import io
import logging
import wave
from typing import Any

from fastapi import HTTPException
from google import genai
from google.genai import types

from app.ai.ports import (
    SpeechSynthesisResult,
    StructuredGenerationResult,
    VoiceReadingGenerationToken,
    VoiceTranslationGenerationResult,
)
from app.ai.providers.gemini.config_manager import GeminiConfigManager
from app.core.config import settings
from app.features.chat_translation.normalizer import normalize_chat_translation_result
from app.features.chat_translation.prompts import build_chat_translation_prompt
from app.features.voice_translation.prompts import build_voice_translation_prompt
from app.schemas.voice_translation import VoiceTranslationProviderPayload

logger = logging.getLogger(__name__)


class GeminiService:
    def __init__(self) -> None:
        self._client = None
        self.model_name = settings.GEMINI_MODEL_NAME
        self.config_manager = GeminiConfigManager()

    @property
    def client(self):
        if self._client is None:
            self._client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        return self._client

    @property
    def ready(self) -> bool:
        return bool(settings.GOOGLE_API_KEY.strip())

    async def warm_up(self) -> None:
        if self.ready:
            _ = self.client

    async def shutdown(self) -> None:
        if self._client is None:
            return
        await self._client.aio.aclose()
        self._client.close()
        self._client = None

    async def call(
        self,
        type_name: str,
        data: str,
        schema: dict | None = None,
    ) -> Any:
        try:
            config = self.config_manager.get_cached_config(
                type_name=type_name,
                schema=schema,
            )

            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=data,
                config=config,
            )

            return response.parsed if schema else response.text
        except Exception as exc:
            logger.error("Gemini API Call Error: %s", exc)
            raise

    async def call_with_metadata(
        self,
        type_name: str,
        data: str,
        schema: dict | None = None,
    ) -> StructuredGenerationResult:
        try:
            config = self.config_manager.get_cached_config(
                type_name=type_name,
                schema=schema,
            )
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=data,
                config=config,
            )
            usage = getattr(response, "usage_metadata", None)
            return StructuredGenerationResult(
                data=response.parsed if schema else response.text,
                input_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
                output_tokens=int(getattr(usage, "candidates_token_count", 0) or 0),
                provider="gemini",
                model=self.model_name,
            )
        except Exception as exc:
            logger.error("Gemini API Metadata Call Error: %s", exc)
            raise

    async def synthesize_speech(
        self,
        *,
        text: str,
        voice: str,
        language: str,
        speed: str,
    ) -> SpeechSynthesisResult:
        instruction = self._build_tts_instruction(
            text=text,
            language=language,
            speed=speed,
        )

        try:
            response = await self.client.aio.models.generate_content(
                model=settings.GEMINI_TTS_MODEL_NAME,
                contents=instruction,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=voice,
                            )
                        )
                    ),
                ),
            )
            candidates = getattr(response, "candidates", None) or []
            if not candidates:
                raise ValueError("Gemini TTS response has no candidate")
            content = getattr(candidates[0], "content", None)
            parts = getattr(content, "parts", None) or []
            if not parts:
                raise ValueError("Gemini TTS response has no audio part")
            inline_data = getattr(parts[0], "inline_data", None)
            raw_audio = getattr(inline_data, "data", None)
            if not raw_audio:
                raise ValueError("Gemini TTS audio output is empty")

            if isinstance(raw_audio, str):
                pcm = base64.b64decode(raw_audio)
            else:
                pcm = bytes(raw_audio)
            wav_bytes = self._pcm_to_wav(pcm)
            return SpeechSynthesisResult(
                audio_bytes=wav_bytes,
                content_type="audio/wav",
                provider="gemini",
                model=settings.GEMINI_TTS_MODEL_NAME,
                duration_seconds=len(pcm) / (24_000 * 2),
            )
        except Exception as exc:
            logger.error("Gemini TTS API Call Error: %s", exc)
            raise

    @staticmethod
    def _build_tts_instruction(*, text: str, language: str, speed: str) -> str:
        pace = "slow and clear" if speed == "SLOW" else "natural conversational"
        return (
            f"Speak the following {language} text exactly. "
            f"Use a {pace} pace suitable for a language learner. "
            "Do not add, remove, translate, or paraphrase any words.\n\n"
            f"{text}"
        )

    @staticmethod
    def _pcm_to_wav(pcm: bytes) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(24_000)
            writer.writeframes(pcm)
        return buffer.getvalue()

    async def call_with_image(
        self,
        type_name: str,
        prompt: str,
        image_bytes: bytes,
        mime_type: str,
        schema: dict | None = None,
    ) -> Any:
        try:
            config = self.config_manager.get_cached_config(
                type_name=type_name,
                schema=schema,
            )

            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type=mime_type,
                    ),
                    prompt,
                ],
                config=config,
            )

            return response.parsed if schema else response.text
        except Exception as exc:
            logger.error("Gemini Vision API Call Error: %s", exc)
            raise

    async def translate_chat_message(
        self,
        text: str,
        target_language_code: str,
        source_language_code: str | None = None,
    ) -> str:
        prompt = build_chat_translation_prompt(
            text=text,
            target_language_code=target_language_code,
            source_language_code=source_language_code,
        )

        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=self.config_manager.get_chat_translation_fast_config(),
        )

        result = response.text

        if not isinstance(result, str) or not result.strip():
            raise HTTPException(
                status_code=502,
                detail="채팅 메시지 번역 결과가 비어 있습니다.",
            )

        normalized_result = normalize_chat_translation_result(result)

        if not normalized_result:
            raise HTTPException(
                status_code=502,
                detail="채팅 메시지 번역 결과가 비어 있습니다.",
            )

        return normalized_result

    async def translate_voice_utterance(
        self,
        *,
        source_text: str,
        source_language: str,
        target_language: str,
    ) -> VoiceTranslationGenerationResult:
        prompt = build_voice_translation_prompt(
            source_text=source_text,
            source_language=source_language,
            target_language=target_language,
        )
        schema = VoiceTranslationProviderPayload.model_json_schema(by_alias=True)

        try:
            response = await self.client.aio.models.generate_content(
                model=settings.AI_VOICE_TRANSLATION_MODEL_NAME,
                contents=prompt,
                config=self.config_manager.get_voice_translation_config(schema),
            )
            payload = VoiceTranslationProviderPayload.model_validate(response.parsed)
            usage = getattr(response, "usage_metadata", None)
            return VoiceTranslationGenerationResult(
                translated_text=payload.translated_text,
                source_reading_tokens=[
                    VoiceReadingGenerationToken(
                        surface=token.surface,
                        reading=token.reading,
                    )
                    for token in payload.source_reading_tokens
                ],
                input_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
                output_tokens=int(getattr(usage, "candidates_token_count", 0) or 0),
                provider="gemini",
                model=settings.AI_VOICE_TRANSLATION_MODEL_NAME,
            )
        except Exception:
            # Voice content and provider raw responses must never enter logs.
            logger.warning("Voice translation provider call failed")
            raise
