from __future__ import annotations

import base64
import hashlib
import json
import logging
from typing import Any

from fastapi import HTTPException
from openai import AsyncOpenAI

from app.ai.model_policy import get_model_name_for_task, get_task_model_policy
from app.ai.ports import (
    StructuredGenerationResult,
    VoiceReadingGenerationToken,
    VoiceTranslationGenerationResult,
)
from app.ai.prompt_registry import get_prompt_rule
from app.ai.providers.openai.schema import build_openai_text_config
from app.core.config import settings
from app.features.chat_translation.normalizer import normalize_chat_translation_result
from app.features.chat_translation.prompts import build_chat_translation_prompt
from app.features.voice_translation.prompts import build_voice_translation_prompt
from app.schemas.voice_translation import VoiceTranslationProviderPayload

logger = logging.getLogger(__name__)


class OpenAIProviderResponseError(RuntimeError):
    """Retryable provider-side response-format/incomplete-output failure."""

    status_code = 502


class OpenAIService:
    provider_name = "openai"

    def __init__(self) -> None:
        self._client: AsyncOpenAI | None = None

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=settings.OPENAI_API_KEY,
                timeout=settings.OPENAI_REQUEST_TIMEOUT_SECONDS,
                # Application services already own retry/idempotency policy. Avoid
                # hidden SDK retries that could multiply cost or duplicate work.
                max_retries=0,
            )
        return self._client

    @property
    def ready(self) -> bool:
        return bool(settings.OPENAI_API_KEY.strip())

    async def warm_up(self) -> None:
        if self.ready:
            _ = self.client

    async def shutdown(self) -> None:
        if self._client is None:
            return
        await self._client.close()
        self._client = None

    def model_name_for(self, type_name: str) -> str:
        return get_model_name_for_task(type_name)

    async def call(
        self,
        type_name: str,
        data: str,
        schema: dict | None = None,
    ) -> Any:
        result = await self.call_with_metadata(
            type_name=type_name,
            data=data,
            schema=schema,
        )
        return result.data

    async def call_with_metadata(
        self,
        type_name: str,
        data: str,
        schema: dict | None = None,
    ) -> StructuredGenerationResult:
        return await self._create_response(
            type_name=type_name,
            user_input=data,
            schema=schema,
        )

    async def call_with_image(
        self,
        type_name: str,
        prompt: str,
        image_bytes: bytes,
        mime_type: str,
        schema: dict | None = None,
    ) -> Any:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        image_url = f"data:{mime_type};base64,{encoded}"
        user_input = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": image_url,
                        # Receipts contain small text; prioritize extraction quality.
                        "detail": "high",
                    },
                ],
            }
        ]
        result = await self._create_response(
            type_name=type_name,
            user_input=user_input,
            schema=schema,
        )
        return result.data

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
        result = await self.call(
            type_name="CHAT_MESSAGE_TRANSLATION",
            data=prompt,
        )
        if not isinstance(result, str) or not result.strip():
            raise HTTPException(
                status_code=502,
                detail="채팅 메시지 번역 결과가 비어 있습니다.",
            )
        normalized = normalize_chat_translation_result(result)
        if not normalized:
            raise HTTPException(
                status_code=502,
                detail="채팅 메시지 번역 결과가 비어 있습니다.",
            )
        return normalized

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
            result = await self.call_with_metadata(
                type_name="VOICE_TRANSLATION",
                data=prompt,
                schema=schema,
            )
            payload = VoiceTranslationProviderPayload.model_validate(result.data)
            return VoiceTranslationGenerationResult(
                translated_text=payload.translated_text,
                source_reading_tokens=[
                    VoiceReadingGenerationToken(
                        surface=token.surface,
                        reading=token.reading,
                    )
                    for token in payload.source_reading_tokens
                ],
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                provider="openai",
                model=result.model,
            )
        except Exception:
            # Voice content and raw provider responses must never enter logs.
            logger.warning("Voice translation provider call failed")
            raise

    async def _create_response(
        self,
        *,
        type_name: str,
        user_input: Any,
        schema: dict | None,
    ) -> StructuredGenerationResult:
        rule = get_prompt_rule(type_name)
        if not rule:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid type: {type_name}",
            )

        policy = get_task_model_policy(type_name)
        model = get_model_name_for_task(type_name)
        text_config = build_openai_text_config(
            type_name=type_name,
            schema=schema,
            verbosity=policy.verbosity,
        )

        try:
            response = await self.client.responses.create(
                model=model,
                instructions=rule,
                input=user_input,
                reasoning={"effort": policy.reasoning_effort},
                max_output_tokens=policy.max_output_tokens,
                text=text_config,
                prompt_cache_key=self._prompt_cache_key(type_name, model),
                store=False,
            )
            status = str(getattr(response, "status", "completed") or "completed")
            if status != "completed":
                raise OpenAIProviderResponseError(
                    f"OpenAI response did not complete: status={status}"
                )

            output_text = response.output_text
            if not isinstance(output_text, str) or not output_text.strip():
                raise OpenAIProviderResponseError("OpenAI response has no output text")

            data: Any = output_text
            if schema is not None:
                try:
                    data = json.loads(output_text)
                except json.JSONDecodeError as exc:
                    raise OpenAIProviderResponseError(
                        "OpenAI structured response is not valid JSON"
                    ) from exc

            usage = getattr(response, "usage", None)
            return StructuredGenerationResult(
                data=data,
                input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
                output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
                provider="openai",
                model=str(getattr(response, "model", None) or model),
            )
        except Exception as exc:
            # Never log prompt data or provider raw response bodies here.
            logger.error(
                "OpenAI API call failed. type=%s model=%s errorType=%s",
                type_name,
                model,
                type(exc).__name__,
            )
            raise

    @staticmethod
    def _prompt_cache_key(type_name: str, model: str) -> str:
        # Stable per task/model bucket. The actual prefix still has to match, so this
        # cannot cross-contaminate unrelated prompts; it only improves cache locality.
        digest = hashlib.sha256(f"{type_name}|{model}".encode("utf-8")).hexdigest()[:24]
        return f"translacat:{digest}"
