from __future__ import annotations

import base64
import hashlib
import logging
from typing import Any, cast

from fastapi import HTTPException
from openai import AsyncOpenAI
from openai.types.responses.response_input_param import ResponseInputParam
from openai.types.responses.response_text_config_param import ResponseTextConfigParam
from openai.types.shared_params.reasoning import Reasoning

from app.ai.model_policy import (
    AiModelTier,
    get_model_name_for_task,
    get_model_name_for_tier,
    get_task_model_policy,
)
from app.ai.ports import (
    StructuredGenerationResult,
    VoiceReadingGenerationToken,
    VoiceTranslationGenerationResult,
)
from app.ai.prompt_registry import get_prompt_rule
from app.ai.providers.openai.error_diagnostics import safe_openai_error_metadata
from app.ai.providers.openai.response import (
    OpenAIProviderResponseError as OpenAIProviderResponseError,
)
from app.ai.providers.openai.response import (
    decode_response,
)
from app.ai.providers.openai.schema import build_explicit_text_config, build_openai_text_config
from app.core.config import settings
from app.features.chat_translation.normalizer import normalize_chat_translation_result
from app.features.chat_translation.prompts import build_chat_translation_prompt
from app.features.voice_translation.prompts import build_voice_translation_prompt
from app.schemas.voice_translation import VoiceTranslationProviderPayload

logger = logging.getLogger(__name__)


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

    async def execute_explicit(
        self,
        *,
        instructions: str,
        messages: list[dict[str, str]],
        tier: AiModelTier,
        reasoning_effort: str,
        verbosity: str,
        max_output_tokens: int,
        remaining_milliseconds: int,
        response_schema: dict | None,
        schema_name_value: str | None,
        strict: bool,
    ) -> StructuredGenerationResult:
        """범용 모델 실행. task/prompt registry나 업무 validator를 참조하지 않는다."""
        model = get_model_name_for_tier(tier)
        text_config = build_explicit_text_config(
            schema=response_schema, schema_name_value=schema_name_value,
            strict=strict, verbosity=verbosity,
        )
        # LL이 전달한 남은 전체 deadline을 줄이는 방향으로만 적용한다.
        provider_timeout = (
            max(85.0, settings.OPENAI_REQUEST_TIMEOUT_SECONDS)
            if tier == AiModelTier.SOL else settings.OPENAI_REQUEST_TIMEOUT_SECONDS
        )
        timeout = min(remaining_milliseconds / 1000.0, provider_timeout)
        try:
            response = await self.client.with_options(timeout=timeout).responses.create(
                model=model,
                instructions=instructions,
                input=cast(ResponseInputParam, messages),
                reasoning=cast(Reasoning, {"effort": reasoning_effort}),
                max_output_tokens=max_output_tokens,
                text=cast(ResponseTextConfigParam, text_config),
                store=False,
            )
            usage = getattr(response, "usage", None)
            return StructuredGenerationResult(
                data=decode_response(response, structured=response_schema is not None),
                input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
                output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
                provider="openai",
                model=str(getattr(response, "model", None) or model),
            )
        except Exception as exc:
            diagnostic = safe_openai_error_metadata(exc, schema=response_schema)
            logger.error(
                "Explicit model execution failed. model=%s errorType=%s "
                "status=%s providerCode=%s schemaSha256=%s",
                model, type(exc).__name__, diagnostic["status_code"],
                diagnostic["provider_code"], diagnostic["schema_sha256"],
            )
            raise

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
            # Reading Sol/high may legitimately take longer than the generic
            # Luna deadline. The service's 80s boundary remains authoritative.
            client = (self.client.with_options(
                timeout=max(85.0, settings.OPENAI_REQUEST_TIMEOUT_SECONDS),
            )
                      if model == "gpt-5.6-sol" else self.client)
            response = await client.responses.create(
                model=model,
                instructions=rule,
                input=user_input,
                reasoning=cast(Reasoning, {"effort": policy.reasoning_effort}),
                max_output_tokens=policy.max_output_tokens,
                text=cast(ResponseTextConfigParam, text_config),
                prompt_cache_key=self._prompt_cache_key(type_name, model),
                store=False,
            )
            data = decode_response(response, structured=schema is not None)

            usage = getattr(response, "usage", None)
            return StructuredGenerationResult(
                data=data,
                input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
                output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
                provider="openai",
                model=str(getattr(response, "model", None) or model),
            )
        except Exception as exc:
            # Prompt/API key/Provider body 전체는 기록하지 않는다. 4xx 원인 판별에 필요한
            # 안전한 구조화 메타데이터와 schema fingerprint만 남긴다.
            diagnostic = safe_openai_error_metadata(exc, schema=schema)
            logger.error(
                "OpenAI API call failed. "
                "type=%s model=%s errorType=%s status=%s providerCode=%s "
                "providerType=%s providerParam=%s providerRequestId=%s "
                "structured=%s schemaSha256=%s providerMessageHint=%s",
                type_name,
                model,
                type(exc).__name__,
                diagnostic["status_code"],
                diagnostic["provider_code"],
                diagnostic["provider_type"],
                diagnostic["provider_param"],
                diagnostic["provider_request_id"],
                diagnostic["structured"],
                diagnostic["schema_sha256"],
                diagnostic["provider_message_hint"],
            )
            raise

    @staticmethod
    def _prompt_cache_key(type_name: str, model: str) -> str:
        # Stable per task/model bucket. The actual prefix still has to match, so this
        # cannot cross-contaminate unrelated prompts; it only improves cache locality.
        digest = hashlib.sha256(f"{type_name}|{model}".encode()).hexdigest()[:24]
        return f"translacat:{digest}"
