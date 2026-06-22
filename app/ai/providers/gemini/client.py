import asyncio
import json
import logging
from typing import Any, List

from fastapi import HTTPException
from google import genai
from google.genai import types

from app.ai.providers.gemini.config_manager import GeminiConfigManager
from app.common.collection_utils import chunk_list
from app.core.config import settings
from app.features.chat_translation.normalizer import normalize_chat_translation_result
from app.features.chat_translation.prompts import build_chat_translation_prompt

logger = logging.getLogger(__name__)


class GeminiService:
    def __init__(self) -> None:
        self.client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        self.model_name = settings.GEMINI_MODEL_NAME
        self.config_manager = GeminiConfigManager()
        self.semaphore = asyncio.Semaphore(20)
        self.batch_size = 20

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
