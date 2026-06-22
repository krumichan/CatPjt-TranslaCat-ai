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

    async def translate_batch(
        self,
        texts: List[str],
        type_name: str,
    ) -> List[str]:
        initial_chunks = list(chunk_list(texts, self.batch_size))
        tasks = [
            self._recursive_translate(chunk=chunk, type_name=type_name)
            for chunk in initial_chunks
        ]

        final_results = await asyncio.gather(*tasks)

        return [item for sublist in final_results for item in sublist]

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

    async def _call_chunk(self, type_name: str, chunk: List[str]) -> List[str]:
        n = len(chunk)
        list_schema = {"type": "ARRAY", "items": {"type": "STRING"}}
        payload = json.dumps(chunk, ensure_ascii=False)

        try:
            results = await self.call(type_name, payload, schema=list_schema)

            if isinstance(results, list):
                if len(results) != n:
                    raise ValueError(f"Count mismatch: expected {n}, got {len(results)}")

                return [str(result).strip() for result in results]

            raise ValueError("Response format error (not a list)")
        except Exception as exc:
            logger.debug("Chunk translation failed: %s", exc)
            raise

    async def single_unit_retry_task(
        self,
        text: str,
        type_name: str,
        max_retries: int = 2,
    ) -> str:
        for attempt in range(max_retries + 1):
            try:
                result = await self.call(type_name, text, schema=None)

                if isinstance(result, str):
                    clean_result = result.strip().strip("[]'\"")
                    if clean_result:
                        return clean_result

                logger.warning("예상치 못한 응답 타입: %s", type(result))
            except Exception as exc:
                logger.error("단일 번역 실패: %s", exc)

            if attempt < max_retries:
                await asyncio.sleep(0.3)

        return ""

    async def _recursive_translate(
        self,
        chunk: List[str],
        type_name: str,
    ) -> List[str]:
        n = len(chunk)

        if n <= 1:
            return [await self.single_unit_retry_task(chunk[0], type_name)]

        async with self.semaphore:
            try:
                results = await self._call_chunk(type_name, chunk)

                if len(results) == n:
                    return results

                logger.warning("Mismatch (%s/%s). Splitting...", len(results), n)
            except Exception:
                logger.warning("Chunk failed (N=%s). Splitting...", n)

        mid = (n // 2) + 1
        if mid >= n:
            mid = n // 2

        left_chunk = chunk[:mid]
        right_chunk = chunk[mid:]

        left_result, right_result = await asyncio.gather(
            self._recursive_translate(left_chunk, type_name),
            self._recursive_translate(right_chunk, type_name),
        )

        return left_result + right_result
