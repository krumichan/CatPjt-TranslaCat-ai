import asyncio
import json
import logging
from typing import Any

from app.ai.ports import TextGenerationProvider
from app.common.collection_utils import chunk_list

logger = logging.getLogger(__name__)


class TranslationService:
    def __init__(
        self,
        provider: TextGenerationProvider,
        batch_size: int = 20,
        concurrency: int = 20,
    ) -> None:
        self.provider = provider
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(concurrency)

    async def translate_single(
        self,
        text: str,
        type_name: str,
    ) -> str:
        result = await self.provider.call(
            type_name=type_name,
            data=text,
            schema=None,
        )

        return self._normalize_single_result(result)

    async def translate_batch(
        self,
        texts: list[str],
        type_name: str,
    ) -> list[str]:
        if not texts:
            return []

        initial_chunks = list(chunk_list(texts, self.batch_size))

        tasks = [
            self._recursive_translate(
                chunk=chunk,
                type_name=type_name,
            )
            for chunk in initial_chunks
        ]

        final_results = await asyncio.gather(*tasks)

        return [
            item
            for sublist in final_results
            for item in sublist
        ]

    async def _call_chunk(
        self,
        type_name: str,
        chunk: list[str],
    ) -> list[str]:
        expected_count = len(chunk)

        list_schema = {
            "type": "ARRAY",
            "items": {
                "type": "STRING",
            },
        }

        payload = json.dumps(
            chunk,
            ensure_ascii=False,
        )

        try:
            results = await self.provider.call(
                type_name=type_name,
                data=payload,
                schema=list_schema,
            )

            if isinstance(results, list):
                if len(results) != expected_count:
                    raise ValueError(
                        f"Count mismatch: expected {expected_count}, got {len(results)}"
                    )

                return [
                    str(result).strip()
                    for result in results
                ]

            raise ValueError("Response format error: response is not a list.")

        except Exception as exc:
            logger.debug("Chunk translation failed: %s", exc)
            raise

    async def _single_unit_retry_task(
        self,
        text: str,
        type_name: str,
        max_retries: int = 2,
    ) -> str:
        for attempt in range(max_retries + 1):
            try:
                result = await self.provider.call(
                    type_name=type_name,
                    data=text,
                    schema=None,
                )

                translated = self._normalize_single_result(result)

                if translated:
                    return translated

            except Exception as exc:
                logger.error("Single translation failed: %s", exc)

            if attempt < max_retries:
                await asyncio.sleep(0.3)

        return ""

    async def _recursive_translate(
        self,
        chunk: list[str],
        type_name: str,
    ) -> list[str]:
        chunk_size = len(chunk)

        if chunk_size <= 1:
            return [
                await self._single_unit_retry_task(
                    text=chunk[0],
                    type_name=type_name,
                )
            ]

        async with self.semaphore:
            try:
                results = await self._call_chunk(
                    type_name=type_name,
                    chunk=chunk,
                )

                if len(results) == chunk_size:
                    return results

                logger.warning(
                    "Translation count mismatch. expected=%s, actual=%s. Splitting...",
                    chunk_size,
                    len(results),
                )

            except Exception:
                logger.warning(
                    "Chunk translation failed. chunk_size=%s. Splitting...",
                    chunk_size,
                )

        mid = (chunk_size // 2) + 1

        if mid >= chunk_size:
            mid = chunk_size // 2

        left_chunk = chunk[:mid]
        right_chunk = chunk[mid:]

        left_result, right_result = await asyncio.gather(
            self._recursive_translate(
                chunk=left_chunk,
                type_name=type_name,
            ),
            self._recursive_translate(
                chunk=right_chunk,
                type_name=type_name,
            ),
        )

        return left_result + right_result

    def _normalize_single_result(
        self,
        result: Any,
    ) -> str:
        if result is None:
            return ""

        if isinstance(result, str):
            return result.strip().strip("[]'\"")

        logger.warning(
            "Unexpected translation response type: %s",
            type(result),
        )

        return str(result).strip()