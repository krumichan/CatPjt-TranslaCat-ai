import asyncio
import json
from typing import Any

from app.ai.ports import TextGenerationProvider
from app.common.collection_utils import chunk_list


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

    async def translate_single(self, text: str, type_name: str) -> str:
        # 단일 문장도 배치와 같은 동시성 제한을 사용한다. 실패는 호출자에게 전달한다.
        async with self.semaphore:
            result = await self.provider.call(type_name=type_name, data=text, schema=None)
        return self._validate_single_result(result)

    async def translate_batch(self, texts: list[str], type_name: str) -> list[str]:
        if not texts:
            return []

        # 최초 조각만 호출한다. 거절·잘못된 출력에 대한 분할/보정 호출은 만들지 않는다.
        tasks = [
            asyncio.create_task(self._translate_chunk(type_name, chunk))
            for chunk in chunk_list(texts, self.batch_size)
        ]
        try:
            results = await asyncio.gather(*tasks)
        except BaseException:
            # 한 조각이 실패하면 대기 중인 호출을 취소하고 부분 성공을 반환하지 않는다.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        return [item for chunk in results for item in chunk]

    async def _translate_chunk(self, type_name: str, chunk: list[str]) -> list[str]:
        # 기존 단일 항목 Provider 입력 형식은 보존한다.
        if len(chunk) == 1:
            return [await self.translate_single(chunk[0], type_name)]

        schema = {"type": "ARRAY", "items": {"type": "STRING"}}
        async with self.semaphore:
            result = await self.provider.call(
                type_name=type_name,
                data=json.dumps(chunk, ensure_ascii=False),
                schema=schema,
            )

        # 모델 출력의 개수·타입·공백 여부는 코드에서 확인한다. 원문/응답은 로그에 남기지 않는다.
        if not isinstance(result, list) or len(result) != len(chunk):
            raise ValueError("TRANSLATION_RESPONSE_INVALID: expected one string per input")
        return [self._validate_single_result(item) for item in result]

    @staticmethod
    def _validate_single_result(result: Any) -> str:
        if not isinstance(result, str) or not result.strip():
            raise ValueError("TRANSLATION_RESPONSE_INVALID: expected a nonblank string")
        # 따옴표와 대괄호는 실제 대사/본문일 수 있으므로 응답을 임의로 복구하지 않는다.
        return result
