import asyncio
import logging
import json
from typing import List, Optional

from google import genai
from google.genai import types
from fastapi import HTTPException
from app.core.config import settings
from app.core.prompts import PROMPT_MAP
from app.core.constants import DEFAULT_GENERATION_CONFIG, DEFAULT_SAFETY_SETTINGS
from app.core.utils import chunk_list

logger = logging.getLogger(__name__)

class GeminiService:
    def __init__(self):
        # 💡 클라이언트는 한 번만 생성하여 재사용
        self.client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        self.model_name = settings.GEMINI_MODEL_NAME
        self._config_cache = {}
        self.semaphore = asyncio.Semaphore(20)
        self.batch_size = 20

    def get_rule(self, type_name: str) -> str:
        rule = PROMPT_MAP.get(type_name)
        if not rule:
            raise HTTPException(status_code=400, detail=f"Invalid type: {type_name}")
        return rule

    def get_cached_config(self, type_name: str, schema: Optional[dict] = None):
        """설정 객체 캐싱 (CPU 부하 감소)"""
        cache_key = f"{type_name}_{hash(str(schema)) if schema else 'none'}"
        if cache_key not in self._config_cache:
            rule = self.get_rule(type_name)
            self._config_cache[cache_key] = types.GenerateContentConfig(
                system_instruction=rule,
                response_mime_type="application/json" if schema else "text/plain",
                response_schema=schema,
                safety_settings=DEFAULT_SAFETY_SETTINGS,
                **DEFAULT_GENERATION_CONFIG
            )
        return self._config_cache[cache_key]

    async def call(self, type_name: str, data: str, schema: Optional[dict] = None):
        """Gemini 호출 핵심 엔진"""
        try:
            config = self.get_cached_config(type_name, schema)
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=data,
                config=config
            )
            #  schema가 있으면 .parsed(SDK 자동 파싱), 없으면 .text 반환
            return response.parsed if schema else response.text
        except Exception as e:
            logger.error(f"Gemini API Call Error: {e}")
            raise e
    
    async def call_with_image(
        self,
        type_name: str,
        prompt: str,
        image_bytes: bytes,
        mime_type: str,
        schema: dict | None = None,
    ):
        try:
            config = self.get_cached_config(type_name, schema)

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
        
    async def translate_batch(self, texts: List[str], type_name: str) -> List[str]:
        """
        여러 문장을 배치 단위로 병렬 번역하며, 실패한 문장은 개별적으로 재시도합니다.

        내부적으로 chunk_list를 사용하여 대량의 문장을 일정 크기로 나누어 병렬 처리하며,
        구글 API의 Rate Limit을 고려하면서도 처리 속도를 극대화합니다.

        Args:
            texts (List[str]): 번역할 원문 문장들의 리스트.
            type_name (str): 번역 프롬프트 타입 (예: 'novel').

        Returns:
            List[str]: 입력 순서가 유지된 번역 결과 리스트. 
                      끝까지 실패한 문장은 "[번역 실패]"로 표시됩니다.
        """

        # 메인 실행: 처음엔 10개씩(또는 20개씩) 청크로 나눠서 재귀 시작점 제공
        initial_chunks = list(chunk_list(texts, self.batch_size))
        tasks = [self._recursive_translate(chunk=c, type_name=type_name) for c in initial_chunks]
        
        final_results = await asyncio.gather(*tasks)
        
        # 2차원 리스트 평탄화
        return [item for sublist in final_results for item in sublist]

    async def _call_chunk(self, type_name: str, chunk: List[str]) -> List[str]:
        """리스트 단위 번역 (JSON 모드)"""
        n = len(chunk)
        # 💡 리스트 형태의 스키마 명시
        list_schema = {"type": "ARRAY", "items": {"type": "STRING"}}
        
        payload = json.dumps(chunk, ensure_ascii=False)

        try:
            # SDK의 .parsed를 쓰기 위해 schema를 넘김.
            results = await self.call(type_name, payload, schema=list_schema)
            if isinstance(results, list):
                if len(results) != n:
                    raise ValueError(f"Count mismatch: expected {n}, got {len(results)}")
                return [str(r).strip() for r in results]
            
            raise ValueError("Response format error (not a list)")
        except Exception as e:
            logger.debug(f"Chunk translation failed: {e}")
            raise e

    async def single_unit_retry_task(self, text: str, type_name: str, max_retries: int = 2) -> str:
        """한 문장씩 정밀 번역 (재시도 포함)"""
        for attempt in range(max_retries + 1):
            try:
                # 단일 문장은 schema 없이 text 모드로 빠르게 호출
                result = await self.call(type_name, text, schema=None)
                if isinstance(result, str):
                    clean_res = result.strip().strip("[]'\"")
                    if clean_res:
                        return clean_res
                else:
                    # 혹시라도 객체 형태로 왔을 경우를 대비한 안전장치
                    logger.warning(f"예상치 못한 응답 타입: {type(result)}")
            except Exception as e:
                logger.error(f"단일 번역 실패: {e}")

            if attempt < max_retries:
                await asyncio.sleep(0.3)

        return ""

    async def _recursive_translate(self, chunk: List[str], type_name: str) -> List[str]:
        """재귀적 이진 분할 로직 (N -> N/2+1 -> 1)"""
        n = len(chunk)
        
        # 1. 탈출 조건: 문장이 1개면 정밀 번역 실행
        if n <= 1:
            return [await self.single_unit_retry_task(chunk[0], type_name)]

        # 2. 현재 청크 시도
        async with self.semaphore:
            try:
                results = await self._call_chunk(type_name, chunk)
                if len(results) == n:
                    return results
                logger.warning(f"Mismatch ({len(results)}/{n}). Splitting...")
            except Exception:
                logger.warning(f"Chunk failed (N={n}). Splitting...")

        # 3. 실패 시 이진 분할 (n/2 + 1)
        mid = (n // 2) + 1
        if mid >= n: mid = n // 2

        left_chunk = chunk[:mid]
        right_chunk = chunk[mid:]

        # 왼쪽/오른쪽을 다시 비동기 재귀 호출
        left_res, right_res = await asyncio.gather(
            self._recursive_translate(left_chunk, type_name),
            self._recursive_translate(right_chunk, type_name)
        )
        return left_res + right_res
