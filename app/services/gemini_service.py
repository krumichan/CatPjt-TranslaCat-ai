import asyncio
import logging
from typing import List, Optional
from app.core.constants import DEFAULT_CHUNK_SIZE, DEFAULT_GENERATION_CONFIG, DEFAULT_SAFETY_SETTINGS
from app.core.prompts import PROMPT_MAP
from app.core.utils import chunk_list
from fastapi import HTTPException
from google import genai
from google.genai import types
from app.core.config import settings

logger = logging.getLogger(__name__)

class GeminiService:
    """
    Google Gemini API를 사용하여 텍스트 번역 및 생성 서비스를 제공하는 클래스.
    
    이 클래스는 단일 문장 번역, 대량의 문장 배치 번역, 그리고 실패 시 
    자동 재시도 로직을 포함하고 있습니다.
    """

    def __init__(self):
        """
        GeminiService를 초기화하고 Google GenAI 클라이언트를 설정합니다.
        """
        self.client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        self.model_name = settings.GEMINI_MODEL_NAME
    
    def get_rule(self, type_name: str) -> str:
        """
        번역 타입(소설, 채팅 등)에 따른 시스템 프롬프트 규칙을 조회합니다.

        Args:
            type_name (str): PROMPT_MAP에 정의된 번역 타입 키.

        Returns:
            str: 해당 타입에 설정된 시스템 instruction 문자열.

        Raises:
            HTTPException: 유효하지 않은 type_name이 전달될 경우 400 에러를 발생시킵니다.
        """
        rule = PROMPT_MAP.get(type_name)
        if not rule:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid translation type: '{type_name}'. Available: {list(PROMPT_MAP.keys())}"
            )
        return rule

    async def call(self, rule:str, data: str, schema: Optional[dict] = None) -> str:
        """
        Gemini 모델에 단일 요청을 보내고 응답 텍스트를 반환합니다.

        Args:
            rule (str): 시스템 instruction으로 설정할 프롬프트 규칙.
            data (str): 번역하거나 처리할 원문 텍스트.
            schema (dict, optional): JSON 출력이 필요할 경우 적용할 JSON 스키마.

        Returns:
            str: Gemini 모델이 생성한 텍스트 결과물.

        Raises:
            ValueError: Gemini가 빈 응답을 반환할 경우 발생합니다.
            Exception: API 통신 중 발생하는 기타 모든 예외를 다시 발생시킵니다.
        """
        try:
            config = types.GenerateContentConfig(
                system_instruction=rule,
                response_mime_type="application/json" if schema else "text/plain",
                response_schema=schema,
                safety_settings=DEFAULT_SAFETY_SETTINGS,
                **DEFAULT_GENERATION_CONFIG
            )

            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=data,
                config=config
            )

            text = response.text

            if not text or text.strip() == "":
                logger.warning("Gemini returned an empty response.")
                raise ValueError("Gemini가 번역 결과를 생성하지 못했습니다. (응답 비어있음)")
            
            return text
        except Exception as e:
            logger.error(f"Error calling Gemini: {e}")
            raise e
    
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
        rule = self.get_rule(type_name)
        final_results = []

        # 원문 리스트를 일정한 크기(DEFAULT_CHUNK_SIZE)로 나누어 순차적으로 처리
        for chunk in chunk_list(texts, DEFAULT_CHUNK_SIZE):
            # 현재 chunk 내의 모든 문장에 대해 비동기 태스크 생성
            tasks = [self.call(rule=rule, data=text) for text in chunk]

            # 1차 시도: asyncio.gather를 통한 병렬 실행 (return_exceptions=True로 부분 실패 허용)
            chunk_translated = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(chunk_translated):
                # 에러가 발생했거나 응답이 비어있는 경우 개별 재시도 수행
                if isinstance(result, Exception) or not result:
                    # 로그에는 문장의 앞부분 20자만 출력하여 가독성 유지
                    logger.warning(f"번역 실패 재시도 중: {chunk[i][:20]}...")
                    try:
                        # 2차 시도: 개별 비동기 호출로 정밀 재시도
                        retry_result = await self.call(rule=rule, data=chunk[i])
                        final_results.append(retry_result)
                    except Exception:
                        # 최종 실패 시 플레이스홀더 추가하여 리스트 순서 유지
                        final_results.append("[번역 실패]")
                else:
                    final_results.append(result)
        
        return final_results