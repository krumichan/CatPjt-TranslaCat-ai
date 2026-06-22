from typing import Optional

from google.genai import types

from app.core.constants import DEFAULT_GENERATION_CONFIG, DEFAULT_SAFETY_SETTINGS


def build_default_gemini_config(
    rule: str,
    schema: Optional[dict] = None,
) -> types.GenerateContentConfig:
    """일반 Gemini 호출 설정 생성"""
    return types.GenerateContentConfig(
        system_instruction=rule,
        response_mime_type="application/json" if schema else "text/plain",
        response_schema=schema,
        safety_settings=DEFAULT_SAFETY_SETTINGS,
        **DEFAULT_GENERATION_CONFIG,
    )


def build_fast_translation_config() -> types.GenerateContentConfig:
    """채팅 메시지 번역 전용 빠른 설정"""
    return types.GenerateContentConfig(
        temperature=0,
        max_output_tokens=128,
        response_mime_type="text/plain",
        safety_settings=DEFAULT_SAFETY_SETTINGS,
        thinking_config=types.ThinkingConfig(
            thinking_budget=0,
        ),
    )