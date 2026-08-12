from google.genai import types

from app.core.constants import DEFAULT_GENERATION_CONFIG, DEFAULT_SAFETY_SETTINGS


def build_default_gemini_config(
    rule: str,
    schema: dict | None = None,
) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=rule,
        response_mime_type="application/json" if schema else "text/plain",
        response_schema=schema,
        safety_settings=DEFAULT_SAFETY_SETTINGS,
        **DEFAULT_GENERATION_CONFIG,
    )


def build_fast_translation_config() -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        temperature=0,
        max_output_tokens=128,
        response_mime_type="text/plain",
        safety_settings=DEFAULT_SAFETY_SETTINGS,
        thinking_config=types.ThinkingConfig(
            thinking_budget=0,
        ),
    )


def build_chat_ai_reply_config(
    rule: str,
    schema: dict,
) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=rule,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        max_output_tokens=1024,
        response_mime_type="application/json",
        response_schema=schema,
        safety_settings=DEFAULT_SAFETY_SETTINGS,
    )



def build_language_learning_generation_config(
    rule: str,
    schema: dict,
) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=rule,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        max_output_tokens=8192,
        response_mime_type="application/json",
        response_schema=schema,
        safety_settings=DEFAULT_SAFETY_SETTINGS,
    )


def build_language_learning_evaluation_config(
    rule: str,
    schema: dict,
) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=rule,
        temperature=0.1,
        top_p=0.9,
        top_k=20,
        max_output_tokens=8192,
        response_mime_type="application/json",
        response_schema=schema,
        safety_settings=DEFAULT_SAFETY_SETTINGS,
    )
