from copy import deepcopy
from typing import Any

from google.genai import types

from app.core.constants import DEFAULT_GENERATION_CONFIG, DEFAULT_SAFETY_SETTINGS


_GEMINI_SCHEMA_KEYS_TO_DROP = frozenset(
    {
        # Gemini structured output only needs the response shape here.
        # Pydantic validates these constraints again after generation.
        "additionalProperties",
        "title",
        "description",
        "default",
        "examples",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "pattern",
        "format",
        "minItems",
        "maxItems",
        "uniqueItems",
        "minProperties",
        "maxProperties",
    }
)


def sanitize_gemini_response_schema(schema: dict | None) -> dict | None:
    """Return a Gemini-compatible copy of a JSON schema.

    Pydantic JSON Schema contains validation metadata and cardinality/range
    constraints that Gemini either does not support or can reject as a schema with
    too many serving states. Keep only the structural contract that helps Gemini
    produce valid JSON; Pydantic remains the source of truth for strict validation
    after the provider response is received.
    """
    if schema is None:
        return None

    return _sanitize_schema_node(deepcopy(schema))


def _sanitize_schema_node(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _sanitize_schema_node(child)
            for key, child in value.items()
            if key not in _GEMINI_SCHEMA_KEYS_TO_DROP
        }
    if isinstance(value, list):
        return [_sanitize_schema_node(child) for child in value]
    return value


def build_default_gemini_config(
    rule: str,
    schema: dict | None = None,
) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=rule,
        response_mime_type="application/json" if schema else "text/plain",
        response_schema=sanitize_gemini_response_schema(schema),
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


def build_voice_translation_config(
    rule: str,
    schema: dict,
    *,
    max_output_tokens: int,
) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=rule,
        temperature=0,
        max_output_tokens=max_output_tokens,
        response_mime_type="application/json",
        response_schema=sanitize_gemini_response_schema(schema),
        safety_settings=DEFAULT_SAFETY_SETTINGS,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
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
        response_schema=sanitize_gemini_response_schema(schema),
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
        response_schema=sanitize_gemini_response_schema(schema),
        safety_settings=DEFAULT_SAFETY_SETTINGS,
    )


def build_language_learning_vocab_design_config(
    rule: str,
    schema: dict,
) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=rule,
        temperature=0.4,
        top_p=0.8,
        top_k=30,
        max_output_tokens=2048,
        response_mime_type="application/json",
        response_schema=sanitize_gemini_response_schema(schema),
        safety_settings=DEFAULT_SAFETY_SETTINGS,
    )


def build_language_learning_vocab_repair_config(
    rule: str,
    schema: dict,
) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=rule,
        temperature=0.2,
        top_p=0.6,
        top_k=20,
        max_output_tokens=4096,
        response_mime_type="application/json",
        response_schema=sanitize_gemini_response_schema(schema),
        safety_settings=DEFAULT_SAFETY_SETTINGS,
    )


def build_language_learning_choice_verification_config(
    rule: str,
    schema: dict,
) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=rule,
        temperature=0,
        top_p=0.2,
        top_k=10,
        max_output_tokens=256,
        response_mime_type="application/json",
        response_schema=sanitize_gemini_response_schema(schema),
        safety_settings=DEFAULT_SAFETY_SETTINGS,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )



def build_language_learning_task_verification_config(
    rule: str,
    schema: dict,
) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=rule,
        temperature=0,
        top_p=0.2,
        top_k=10,
        max_output_tokens=512,
        response_mime_type="application/json",
        response_schema=sanitize_gemini_response_schema(schema),
        safety_settings=DEFAULT_SAFETY_SETTINGS,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
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
        response_schema=sanitize_gemini_response_schema(schema),
        safety_settings=DEFAULT_SAFETY_SETTINGS,
    )


def build_language_learning_writing_verification_config(
    rule: str,
    schema: dict,
) -> types.GenerateContentConfig:
    # Bound evidence + IDs require more room than the 512-token sufficiency gate.
    return types.GenerateContentConfig(
        system_instruction=rule,
        temperature=0,
        top_p=0.2,
        top_k=10,
        max_output_tokens=4096,
        response_mime_type="application/json",
        response_schema=sanitize_gemini_response_schema(schema),
        safety_settings=DEFAULT_SAFETY_SETTINGS,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
