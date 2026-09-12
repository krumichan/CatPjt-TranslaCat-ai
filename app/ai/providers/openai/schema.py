from __future__ import annotations

import re
from copy import deepcopy
from typing import Any


_SIMPLE_PATTERN_ENUMS: dict[str, list[str]] = {
    "^[AB]$": ["A", "B"],
    "^[A-D]$": ["A", "B", "C", "D"],
}


_TYPE_MAP = {
    "OBJECT": "object",
    "ARRAY": "array",
    "STRING": "string",
    "INTEGER": "integer",
    "NUMBER": "number",
    "BOOLEAN": "boolean",
    "NULL": "null",
}


def normalize_openai_response_schema(schema: dict | None) -> dict | None:
    """Normalize legacy Gemini-style JSON schemas for OpenAI Structured Outputs.

    The application still performs Pydantic/domain validation after the provider
    response. We therefore keep the service-owned schema semantics and only convert
    provider-specific syntax such as uppercase type names and `nullable`.
    """
    if schema is None:
        return None
    return _normalize_node(deepcopy(schema))


def _normalize_node(value: Any) -> Any:
    if isinstance(value, list):
        return [_normalize_node(child) for child in value]
    if not isinstance(value, dict):
        return value

    result: dict[str, Any] = {}
    nullable = bool(value.get("nullable", False))

    for key, child in value.items():
        if key == "nullable":
            continue
        if key == "type":
            result[key] = _normalize_type(child, nullable=nullable)
        else:
            result[key] = _normalize_node(child)

    # `nullable: true` can appear without an explicit type in some generated schemas.
    if nullable and "type" not in result:
        any_of = result.pop("anyOf", None)
        if isinstance(any_of, list):
            result["anyOf"] = [*any_of, {"type": "null"}]

    # A small finite pattern is more reliably followed as an enum by providers.
    # This is semantics-preserving and particularly important for answer/plan IDs.
    pattern = result.get("pattern")
    if isinstance(pattern, str) and pattern in _SIMPLE_PATTERN_ENUMS:
        result.pop("pattern", None)
        result["enum"] = list(_SIMPLE_PATTERN_ENUMS[pattern])

    return result


def _normalize_type(value: Any, *, nullable: bool) -> Any:
    if isinstance(value, list):
        normalized = [_normalize_single_type(item) for item in value]
        if nullable and "null" not in normalized:
            normalized.append("null")
        return normalized

    normalized = _normalize_single_type(value)
    if nullable and normalized != "null":
        return [normalized, "null"]
    return normalized


def _normalize_single_type(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    return _TYPE_MAP.get(value, value.lower())


def schema_name(type_name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", type_name).strip("_")
    return (normalized or "translacat_response")[:64]


# Only these closed, required-field schemas opt in. Legacy optional-field callers
# retain their existing migration contract. Semantic/binding checks remain mandatory.
_STRICT_REVIEW_TASKS = frozenset({
    "LANGUAGE_LEARNING_DAILY_WRITING_GENERATION",
    "LANGUAGE_LEARNING_WRITING_SOURCE_LOCALIZATION",
    "LANGUAGE_LEARNING_WRITING_DIFFICULTY_PRESCREEN",
    "LANGUAGE_LEARNING_WRITING_TASK_VERIFICATION",
    "LANGUAGE_LEARNING_WRITING_DIFFICULTY_VERIFICATION",
    "LANGUAGE_LEARNING_WRITING_NOTE_VERIFICATION",
    "LANGUAGE_LEARNING_WRITING_NOTE_LOCALIZATION",
})


class OpenAISchemaConfigurationError(ValueError):
    status_code = 400


def _assert_strict_objects(node: Any) -> None:
    if isinstance(node, list):
        for child in node:
            _assert_strict_objects(child)
    elif isinstance(node, dict):
        if node.get("type") == "object":
            if node.get("additionalProperties") is not False or set(node.get("required", [])) != set(node.get("properties", {})):
                raise OpenAISchemaConfigurationError("Strict Writing review schemas require closed objects and every property required")
        for child in node.values():
            _assert_strict_objects(child)


def build_openai_text_config(
    *,
    type_name: str,
    schema: dict | None,
    verbosity: str,
) -> dict[str, Any]:
    config: dict[str, Any] = {"verbosity": verbosity}
    if schema is None:
        return config

    normalized_schema = normalize_openai_response_schema(schema)
    assert normalized_schema is not None
    strict = type_name in _STRICT_REVIEW_TASKS
    if strict:
        _assert_strict_objects(normalized_schema)
    config["format"] = {
        "type": "json_schema",
        "name": schema_name(type_name),
        "strict": strict,
        "schema": normalized_schema,
    }
    return config
