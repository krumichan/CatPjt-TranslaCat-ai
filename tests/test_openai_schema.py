from app.ai.providers.openai.schema import (
    build_openai_text_config,
    normalize_openai_response_schema,
)


def test_normalizes_legacy_gemini_schema_types_and_nullable():
    schema = {
        "type": "OBJECT",
        "properties": {
            "amount": {"type": "INTEGER", "nullable": True},
            "items": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
            },
        },
        "required": ["amount", "items"],
    }

    normalized = normalize_openai_response_schema(schema)

    assert normalized == {
        "type": "object",
        "properties": {
            "amount": {"type": ["integer", "null"]},
            "items": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["amount", "items"],
    }


def test_structured_output_keeps_migration_schema_non_strict():
    config = build_openai_text_config(
        type_name="RECEIPT_ANALYSIS",
        schema={"type": "OBJECT", "properties": {}},
        verbosity="low",
    )

    assert config["format"]["type"] == "json_schema"
    assert config["format"]["strict"] is False
    assert config["format"]["schema"]["type"] == "object"
