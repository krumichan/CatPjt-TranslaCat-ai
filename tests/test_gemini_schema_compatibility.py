from app.ai.providers.gemini.configs import sanitize_gemini_response_schema


def test_sanitize_gemini_response_schema_removes_additional_properties_recursively():
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "metadata": {
                            "type": "object",
                            "additionalProperties": {"type": "string"},
                        }
                    },
                },
            }
        },
    }

    sanitized = sanitize_gemini_response_schema(schema)

    assert "additionalProperties" not in sanitized
    item_schema = sanitized["properties"]["items"]["items"]
    assert "additionalProperties" not in item_schema
    assert "additionalProperties" not in item_schema["properties"]["metadata"]


def test_sanitize_gemini_response_schema_does_not_mutate_original_schema():
    schema = {
        "type": "object",
        "additionalProperties": False,
    }

    sanitized = sanitize_gemini_response_schema(schema)

    assert schema["additionalProperties"] is False
    assert "additionalProperties" not in sanitized


def test_sanitize_gemini_response_schema_accepts_none():
    assert sanitize_gemini_response_schema(None) is None


def test_sanitize_gemini_response_schema_compacts_validation_constraints_recursively():
    schema = {
        "title": "ListeningGenerationPayload",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "items": {
                "title": "Items",
                "type": "array",
                "minItems": 1,
                "maxItems": 30,
                "items": {
                    "type": "object",
                    "properties": {
                        "sourceText": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 4000,
                            "pattern": ".+",
                        },
                        "estimatedAudioSeconds": {
                            "type": "number",
                            "minimum": 1,
                            "maximum": 60,
                        },
                        "status": {
                            "type": "string",
                            "enum": ["READY", "FAILED"],
                        },
                    },
                    "required": [
                        "sourceText",
                        "estimatedAudioSeconds",
                        "status",
                    ],
                },
            }
        },
        "required": ["items"],
    }

    sanitized = sanitize_gemini_response_schema(schema)

    assert sanitized == {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "sourceText": {"type": "string"},
                        "estimatedAudioSeconds": {"type": "number"},
                        "status": {
                            "type": "string",
                            "enum": ["READY", "FAILED"],
                        },
                    },
                    "required": [
                        "sourceText",
                        "estimatedAudioSeconds",
                        "status",
                    ],
                },
            }
        },
        "required": ["items"],
    }
