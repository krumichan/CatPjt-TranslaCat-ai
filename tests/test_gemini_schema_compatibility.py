from app.ai.providers.gemini.configs import (
    build_language_learning_task_verification_config,
    build_language_learning_vocab_design_config,
    build_language_learning_vocab_repair_config,
    sanitize_gemini_response_schema,
)


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


def test_level_test_reference_payload_properties_survive_gemini_sanitization():
    from app.schemas.language_learning_level_test import LevelTestQuestionGenerationPayload

    sanitized = sanitize_gemini_response_schema(
        LevelTestQuestionGenerationPayload.model_json_schema()
    )

    reference_schema = sanitized["$defs"]["LevelTestReferencePayload"]
    expected = {
        "sourceText",
        "referenceMeanings",
        "keyMeaningUnits",
        "referenceText",
        "translationSourceText",
        "emphasisText",
        "providedFacts",
        "requiredIntents",
        "responseConstraints",
        "readingPassage",
        "readingQuestion",
        "listeningQuestion",
    }
    assert set(reference_schema["properties"]) == expected
    assert set(reference_schema["required"]) == expected
    assert "additionalProperties" not in reference_schema


def test_vocab_context_multistage_configs_use_role_specific_sampling():
    from app.schemas.language_learning_level_test import (
        LevelTestVocabContextDesignPayload,
        LevelTestVocabContextRepairPayload,
    )

    design = build_language_learning_vocab_design_config(
        "design",
        LevelTestVocabContextDesignPayload.model_json_schema(),
    )
    repair = build_language_learning_vocab_repair_config(
        "repair",
        LevelTestVocabContextRepairPayload.model_json_schema(),
    )

    assert design.temperature == 0.4
    assert design.max_output_tokens == 2048
    assert repair.temperature == 0.2
    assert repair.max_output_tokens == 4096


def test_choice_verification_schema_exposes_best_answer_contract_after_sanitization():
    from app.schemas.language_learning_level_test import LevelTestChoiceSemanticVerificationPayload

    sanitized = sanitize_gemini_response_schema(
        LevelTestChoiceSemanticVerificationPayload.model_json_schema()
    )

    properties = sanitized["properties"]
    assert set(properties) == {
        "verifiable",
        "plausibleOptionKeys",
        "nearEquivalentOptionKeys",
        "bestOptionKey",
        "bestOptionAdvantage",
    }
    assert set(sanitized["required"]) == {
        "verifiable",
        "bestOptionKey",
        "bestOptionAdvantage",
    }
    advantage_ref = properties["bestOptionAdvantage"]["$ref"].split("/")[-1]
    assert sanitized["$defs"][advantage_ref]["enum"] == ["CLEAR", "WEAK", "NONE"]


def test_vocab_context_design_schema_is_compact_and_provider_visible():
    from app.schemas.language_learning_level_test import LevelTestVocabContextDesignPayload

    sanitized = sanitize_gemini_response_schema(
        LevelTestVocabContextDesignPayload.model_json_schema()
    )

    design_schema = sanitized["$defs"]["LevelTestVocabContextDesign"]
    assert set(design_schema["properties"]) == {
        "designId",
        "targetExpression",
        "targetMeaning",
        "semanticConstraint",
        "scenarioCategory",
        "communicativeIntent",
    }
    assert "contrastDimension" not in design_schema["properties"]
    assert "distractorConstraints" not in design_schema["properties"]


def test_task_sufficiency_verification_config_is_deterministic_and_compact():
    from app.schemas.language_learning_level_test import LevelTestTaskSufficiencyVerificationPayload

    config = build_language_learning_task_verification_config(
        "task-check",
        LevelTestTaskSufficiencyVerificationPayload.model_json_schema(),
    )

    assert config.temperature == 0
    assert config.top_p == 0.2
    assert config.top_k == 10
    assert config.max_output_tokens == 512
    assert config.thinking_config.thinking_budget == 0
