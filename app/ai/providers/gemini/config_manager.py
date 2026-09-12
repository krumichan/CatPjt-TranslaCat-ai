from fastapi import HTTPException

from app.ai.prompt_registry import get_prompt_rule
from app.ai.providers.gemini.configs import (
    build_chat_ai_reply_config,
    build_default_gemini_config,
    build_fast_translation_config,
    build_language_learning_choice_verification_config,
    build_language_learning_evaluation_config,
    build_language_learning_generation_config,
    build_language_learning_task_verification_config,
    build_language_learning_writing_verification_config,
    build_language_learning_vocab_design_config,
    build_language_learning_vocab_repair_config,
    build_voice_translation_config,
)
from app.core.config import settings


# Candidate-bound review schemas contain request-specific IDs/hashes. Keeping
# each in the shared cache would retain every reviewed candidate indefinitely.
_UNCACHED_WRITING_REVIEW_TASKS = frozenset({
    "LANGUAGE_LEARNING_WRITING_SOURCE_LOCALIZATION",
    "LANGUAGE_LEARNING_WRITING_DIFFICULTY_PRESCREEN",
    "LANGUAGE_LEARNING_WRITING_TASK_VERIFICATION",
    "LANGUAGE_LEARNING_WRITING_DIFFICULTY_VERIFICATION",
    "LANGUAGE_LEARNING_WRITING_NOTE_VERIFICATION",
    "LANGUAGE_LEARNING_WRITING_NOTE_LOCALIZATION",
})


class GeminiConfigManager:
    def __init__(self) -> None:
        self._config_cache = {}

    def get_rule(self, type_name: str) -> str:
        rule = get_prompt_rule(type_name)

        if not rule:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid type: {type_name}",
            )

        return rule

    def get_cached_config(
        self,
        type_name: str,
        schema: dict | None = None,
    ):
        if type_name in _UNCACHED_WRITING_REVIEW_TASKS and schema is not None:
            return build_language_learning_writing_verification_config(
                rule=self.get_rule(type_name), schema=schema,
            )

        cache_key = self._build_cache_key(
            type_name=type_name,
            schema=schema,
        )

        if cache_key not in self._config_cache:
            rule = self.get_rule(type_name)

            if type_name == "AI_CHAT_REPLY" and schema is not None:
                self._config_cache[cache_key] = build_chat_ai_reply_config(
                    rule=rule,
                    schema=schema,
                )
            elif (
                type_name
                in {
                    "LANGUAGE_LEARNING_DAILY_WRITING_GENERATION",
                    "LANGUAGE_LEARNING_LEVEL_TEST_QUESTION",
                    "LANGUAGE_LEARNING_LEVEL_TEST_GENERATION",
                    "LANGUAGE_LEARNING_SPEAKING_CONVERSATION",
                    "LANGUAGE_LEARNING_LISTENING_GENERATION",
                    "LANGUAGE_LEARNING_LISTENING_EXPLANATION",
                }
                and schema is not None
            ):
                self._config_cache[cache_key] = (
                    build_language_learning_generation_config(
                        rule=rule,
                        schema=schema,
                    )
                )
            elif (
                type_name == "LANGUAGE_LEARNING_LEVEL_TEST_VOCAB_CONTEXT_DESIGN"
                and schema is not None
            ):
                self._config_cache[cache_key] = (
                    build_language_learning_vocab_design_config(
                        rule=rule,
                        schema=schema,
                    )
                )
            elif (
                type_name == "LANGUAGE_LEARNING_LEVEL_TEST_VOCAB_CONTEXT_REPAIR"
                and schema is not None
            ):
                self._config_cache[cache_key] = (
                    build_language_learning_vocab_repair_config(
                        rule=rule,
                        schema=schema,
                    )
                )
            elif (
                type_name == "LANGUAGE_LEARNING_LEVEL_TEST_CHOICE_VERIFICATION"
                and schema is not None
            ):
                self._config_cache[cache_key] = (
                    build_language_learning_choice_verification_config(
                        rule=rule,
                        schema=schema,
                    )
                )
            elif (
                type_name == "LANGUAGE_LEARNING_LEVEL_TEST_TASK_SUFFICIENCY_VERIFICATION"
                and schema is not None
            ):
                self._config_cache[cache_key] = (
                    build_language_learning_task_verification_config(
                        rule=rule,
                        schema=schema,
                    )
                )
            elif (
                type_name
                in {
                    "LANGUAGE_LEARNING_WRITING_EVALUATION",
                    "LANGUAGE_LEARNING_SPEAKING_EVALUATION",
                    "LANGUAGE_LEARNING_LISTENING_INTERPRETATION",
                    "LANGUAGE_LEARNING_LISTENING_SUMMARY_EVALUATION",
                    "LANGUAGE_LEARNING_LEVEL_TEST_SPEAKING_EVALUATION",
                }
                and schema is not None
            ):
                self._config_cache[cache_key] = (
                    build_language_learning_evaluation_config(
                        rule=rule,
                        schema=schema,
                    )
                )
            else:
                self._config_cache[cache_key] = build_default_gemini_config(
                    rule=rule,
                    schema=schema,
                )

        return self._config_cache[cache_key]

    def get_chat_translation_fast_config(self):
        cache_key = "CHAT_TRANSLATION_FAST"

        if cache_key not in self._config_cache:
            self._config_cache[cache_key] = build_fast_translation_config()

        return self._config_cache[cache_key]

    def get_voice_translation_config(self, schema: dict):
        cache_key = self._build_cache_key("VOICE_TRANSLATION_FAST", schema)
        if cache_key not in self._config_cache:
            self._config_cache[cache_key] = build_voice_translation_config(
                rule=self.get_rule("VOICE_TRANSLATION"),
                schema=schema,
                max_output_tokens=settings.AI_VOICE_TRANSLATION_MAX_OUTPUT_TOKENS,
            )
        return self._config_cache[cache_key]

    def _build_cache_key(
        self,
        type_name: str,
        schema: dict | None = None,
    ) -> str:
        if schema is None:
            return f"{type_name}_none"

        return f"{type_name}_{str(schema)}"
