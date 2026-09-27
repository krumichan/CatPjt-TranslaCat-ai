from fastapi import HTTPException

from app.ai.prompt_registry import get_prompt_rule
from app.ai.providers.gemini.configs import (
    build_default_gemini_config,
    build_voice_translation_config,
)
from app.core.config import settings


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
        cache_key = self._build_cache_key(
            type_name=type_name,
            schema=schema,
        )

        if cache_key not in self._config_cache:
            rule = self.get_rule(type_name)

            self._config_cache[cache_key] = build_default_gemini_config(
                rule=rule,
                schema=schema,
            )

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
