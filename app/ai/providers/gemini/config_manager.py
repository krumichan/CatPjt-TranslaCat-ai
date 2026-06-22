from fastapi import HTTPException

from app.ai.prompt_registry import get_prompt_rule
from app.ai.providers.gemini.configs import (
    build_default_gemini_config,
    build_fast_translation_config,
)


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

    def get_chat_translation_fast_config(self):
        cache_key = "CHAT_TRANSLATION_FAST"

        if cache_key not in self._config_cache:
            self._config_cache[cache_key] = build_fast_translation_config()

        return self._config_cache[cache_key]

    def _build_cache_key(
        self,
        type_name: str,
        schema: dict | None = None,
    ) -> str:
        if schema is None:
            return f"{type_name}_none"

        return f"{type_name}_{str(schema)}"
