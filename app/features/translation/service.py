from typing import List

from app.ai.ports import TextGenerationProvider


class TranslationService:
    def __init__(self, provider: TextGenerationProvider) -> None:
        self.provider = provider

    async def translate_single(self, text: str, type_name: str) -> str:
        result = await self.provider.call(type_name=type_name, data=text)
        return str(result or "")

    async def translate_batch(self, texts: List[str], type_name: str) -> List[str]:
        if not hasattr(self.provider, "translate_batch"):
            raise RuntimeError("Current AI provider does not support batch translation.")

        return await self.provider.translate_batch(texts, type_name)  # type: ignore[attr-defined]
