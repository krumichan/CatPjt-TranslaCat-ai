from app.ai.ports import ChatTranslationProvider


class ChatTranslationService:
    def __init__(self, provider: ChatTranslationProvider) -> None:
        self.provider = provider

    async def translate(
        self,
        text: str,
        target_language_code: str,
        source_language_code: str | None = None,
    ) -> str:
        return await self.provider.translate_chat_message(
            text=text,
            target_language_code=target_language_code,
            source_language_code=source_language_code,
        )
