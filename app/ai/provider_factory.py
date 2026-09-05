from app.ai.provider_pool import AiProviderPool
from app.ai.providers.gemini.client import GeminiService
from app.ai.providers.openai.client import OpenAIService
from app.core.config import settings


def create_text_generation_provider() -> AiProviderPool:
    pool = AiProviderPool()
    provider_names = [
        value.strip().lower()
        for value in settings.AI_TEXT_PROVIDER.split(",")
        if value.strip()
    ]

    for provider_name in provider_names:
        if provider_name == "openai":
            provider = OpenAIService()
        elif provider_name == "gemini":
            provider = GeminiService()
        else:
            raise ValueError(f"Unsupported AI_TEXT_PROVIDER: {provider_name}")

        pool.dock(
            name=provider_name,
            provider=provider,
            max_concurrency=settings.AI_TEXT_PROVIDER_MAX_CONCURRENCY,
            cooldown_seconds=settings.AI_TEXT_PROVIDER_FAILURE_COOLDOWN_SECONDS,
        )

    if not provider_names:
        raise ValueError("AI_TEXT_PROVIDER must contain at least one provider")
    return pool


def create_speech_synthesis_provider() -> GeminiService:
    """Keep current Gemini TTS isolated from the text-provider migration."""
    return GeminiService()
