from app.ai.provider_pool import AiProviderPool
from app.ai.providers.openai.client import OpenAIService
from app.ai.providers.openai.speech import OpenAISpeechService
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
            # Historical Gemini artifacts are read from storage, never created
            # by routing a new production request to a paid Gemini endpoint.
            raise ValueError("Gemini generation is disabled for new requests")
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


def create_speech_synthesis_provider() -> OpenAISpeechService:
    return OpenAISpeechService()
