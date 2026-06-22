from app.ai.providers.gemini.client import GeminiService
from app.core.config import settings


def create_text_generation_provider():
    provider = settings.AI_TEXT_PROVIDER.strip().lower()

    if provider == "gemini":
        return GeminiService()

    raise ValueError(f"Unsupported AI_TEXT_PROVIDER: {settings.AI_TEXT_PROVIDER}")
