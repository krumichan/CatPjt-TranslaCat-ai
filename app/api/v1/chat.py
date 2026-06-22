from fastapi import APIRouter, Depends

from app.api.dependencies import get_gemini_service
from app.schemas.chat import ChatTranslationRequest, ChatTranslationResponse
from app.services.gemini_service import GeminiService

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


@router.post("/translate", response_model=ChatTranslationResponse)
async def translate_chat_message(
    request: ChatTranslationRequest,
    service: GeminiService = Depends(get_gemini_service),
) -> ChatTranslationResponse:
    translated_text = await service.translate_chat_message(
        text=request.text,
        target_language_code=request.target_language_code,
        source_language_code=request.source_language_code,
    )

    return ChatTranslationResponse(
        translated_text=translated_text,
    )