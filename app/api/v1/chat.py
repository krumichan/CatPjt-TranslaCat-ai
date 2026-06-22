from fastapi import APIRouter, Depends

from app.api.dependencies import get_chat_translation_service
from app.features.chat_translation.service import ChatTranslationService
from app.schemas.chat import ChatTranslationRequest, ChatTranslationResponse

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


@router.post("/translate", response_model=ChatTranslationResponse)
async def translate_chat_message(
    request: ChatTranslationRequest,
    service: ChatTranslationService = Depends(get_chat_translation_service),
) -> ChatTranslationResponse:
    translated_text = await service.translate(
        text=request.text,
        target_language_code=request.target_language_code,
        source_language_code=request.source_language_code,
    )

    return ChatTranslationResponse(
        translated_text=translated_text
    )
