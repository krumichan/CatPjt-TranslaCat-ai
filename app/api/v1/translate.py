from fastapi import APIRouter, Depends

from app.api.dependencies import get_translation_service
from app.features.translation.service import TranslationService
from app.schemas.translation import BatchTranslationRequest, SingleTranslationRequest

router = APIRouter(
    prefix="/translate",
    tags=["Translation"],
)


@router.post("/single")
async def translate_single(
    request: SingleTranslationRequest,
    service: TranslationService = Depends(get_translation_service),
):
    translated_text = await service.translate_single(
        text=request.text,
        type_name=request.type,
    )

    return {"translated": translated_text}


@router.post("/batch")
async def translate_batch(
    request: BatchTranslationRequest,
    service: TranslationService = Depends(get_translation_service),
):
    results = await service.translate_batch(
        texts=request.texts,
        type_name=request.type,
    )

    return {"translated": results}
