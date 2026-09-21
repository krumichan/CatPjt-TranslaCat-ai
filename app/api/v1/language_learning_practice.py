from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_language_learning_reading_vocabulary_service
from app.features.language_learning.reading_vocabulary.service import ReadingVocabularyGenerationService
from app.schemas.language_learning_practice import (
    PracticeDomain,
    PracticeGenerationRequest,
    PracticeGenerationResponse,
)

router = APIRouter(
    prefix="/language-learning/practice",
    tags=["Language Learning - Reading/Vocabulary"],
)


@router.post("/generate", response_model=PracticeGenerationResponse)
async def generate_practice(
    request: PracticeGenerationRequest,
    service: ReadingVocabularyGenerationService = Depends(
        get_language_learning_reading_vocabulary_service
    ),
) -> PracticeGenerationResponse:
    # Preserve legacy DTOs and offline fixtures, but never start new paid work
    # for the retired standalone product (including plan-only requests).
    if request.domain == PracticeDomain.VOCABULARY:
        raise HTTPException(
            status_code=410,
            detail={"code": "DAILY_VOCABULARY_RETIRED", "policyVersion": "reading-first-v1"},
        )
    return await service.generate(request)
