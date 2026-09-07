from fastapi import APIRouter, Depends

from app.api.dependencies import get_language_learning_reading_vocabulary_service
from app.features.language_learning.reading_vocabulary.service import ReadingVocabularyGenerationService
from app.schemas.language_learning_practice import PracticeGenerationRequest, PracticeGenerationResponse

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
    return await service.generate(request)
