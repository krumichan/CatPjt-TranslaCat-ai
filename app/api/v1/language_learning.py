from fastapi import APIRouter, Depends

from app.api.dependencies import get_language_learning_writing_service
from app.features.language_learning.writing.service import LanguageLearningWritingService
from app.schemas.language_learning import (
    DailyWritingGenerationRequest,
    DailyWritingGenerationResponse,
    LevelTestQuestionRequest,
    LevelTestQuestionResponse,
    WritingEvaluationRequest,
    WritingEvaluationResponse,
)

router = APIRouter(
    prefix="/language-learning/writing",
    tags=["Language Learning - Writing"],
)


@router.post(
    "/daily/generate",
    response_model=DailyWritingGenerationResponse,
)
async def generate_daily_writing(
    request: DailyWritingGenerationRequest,
    service: LanguageLearningWritingService = Depends(
        get_language_learning_writing_service
    ),
) -> DailyWritingGenerationResponse:
    return await service.generate_daily(request)


@router.post(
    "/evaluate",
    response_model=WritingEvaluationResponse,
)
async def evaluate_writing(
    request: WritingEvaluationRequest,
    service: LanguageLearningWritingService = Depends(
        get_language_learning_writing_service
    ),
) -> WritingEvaluationResponse:
    return await service.evaluate(request)


@router.post(
    "/level-test/question",
    response_model=LevelTestQuestionResponse,
)
async def generate_level_test_question(
    request: LevelTestQuestionRequest,
    service: LanguageLearningWritingService = Depends(
        get_language_learning_writing_service
    ),
) -> LevelTestQuestionResponse:
    return await service.generate_level_test_question(request)
