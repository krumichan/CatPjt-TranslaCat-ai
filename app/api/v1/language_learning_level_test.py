from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import ValidationError

from app.api.dependencies import get_language_learning_level_test_service
from app.features.language_learning.level_test.service import LevelTestService
from app.schemas.language_learning_level_test import (
    LevelTestEvaluationResponse,
    LevelTestQuestionGenerationRequest,
    LevelTestQuestionGenerationResponse,
    LevelTestSpeakingEvaluationContext,
    LevelTestTextEvaluationRequest,
)

router = APIRouter(
    prefix="/language-learning/level-test",
    tags=["Language Learning - Level Test"],
)


@router.post("/questions/generate", response_model=LevelTestQuestionGenerationResponse)
async def generate_level_test_question(
    request: LevelTestQuestionGenerationRequest,
    service: LevelTestService = Depends(get_language_learning_level_test_service),
) -> LevelTestQuestionGenerationResponse:
    return await service.generate_question(request)


@router.post("/evaluate/text", response_model=LevelTestEvaluationResponse)
async def evaluate_level_test_text(
    request: LevelTestTextEvaluationRequest,
    service: LevelTestService = Depends(get_language_learning_level_test_service),
) -> LevelTestEvaluationResponse:
    return await service.evaluate_text(request)


@router.post("/evaluate/speaking", response_model=LevelTestEvaluationResponse)
async def evaluate_level_test_speaking(
    context: str = Form(...),
    audio: UploadFile = File(...),
    service: LevelTestService = Depends(get_language_learning_level_test_service),
) -> LevelTestEvaluationResponse:
    try:
        request = LevelTestSpeakingEvaluationContext.model_validate_json(context)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    return await service.evaluate_speaking(
        request,
        audio_bytes=await audio.read(),
        file_name=audio.filename,
        content_type=audio.content_type,
    )
