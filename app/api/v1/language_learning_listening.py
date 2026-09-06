from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import ValidationError

from app.api.dependencies import (
    get_language_learning_listening_audio_store,
    get_language_learning_listening_comprehension_service,
    get_language_learning_listening_dictation_service,
    get_language_learning_listening_explanation_service,
    get_language_learning_listening_generation_service,
    get_language_learning_listening_interpretation_service,
    get_language_learning_listening_repeat_service,
    get_language_learning_listening_summary_service,
    get_language_learning_listening_tts_service,
)
from app.features.language_learning.listening.audio_store import (
    TemporaryListeningAudioStore,
)
from app.features.language_learning.listening.comprehension_service import ListeningComprehensionService
from app.features.language_learning.listening.dictation_service import (
    ListeningDictationService,
)
from app.features.language_learning.listening.errors import ListeningStageException
from app.features.language_learning.listening.explanation_service import (
    ListeningExplanationService,
)
from app.features.language_learning.listening.generation_service import (
    ListeningGenerationService,
)
from app.features.language_learning.listening.interpretation_service import (
    ListeningInterpretationService,
)
from app.features.language_learning.listening.repeat_service import (
    ListeningRepeatService,
)
from app.features.language_learning.listening.summary_service import ListeningSummaryService
from app.features.language_learning.listening.tts_service import ListeningTtsService
from app.schemas.language_learning_listening import (
    ComprehensionEvaluationRequest,
    DictationEvaluationRequest,
    InterpretationEvaluationRequest,
    ListeningEvaluationResponse,
    ListeningSetGenerationRequest,
    ListeningSetGenerationResponse,
    ListeningTtsRequest,
    ListeningTtsResponse,
    RecommendationExplanationRequest,
    RecommendationExplanationResponse,
    RepeatEvaluationContext,
    SummaryEvaluationRequest,
)

logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/language-learning/listening",
    tags=["Language Learning - Listening"],
)


def _raise_listening_error(exc: ListeningStageException) -> None:
    status_code = 502 if exc.retryable else 422
    if exc.code.value == "PROVIDER_TIMEOUT":
        status_code = 504
    elif exc.code.value == "PROVIDER_RATE_LIMITED":
        status_code = 429
    raise HTTPException(
        status_code=status_code,
        detail=exc.to_schema().model_dump(mode="json", by_alias=True),
    ) from exc


@router.post("/sets/generate", response_model=ListeningSetGenerationResponse)
async def generate_listening_set(
    request: ListeningSetGenerationRequest,
    service: ListeningGenerationService = Depends(
        get_language_learning_listening_generation_service
    ),
) -> ListeningSetGenerationResponse:
    try:
        return await service.generate(request)
    except ListeningStageException as exc:
        _raise_listening_error(exc)
        raise AssertionError("unreachable")


@router.post("/tts", response_model=ListeningTtsResponse)
async def synthesize_listening_tts(
    request: ListeningTtsRequest,
    service: ListeningTtsService = Depends(get_language_learning_listening_tts_service),
) -> ListeningTtsResponse:
    try:
        return await service.synthesize(request)
    except ListeningStageException as exc:
        _raise_listening_error(exc)
        raise AssertionError("unreachable")


@router.get("/audio/{audio_reference}")
async def get_listening_audio(
    audio_reference: str,
    store: TemporaryListeningAudioStore = Depends(
        get_language_learning_listening_audio_store
    ),
):
    logger.info(
        "Listening audio download requested. audio_reference=%s",
        audio_reference,
    )
    stored = store.get(audio_reference)
    if stored is None:
        logger.warning(
            "Listening audio download miss. audio_reference=%s",
            audio_reference,
        )
        raise HTTPException(
            status_code=404, detail="Listening TTS Audio를 찾을 수 없습니다."
        )
    logger.info(
        "Listening audio download hit. audio_reference=%s bytes=%d "
        "content_type=%s duration_seconds=%.3f checksum=%s",
        audio_reference,
        stored.path.stat().st_size,
        stored.content_type,
        stored.duration_seconds,
        stored.checksum[:12],
    )
    return FileResponse(
        stored.path,
        media_type=stored.content_type,
        filename=f"{audio_reference}.audio",
    )


@router.post("/evaluate/dictation", response_model=ListeningEvaluationResponse)
async def evaluate_dictation(
    request: DictationEvaluationRequest,
    service: ListeningDictationService = Depends(
        get_language_learning_listening_dictation_service
    ),
) -> ListeningEvaluationResponse:
    return await service.evaluate(request)


@router.post("/evaluate/comprehension", response_model=ListeningEvaluationResponse)
async def evaluate_comprehension(
    request: ComprehensionEvaluationRequest,
    service: ListeningComprehensionService = Depends(
        get_language_learning_listening_comprehension_service
    ),
) -> ListeningEvaluationResponse:
    return await service.evaluate(request)


@router.post("/evaluate/interpretation", response_model=ListeningEvaluationResponse)
async def evaluate_interpretation(
    request: InterpretationEvaluationRequest,
    service: ListeningInterpretationService = Depends(
        get_language_learning_listening_interpretation_service
    ),
) -> ListeningEvaluationResponse:
    try:
        return await service.evaluate(request)
    except ListeningStageException as exc:
        _raise_listening_error(exc)
        raise AssertionError("unreachable")


@router.post("/evaluate/summary", response_model=ListeningEvaluationResponse)
async def evaluate_summary(
    request: SummaryEvaluationRequest,
    service: ListeningSummaryService = Depends(
        get_language_learning_listening_summary_service
    ),
) -> ListeningEvaluationResponse:
    try:
        return await service.evaluate(request)
    except ListeningStageException as exc:
        _raise_listening_error(exc)
        raise AssertionError("unreachable")


@router.post("/evaluate/repeat", response_model=ListeningEvaluationResponse)
async def evaluate_repeat(
    context: str = Form(...),
    audio: UploadFile = File(...),
    service: ListeningRepeatService = Depends(
        get_language_learning_listening_repeat_service
    ),
) -> ListeningEvaluationResponse:
    try:
        request = RepeatEvaluationContext.model_validate_json(context)
        audio_bytes = await audio.read()
        return await service.evaluate(
            request,
            audio_bytes=audio_bytes,
            file_name=audio.filename,
            content_type=audio.content_type,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    except ListeningStageException as exc:
        _raise_listening_error(exc)
        raise AssertionError("unreachable")


@router.post(
    "/recommendations/explain",
    response_model=RecommendationExplanationResponse,
)
async def explain_listening_recommendation(
    request: RecommendationExplanationRequest,
    service: ListeningExplanationService = Depends(
        get_language_learning_listening_explanation_service
    ),
) -> RecommendationExplanationResponse:
    try:
        return await service.explain(request)
    except ListeningStageException as exc:
        _raise_listening_error(exc)
        raise AssertionError("unreachable")
