from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import ValidationError

from app.api.dependencies import (
    get_language_learning_speaking_audio_store,
    get_language_learning_speaking_conversation_service,
    get_language_learning_speaking_evaluation_service,
    get_language_learning_speaking_tts_service,
    get_language_learning_speaking_turn_service,
)
from app.features.language_learning.speaking.audio_store import TemporaryTtsAudioStore
from app.features.language_learning.speaking.conversation_service import SpeakingConversationService
from app.features.language_learning.speaking.errors import SpeakingStageException
from app.features.language_learning.speaking.evaluation_service import SpeakingEvaluationService
from app.features.language_learning.speaking.tts_service import SpeakingTtsService
from app.features.language_learning.speaking.turn_service import SpeakingTurnService
from app.schemas.language_learning_speaking import (
    ConversationGenerationRequest,
    ConversationGenerationResponse,
    SessionStartRequest,
    SessionStartResponse,
    SpeakingEvaluationRequest,
    SpeakingEvaluationResponse,
    SttRequestContext,
    SttResponse,
    TtsRequest,
    TtsResponse,
    TurnProcessResponse,
)

router = APIRouter(
    prefix="/language-learning/speaking",
    tags=["Language Learning - Speaking"],
)


def _raise_speaking_error(exc: SpeakingStageException) -> None:
    status_code = 422 if not exc.retryable else 502
    if exc.code.value == "PROVIDER_TIMEOUT":
        status_code = 504
    elif exc.code.value == "PROVIDER_RATE_LIMITED":
        status_code = 429
    raise HTTPException(
        status_code=status_code,
        detail=exc.to_schema().model_dump(mode="json", by_alias=True),
    ) from exc


@router.post("/sessions/start", response_model=SessionStartResponse)
async def start_speaking_session(
    request: SessionStartRequest,
    service: SpeakingTurnService = Depends(get_language_learning_speaking_turn_service),
) -> SessionStartResponse:
    try:
        return await service.start_session(request)
    except SpeakingStageException as exc:
        _raise_speaking_error(exc)
        raise AssertionError("unreachable")


@router.post("/turns/transcribe", response_model=SttResponse)
async def transcribe_speaking_turn(
    context: str = Form(...),
    audio: UploadFile = File(...),
    service: SpeakingTurnService = Depends(get_language_learning_speaking_turn_service),
) -> SttResponse:
    try:
        request = SttRequestContext.model_validate_json(context)
        audio_bytes = await audio.read()
        return await service.transcribe_audio(
            context=request,
            audio_bytes=audio_bytes,
            file_name=audio.filename,
            content_type=audio.content_type,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    except SpeakingStageException as exc:
        _raise_speaking_error(exc)
        raise AssertionError("unreachable")


@router.post("/turns/respond", response_model=ConversationGenerationResponse)
async def generate_speaking_response(
    request: ConversationGenerationRequest,
    service: SpeakingConversationService = Depends(
        get_language_learning_speaking_conversation_service
    ),
) -> ConversationGenerationResponse:
    try:
        return await service.generate(request)
    except SpeakingStageException as exc:
        _raise_speaking_error(exc)
        raise AssertionError("unreachable")


@router.post("/turns/process", response_model=TurnProcessResponse)
async def process_speaking_turn(
    context: str = Form(...),
    audio: UploadFile = File(...),
    service: SpeakingTurnService = Depends(get_language_learning_speaking_turn_service),
) -> TurnProcessResponse:
    try:
        request = ConversationGenerationRequest.model_validate_json(context)
        audio_bytes = await audio.read()
        return await service.process_turn(
            context=request,
            audio_bytes=audio_bytes,
            file_name=audio.filename,
            content_type=audio.content_type,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc


@router.post("/tts", response_model=TtsResponse)
async def synthesize_speaking_tts(
    request: TtsRequest,
    service: SpeakingTtsService = Depends(get_language_learning_speaking_tts_service),
) -> TtsResponse:
    try:
        return await service.synthesize(request)
    except SpeakingStageException as exc:
        _raise_speaking_error(exc)
        raise AssertionError("unreachable")


@router.get("/audio/{audio_reference}")
async def get_speaking_audio(
    audio_reference: str,
    store: TemporaryTtsAudioStore = Depends(get_language_learning_speaking_audio_store),
):
    stored = store.get(audio_reference)
    if stored is None:
        raise HTTPException(status_code=404, detail="TTS Audio를 찾을 수 없습니다.")
    return FileResponse(
        stored.path,
        media_type=stored.content_type,
        filename=f"{audio_reference}.wav",
    )


@router.post("/evaluate", response_model=SpeakingEvaluationResponse)
async def evaluate_speaking_session(
    request: SpeakingEvaluationRequest,
    service: SpeakingEvaluationService = Depends(
        get_language_learning_speaking_evaluation_service
    ),
) -> SpeakingEvaluationResponse:
    try:
        return await service.evaluate(request)
    except SpeakingStageException as exc:
        _raise_speaking_error(exc)
        raise AssertionError("unreachable")
