from fastapi import APIRouter

from app.api.internal import model_execution, speech_execution, speech_transcription, voice

internal_router = APIRouter()
internal_router.include_router(voice.router)
internal_router.include_router(model_execution.router)
internal_router.include_router(speech_execution.router)
internal_router.include_router(speech_transcription.router)
