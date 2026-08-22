from fastapi import APIRouter

from app.api.internal import voice

internal_router = APIRouter()
internal_router.include_router(voice.router)
