from fastapi import APIRouter

from app.api.v1 import (
    chat,
    language_learning,
    language_learning_listening,
    language_learning_level_test,
    language_learning_speaking,
    receipt,
    stt,
    translate,
)

api_router = APIRouter()

api_router.include_router(translate.router)
api_router.include_router(stt.router)
api_router.include_router(receipt.router)
api_router.include_router(chat.router)
api_router.include_router(language_learning.router)
api_router.include_router(language_learning_listening.router)
api_router.include_router(language_learning_level_test.router)
api_router.include_router(language_learning_speaking.router)
