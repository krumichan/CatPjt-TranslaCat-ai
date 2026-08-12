from fastapi import APIRouter

from app.api.v1 import chat, language_learning, receipt, stt, translate

api_router = APIRouter()

api_router.include_router(translate.router)
api_router.include_router(stt.router)
api_router.include_router(receipt.router)
api_router.include_router(chat.router)
api_router.include_router(language_learning.router)
