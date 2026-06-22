from fastapi import APIRouter

from app.api.v1 import chat, receipt, stt, translate

api_router = APIRouter()

api_router.include_router(translate.router)
api_router.include_router(stt.router)
api_router.include_router(receipt.router)
api_router.include_router(chat.router)
