from fastapi import APIRouter
from app.api.v1 import translate, stt

api_router = APIRouter()
api_router.include_router(translate.router)
api_router.include_router(stt.router)