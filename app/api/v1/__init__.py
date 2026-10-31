from fastapi import APIRouter

from app.api.v1 import (
    receipt,
    spending_analysis,
    stt,
    translate,
)

api_router = APIRouter()

api_router.include_router(translate.router)
api_router.include_router(stt.router)
api_router.include_router(receipt.router)
api_router.include_router(spending_analysis.router)
