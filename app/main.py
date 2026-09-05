import secrets
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.dependencies import (
    get_ai_provider,
    get_ocr_service,
    get_speech_runtime,
    get_speech_synthesis_provider,
    get_voice_speech_evidence_guard,
    get_voice_stream_service,
    get_voice_translation_service,
)
from app.api.internal import internal_router
from app.api.v1 import api_router
from app.core.config import settings
from app.core.config_logger import setup_logging
from app.core.openapi import set_custom_openapi
from app.schemas.voice_translation import VoiceErrorCode, VoiceStage

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.AI_VOICE_ENABLED:
        await get_voice_translation_service().warm_up()
        await get_voice_speech_evidence_guard().warm_up()
        await get_speech_runtime().warm_up()

    if settings.OCR_WARM_UP:
        await get_ocr_service().warm_up()

    try:
        yield
    finally:
        await get_voice_stream_service().shutdown()
        await get_speech_runtime().shutdown()
        await get_voice_speech_evidence_guard().shutdown()
        shutdown_provider = getattr(get_ai_provider(), "shutdown", None)
        if shutdown_provider is not None:
            await shutdown_provider()
        shutdown_speech_provider = getattr(
            get_speech_synthesis_provider(), "shutdown", None
        )
        if shutdown_speech_provider is not None:
            await shutdown_speech_provider()


app = FastAPI(
    title="Project Cat: AI Server",
    swagger_ui_parameters={"displayRequestDuration": True},
    lifespan=lifespan,
)

app.openapi = lambda: set_custom_openapi(app)


@app.middleware("http")
async def check_api_key_middleware(request: Request, call_next):
    exempt_paths = ["/", "/docs", "/redoc", "/openapi.json"]

    if request.url.path in exempt_paths:
        return await call_next(request)

    provided_api_key = request.headers.get("X-API-KEY")
    if (
        not settings.SERVER_API_KEY
        or not provided_api_key
        or not secrets.compare_digest(provided_api_key, settings.SERVER_API_KEY)
    ):
        if request.url.path.startswith("/internal/v1/"):
            return JSONResponse(
                status_code=401,
                content={
                    "detail": {
                        "code": VoiceErrorCode.UNAUTHORIZED.value,
                        "stage": VoiceStage.AUTH.value,
                        "retryable": False,
                        "message": "인증되지 않은 Internal Service 요청입니다.",
                    }
                },
            )
        return JSONResponse(
            status_code=401,
            content={"detail": "인증되지 않은 요청입니다."},
        )

    return await call_next(request)


app.include_router(api_router, prefix="/api/v1")
app.include_router(internal_router, prefix="/internal/v1")


@app.get("/")
def home():
    return {
        "message": "AI Server is Ready!",
        "version": "v1",
    }
