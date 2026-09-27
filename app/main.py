import secrets
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.dependencies import (
    get_ai_provider,
    get_model_execution_provider,
    get_ocr_service,
    get_speaking_speech_runtime,
    get_speech_runtime,
    get_speech_synthesis_provider,
    get_voice_speech_evidence_guard,
    get_voice_stream_service,
    get_voice_translation_service,
)
from app.api.internal import internal_router
from app.api.internal.speech_transcription import get_speech_evidence_runtime
from app.api.v1 import api_router
from app.core.chat_auth import authorize_chat_api_key, validate_chat_auth_configuration
from app.core.config import settings
from app.core.config_logger import setup_logging
from app.core.openapi import set_custom_openapi
from app.schemas.voice_translation import VoiceErrorCode, VoiceStage

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 명시 활성화한 Chat 인증 오류는 Provider warm-up보다 먼저, 비밀값 없이 거부한다.
    validate_chat_auth_configuration(settings)

    if settings.AI_VOICE_ENABLED:
        await get_voice_translation_service().warm_up()
        await get_voice_speech_evidence_guard().warm_up()
        await get_speech_runtime().warm_up()

    if settings.OCR_WARM_UP and not settings.RECEIPT_VISION_DISABLE_OCR_WARMUP:
        await get_ocr_service().warm_up()

    try:
        yield
    finally:
        await get_voice_stream_service().shutdown()
        await get_speech_runtime().shutdown()
        if get_speaking_speech_runtime() is not get_speech_runtime():
            await get_speaking_speech_runtime().shutdown()
        # LL이 요청하는 범용 음성 근거 검사기의 자원을 종료한다.
        await get_speech_evidence_runtime().shutdown()
        await get_voice_speech_evidence_guard().shutdown()
        shutdown_provider = getattr(get_ai_provider(), "shutdown", None)
        if shutdown_provider is not None:
            await shutdown_provider()
        await get_model_execution_provider().shutdown()
        shutdown_speech_provider = getattr(get_speech_synthesis_provider(), "shutdown", None)
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

    # 전용 Chat 키는 범용 모델 실행 POST에만 유효하다. LL/Voice 키와 오류 wire는 유지한다.
    provided_api_key = request.headers.get("X-API-KEY")
    chat_decision = authorize_chat_api_key(
        settings, request.method, request.url.path, provided_api_key
    )
    if chat_decision is True:
        return await call_next(request)

    if chat_decision is None and request.url.path in exempt_paths:
        return await call_next(request)

    if (
        chat_decision is False
        or not settings.SERVER_API_KEY
        or not provided_api_key
        or not provided_api_key.isascii()
        or not secrets.compare_digest(provided_api_key, settings.SERVER_API_KEY)
    ):
        if request.url.path.startswith("/internal/v1/"):
            if request.url.path == "/internal/v1/model/execute":
                return JSONResponse(
                    status_code=401,
                    content={
                        "detail": {"code": "MODEL_EXECUTION_UNAUTHORIZED", "retryable": False}
                    },
                )
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
