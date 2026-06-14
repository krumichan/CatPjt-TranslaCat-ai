from contextlib import asynccontextmanager

from app.core.config_logger import setup_logging

setup_logging()

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.dependencies import get_ocr_service
from app.api.v1 import api_router
from app.core.config import settings
from app.core.openapi import set_custom_openapi


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.OCR_WARM_UP:
        await get_ocr_service().warm_up()

    yield

app = FastAPI(
    title="Project Cat: AI Server",
    swagger_ui_parameters={"displayRequestDuration": True},
    lifespan=lifespan,
)

# --- OpenAPI 설정 적용 ---
app.openapi = lambda: set_custom_openapi(app)

# --- 미들웨어 (보안) ---
@app.middleware("http")
async def check_api_key_middleware(request: Request, call_next):
    EXEMPT_PATHS = ["/", "/docs", "/redoc", "/openapi.json"]
    
    if request.url.path in EXEMPT_PATHS:
        return await call_next(request)

    if request.headers.get("X-API-KEY") != settings.SERVER_API_KEY:
        return JSONResponse(
            status_code=401,
            content={"detail": "인증되지 않은 요청입니다."}
        )

    return await call_next(request)

app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def home():
    return {"message": "AI Server is Ready!", "version": "v1"}