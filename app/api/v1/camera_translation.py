"""BE가 인증하여 호출하는 카메라 전용 추가 OCR·영역 번역 진입점."""

from collections.abc import Awaitable, Callable
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response
from fastapi.routing import APIRoute

from app.ai.providers.openai.client import OpenAIService
from app.features.camera_translation.contracts import (
    CameraLanguage,
    CameraTranslationRequest,
    CameraTranslationResult,
    CommonOcrResult,
)
from app.features.camera_translation.errors import CameraServiceError
from app.features.camera_translation.ocr import recognize_crop
from app.features.camera_translation.segments import translate_segments
from app.services.ocr_service import OCRService


class _CameraRoute(APIRoute):
    """입력의 원문과 이미지 정보를 내부 API 오류 본문에서 제외한다."""

    def get_route_handler(self) -> Callable[[Request], Awaitable[Response]]:
        """검증 실패와 업무 오류를 고정된 코드로 반환한다."""
        route_handler = super().get_route_handler()

        # FastAPI의 기본 검증 응답에는 실패한 입력값이 포함될 수 있다.
        async def handle(request: Request) -> Response:
            try:
                return await route_handler(request)
            except RequestValidationError as exc:
                raise HTTPException(422, detail={"code": "INVALID_INPUT"}) from exc
            except CameraServiceError as exc:
                raise HTTPException(exc.status_code, detail={"code": exc.code}) from exc

        return handle


router = APIRouter(
    prefix="/camera-translation",
    tags=["Camera translation"],
    route_class=_CameraRoute,
)


def get_camera_ocr_service() -> OCRService:
    """기존 OCR 모델과 동시성 제한을 공유하는 의존성을 얻는다."""
    from app.api.dependencies import get_ocr_service

    return get_ocr_service()


def get_camera_model_provider() -> OpenAIService:
    """명시적 프롬프트와 비저장 실행을 지원하는 기존 제공자를 얻는다."""
    from app.api.dependencies import get_model_execution_provider

    return get_model_execution_provider()


@router.post("/ocr", response_model=CommonOcrResult)
async def camera_ocr(
    image: Annotated[UploadFile, File()],
    image_id: Annotated[UUID, Form(alias="imageId")],
    source_language: Annotated[CameraLanguage, Form(alias="sourceLanguage")],
    service: Annotated[OCRService, Depends(get_camera_ocr_service)],
) -> CommonOcrResult:
    """추가로 인식할 JPEG/PNG 범위 하나의 공통 OCR 결과를 반환한다."""
    return await recognize_crop(
        upload=image,
        image_id=image_id,
        source_language=source_language,
        ocr_service=service,
    )


@router.post("/segments", response_model=CameraTranslationResult)
async def camera_segments(
    request: CameraTranslationRequest,
    provider: Annotated[OpenAIService, Depends(get_camera_model_provider)],
) -> CameraTranslationResult:
    """텍스트 영역을 번역하고 각 입력 ID에 응답을 연결한다."""
    return await translate_segments(request, provider)
