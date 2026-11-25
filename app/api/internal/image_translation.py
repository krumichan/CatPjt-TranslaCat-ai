from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.features.image_translation.ocr import MAX_IMAGE_BYTES, ImageInputError
from app.features.image_translation.service import ImageOcrUnavailable, ImageTranslationService
from app.schemas.image_translation import ImageLanguage, ImagePageTranslation

router = APIRouter(prefix="/image-translation", tags=["Internal Image Translation"])


@lru_cache(maxsize=1)
def get_image_translation_service() -> ImageTranslationService:
    """アプリケーションが共有する OCR とテキスト翻訳をページ処理へ接続する。"""
    from app.api.dependencies import get_ocr_service, get_translation_service

    return ImageTranslationService(get_ocr_service(), get_translation_service())


@router.post("/page", response_model=ImagePageTranslation, response_model_exclude_unset=True)
async def translate_page(
    file: Annotated[UploadFile, File()],
    service: Annotated[ImageTranslationService, Depends(get_image_translation_service)],
    source_language: Annotated[ImageLanguage, Form(alias="sourceLanguage")],
    target_language: Annotated[ImageLanguage, Form(alias="targetLanguage")],
) -> ImagePageTranslation:
    """認証済み内部リクエストの画像一枚を OCR・翻訳し、ページ結果を返す。"""
    # 内部 HTTP 境界で上限を超えたファイルを読み込まず、未知の形式も後続で検証する。
    if file.size is not None and file.size > MAX_IMAGE_BYTES:
        raise HTTPException(413, detail={"code": "IMAGE_LIMIT_EXCEEDED", "retryable": False})
    contents = await file.read(MAX_IMAGE_BYTES + 1)

    # OCR の実行失敗を画像形式・言語の入力エラーと区別する。
    try:
        return await service.translate_page(
            contents, file.content_type or "", source_language, target_language
        )
    except ImageInputError as exc:
        raise HTTPException(
            exc.status_code, detail={"code": exc.code, "retryable": False}
        ) from exc
    except ImageOcrUnavailable as exc:
        raise HTTPException(
            503, detail={"code": "OCR_UNAVAILABLE", "retryable": True}
        ) from exc
