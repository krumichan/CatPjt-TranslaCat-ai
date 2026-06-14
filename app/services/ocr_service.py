import asyncio
import io
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import HTTPException, UploadFile
from PIL import Image, ImageOps

from app.core.config import settings

logger = logging.getLogger(__name__)


class OCRService:
    def __init__(self) -> None:
        self._ocr_cache: dict[tuple[str, str], Any] = {}
        self._lock = asyncio.Lock()
    
    async def warm_up(self) -> None:
        await self._get_ocr(settings.OCR_LANGUAGE)

    async def extract_text_from_upload(
        self,
        file: UploadFile,
        ocr_language: str | None = None,
    ) -> str:
        self._validate_file(file)

        contents = await file.read()

        if len(contents) > settings.OCR_MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="이미지 파일 크기가 너무 큽니다.",
            )

        suffix = Path(file.filename or "").suffix.lower()
        temp_file_path = self._write_temp_file(contents, suffix)

        try:
            language = (ocr_language or settings.OCR_LANGUAGE).strip()
            ocr = await self._get_ocr(language)

            return await asyncio.to_thread(
                self._extract_text_from_path,
                ocr,
                temp_file_path,
            )
        finally:
            try:
                os.remove(temp_file_path)
            except FileNotFoundError:
                pass
    
    def _validate_file(self, file: UploadFile) -> None:
        content_type = file.content_type
        suffix = Path(file.filename or "").suffix.lower()

        if content_type not in settings.OCR_ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=400,
                detail="지원하지 않는 이미지 형식입니다.",
            )

        if suffix not in settings.OCR_ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail="지원하지 않는 이미지 확장자입니다.",
            )
    
    def _write_temp_file(self, contents: bytes, suffix: str) -> str:
        processed_contents, processed_suffix = self._preprocess_image(contents, suffix)

        with tempfile.NamedTemporaryFile(delete=False, suffix=processed_suffix) as temp_file:
            temp_file.write(processed_contents)

            return temp_file.name
    
    def _preprocess_image(self, contents: bytes, suffix: str) -> tuple[bytes, str]:
        try:
            image = Image.open(io.BytesIO(contents))
            image = ImageOps.exif_transpose(image)

            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")

            width, height = image.size

            if width > settings.OCR_MAX_IMAGE_WIDTH:
                ratio = settings.OCR_MAX_IMAGE_WIDTH / width
                new_height = int(height * ratio)
                image = image.resize(
                    (settings.OCR_MAX_IMAGE_WIDTH, new_height),
                    Image.LANCZOS,
                )

            output = io.BytesIO()

            if image.mode == "L":
                image = image.convert("RGB")

            image.save(
                output,
                format="JPEG",
                quality=settings.OCR_IMAGE_QUALITY,
                optimize=True,
            )

            return output.getvalue(), ".jpg"
        except Exception:
            logger.warning(
                "이미지 전처리에 실패하여 원본 파일로 OCR을 진행합니다.",
                exc_info=True,
            )

            return contents, suffix

    async def _get_ocr(self, language: str) -> Any:
        cache_key = (language, settings.OCR_VERSION)

        if cache_key in self._ocr_cache:
            return self._ocr_cache[cache_key]

        async with self._lock:
            if cache_key not in self._ocr_cache:
                self._ocr_cache[cache_key] = await asyncio.to_thread(
                    self._create_ocr,
                    language,
                    settings.OCR_VERSION,
                )

        return self._ocr_cache[cache_key]

    def _create_ocr(
        self,
        language: str,
        ocr_version: str,
    ) -> Any:
        os.environ.setdefault(
            "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK",
            str(settings.PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK),
        )

        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR가 설치되어 있지 않습니다. requirements.txt를 확인해주세요."
            ) from exc

        try:
            return PaddleOCR(
                lang=language,
                ocr_version=ocr_version,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        except (TypeError, ValueError):
            logger.info(
                "PaddleOCR 3.x 파라미터 초기화에 실패하여 2.x 호환 파라미터로 재시도합니다."
            )

        return PaddleOCR(
            use_angle_cls=True,
            lang=language,
            ocr_version=ocr_version,
        )

    def _extract_text_from_path(self, ocr: Any, image_path: str) -> str:
        try:
            if hasattr(ocr, "predict"):
                result = ocr.predict(input=image_path)
            else:
                result = ocr.ocr(image_path, cls=True)

            texts = self._collect_texts(result)
        except Exception as exc:
            logger.exception("PaddleOCR 처리에 실패했습니다.")

            raise HTTPException(
                status_code=500,
                detail="OCR 처리 중 오류가 발생했습니다.",
            ) from exc

        normalized_texts = [text.strip() for text in texts if text.strip()]

        return "\n".join(dict.fromkeys(normalized_texts))

    def _collect_texts(self, value: Any) -> list[str]:
        texts: list[str] = []

        if value is None:
            return texts

        if isinstance(value, str):
            return [value]

        if isinstance(value, dict):
            for key in ("rec_texts", "texts"):
                nested_value = value.get(key)

                if isinstance(nested_value, list):
                    texts.extend(
                        str(item)
                        for item in nested_value
                        if item is not None
                    )

            for key in ("text", "label"):
                nested_value = value.get(key)

                if isinstance(nested_value, str):
                    texts.append(nested_value)

            for nested_value in value.values():
                if isinstance(nested_value, (dict, list, tuple)):
                    texts.extend(self._collect_texts(nested_value))

            return texts

        if isinstance(value, (list, tuple)):
            # PaddleOCR 2.x: [box, (text, score)] 구조 대응
            if (
                len(value) >= 2
                and isinstance(value[1], (list, tuple))
                and value[1]
                and isinstance(value[1][0], str)
            ):
                texts.append(value[1][0])

            for item in value:
                texts.extend(self._collect_texts(item))

            return texts

        if hasattr(value, "json"):
            json_attr = getattr(value, "json")

            try:
                json_value = json_attr() if callable(json_attr) else json_attr
            except TypeError:
                json_value = None

            if isinstance(json_value, dict):
                texts.extend(self._collect_texts(json_value))

        if hasattr(value, "__dict__"):
            texts.extend(self._collect_texts(vars(value)))

        return texts