import asyncio
import io
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import HTTPException, UploadFile
from PIL import Image, ImageOps

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OCRLine:
    text: str
    bounding_box: list[list[float]] | None = None
    confidence: float | None = None

    def as_payload(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "bounding_box": self.bounding_box,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class OCRDocument:
    lines: list[OCRLine] = field(default_factory=list)

    @property
    def text(self) -> str:
        # Equal text at different positions belongs to different receipts.
        return "\n".join(line.text for line in self.lines)


class OCRService:
    def __init__(self) -> None:
        self._ocr_cache: dict[tuple[str, str], Any] = {}
        self._create_lock = asyncio.Lock()
        self._predict_lock = asyncio.Lock()

    async def warm_up(self) -> None:
        await self._get_ocr(settings.OCR_LANGUAGE)

    async def extract_text_from_upload(
        self,
        file: UploadFile,
        ocr_language: str | None = None,
    ) -> str:
        return (await self.extract_document_from_upload(file, ocr_language)).text

    async def extract_document_from_upload(
        self,
        file: UploadFile,
        ocr_language: str | None = None,
    ) -> OCRDocument:
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

            async with self._predict_lock:
                return await asyncio.to_thread(
                    self._extract_document_from_path,
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

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=processed_suffix
        ) as temp_file:
            temp_file.write(processed_contents)
            return temp_file.name

    def _preprocess_image(self, contents: bytes, suffix: str) -> tuple[bytes, str]:
        try:
            image = Image.open(io.BytesIO(contents))
            image = ImageOps.exif_transpose(image)

            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")

            image = self._resize_image_safely(image)

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
        except Exception as exc:
            logger.warning(
                "Receipt image preprocessing failed. error_type=%s", type(exc).__name__
            )
            return contents, suffix

    def _resize_image_safely(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        max_width = settings.OCR_MAX_IMAGE_WIDTH
        max_height = settings.OCR_MAX_IMAGE_HEIGHT
        max_pixels = settings.OCR_MAX_IMAGE_PIXELS

        ratio = min(
            max_width / width if width > max_width else 1.0,
            max_height / height if height > max_height else 1.0,
        )

        if width * height > max_pixels:
            pixel_ratio = (max_pixels / (width * height)) ** 0.5
            ratio = min(ratio, pixel_ratio)

        if ratio < 1.0:
            new_width = max(1, int(width * ratio))
            new_height = max(1, int(height * ratio))
            logger.info(
                "OCR 이미지 리사이즈: %sx%s -> %sx%s",
                width,
                height,
                new_width,
                new_height,
            )
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        return image

    async def _get_ocr(self, language: str) -> Any:
        cache_key = (language, settings.OCR_VERSION)

        if cache_key in self._ocr_cache:
            return self._ocr_cache[cache_key]

        async with self._create_lock:
            if cache_key not in self._ocr_cache:
                self._ocr_cache[cache_key] = await asyncio.to_thread(
                    self._create_ocr,
                    language,
                    settings.OCR_VERSION,
                )

        return self._ocr_cache[cache_key]

    def _create_ocr(self, language: str, ocr_version: str) -> Any:
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
                text_recognition_batch_size=settings.OCR_TEXT_RECOGNITION_BATCH_SIZE,
                text_det_limit_side_len=settings.OCR_TEXT_DET_LIMIT_SIDE_LEN,
                text_det_limit_type=settings.OCR_TEXT_DET_LIMIT_TYPE,
                enable_mkldnn=settings.OCR_ENABLE_MKLDNN,
                cpu_threads=settings.OCR_CPU_THREADS,
            )
        except (TypeError, ValueError):
            logger.info(
                "PaddleOCR 3.x 파라미터 초기화에 실패하여 2.x 호환 파라미터로 재시도합니다."
            )
            return PaddleOCR(
                use_angle_cls=True,
                lang=language,
                ocr_version=ocr_version,
                enable_mkldnn=settings.OCR_ENABLE_MKLDNN,
                cpu_threads=settings.OCR_CPU_THREADS,
            )

    def _extract_document_from_path(self, ocr: Any, image_path: str) -> OCRDocument:
        try:
            if hasattr(ocr, "predict"):
                result = ocr.predict(input=image_path)
            else:
                result = ocr.ocr(image_path, cls=True)

            lines = self._collect_lines(result)
        except Exception as exc:
            logger.warning("OCR failed. error_type=%s", type(exc).__name__)
            raise HTTPException(
                status_code=500,
                detail="OCR 처리 중 오류가 발생했습니다.",
            ) from exc

        return OCRDocument(lines=lines)

    def _collect_lines(self, value: Any) -> list[OCRLine]:
        if value is None:
            return []
        if not isinstance(value, (dict, list, tuple, str)) and hasattr(value, "json"):
            json_value = value.json() if callable(value.json) else value.json
            return self._collect_lines(json_value)
        if isinstance(value, dict):
            texts = value.get("rec_texts")
            if texts is not None:
                boxes = value.get("rec_polys")
                if boxes is None:
                    boxes = value.get("dt_polys")
                scores = value.get("rec_scores")
                return [
                    OCRLine(
                        text=str(text).strip(),
                        bounding_box=self._coerce_box(boxes[index])
                        if boxes is not None and index < len(boxes)
                        else None,
                        confidence=float(scores[index])
                        if scores is not None and index < len(scores)
                        else None,
                    )
                    for index, text in enumerate(texts)
                    if str(text).strip()
                ]
            lines: list[OCRLine] = []
            for nested in value.values():
                if isinstance(nested, (dict, list, tuple)):
                    lines.extend(self._collect_lines(nested))
            return lines
        if isinstance(value, (list, tuple)):
            if (
                len(value) == 2
                and isinstance(value[1], (list, tuple))
                and value[1]
                and isinstance(value[1][0], str)
            ):
                return [
                    OCRLine(
                        text=value[1][0].strip(),
                        bounding_box=self._coerce_box(value[0]),
                        confidence=float(value[1][1]) if len(value[1]) > 1 else None,
                    )
                ]
            return [line for item in value for line in self._collect_lines(item)]
        # Paddle predict can return a generator of per-image results.
        if not isinstance(value, str) and hasattr(value, "__iter__"):
            return [line for item in value for line in self._collect_lines(item)]
        return []

    @staticmethod
    def _coerce_box(value: Any) -> list[list[float]] | None:
        try:
            return [[float(point[0]), float(point[1])] for point in value]
        except (TypeError, ValueError, IndexError):
            return None
