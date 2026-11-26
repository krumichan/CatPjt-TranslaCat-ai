"""업로드한 범위의 PaddleOCR 결과를 공통 v1 좌표 공간으로 옮긴다."""

import io
import math
from importlib.metadata import PackageNotFoundError, version
from typing import Literal, cast
from uuid import UUID

from anyio import current_time, fail_after
from anyio.to_thread import run_sync
from fastapi import UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

from app.core.config import settings
from app.features.camera_translation.contracts import (
    Affine,
    CameraLanguage,
    CommonOcrResult,
    OcrEngine,
    OcrImageGeometry,
    OcrRegion,
    Point,
)
from app.features.camera_translation.errors import CameraServiceError
from app.services.ocr_service import OCRDocument, OCRService

_UPLOAD_LIMIT_BYTES = 8 * 1024 * 1024
_MAX_EDGE_PX = 3072
_MAX_PIXELS = 12_000_000
_OCR_TIMEOUT_SECONDS = 7.5
_INPUT_FORMATS = {"image/jpeg": "JPEG", "image/png": "PNG"}
_PADDLE_LANGUAGES = {"ja": "japan", "ko": "korean", "en": "en"}


async def recognize_crop(
    *,
    upload: UploadFile,
    image_id: UUID,
    source_language: CameraLanguage,
    ocr_service: OCRService,
) -> CommonOcrResult:
    """유효한 JPEG/PNG 한 장을 인식하고 방향이 보정된 이미지 좌표를 반환한다."""
    # MIME와 바이트 한도를 디코딩 및 모델 사용보다 먼저 확인한다.
    expected_format = _INPUT_FORMATS.get(upload.content_type or "")
    if expected_format is None:
        raise CameraServiceError(415, "OCR_DECODE_FAILED")
    uploaded_bytes = await upload.read(_UPLOAD_LIMIT_BYTES + 1)
    if len(uploaded_bytes) > _UPLOAD_LIMIT_BYTES:
        raise CameraServiceError(413, "OCR_DECODE_FAILED")

    # 입력 디코딩과 모델 대기·실행을 같은 시간 제한에 둔다.
    paddle_language = _PADDLE_LANGUAGES[source_language]
    try:
        with fail_after(_OCR_TIMEOUT_SECONDS) as deadline:
            image, input_to_image = await run_sync(
                _decode_crop, uploaded_bytes, expected_format
            )
            document = await ocr_service.extract_document_from_image(
                image, ocr_language=paddle_language, ocr_version=settings.OCR_VERSION
            )
            if current_time() >= deadline.deadline:
                raise TimeoutError
    except CameraServiceError:
        raise
    except TimeoutError as exc:
        raise CameraServiceError(504, "OCR_DEADLINE_EXCEEDED") from exc
    except Exception as exc:
        raise CameraServiceError(503, "OCR_UNAVAILABLE") from exc

    # 공용 어댑터의 인식 좌표는 보정해 제출한 이미지와 동일한 픽셀 공간이다.
    regions, has_invalid_geometry = _regions_in_image(document, image.size)
    status: Literal["OK", "NO_TEXT", "UNCERTAIN"] = "NO_TEXT"
    if has_invalid_geometry:
        status = "UNCERTAIN"
    elif regions:
        status = "OK"
    return CommonOcrResult(
        image_id=image_id,
        image=OcrImageGeometry(
            width_px=image.width,
            height_px=image.height,
            input_to_image=input_to_image,
        ),
        status=status,
        regions=regions,
        engine=OcrEngine(
            provider="paddleocr",
            model=f"{paddle_language}_{settings.OCR_VERSION}",
            version=_paddle_version(),
        ),
    )


# 실제 이미지 형식과 크기를 확인하고 카메라의 EXIF 방향을 보정한다.
def _decode_crop(contents: bytes, expected_format: str) -> tuple[Image.Image, Affine]:
    try:
        with Image.open(io.BytesIO(contents), formats=["JPEG", "PNG"]) as raw_image:
            width, height = raw_image.size
            if raw_image.format != expected_format or getattr(raw_image, "n_frames", 1) != 1:
                raise CameraServiceError(415, "OCR_DECODE_FAILED")
            if max(width, height) > _MAX_EDGE_PX or width * height > _MAX_PIXELS:
                raise CameraServiceError(413, "OCR_DECODE_FAILED")

            # 픽셀 모서리 좌표의 방향 변환을 원본 EXIF에서 읽어 둔다.
            orientation = raw_image.getexif().get(274, 1)
            transform = _orientation_transform(orientation, width, height)
            with ImageOps.exif_transpose(raw_image) as oriented:
                return oriented.convert("RGB"), transform
    except Image.DecompressionBombError as exc:
        raise CameraServiceError(413, "OCR_DECODE_FAILED") from exc
    except (UnidentifiedImageError, OSError, TypeError, ValueError, SyntaxError) as exc:
        raise CameraServiceError(400, "OCR_DECODE_FAILED") from exc


# EXIF 전체 방향을 보정 전 → 보정 후 픽셀 모서리 좌표로 표현한다.
def _orientation_transform(orientation: int, width: int, height: int) -> Affine:
    transforms: dict[int, Affine] = {
        1: (1, 0, 0, 0, 1, 0),
        2: (-1, 0, width, 0, 1, 0),
        3: (-1, 0, width, 0, -1, height),
        4: (1, 0, 0, 0, -1, height),
        5: (0, 1, 0, 1, 0, 0),
        6: (0, -1, height, 1, 0, 0),
        7: (0, -1, height, -1, 0, width),
        8: (0, 1, 0, -1, 0, width),
    }
    return transforms.get(orientation, transforms[1])


# 유효한 영역의 읽기 순서를 유지하고 좌표가 빠진 인식은 불확실로 알린다.
def _regions_in_image(
    document: OCRDocument,
    image_size: tuple[int, int],
) -> tuple[list[OcrRegion], bool]:
    regions = []
    has_invalid_geometry = False
    for line in document.lines:
        source_text = line.text.strip()
        if not source_text:
            continue
        polygon = _ordered_polygon(line.bounding_box, image_size)
        if polygon is None:
            has_invalid_geometry = True
            continue

        # 영역 ID는 이번 결과의 순서에만 연결한다. 보정되지 않은 점수는 보내지 않는다.
        index = len(regions)
        regions.append(
            OcrRegion(
                segment_id=f"line-{index}",
                reading_order=index,
                polygon=polygon,
                source_text=source_text,
                writing_mode="unknown",
                confidence=None,
            )
        )
    return regions, has_invalid_geometry


# 네 점이 유효한 볼록 사각형일 때 상단 꼭짓점부터 시계 방향으로 정렬한다.
def _ordered_polygon(
    bounds: list[list[float]] | None,
    image_size: tuple[int, int],
) -> tuple[Point, Point, Point, Point] | None:
    if bounds is None or len(bounds) != 4 or any(len(point) != 2 for point in bounds):
        return None
    width, height = image_size
    points = [(x, y) for x, y in bounds]
    if any(not (0 <= x <= width and 0 <= y <= height) for x, y in points):
        return None

    # 이미지의 Y축은 아래로 증가하므로 atan2 오름차순이 화면의 시계 방향이다.
    center_x = sum(x for x, _ in points) / 4
    center_y = sum(y for _, y in points) / 4
    points.sort(key=lambda point: math.atan2(point[1] - center_y, point[0] - center_x))
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % 4]
        x3, y3 = points[(index + 2) % 4]
        if (x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2) <= 0:
            return None
    start = min(range(4), key=lambda index: (points[index][1], points[index][0]))
    return cast(tuple[Point, Point, Point, Point], tuple(points[start:] + points[:start]))


# 인식 모델의 구성에 실제로 설치된 PaddleOCR 패키지 버전을 기록한다.
def _paddle_version() -> str:
    try:
        return version("paddleocr")
    except PackageNotFoundError:
        return "unknown"
