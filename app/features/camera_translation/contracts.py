"""카메라용 AI 내부 요청 및 공통 OCR v1 응답."""

from __future__ import annotations

from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator
from pydantic.alias_generators import to_camel

CameraLanguage = Literal["ja", "ko", "en"]
SegmentId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,64}$")]
SegmentText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
Point = tuple[float, float]
Affine = tuple[float, float, float, float, float, float]

_SUPPORTED_LANGUAGE_PAIRS = frozenset({("ja", "ko"), ("ko", "ja"), ("en", "ko")})
_MAX_REQUEST_CHARS = 4_000


class CameraPayload(BaseModel):
    """BE에 보이는 JSON 필드 이름을 camelCase로 유지한다."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")


class CameraTranslationSegment(CameraPayload):
    """사용자가 확인한 원문 하나와 선택 범위에 속하는 문맥."""

    id: SegmentId
    text: SegmentText
    context: SegmentText | None = None


class CameraTranslationRequest(CameraPayload):
    """BE가 동일한 언어 방향으로 번역할 표현을 한 번에 제출한다."""

    request_id: UUID
    mode: Literal["STILL", "PREVIEW"]
    source_language: CameraLanguage
    target_language: CameraLanguage
    segments: list[CameraTranslationSegment] = Field(min_length=1, max_length=12)

    @model_validator(mode="after")
    def validate_segment_batch(self) -> CameraTranslationRequest:
        """지원 언어 방향, ID 유일성, 원문과 문맥의 전체 길이를 확인한다."""
        if (self.source_language, self.target_language) not in _SUPPORTED_LANGUAGE_PAIRS:
            raise ValueError("Unsupported camera translation language pair")
        if len({segment.id for segment in self.segments}) != len(self.segments):
            raise ValueError("Camera translation segment IDs must be unique")

        # 요청의 각 문맥도 단일 모델 호출의 문자 상한에 포함한다.
        total_chars = sum(
            len(segment.text) + len(segment.context or "") for segment in self.segments
        )
        if total_chars > _MAX_REQUEST_CHARS:
            raise ValueError("Camera translation text is too long")
        return self


class CameraSegmentResult(CameraPayload):
    """입력 표현 하나의 번역 또는 안전한 실패 코드."""

    id: SegmentId
    status: Literal["TRANSLATED", "FAILED"]
    translated_text: str | None
    error_code: Literal["INVALID_MODEL_OUTPUT", "SEGMENT_TEMPORARILY_UNAVAILABLE"] | None


class CameraTranslationResult(CameraPayload):
    """입력 요청 ID와 순서를 보존하는 번역 결과."""

    request_id: UUID
    segments: list[CameraSegmentResult]


class OcrImageGeometry(CameraPayload):
    """방향 보정한 업로드 이미지와 보정 전 픽셀 공간 사이의 변환."""

    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    input_to_image: Affine


class OcrRegion(CameraPayload):
    """한 OCR 결과에 속하는 좌상단 픽셀 좌표 영역."""

    segment_id: str
    reading_order: int = Field(ge=0)
    polygon: tuple[Point, Point, Point, Point]
    source_text: str = Field(min_length=1)
    writing_mode: Literal[
        "horizontal-ltr", "horizontal-rtl", "vertical-rtl", "vertical-ltr", "unknown"
    ]
    confidence: float | None = Field(ge=0, le=1)


class OcrEngine(CameraPayload):
    """인식 엔진의 제공자, 모델, 구성 버전 식별자."""

    provider: str
    model: str
    version: str


class CommonOcrResult(CameraPayload):
    """촬영 범위의 공통 OCR v1 결과."""

    schema_version: Literal[1] = 1
    image_id: UUID
    image: OcrImageGeometry
    coordinate_space: Literal["image-pixels-top-left"] = "image-pixels-top-left"
    status: Literal["OK", "NO_TEXT", "UNCERTAIN"]
    regions: list[OcrRegion]
    engine: OcrEngine
