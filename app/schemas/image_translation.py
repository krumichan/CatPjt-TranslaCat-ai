from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

ImageLanguage = Literal["ja", "ko", "en"]
ImagePageState = Literal["READY", "NO_TEXT", "FAILED"]
WritingMode = Literal[
    "horizontal-ltr", "horizontal-rtl", "vertical-rtl", "vertical-ltr", "unknown"
]
Point = tuple[float, float]


class ImageTranslationSegment(BaseModel):
    """OCR v1 の領域と、その領域に対応する翻訳を返す。"""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    segment_id: str
    reading_order: int = Field(ge=0)
    source_text: str
    translated_text: str | None
    polygon: list[Point] = Field(min_length=4, max_length=4)
    confidence: float | None = Field(ge=0, le=1)
    writing_mode: WritingMode
    needs_review: bool


class ImagePageTranslation(BaseModel):
    """方向補正後の画像のピクセル空間と、ページの処理結果を返す。"""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    ocr_contract_version: Literal[1] = 1
    segments: list[ImageTranslationSegment]
    state: ImagePageState
    error_code: str | None = None
    retryable: bool | None = None
