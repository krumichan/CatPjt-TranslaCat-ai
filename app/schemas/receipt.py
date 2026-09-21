from datetime import date
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field


class ReceiptAnalysisMode(str, Enum):
    OCR_WITH_AI = "OCR_WITH_AI"
    VISION_ONLY = "VISION_ONLY"
    VISION_FIRST = "VISION_FIRST"
    OCR_ONLY = "OCR_ONLY"


class ReceiptStatus(str, Enum):
    READY = "READY"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    UNREADABLE = "UNREADABLE"


class ReceiptAnalysisItem(BaseModel):
    receipt_id: str
    title: str | None = None
    store_name: str | None = None
    original_amount: Decimal | None = Field(default=None, gt=0)
    detected_currency_code: str | None = None
    transaction_date: date | None = None
    category_name: str | None = None
    memo: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    currency_confidence: float | None = Field(default=None, ge=0, le=1)
    detected_language: str | None = None
    status: ReceiptStatus = ReceiptStatus.NEEDS_REVIEW
    warnings: list[str] = Field(default_factory=list)


class ReceiptAnalysisResponse(BaseModel):
    receipts: list[ReceiptAnalysisItem] = Field(default_factory=list)
    receipt_count: int = Field(default=0, ge=0)
    warnings: list[str] = Field(default_factory=list)
    ocr_engine: str = "none"
    used_ai: bool = False


class ReceiptAnalysisOptions(BaseModel):
    category_candidates: list[str] = Field(default_factory=list, max_length=500)
    currency_code: str | None = Field(
        default=None,
        description="Deprecated target currency; accepted but ignored for source detection",
    )
    ocr_language: str | None = Field(
        default=None,
        description="Explicit OCR fallback language hint only; never derived from target currency",
    )
    analysis_mode: ReceiptAnalysisMode | None = Field(
        default=None,
        description="영수증 분석 방식",
    )
    stop_keywords: list[str] | None = Field(
        default=None,
        description="Advisory keywords only; never truncate subsequent receipts",
    )
    important_keywords: list[str] | None = Field(
        default=None,
        description="AI 분석용 OCR 텍스트 압축 시 유지할 중요 키워드 목록",
    )
    exclude_item_keywords: list[str] | None = Field(
        default=None,
        description="상품명 후보에서 제외할 광고/안내문 키워드 목록",
    )
