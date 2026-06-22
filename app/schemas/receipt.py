from enum import Enum

from pydantic import BaseModel, Field


class ReceiptAnalysisMode(str, Enum):
    OCR_WITH_AI = "OCR_WITH_AI"
    VISION_ONLY = "VISION_ONLY"
    VISION_FIRST = "VISION_FIRST"
    OCR_ONLY = "OCR_ONLY"


class ReceiptAnalysisResponse(BaseModel):
    title: str | None = Field(None, description="거래명 후보")
    store_name: str | None = Field(None, description="점포명 후보")
    amount: int | None = Field(None, description="합계 금액 후보")
    transaction_date: str | None = Field(None, description="거래일 후보(yyyy-MM-dd)")
    category_name: str | None = Field(None, description="카테고리 후보")
    memo: str | None = Field(None, description="메모 후보")
    confidence: float | None = Field(None, ge=0, le=1, description="분석 신뢰도")
    raw_text: str | None = Field(None, description="OCR 원문 텍스트")
    ocr_engine: str = Field("paddleocr", description="사용한 OCR 엔진")
    used_ai: bool = Field(False, description="AI 구조화 사용 여부")


class ReceiptAnalysisOptions(BaseModel):
    currency_code: str | None = Field(
        default=None,
        description="가계부 기준 통화 코드. 예: JPY, KRW, USD",
    )
    ocr_language: str | None = Field(
        default=None,
        description="OCR 언어 코드. 예: japan, korean, en",
    )
    analysis_mode: ReceiptAnalysisMode | None = Field(
        default=None,
        description="영수증 분석 방식",
    )
    stop_keywords: list[str] | None = Field(
        default=None,
        description="AI 분석용 OCR 텍스트 압축 시 이후 내용을 잘라낼 키워드 목록",
    )
    important_keywords: list[str] | None = Field(
        default=None,
        description="AI 분석용 OCR 텍스트 압축 시 유지할 중요 키워드 목록",
    )
    exclude_item_keywords: list[str] | None = Field(
        default=None,
        description="상품명 후보에서 제외할 광고/안내문 키워드 목록",
    )
