from datetime import date, datetime, time
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


class ReceiptAmountReviewStatus(str, Enum):
    READY = "READY"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    EXCLUDED = "EXCLUDED"


class ReceiptCategorySource(str, Enum):
    EXISTING = "EXISTING"
    DEFAULT = "DEFAULT"
    NEW = "NEW"
    FALLBACK = "FALLBACK"
    USER = "USER"


class ReceiptPaymentType(str, Enum):
    LOYALTY_POINTS = "LOYALTY_POINTS"
    CASH = "CASH"
    CREDIT_CARD = "CREDIT_CARD"
    DEBIT_CARD = "DEBIT_CARD"
    ELECTRONIC_MONEY = "ELECTRONIC_MONEY"
    GIFT_CARD = "GIFT_CARD"
    VOUCHER = "VOUCHER"
    OTHER_PAID = "OTHER_PAID"
    UNKNOWN = "UNKNOWN"


class ReceiptPaymentItem(BaseModel):
    payment_type: ReceiptPaymentType
    amount: Decimal = Field(gt=0)
    evidence: str | None = None
    duplicate_group: str | None = None


class ReceiptRuntimeIdentity(BaseModel):
    run_id: str
    source_fingerprint: str
    started_at: datetime
    process_id: int
    working_directory: str
    command_fingerprint: str
    git_head: str | None = None
    provider_call_count: int = Field(ge=0)


class ReceiptAnalysisItem(BaseModel):
    receipt_id: str
    title: str | None = None
    store_name: str | None = None
    branch_name: str | None = None
    merchant_evidence: str | None = None
    branch_evidence: str | None = None
    bounding_box: list[float] | None = None
    identity_source_box: list[float] | None = None
    identity_verification: str | None = None
    financial_source_box: list[float] | None = None
    financial_recovery_provenance: str | None = None
    purchase_total: Decimal | None = Field(default=None, gt=0)
    payment_breakdown: list[ReceiptPaymentItem] = Field(default_factory=list)
    cash_tendered: Decimal | None = Field(default=None, gt=0)
    change: Decimal | None = Field(default=None, ge=0)
    book_amount: Decimal | None = Field(default=None, ge=0)
    amount_policy_version: str = "receipt-book-amount-v1"
    amount_reason: str | None = None
    review_status: ReceiptAmountReviewStatus = ReceiptAmountReviewStatus.NEEDS_REVIEW
    # Backward-compatible wire name. It is the deterministic bookkeeping amount,
    # never the provider's purchase total.
    original_amount: Decimal | None = Field(default=None, ge=0)
    detected_currency_code: str | None = None
    transaction_date: date | None = None
    transaction_time: time | None = None
    category_name: str | None = None
    category_source: ReceiptCategorySource = ReceiptCategorySource.FALLBACK
    category_reason: str | None = None
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
    analysis_trace_id: str | None = None
    runtime_identity: ReceiptRuntimeIdentity | None = None


class ReceiptAnalysisOptions(BaseModel):
    category_candidates: list[str] = Field(default_factory=list, max_length=500)
    default_category_candidates: list[str] = Field(default_factory=list, max_length=100)
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
