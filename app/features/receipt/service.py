import logging
import time
from typing import Any

from fastapi import HTTPException, UploadFile

from app.ai.ports import TextGenerationProvider
from app.core.config import settings
from app.features.receipt.parser import extract_receipt_candidates
from app.features.receipt.prompts import (
    build_receipt_text_analysis_prompt,
    build_receipt_vision_prompt,
)
from app.features.receipt.validation import validate_response
from app.schemas.receipt import (
    ReceiptAnalysisMode,
    ReceiptAnalysisOptions,
    ReceiptAnalysisResponse,
    ReceiptAmountReviewStatus,
    ReceiptStatus,
)
from app.services.ocr_service import OCRService

logger = logging.getLogger(__name__)

_NULLABLE_TEXT = {"type": "STRING", "nullable": True}
_ITEM_PROPERTIES: dict[str, Any] = {
    name: dict(_NULLABLE_TEXT)
    for name in (
        "title",
        "store_name",
        "branch_name",
        "purchase_total",
        "original_amount",
        "cash_tendered",
        "change",
        "detected_currency_code",
        "currency_evidence",
        "transaction_date",
        "transaction_time",
        "source_date",
        "source_time",
        "date_order",
        "date_order_evidence",
        "category_name",
        "memo",
        "detected_language",
    )
}
_ITEM_PROPERTIES.update(
    {
        "payment_breakdown": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "payment_type": {
                        "type": "STRING",
                        "enum": [
                            "LOYALTY_POINTS",
                            "CASH",
                            "CREDIT_CARD",
                            "DEBIT_CARD",
                            "ELECTRONIC_MONEY",
                            "GIFT_CARD",
                            "VOUCHER",
                            "OTHER_PAID",
                            "UNKNOWN",
                        ],
                    },
                    "amount": dict(_NULLABLE_TEXT),
                    "evidence": dict(_NULLABLE_TEXT),
                    "duplicate_group": dict(_NULLABLE_TEXT),
                },
                "required": [
                    "payment_type",
                    "amount",
                    "evidence",
                    "duplicate_group",
                ],
            },
        },
        "confidence": {"type": "NUMBER", "nullable": True},
        "currency_confidence": {"type": "NUMBER", "nullable": True},
        "bounding_box": {
            "type": "ARRAY",
            "items": {"type": "NUMBER"},
            "nullable": True,
        },
        "status": {"type": "STRING", "enum": ["READY", "NEEDS_REVIEW", "UNREADABLE"]},
    }
)
_RECEIPT_ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "receipts": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": _ITEM_PROPERTIES,
                "required": list(_ITEM_PROPERTIES),
            },
        },
    },
    "required": ["receipts"],
}


class ReceiptAnalysisService:
    def __init__(
        self, ocr_service: OCRService, ai_provider: TextGenerationProvider
    ) -> None:
        self.ocr_service = ocr_service
        self.ai_provider = ai_provider

    async def analyze(
        self, file: UploadFile, options: ReceiptAnalysisOptions | None = None
    ) -> ReceiptAnalysisResponse:
        started = time.perf_counter()
        options = options or ReceiptAnalysisOptions()
        self.ocr_service._validate_file(file)
        # Enforce the same upload policy for vision and OCR; limit the read as well.
        image_bytes = await file.read(settings.OCR_MAX_FILE_SIZE + 1)
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty receipt image")
        if len(image_bytes) > settings.OCR_MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="Receipt image is too large")
        await file.seek(0)
        mode = self._resolve_analysis_mode(options)
        fallback: list[str] = []
        response: ReceiptAnalysisResponse | None = None
        if mode in (ReceiptAnalysisMode.VISION_FIRST, ReceiptAnalysisMode.VISION_ONLY):
            try:
                result = await self.ai_provider.call_with_image(
                    type_name="RECEIPT_ANALYSIS",
                    prompt=build_receipt_vision_prompt(options),
                    image_bytes=image_bytes,
                    mime_type=file.content_type or "image/jpeg",
                    schema=_RECEIPT_ANALYSIS_SCHEMA,
                )
                response = validate_response(
                    result, options, ocr_engine="vision", used_ai=True
                )
                # Keep partial and low-confidence objects. A second whole-image pass
                # must never replace good receipts or merge them with an unreadable one.
                if not response.receipts and mode == ReceiptAnalysisMode.VISION_FIRST:
                    fallback.append("VISION_NO_RECEIPTS_OCR_FALLBACK")
                    response = None
            except Exception as exc:
                logger.warning(
                    "Receipt vision failed. error_type=%s", type(exc).__name__
                )
                fallback.append("VISION_UNAVAILABLE")
                if mode == ReceiptAnalysisMode.VISION_ONLY:
                    response = ReceiptAnalysisResponse(
                        ocr_engine="vision", warnings=["ANALYSIS_UNAVAILABLE"]
                    )
        if response is None:
            response = await self._analyze_ocr(
                file, options, use_ai=mode != ReceiptAnalysisMode.OCR_ONLY
            )
            if fallback:
                fallback.append("OCR_FALLBACK_USED")
        response.warnings = list(dict.fromkeys([*fallback, *response.warnings]))
        logger.info(
            "Receipt analysis mode=%s count=%d ready=%d review=%d unreadable=%d currencies=%s engine=%s fallback=%s latency_ms=%d",
            mode.value,
            response.receipt_count,
            sum(item.status == ReceiptStatus.READY for item in response.receipts),
            sum(
                item.status == ReceiptStatus.NEEDS_REVIEW for item in response.receipts
            ),
            sum(item.status == ReceiptStatus.UNREADABLE for item in response.receipts),
            sorted(
                {
                    item.detected_currency_code
                    for item in response.receipts
                    if item.detected_currency_code
                }
            ),
            response.ocr_engine,
            bool(fallback),
            round((time.perf_counter() - started) * 1000),
        )
        return response

    @staticmethod
    def _resolve_analysis_mode(options: ReceiptAnalysisOptions) -> ReceiptAnalysisMode:
        mode_value = options.analysis_mode or settings.RECEIPT_ANALYSIS_MODE
        try:
            return ReceiptAnalysisMode(
                str(
                    mode_value.value
                    if isinstance(mode_value, ReceiptAnalysisMode)
                    else mode_value
                ).upper()
            )
        except ValueError:
            logger.warning("Invalid receipt analysis mode; using VISION_FIRST")
            return ReceiptAnalysisMode.VISION_FIRST

    async def _analyze_ocr(
        self, file: UploadFile, options: ReceiptAnalysisOptions, *, use_ai: bool
    ) -> ReceiptAnalysisResponse:
        try:
            document = await self.ocr_service.extract_document_from_upload(
                file=file,
                ocr_language=options.ocr_language or settings.OCR_LANGUAGE,
            )
        except HTTPException as exc:
            if exc.status_code in (400, 413):
                raise
            logger.warning("Receipt OCR failed. error_type=%s", type(exc).__name__)
            return ReceiptAnalysisResponse(
                ocr_engine="paddleocr", warnings=["OCR_UNAVAILABLE"]
            )
        except Exception as exc:
            logger.warning("Receipt OCR failed. error_type=%s", type(exc).__name__)
            return ReceiptAnalysisResponse(
                ocr_engine="paddleocr", warnings=["OCR_UNAVAILABLE"]
            )
        if not document.lines:
            return ReceiptAnalysisResponse(
                ocr_engine="paddleocr", warnings=["OCR_NO_TEXT"]
            )
        candidates = extract_receipt_candidates(document.text)
        warnings = ["OCR_LANGUAGE_HINT_LIMITATION"]
        if use_ai:
            try:
                result = await self.ai_provider.call(
                    type_name="RECEIPT_ANALYSIS",
                    data=build_receipt_text_analysis_prompt(
                        document.text,
                        candidates,
                        options,
                        spatial_lines=[line.as_payload() for line in document.lines],
                    ),
                    schema=_RECEIPT_ANALYSIS_SCHEMA,
                )
                response = validate_response(
                    result, options, ocr_engine="paddleocr", used_ai=True
                )
                if response.receipts:
                    if not all(line.bounding_box for line in document.lines):
                        for item in response.receipts:
                            item.book_amount = None
                            item.original_amount = None
                            item.review_status = ReceiptAmountReviewStatus.NEEDS_REVIEW
                            item.status = ReceiptStatus.NEEDS_REVIEW
                            item.warnings.append("OCR_RECEIPT_BOUNDARIES_UNVERIFIED")
                    response.warnings.extend(warnings)
                    return response
            except Exception as exc:
                logger.warning(
                    "Receipt OCR structuring failed. error_type=%s", type(exc).__name__
                )
            warnings.append("OCR_AI_UNAVAILABLE_RULE_FALLBACK")
        # No local image-segmentation heuristic claims to establish physical receipt
        # boundaries. This fallback is an explicitly unconfirmed manual-review draft.
        response = validate_response(
            {"receipts": [candidates]}, options, ocr_engine="paddleocr", used_ai=False
        )
        item = response.receipts[0]
        item.status = ReceiptStatus.NEEDS_REVIEW
        item.warnings.append("OCR_RECEIPT_BOUNDARIES_UNVERIFIED")
        if candidates["total_line_count"] != 1:
            item.book_amount = None
            item.original_amount = None
            item.review_status = ReceiptAmountReviewStatus.NEEDS_REVIEW
            item.warnings.append("OCR_MULTIPLE_OR_UNKNOWN_TOTALS")
        response.warnings.extend(warnings)
        return response
