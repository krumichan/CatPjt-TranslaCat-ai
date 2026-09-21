"""Validate each provider object independently; never trust generated financial data."""

import math
import re
from datetime import date
from typing import Any

from app.features.receipt.currencies import normalize_currency
from app.features.receipt.parser import normalize_amount, normalize_date
from app.schemas.receipt import (
    ReceiptAnalysisItem,
    ReceiptAnalysisOptions,
    ReceiptAnalysisResponse,
    ReceiptStatus,
)


def _text(value: object, limit: int = 1000) -> str | None:
    return value.strip()[:limit] or None if isinstance(value, str) else None


def _confidence(value: object, warnings: list[str], name: str) -> float | None:
    if value is None:
        return None
    try:
        number = (
            float(value)
            if isinstance(value, (int, float, str)) and not isinstance(value, bool)
            else math.nan
        )
    except (ValueError, TypeError, OverflowError):
        number = math.nan
    if not math.isfinite(number) or not 0 <= number <= 1:
        warnings.append(f"INVALID_{name}")
        return None
    return number


def _box(value: object) -> tuple[float, ...] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    if any(
        not isinstance(x, (float, int)) or isinstance(x, bool) or not 0 <= x <= 1
        for x in value
    ):
        return None
    if value[0] >= value[2] or value[1] >= value[3]:
        return None
    return tuple(value)


def validate_item(
    raw: object, index: int, options: ReceiptAnalysisOptions
) -> ReceiptAnalysisItem:
    receipt_id = f"receipt-{index + 1}"
    if not isinstance(raw, dict):
        return ReceiptAnalysisItem(
            receipt_id=receipt_id,
            status=ReceiptStatus.UNREADABLE,
            warnings=["MALFORMED_RECEIPT"],
        )
    warnings: list[str] = []
    confidence = _confidence(raw.get("confidence"), warnings, "CONFIDENCE")
    currency_confidence = _confidence(
        raw.get("currency_confidence"), warnings, "CURRENCY_CONFIDENCE"
    )
    if raw.get("currency_confidence") is not None and currency_confidence is None:
        warnings.append("UNVERIFIED_CURRENCY")
    currency = normalize_currency(raw.get("detected_currency_code"))
    evidence = _text(raw.get("currency_evidence"))
    if evidence and re.fullmatch(r"[$¥￥\s]+", evidence):
        currency = None
        warnings.append("AMBIGUOUS_CURRENCY_SYMBOL")
    if (
        currency_confidence is not None and currency_confidence < 0.75
    ) or "UNVERIFIED_CURRENCY" in warnings:
        currency = None
        warnings.append("LOW_CURRENCY_CONFIDENCE")
    if currency is None:
        warnings.append("UNKNOWN_CURRENCY")
    amount = normalize_amount(raw.get("original_amount"), currency)
    if amount is None:
        warnings.append("INVALID_OR_MISSING_AMOUNT")
    date_order = (
        _text(raw.get("date_order")) if _text(raw.get("date_order_evidence")) else None
    )
    source_date = _text(raw.get("source_date"))
    date_value = normalize_date(source_date or raw.get("transaction_date"), date_order)
    if source_date and date_value is None:
        # For named months in any script the provider can supply the ISO form.
        # Numeric dates embedded in surrounding text still require an explicit order.
        numeric_date = re.search(
            r"\d{4}[./\-年]\d{1,2}[./\-月]\d{1,2}日?|\d{1,2}[./\-]\d{1,2}[./\-]\d{4}",
            source_date,
        )
        if numeric_date:
            date_value = normalize_date(numeric_date.group(), date_order)
        elif any(char.isalpha() for char in source_date) and any(
            char.isdigit() for char in source_date
        ):
            date_value = normalize_date(raw.get("transaction_date"))
    if date_value is None:
        warnings.append("INVALID_OR_AMBIGUOUS_DATE")
    category = _text(raw.get("category_name"), 50)
    if category is not None and category not in options.category_candidates:
        category = None
        warnings.append("CATEGORY_NOT_IN_ACCOUNT_BOOK")
    title = _text(raw.get("title"), 100)
    store = _text(raw.get("store_name"), 100)
    if not title and not store:
        warnings.append("MISSING_MERCHANT")
    status = ReceiptStatus.READY
    if (
        warnings
        or confidence is None
        or confidence < 0.75
        or raw.get("status") == "NEEDS_REVIEW"
    ):
        status = ReceiptStatus.NEEDS_REVIEW
    if raw.get("status") == "UNREADABLE" or not any((title, store, amount, date_value)):
        status = ReceiptStatus.UNREADABLE
    if status == ReceiptStatus.UNREADABLE:
        # An explicitly unreadable object cannot retain an invented authoritative amount.
        amount = None
    return ReceiptAnalysisItem(
        receipt_id=receipt_id,
        title=title,
        store_name=store,
        original_amount=amount,
        detected_currency_code=currency,
        transaction_date=date.fromisoformat(date_value) if date_value else None,
        category_name=category,
        memo=_text(raw.get("memo"), 500),
        confidence=confidence,
        currency_confidence=currency_confidence,
        detected_language=_text(raw.get("detected_language"), 35),
        status=status,
        warnings=warnings,
    )


def validate_response(
    result: object,
    options: ReceiptAnalysisOptions,
    *,
    ocr_engine: str,
    used_ai: bool,
) -> ReceiptAnalysisResponse:
    raw_items = result.get("receipts") if isinstance(result, dict) else None
    if not isinstance(raw_items, list):
        raise ValueError("Invalid receipt batch envelope")
    warnings: list[str] = []
    if len(raw_items) > 30:
        raw_items = raw_items[:30]
        warnings.append("RECEIPT_LIMIT_REACHED")
    receipts: list[ReceiptAnalysisItem] = []
    seen_positions: set[tuple[Any, ...]] = set()
    seen_fields: dict[tuple[Any, ...], ReceiptAnalysisItem] = {}
    for index, raw in enumerate(raw_items):
        item = validate_item(raw, index, options)
        position = _box(raw.get("bounding_box")) if isinstance(raw, dict) else None
        fields = (
            item.store_name,
            item.original_amount,
            item.detected_currency_code,
            item.transaction_date,
        )
        position_key = (*position, *fields) if position is not None else None
        if position_key is not None and position_key in seen_positions:
            warnings.append("DUPLICATE_RECEIPT_REMOVED")
            continue
        if position_key is not None:
            seen_positions.add(position_key)
        elif item.original_amount is not None and fields in seen_fields:
            item.warnings.append("POSSIBLE_DUPLICATE_RECEIPT")
            item.status = ReceiptStatus.NEEDS_REVIEW
            previous = seen_fields[fields]
            if "POSSIBLE_DUPLICATE_RECEIPT" not in previous.warnings:
                previous.warnings.append("POSSIBLE_DUPLICATE_RECEIPT")
                previous.status = ReceiptStatus.NEEDS_REVIEW
        if position is None:
            seen_fields[fields] = item
        receipts.append(item)
    if not receipts:
        warnings.append("NO_RECEIPTS_DETECTED")
    return ReceiptAnalysisResponse(
        receipts=receipts,
        receipt_count=len(receipts),
        warnings=list(dict.fromkeys(warnings)),
        ocr_engine=ocr_engine,
        used_ai=used_ai,
    )
