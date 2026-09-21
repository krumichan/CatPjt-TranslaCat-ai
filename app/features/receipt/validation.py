"""Validate each provider object independently; never trust generated financial data."""

import math
import re
from datetime import date, time
from decimal import Decimal
from typing import Any

from app.features.receipt.currencies import normalize_currency
from app.features.receipt.parser import normalize_amount, normalize_date
from app.schemas.receipt import (
    ReceiptAnalysisItem,
    ReceiptAnalysisOptions,
    ReceiptAnalysisResponse,
    ReceiptAmountReviewStatus,
    ReceiptPaymentItem,
    ReceiptPaymentType,
    ReceiptStatus,
)

_AMOUNT_POLICY_VERSION = "receipt-book-amount-v1"
_READY_CONFIDENCE_THRESHOLD = 0.95
_NON_PAID_TYPES = {ReceiptPaymentType.LOYALTY_POINTS}


def _non_negative_amount(value: object, currency: str | None) -> Decimal | None:
    if value in (0, "0", "0.0", "0.00"):
        return Decimal("0")
    return normalize_amount(value, currency)


def _payment_items(
    value: object, currency: str | None, warnings: list[str]
) -> list[ReceiptPaymentItem]:
    if value is None:
        return []
    if not isinstance(value, list):
        warnings.append("INVALID_PAYMENT_BREAKDOWN")
        return []
    result: list[ReceiptPaymentItem] = []
    for raw in value[:30]:
        if not isinstance(raw, dict):
            warnings.append("INVALID_PAYMENT_ITEM")
            continue
        try:
            payment_type = ReceiptPaymentType(str(raw.get("payment_type")))
        except ValueError:
            warnings.append("UNKNOWN_PAYMENT_TYPE")
            continue
        amount = normalize_amount(raw.get("amount"), currency)
        if amount is None:
            warnings.append("INVALID_PAYMENT_AMOUNT")
            continue
        result.append(
            ReceiptPaymentItem(
                payment_type=payment_type,
                amount=amount,
                evidence=_text(raw.get("evidence"), 160),
                duplicate_group=_text(raw.get("duplicate_group"), 80),
            )
        )
    if isinstance(value, list) and len(value) > 30:
        warnings.append("PAYMENT_ITEM_LIMIT_REACHED")
    return result


def _book_amount(
    purchase_total: Decimal | None,
    payments: list[ReceiptPaymentItem],
    cash_tendered: Decimal | None,
    change: Decimal | None,
    warnings: list[str],
) -> tuple[Decimal | None, ReceiptAmountReviewStatus, str]:
    if purchase_total is None:
        return None, ReceiptAmountReviewStatus.NEEDS_REVIEW, "MISSING_PURCHASE_TOTAL"
    if not payments:
        warnings.append("PURCHASE_TOTAL_WITHOUT_ALLOCATION")
        return (
            purchase_total,
            ReceiptAmountReviewStatus.READY,
            "PURCHASE_TOTAL_NO_PAYMENT_ALLOCATION",
        )

    collapsed: list[ReceiptPaymentItem] = []
    groups: dict[str, ReceiptPaymentItem] = {}
    group_conflict = False
    for payment in payments:
        group = payment.duplicate_group
        if not group:
            collapsed.append(payment)
            continue
        previous = groups.get(group)
        if previous is None:
            groups[group] = payment
            collapsed.append(payment)
        elif (
            previous.payment_type != payment.payment_type
            or previous.amount != payment.amount
        ):
            group_conflict = True
    if group_conflict:
        warnings.append("CONFLICTING_DUPLICATE_PAYMENT_GROUP")
        return None, ReceiptAmountReviewStatus.NEEDS_REVIEW, "PAYMENT_DUPLICATE_CONFLICT"
    if len(collapsed) < len(payments):
        warnings.append("DUPLICATE_PAYMENT_DETAIL_COLLAPSED")

    if any(item.payment_type == ReceiptPaymentType.UNKNOWN for item in collapsed):
        warnings.append("UNKNOWN_PAYMENT_ALLOCATION")
        return None, ReceiptAmountReviewStatus.NEEDS_REVIEW, "UNKNOWN_PAYMENT_ALLOCATION"

    points = sum(
        (item.amount for item in collapsed if item.payment_type in _NON_PAID_TYPES),
        Decimal("0"),
    )
    paid_items = [item for item in collapsed if item.payment_type not in _NON_PAID_TYPES]
    cash_items = [item for item in paid_items if item.payment_type == ReceiptPaymentType.CASH]
    non_cash_paid = sum(
        (item.amount for item in paid_items if item.payment_type != ReceiptPaymentType.CASH),
        Decimal("0"),
    )
    cash_paid = sum((item.amount for item in cash_items), Decimal("0"))
    if cash_tendered is not None:
        if change is None or cash_tendered < change:
            warnings.append("INVALID_CASH_TENDERED_CHANGE")
            return None, ReceiptAmountReviewStatus.NEEDS_REVIEW, "INVALID_CASH_FACTS"
        net_cash = cash_tendered - change
        if cash_items and cash_paid not in (cash_tendered, net_cash):
            warnings.append("CASH_ALLOCATION_DISAGREES")
            return None, ReceiptAmountReviewStatus.NEEDS_REVIEW, "INVALID_CASH_FACTS"
        if cash_items and cash_paid == cash_tendered and cash_paid != net_cash:
            warnings.append("CASH_TENDERED_NOT_DOUBLE_COUNTED")
        cash_paid = net_cash
    elif not cash_items and change not in (None, Decimal("0")):
        warnings.append("INVALID_CASH_TENDERED_CHANGE")
        return None, ReceiptAmountReviewStatus.NEEDS_REVIEW, "INVALID_CASH_FACTS"
    paid = non_cash_paid + cash_paid

    if points + paid != purchase_total:
        # Some receipts repeat the same card amount in a top summary and in a lower
        # card-detail block without labelling the relationship. Collapse exact paid
        # duplicates only when that is the unique way to reconcile the printed total.
        ungrouped_paid = [
            item
            for item in paid_items
            if item.duplicate_group is None
            and item.payment_type != ReceiptPaymentType.CASH
        ]
        unique_values = {
            (item.payment_type, item.amount) for item in ungrouped_paid
        }
        duplicate_present = len(unique_values) < len(ungrouped_paid)
        other_paid = cash_paid + sum(
            (
                item.amount
                for item in paid_items
                if item.payment_type != ReceiptPaymentType.CASH
                and item.duplicate_group is not None
            ),
            Decimal("0"),
        )
        deduplicated_paid = other_paid + sum(
            (amount for _, amount in unique_values), Decimal("0")
        )
        if duplicate_present and points + deduplicated_paid == purchase_total:
            paid = deduplicated_paid
            warnings.append("DUPLICATE_PAYMENT_DETAIL_COLLAPSED")
        elif (
            points > 0
            and points < purchase_total
            and cash_paid == 0
            and non_cash_paid == purchase_total
        ):
            # Some point-redemption receipts print the gross total beside a card
            # heading, then list the actual split lower down. When the only paid
            # observation is exactly the gross total, treating it as another
            # allocation is arithmetically impossible. The net settlement is the
            # printed total less the separately printed loyalty redemption.
            paid = purchase_total - points
            warnings.append("GROSS_PAYMENT_LINE_REPLACED_BY_NET_SETTLEMENT")
        else:
            warnings.append("PAYMENT_TOTAL_MISMATCH")
            return None, ReceiptAmountReviewStatus.NEEDS_REVIEW, "PAYMENT_TOTAL_MISMATCH"

    if paid == 0 and points == purchase_total:
        warnings.append("ZERO_BOOK_AMOUNT_EXCLUDED")
        return Decimal("0"), ReceiptAmountReviewStatus.EXCLUDED, "FULL_LOYALTY_REDEMPTION"
    if paid <= 0:
        warnings.append("INVALID_BOOK_AMOUNT")
        return None, ReceiptAmountReviewStatus.NEEDS_REVIEW, "INVALID_BOOK_AMOUNT"
    return paid, ReceiptAmountReviewStatus.READY, "SETTLED_PAYMENT_EXCLUDING_LOYALTY_POINTS"


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


def _source_meridiem_time(value: str | None) -> time | None:
    if value is None:
        return None
    match = re.search(
        r"(?<!\d)(\d{1,2}):([0-5]\d)(?::([0-5]\d))?\s*([AP])\.?M\.?(?!\w)",
        value,
        re.IGNORECASE,
    )
    if not match:
        return None
    hour, minute, second, meridiem = match.groups()
    hour_value = int(hour)
    if not 1 <= hour_value <= 12:
        return None
    hour_value %= 12
    if meridiem.upper() == "P":
        hour_value += 12
    return time(hour_value, int(minute), int(second or 0))


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
    purchase_total = normalize_amount(
        raw.get("purchase_total")
        if raw.get("purchase_total") is not None
        else raw.get("original_amount"),
        currency,
    )
    if purchase_total is None:
        warnings.append("INVALID_OR_MISSING_AMOUNT")
    payments = _payment_items(raw.get("payment_breakdown"), currency, warnings)
    cash_tendered = _non_negative_amount(raw.get("cash_tendered"), currency)
    change = _non_negative_amount(raw.get("change"), currency)
    book_amount, amount_review, amount_reason = _book_amount(
        purchase_total, payments, cash_tendered, change, warnings
    )
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
    source_time = _text(raw.get("source_time"), 40)
    raw_time = _text(raw.get("transaction_time"), 16)
    time_value: time | None = None
    source_meridiem = _source_meridiem_time(source_time)
    if source_meridiem is not None:
        time_value = source_meridiem
        if raw_time is not None and raw_time != source_meridiem.isoformat():
            warnings.append("TIME_NORMALIZED_FROM_SOURCE")
    elif raw_time is not None:
        try:
            if not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d(?::[0-5]\d)?", raw_time):
                raise ValueError("invalid receipt time")
            time_value = time.fromisoformat(raw_time)
        except ValueError:
            warnings.append("INVALID_TIME")
    elif source_time is not None:
        warnings.append("INVALID_TIME")
    category = _text(raw.get("category_name"), 50)
    if category is not None and category not in options.category_candidates:
        category = None
        warnings.append("CATEGORY_NOT_IN_ACCOUNT_BOOK")
    title = _text(raw.get("title"), 100)
    store = _text(raw.get("store_name"), 100)
    branch = _text(raw.get("branch_name"), 100)
    if not title and not store:
        warnings.append("MISSING_MERCHANT")
    non_blocking_warnings = {
        "PURCHASE_TOTAL_WITHOUT_ALLOCATION",
        "DUPLICATE_PAYMENT_DETAIL_COLLAPSED",
        "GROSS_PAYMENT_LINE_REPLACED_BY_NET_SETTLEMENT",
        "TIME_NORMALIZED_FROM_SOURCE",
        "CASH_TENDERED_NOT_DOUBLE_COUNTED",
    }
    status = ReceiptStatus.READY
    if (
        any(warning not in non_blocking_warnings for warning in warnings)
        or confidence is None
        or confidence < _READY_CONFIDENCE_THRESHOLD
        or amount_review != ReceiptAmountReviewStatus.READY
        or raw.get("status") == "NEEDS_REVIEW"
    ):
        status = ReceiptStatus.NEEDS_REVIEW
    if raw.get("status") == "UNREADABLE" or not any(
        (title, store, purchase_total, date_value)
    ):
        status = ReceiptStatus.UNREADABLE
    if status == ReceiptStatus.UNREADABLE:
        # An explicitly unreadable object cannot retain an invented authoritative amount.
        book_amount = None
        amount_review = ReceiptAmountReviewStatus.NEEDS_REVIEW
    return ReceiptAnalysisItem(
        receipt_id=receipt_id,
        title=title,
        store_name=store,
        branch_name=branch,
        purchase_total=purchase_total,
        payment_breakdown=payments,
        cash_tendered=cash_tendered,
        change=change,
        book_amount=book_amount,
        amount_policy_version=_AMOUNT_POLICY_VERSION,
        amount_reason=amount_reason,
        review_status=amount_review,
        original_amount=book_amount,
        detected_currency_code=currency,
        transaction_date=date.fromisoformat(date_value) if date_value else None,
        transaction_time=time_value,
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
