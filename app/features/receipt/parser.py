"""Conservative parsing: uncertain separators/dates are never silently corrected."""

import re
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

from app.features.receipt.currencies import CURRENCY_MINOR_UNITS, normalize_currency

_TOTAL = re.compile(
    r"\b(?:grand total|total(?:\s+ttc)?|amount due|amount paid|gesamt|summe|totale|totaal|summa|razem)\b|合計|総合計|お買上計|합계|총액|결제금액|итого|المجموع",
    re.IGNORECASE,
)
_NOT_TOTAL = re.compile(
    r"sub\s*total|sous.total|tax|change|tendered|cash received|小計|釣|거스름|부가세",
    re.IGNORECASE,
)


def normalize_amount(value: object, currency_code: str | None = None) -> Decimal | None:
    if isinstance(value, bool) or isinstance(value, float):
        # Providers are asked for strings; an already-rounded float is not money.
        return None
    if isinstance(value, int) and (value <= 0 or value >= 10**20):
        return None
    if isinstance(value, (int, Decimal)):
        text = str(value)
    elif isinstance(value, str):
        text = value.strip()
    else:
        return None
    if len(text) > 80:
        return None
    if currency_code:
        text = re.sub(
            rf"^(?:{currency_code})\s*|\s*(?:{currency_code})$",
            "",
            text,
            flags=re.IGNORECASE,
        )
    text = text.strip()
    if not re.fullmatch(r"[0-9]+(?:[., '\u00a0\u202f][0-9]+)*", text):
        return None
    if re.search(r"[ '\u00a0\u202f]", text):
        if not re.fullmatch(
            r"[0-9]{1,3}(?:[ '\u00a0\u202f][0-9]{3})+(?:[.,][0-9]+)?", text
        ):
            return None
        text = re.sub(r"[ '\u00a0\u202f]", "", text)
    if "." in text and "," in text:
        separator = "." if text.rfind(".") > text.rfind(",") else ","
        grouping = "," if separator == "." else "."
        if not re.fullmatch(
            rf"[0-9]{{1,3}}(?:{re.escape(grouping)}[0-9]{{3}})+{re.escape(separator)}[0-9]+",
            text,
        ):
            return None
        text = text.replace(grouping, "").replace(separator, ".")
    elif "," in text:
        parts = text.split(",")
        minor_units = CURRENCY_MINOR_UNITS.get(currency_code or "")
        if len(parts) == 2 and len(parts[1]) != 3:
            text = text.replace(",", ".")
        elif minor_units == 0 and all(len(p) == 3 for p in parts[1:]):
            text = text.replace(",", "")
        elif (
            len(parts) > 2 and minor_units != 3 and all(len(p) == 3 for p in parts[1:])
        ):
            text = text.replace(",", "")
        else:
            return None
    elif "." in text:
        parts = text.split(".")
        # Indonesian receipts conventionally print rupiah grouping with a dot and
        # omit sen. Vision models sometimes preserve that separator despite the
        # normalized-decimal contract, so interpret only the unambiguous 3-digit
        # IDR grouping shape here.
        if currency_code == "IDR" and all(len(part) == 3 for part in parts[1:]):
            text = text.replace(".", "")
        elif len(parts) > 2:
            return None
    try:
        amount = Decimal(text)
    except InvalidOperation:
        return None
    if not amount.is_finite() or amount <= 0 or amount.adjusted() >= 20:
        return None
    exponent = amount.as_tuple().exponent
    if not isinstance(exponent, int) or exponent < -8:
        return None
    minor_units = CURRENCY_MINOR_UNITS.get(currency_code or "")
    normalized_exponent = amount.normalize().as_tuple().exponent
    if (
        minor_units is not None
        and isinstance(normalized_exponent, int)
        and normalized_exponent < -minor_units
    ):
        return None
    return amount


def normalize_date(value: object, date_order: str | None = None) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    ymd = re.fullmatch(r"(\d{4})[./\-年](\d{1,2})[./\-月](\d{1,2})日?", text)
    if ymd:
        year, month, day = map(int, ymd.groups())
    else:
        match = re.fullmatch(r"(\d{1,2})[./\-](\d{1,2})[./\-](\d{4})", text)
        if not match:
            return None
        first, second, year = map(int, match.groups())
        if date_order == "DMY" or first > 12:
            day, month = first, second
        elif date_order == "MDY" or second > 12:
            month, day = first, second
        elif first == second:
            day, month = first, second
        else:
            return None
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def extract_receipt_candidates(raw_text: str) -> dict[str, Any]:
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    codes = {
        code
        for code in re.findall(r"\b[A-Z]{3}\b", raw_text.upper())
        if normalize_currency(code)
    }
    currency = next(iter(codes)) if len(codes) == 1 else None
    total_lines = [
        line for line in lines if _TOTAL.search(line) and not _NOT_TOTAL.search(line)
    ]
    amounts: list[Decimal] = []
    if len(total_lines) == 1:
        tokens = re.findall(r"[0-9]+(?:[.,][0-9]+)*", total_lines[0])
        if len(tokens) == 1:
            amount = normalize_amount(tokens[0], currency)
            if amount is not None:
                amounts.append(amount)
    dates = {
        normalized
        for line in lines
        for token in re.findall(
            r"\d{4}[./\-年]\d{1,2}[./\-月]\d{1,2}日?|\d{1,2}[./\-]\d{1,2}[./\-]\d{4}",
            line,
        )
        if (normalized := normalize_date(token)) is not None
    }
    store = next(
        (
            line
            for line in lines[:3]
            if not re.search(r"\d|[@:]", line) and not _TOTAL.search(line)
        ),
        None,
    )
    return {
        "purchase_total": str(amounts[0]) if len(amounts) == 1 else None,
        "payment_breakdown": [],
        "cash_tendered": None,
        "change": None,
        # Compatibility for existing local OCR callers. Validation treats
        # purchase_total as authoritative and recalculates the book amount.
        "original_amount": str(amounts[0]) if len(amounts) == 1 else None,
        "detected_currency_code": currency,
        "transaction_date": next(iter(dates)) if len(dates) == 1 else None,
        "store_name": store,
        "title": store,
        "confidence": 0.45,
        "total_line_count": len(total_lines),
    }
