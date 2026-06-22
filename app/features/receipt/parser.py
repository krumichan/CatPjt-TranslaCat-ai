import re
from datetime import date
from typing import Any

_AMOUNT_KEYWORDS = (
    "合計",
    "税込合計",
    "総合計",
    "お買上計",
    "小計",
    "計",
    "TOTAL",
    "Total",
    "total",
    "합계",
    "총액",
    "결제금액",
)
_EXCLUDE_AMOUNT_KEYWORDS = (
    "お預り",
    "お釣",
    "おつり",
    "釣銭",
    "預り",
    "現金",
    "クレジット",
    "支払",
    "받은금액",
    "거스름돈",
)


def normalize_amount(value: str) -> int | None:
    digits = re.sub(r"[^0-9]", "", value)

    if not digits:
        return None

    try:
        return int(digits)
    except ValueError:
        return None


def normalize_date(value: str) -> str | None:
    value = value.strip()
    patterns = [
        r"(?P<year>20\d{2})[./\-年](?P<month>\d{1,2})[./\-月](?P<day>\d{1,2})日?",
        r"(?P<year>\d{2})[./\-](?P<month>\d{1,2})[./\-](?P<day>\d{1,2})",
    ]

    for pattern in patterns:
        match = re.search(pattern, value)
        if not match:
            continue

        year = int(match.group("year"))
        if year < 100:
            year += 2000

        month = int(match.group("month"))
        day = int(match.group("day"))

        try:
            return date(year, month, day).isoformat()
        except ValueError:
            return None

    return None


def _extract_amount_candidates(lines: list[str]) -> list[int]:
    weighted_candidates: list[tuple[int, int]] = []

    for line_index, line in enumerate(lines):
        amounts = [
            amount
            for amount in (
                normalize_amount(value)
                for value in re.findall(r"(?:¥|￥)?\s*[0-9][0-9,]*", line)
            )
            if amount is not None and amount > 0
        ]

        if not amounts:
            continue

        has_total_keyword = any(keyword in line for keyword in _AMOUNT_KEYWORDS)
        has_exclude_keyword = any(keyword in line for keyword in _EXCLUDE_AMOUNT_KEYWORDS)

        for amount in amounts:
            score = 10
            if has_total_keyword:
                score += 100
            if has_exclude_keyword:
                score -= 80

            score += min(line_index, 20)
            weighted_candidates.append((score, amount))

    weighted_candidates.sort(key=lambda item: item[0], reverse=True)

    result: list[int] = []
    for _, amount in weighted_candidates:
        if amount not in result:
            result.append(amount)

    return result[:5]


def _extract_date_candidates(lines: list[str]) -> list[str]:
    result: list[str] = []

    for line in lines:
        normalized = normalize_date(line)
        if normalized and normalized not in result:
            result.append(normalized)

    return result[:5]


def _extract_store_candidates(lines: list[str]) -> list[str]:
    candidates: list[str] = []

    for line in lines[:8]:
        value = line.strip()
        if not value:
            continue
        if len(value) <= 1:
            continue
        if any(keyword in value for keyword in _AMOUNT_KEYWORDS):
            continue
        if normalize_date(value):
            continue
        if re.search(r"[0-9]{3,}", value):
            continue

        candidates.append(value)

    return candidates[:5]


def extract_receipt_candidates(raw_text: str) -> dict[str, Any]:
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

    return {
        "amountCandidates": _extract_amount_candidates(lines),
        "dateCandidates": _extract_date_candidates(lines),
        "storeNameCandidates": _extract_store_candidates(lines),
        "topLines": lines[:10],
    }
