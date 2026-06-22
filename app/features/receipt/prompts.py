import json

from app.core.config import settings
from app.schemas.receipt import ReceiptAnalysisOptions

RECEIPT_ANALYSIS_PROMPT = """
# Role
You are a strict receipt-to-transaction JSON extraction engine.

# Task
Extract account-book transaction candidate fields from OCR text of a receipt.

# Output Rules
Return ONLY a JSON object matching the response schema.
Do not include markdown, explanations, code fences, or extra keys.

# Field Rules
- title: short transaction title. Prefer store name if clear.
- store_name: store/shop name exactly as inferred from receipt. Use null if unclear.
- amount: final paid total amount as an integer.
  Prefer total keywords such as 合計, 税込合計, お買上計, total, 합계.
  Do NOT use change, deposit, points, tax-only, or cash received amounts.
- transaction_date: yyyy-MM-dd. Use null if no valid date exists.
- category_name: infer one simple Korean category, such as 식비, 교통비, 생활, 쇼핑, 통신비, 의료, 주거, 기타.
- memo: short summary of major purchased items. Keep item names in the original receipt language.
  Do not translate item names. Do not mix languages.
- confidence: number from 0 to 1. Be conservative.

# Strictness
- Do not invent values not supported by OCR text.
- If unsure, return null for that field.
"""


def build_receipt_text_analysis_prompt(
    raw_text: str,
    candidates: dict,
    options: ReceiptAnalysisOptions,
) -> str:
    return json.dumps(
        {
            "rawText": raw_text,
            "candidates": candidates,
            "currencyCode": options.currency_code,
            "ocrLanguage": options.ocr_language or settings.OCR_LANGUAGE,
        },
        ensure_ascii=False,
    )


def build_receipt_vision_prompt(
    options: ReceiptAnalysisOptions,
) -> str:
    return json.dumps(
        {
            "instruction": (
                "이 이미지는 영수증입니다. "
                "점포명, 총액, 거래일, 거래명, 카테고리, 메모 후보를 추출하세요. "
                "반드시 response_schema 형식의 JSON으로만 응답하세요. "
                "총액은 合計, 税込合計, 総合計, お買上計, TOTAL, 결제금액에 해당하는 최종 금액을 우선하세요. "
                "카드번호, 승인번호, 등록번호, 전화번호는 금액으로 선택하지 마세요."
            ),
            "currencyCode": options.currency_code,
            "ocrLanguage": options.ocr_language or settings.OCR_LANGUAGE,
        },
        ensure_ascii=False,
    )
