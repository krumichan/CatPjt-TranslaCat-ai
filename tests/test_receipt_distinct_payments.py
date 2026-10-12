"""같은 상호의 독립 결제 보존. provider 재생은 LIVE 정확도 결과가 아니다."""
from decimal import Decimal

import pytest

from app.features.receipt.validation import validate_response
from app.schemas.receipt import ReceiptAnalysisOptions


@pytest.mark.parametrize("second_amount", ["4030", "5500"])
def test_same_merchant_separate_regions_keep_independent_payments(second_amount):
    # 준비: 10/8 동일 상호의 두 결제. 동액 결제도 별개 위치/시각이면 보존한다.
    base = {
        "store_name": "ちいかわレストラン",
        "title": "ちいかわレストラン 沖縄パルコシティ",
        "transaction_date": "2026-10-08",
        "detected_currency_code": "JPY",
        "currency_evidence": "日本 JPY",
        "currency_confidence": 0.98,
        "confidence": 0.98,
        "category_name": "食費",
        "status": "READY",
    }
    amounts = ["5500", second_amount]
    times = ["18:05:51", "17:59:26"]
    boxes = [[0.063, 0.272, 0.475, 0.906], [0.476, 0.214, 0.917, 0.933]]
    candidates = [
        {
            **base,
            "purchase_total": amount,
            "transaction_time": receipt_time,
            "bounding_box": box,
            "payment_breakdown": [{"payment_type": "CREDIT_CARD", "amount": amount}],
        }
        for amount, receipt_time, box in zip(amounts, times, boxes, strict=True)
    ]

    # 실행: 실제 정규화/중복 판정 코드만 실행한다. 모델이나 원본 인식 정확도를 흉내 내지 않는다.
    result = validate_response(
        {"receipts": candidates}, ReceiptAnalysisOptions(category_candidates=["食費"]),
        ocr_engine="vision", used_ai=True,
    )

    # 검증: 금액/시각/출처가 항목별로 유지되며 같은 상호로 중복 제거하지 않는다.
    assert result.receipt_count == 2
    assert [item.book_amount for item in result.receipts] == [Decimal(v) for v in amounts]
    assert [item.transaction_time.isoformat() for item in result.receipts] == times
    assert [item.bounding_box for item in result.receipts] == boxes
    assert len({item.receipt_id for item in result.receipts}) == 2
    assert "DUPLICATE_RECEIPT_REMOVED" not in result.warnings
