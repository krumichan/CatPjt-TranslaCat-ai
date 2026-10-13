from __future__ import annotations

import io
from copy import deepcopy
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest
from fastapi import UploadFile
from PIL import Image
from starlette.datastructures import Headers

from app.ai.ports import TextGenerationProvider
from app.core.config import settings
from app.features.receipt.parser import normalize_date
from app.features.receipt.service import ReceiptAnalysisService
from app.features.receipt.validation import validate_response
from app.schemas.receipt import ReceiptAnalysisMode, ReceiptAnalysisOptions
from app.services.ocr_service import OCRService


def candidate(**changes):
    return {
        "title": "サンプル店", "store_name": "サンプル店", "merchant_evidence": "サンプル店",
        "purchase_total": "1188", "original_amount": None,
        "detected_currency_code": "JPY", "transaction_date": "2026-10-12",
        "confidence": 0.99, "category_name": "Food",
        "payment_breakdown": [
            {"payment_type": "CREDIT_CARD", "amount": "1188", "evidence": "クレジット 1,188"},
        ],
        **changes,
    }


def validated(raw):
    return validate_response(
        {"receipts": [raw]}, ReceiptAnalysisOptions(category_candidates=["Food"]),
        ocr_engine="vision", used_ai=True,
    ).receipts[0]


@pytest.mark.parametrize("printed,expected", [
    ("令和8年10月13日", "2026-10-13"),
    ("令和 ８ 年 １０ 月 １３ 日", "2026-10-13"),
    ("令和元年5月1日", "2019-05-01"),
    ("平成31年4月30日", "2019-04-30"),
    ("平成元年1月8日", "1989-01-08"),
    ("昭和64年1月7日", "1989-01-07"),
    ("昭和元年12月25日", "1926-12-25"),
    ("令和2年2月29日", "2020-02-29"),
    ("令和元年4月30日", None),
    ("平成31年5月1日", None),
    ("昭和64年1月8日", None),
    ("令和0年10月13日", None),
    ("令和8年2月29日", None),
    ("8年10月13日", None),
])
def test_explicit_japanese_era_dates_respect_calendar_boundaries(printed, expected):
    # 실행 / 검증: 명시된 연호만 변환하고 잘못된 날짜는 추측하지 않는다.
    assert normalize_date(printed) == expected


def test_hospital_printed_date_is_normalized_before_recovery():
    # 준비: 실제 실패 응답처럼 ISO 필드는 비어 있고 원문 연호 날짜는 읽혀 있다.
    raw = candidate(source_date="令和8年10月13日", transaction_date=None)
    original = deepcopy(raw)

    # 실행
    item = validated(raw)
    plan = ReceiptAnalysisService._recovery_plan(raw)

    # 검증: 별도 AI 재판독 없이 날짜를 얻고 원본 응답을 수정하지 않는다.
    assert item.transaction_date.isoformat() == "2026-10-13"
    assert "INVALID_OR_AMBIGUOUS_DATE" not in item.warnings
    assert plan is None
    assert raw == original


def test_invalid_era_source_does_not_fall_back_to_invented_iso_date():
    # 준비
    raw = candidate(source_date="平成31年5月1日", transaction_date="2019-05-01")

    # 실행 / 검증
    assert validated(raw).transaction_date is None


def test_tax_subtotal_misclassified_as_points_cannot_produce_book_amount():
    # 준비: 합계는 맞지만 세전 금액과 세금을 결제 배분으로 만든 실패를 재현한다.
    raw = candidate(payment_breakdown=[
        {"payment_type": "LOYALTY_POINTS", "amount": "1080",
         "evidence": "今回買上金額(税抜) 1,080", "duplicate_group": "points1"},
        {"payment_type": "CREDIT_CARD", "amount": "108",
         "evidence": "今回決済金額 108", "duplicate_group": "card2"},
    ])
    original = deepcopy(raw)

    # 실행
    item = validated(raw)

    # 검증: 잘못된 108엔이나 임의의 1,188엔을 확정하지 않고 BE도 거부하는 UNKNOWN으로 보낸다.
    assert item.book_amount is None
    assert item.original_amount is None
    assert item.review_status == "NEEDS_REVIEW"
    assert item.payment_breakdown[0].payment_type == "UNKNOWN"
    assert item.payment_breakdown[0].amount == Decimal("1080")
    assert item.payment_breakdown[0].evidence == raw["payment_breakdown"][0]["evidence"]
    assert "PAYMENT_EVIDENCE_CONFLICT" in item.warnings
    assert ReceiptAnalysisService._recovery_plan(raw) is None
    assert raw == original


@pytest.mark.parametrize("evidence", [
    "前回残ポイント 1,080P", "今回お買上ポイント 1,080P", "累計ポイント 1,080P",
    "ポイント残高 1,080P", "獲得ポイント 1,080P", "付与ポイント 1,080P",
    "earned points 1080", "points balance 1080", "적립 포인트 1080",
])
def test_point_balance_and_earnings_are_not_redeemed_points(evidence):
    # 준비
    raw = candidate(payment_breakdown=[
        {"payment_type": "LOYALTY_POINTS", "amount": "1080", "evidence": evidence},
        {"payment_type": "CREDIT_CARD", "amount": "108", "evidence": "card 108"},
    ])

    # 실행 / 검증
    assert validated(raw).book_amount is None


@pytest.mark.parametrize("evidence", ["消費税額 108", "小計 108", "tax amount 108"])
def test_tax_and_subtotal_cannot_be_payment_evidence(evidence):
    # 준비
    raw = candidate(purchase_total="108", payment_breakdown=[
        {"payment_type": "CREDIT_CARD", "amount": "108", "evidence": evidence},
    ])

    # 실행 / 검증
    assert validated(raw).book_amount is None


def test_real_point_redemption_and_duplicate_card_remain_5020():
    # 준비: 정상 포인트 사용과 중복 카드 명세를 보존한다.
    raw = candidate(purchase_total="7089", payment_breakdown=[
        {"payment_type": "LOYALTY_POINTS", "amount": "2069", "evidence": "ポイント利用 2,069"},
        {"payment_type": "CREDIT_CARD", "amount": "5020", "evidence": "カード支払 5,020",
         "duplicate_group": "card"},
        {"payment_type": "CREDIT_CARD", "amount": "5020", "evidence": "クレジット売上票 5,020",
         "duplicate_group": "card"},
    ])

    # 실행 / 검증
    assert validated(raw).book_amount == Decimal("5020")


def test_user_corrected_allocation_revalidates_without_sticky_warning():
    # 준비: 원본을 확인한 사용자가 포인트 오분류를 실제 카드 지급액으로 고쳤다.
    raw = candidate(warnings=["PAYMENT_EVIDENCE_CONFLICT"], payment_breakdown=[
        {"payment_type": "CREDIT_CARD", "amount": "1188", "evidence": "クレジット 1,188(税込)"},
    ])

    # 실행 / 검증: 과거 오류를 현재 금액의 영구 차단 상태로 사용하지 않는다.
    item = validated(raw)
    assert item.book_amount == Decimal("1188")
    assert item.review_status == "READY"
    assert "PAYMENT_EVIDENCE_CONFLICT" not in item.warnings


def test_ambiguous_source_date_still_needs_financial_recovery():
    # 준비: 근거 없이 지정된 MDY를 이용해 재판독을 건너뛰면 안 된다.
    raw = candidate(transaction_date=None, source_date="03/04/2026", date_order="MDY")

    # 실행 / 검증
    plan = ReceiptAnalysisService._recovery_plan(raw)
    assert plan is not None and plan.financial_needed


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["era", "conflicting_payment", "normal"])
async def test_fast_path_does_not_add_provider_or_ocr_calls(monkeypatch, case):
    # 준비: 이미지 전처리·스케줄러·후처리는 실제 코드, 외부 AI만 합성 응답이다.
    monkeypatch.setattr(settings, "RECEIPT_ANALYSIS_MODE", "VISION_ONLY")
    raw = candidate()
    if case == "era":
        raw.update(source_date="令和8年10月13日", transaction_date=None)
    elif case == "conflicting_payment":
        raw["payment_breakdown"] = [
            {"payment_type": "LOYALTY_POINTS", "amount": "1080", "evidence": "税抜 1080"},
            {"payment_type": "CREDIT_CARD", "amount": "108", "evidence": "card 108"},
        ]
    provider = AsyncMock(spec=TextGenerationProvider)
    provider.call_with_image.return_value = {"receipts": [raw]}
    ocr = OCRService()
    ocr.extract_document = AsyncMock(side_effect=AssertionError("OCR must not run"))
    buffer = io.BytesIO()
    Image.new("RGB", (480, 640), "white").save(buffer, format="JPEG")
    upload = UploadFile(filename="synthetic.jpg", file=io.BytesIO(buffer.getvalue()),
                        headers=Headers({"content-type": "image/jpeg"}))

    # 실행
    response = await ReceiptAnalysisService(ocr, provider).analyze(
        upload, ReceiptAnalysisOptions(analysis_mode=ReceiptAnalysisMode.VISION_ONLY,
                                       category_candidates=["Food"]),
    )

    # 검증: 정상/변환 가능 날짜/결제 모순 모두 추가 호출 없이 결과 또는 검토 상태를 반환한다.
    provider.call_with_image.assert_awaited_once()
    ocr.extract_document.assert_not_awaited()
    if case == "era":
        assert response.receipts[0].transaction_date.isoformat() == "2026-10-13"
    elif case == "conflicting_payment":
        assert response.receipts[0].book_amount is None
    else:
        assert response.receipts[0].book_amount == Decimal("1188")
