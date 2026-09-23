import asyncio
import io
import json
import time
from decimal import Decimal
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from fastapi import UploadFile
from PIL import Image
from starlette.datastructures import Headers

from app.ai.ports import StructuredGenerationResult, TextGenerationProvider
from app.core.config import settings
from app.features.receipt.parser import (
    extract_receipt_candidates,
    normalize_amount,
    normalize_date,
)
from app.features.receipt.prompts import build_receipt_vision_prompt
from app.features.receipt.service import (
    ReceiptAnalysisService,
    _RECEIPT_ANALYSIS_SCHEMA,
)
from app.features.receipt.validation import validate_response
from app.schemas.receipt import (
    ReceiptAnalysisMode,
    ReceiptAnalysisOptions,
    ReceiptStatus,
)
from app.services.ocr_service import OCRDocument, OCRLine, OCRService


def receipt(**changes):
    return {
        "title": "Coffee",
        "store_name": "Cafe",
        "original_amount": "12.34",
        "detected_currency_code": "USD",
        "transaction_date": "2026-09-15",
        "category_name": "Food",
        "memo": "Coffee and cake",
        "confidence": 0.95,
        "detected_language": "en",
        "currency_confidence": 0.98,
        **changes,
    }


def validate(*items, categories=None):
    return validate_response(
        {"receipts": list(items)},
        ReceiptAnalysisOptions(category_candidates=categories or ["Food"]),
        ocr_engine="vision",
        used_ai=True,
    )


@pytest.mark.parametrize(
    "raw,currency,expected",
    [
        ("JPY 1,280", "JPY", "1280"),
        ("KRW 12,000", "KRW", "12000"),
        ("USD 12.34", "USD", "12.34"),
        ("EUR 1.234,56", "EUR", "1234.56"),
        ("EUR 1 234,56", "EUR", "1234.56"),
        ("KWD 10.125", "KWD", "10.125"),
        ("CLF 10.1234", "CLF", "10.1234"),
        (1280, "JPY", "1280"),
        ("1,234.56", "USD", "1234.56"),
        ("60.000", "IDR", "60000"),
        (Decimal("12.34"), "USD", "12.34"),
    ],
)
def test_decimal_formats(raw, currency, expected):
    assert normalize_amount(raw, currency) == Decimal(expected)


@pytest.mark.parametrize(
    "raw,currency",
    [
        (0, "USD"),
        (-1, "USD"),
        (True, "USD"),
        (12.34, "USD"),
        ("-12.34", "USD"),
        ("NaN", "USD"),
        ("Infinity", "USD"),
        ("1,234", "USD"),
        ("1.234", "EUR"),
        ("10,125", "KWD"),
        ("1.23.45", "USD"),
        ("12/34", "USD"),
        ("USD 12 CAD 34", "USD"),
        ("1e99", "USD"),
        ("12.34", "JPY"),
        ("1.234567891", None),
        ("999999999999999999999999", "USD"),
    ],
)
def test_invalid_or_ambiguous_money_is_not_repaired(raw, currency):
    assert normalize_amount(raw, currency) is None


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2026-09-15", "2026-09-15"),
        ("2026年9月15日", "2026-09-15"),
        ("15/09/2026", "2026-09-15"),
        ("09/15/2026", "2026-09-15"),
        ("03/04/2026", None),
        ("2026-02-30", None),
        (None, None),
    ],
)
def test_dates_do_not_guess_locale(raw, expected):
    assert normalize_date(raw) == expected


def test_ambiguous_date_uses_explicit_source_evidence_only():
    item = receipt(
        transaction_date="2026-03-04", source_date="03/04/2026", date_order="MDY"
    )
    assert validate(item).receipts[0].transaction_date is None
    item["date_order_evidence"] = "US receipt locale"
    transaction_date = validate(item).receipts[0].transaction_date
    assert transaction_date is not None
    assert transaction_date.isoformat() == "2026-03-04"


def test_turkish_fiscal_date_uses_matching_currency_and_language_locale_evidence():
    item = receipt(
        transaction_date=None,
        source_date="06-11-2023",
        date_order=None,
        date_order_evidence=None,
        detected_currency_code="TRY",
        detected_language="tr",
    )

    result = validate(item).receipts[0]

    assert result.transaction_date is not None
    assert result.transaction_date.isoformat() == "2023-11-06"
    assert "DATE_ORDER_FROM_TURKISH_LOCALE" in result.warnings


def test_single_receipt_decimal_json_contract():
    response = validate(receipt())
    assert response.receipt_count == 1
    assert response.receipts[0].status == ReceiptStatus.READY
    assert (
        json.loads(response.model_dump_json())["receipts"][0]["original_amount"]
        == "12.34"
    )


def test_printed_transaction_time_is_validated_and_preserved():
    item = validate(receipt(transaction_time="20:13:39", source_time="8:13:39 PM")).receipts[0]
    assert item.transaction_time is not None
    assert item.transaction_time.isoformat() == "20:13:39"

    invalid = validate(receipt(transaction_time="25:99", source_time="25:99")).receipts[0]
    assert invalid.transaction_time is None
    assert "INVALID_TIME" in invalid.warnings
    assert invalid.status == ReceiptStatus.NEEDS_REVIEW

    corrected = validate(
        receipt(transaction_time="08:13:39", source_time="8:13:39 PM")
    ).receipts[0]
    assert corrected.transaction_time is not None
    assert corrected.transaction_time.isoformat() == "20:13:39"
    assert "TIME_NORMALIZED_FROM_SOURCE" in corrected.warnings
    assert corrected.status == ReceiptStatus.READY


def test_prompt_forbids_shortening_source_language_brand_tokens():
    prompt = build_receipt_vision_prompt(ReceiptAnalysisOptions())
    assert "never translate" in prompt
    assert "abbreviate, or shorten" in prompt
    assert "どらっぐ" in prompt
    assert "IDR 60.000 means 60000" in prompt
    assert "8:13:39 PM -> 20:13:39" in prompt


def test_papasu_points_and_duplicate_card_detail_produce_5020_book_amount():
    item = validate(
        receipt(
            title="どらっぐ ぱぱす 船堀店",
            store_name="どらっぐ ぱぱす",
            branch_name="船堀店",
            purchase_total="7089",
            original_amount=None,
            payment_breakdown=[
                {
                    "payment_type": "LOYALTY_POINTS",
                    "amount": "2069",
                    "evidence": "ポイント支払",
                    "duplicate_group": None,
                },
                {
                    "payment_type": "CREDIT_CARD",
                    "amount": "5020",
                    "evidence": "クレジット（他クレ）",
                    "duplicate_group": "card-1",
                },
                {
                    "payment_type": "CREDIT_CARD",
                    "amount": "5020",
                    "evidence": "カード明細",
                    "duplicate_group": "card-1",
                },
            ],
            change="0",
        )
    ).receipts[0]
    assert item.purchase_total == Decimal("7089")
    assert item.book_amount == Decimal("5020")
    assert item.original_amount == Decimal("5020")
    assert item.review_status.value == "READY"
    assert item.status == ReceiptStatus.READY
    assert "DUPLICATE_PAYMENT_DETAIL_COLLAPSED" in item.warnings


def test_cash_tendered_is_not_double_counted_as_the_cash_allocation():
    item = validate(
        receipt(
            purchase_total="9.00",
            original_amount=None,
            payment_breakdown=[{"payment_type": "CASH", "amount": "10.00"}],
            cash_tendered="10.00",
            change="1.00",
        )
    ).receipts[0]
    assert item.book_amount == Decimal("9.00")
    assert item.status == ReceiptStatus.READY
    assert "CASH_TENDERED_NOT_DOUBLE_COUNTED" in item.warnings


def test_cash_tendered_without_change_keeps_a_reconciled_cash_payment():
    item = validate(
        receipt(
            purchase_total="70.00",
            original_amount=None,
            payment_breakdown=[{"payment_type": "CASH", "amount": "70.00"}],
            cash_tendered="70.00",
            change=None,
        )
    ).receipts[0]
    assert item.book_amount == Decimal("70.00")
    assert item.original_amount == Decimal("70.00")
    assert item.status == ReceiptStatus.READY
    assert "CASH_TENDERED_DUPLICATE_WITHOUT_CHANGE" in item.warnings


def test_zero_cash_tendered_is_normalized_to_missing_without_rejecting_the_batch():
    response = validate(
        receipt(
            purchase_total="12.34",
            original_amount=None,
            payment_breakdown=[],
            cash_tendered="0",
            change="0",
        ),
        receipt(purchase_total="9.99", original_amount=None),
    )
    assert response.receipt_count == 2
    assert response.receipts[0].cash_tendered is None
    assert response.receipts[0].change == Decimal("0")
    assert response.receipts[0].book_amount == Decimal("12.34")


def test_unlabelled_exact_card_duplicate_is_collapsed_only_to_reconcile_total():
    item = validate(
        receipt(
            purchase_total="7089",
            original_amount=None,
            payment_breakdown=[
                {"payment_type": "LOYALTY_POINTS", "amount": "2069"},
                {"payment_type": "CREDIT_CARD", "amount": "5020"},
                {"payment_type": "CREDIT_CARD", "amount": "5020"},
            ],
        )
    ).receipts[0]
    assert item.book_amount == Decimal("5020")
    assert item.status == ReceiptStatus.READY


def test_gross_card_heading_plus_points_derives_net_settlement():
    item = validate(
        receipt(
            title="どらっぐ ぱぱす 船堀店",
            store_name="どらっぐ ぱぱす",
            branch_name="船堀店",
            purchase_total="7089",
            original_amount=None,
            payment_breakdown=[
                {"payment_type": "CREDIT_CARD", "amount": "7089"},
                {"payment_type": "LOYALTY_POINTS", "amount": "2069"},
            ],
        )
    ).receipts[0]
    assert item.book_amount == Decimal("5020")
    assert item.original_amount == Decimal("5020")
    assert item.status == ReceiptStatus.READY
    assert "GROSS_PAYMENT_LINE_REPLACED_BY_NET_SETTLEMENT" in item.warnings


def test_payment_mismatch_is_not_an_authoritative_book_amount():
    item = validate(
        receipt(
            purchase_total="7089",
            original_amount="7089",
            payment_breakdown=[
                {"payment_type": "LOYALTY_POINTS", "amount": "1000"},
                {"payment_type": "CREDIT_CARD", "amount": "5020"},
            ],
        )
    ).receipts[0]
    assert item.book_amount is None
    assert item.original_amount is None
    assert item.review_status.value == "NEEDS_REVIEW"
    assert item.status == ReceiptStatus.NEEDS_REVIEW


def test_full_loyalty_redemption_is_excluded_from_registration():
    item = validate(
        receipt(
            purchase_total="1000",
            original_amount=None,
            payment_breakdown=[
                {"payment_type": "LOYALTY_POINTS", "amount": "1000"}
            ],
        )
    ).receipts[0]
    assert item.book_amount == Decimal("0")
    assert item.review_status.value == "EXCLUDED"
    assert item.status == ReceiptStatus.NEEDS_REVIEW


def test_multiple_receipts_mixed_languages_and_currencies():
    response = validate(
        receipt(
            store_name="コンビニ",
            original_amount="1,280",
            detected_currency_code="JPY",
            detected_language="ja",
        ),
        receipt(
            store_name="Café Paris",
            original_amount="1.234,56",
            detected_currency_code="EUR",
            detected_language="fr",
        ),
        receipt(
            store_name="택시",
            original_amount="12,000",
            detected_currency_code="KRW",
            detected_language="ko",
        ),
    )
    assert response.receipt_count == 3
    assert [item.original_amount for item in response.receipts] == [
        Decimal("1280"),
        Decimal("1234.56"),
        Decimal("12000"),
    ]
    assert len({item.receipt_id for item in response.receipts}) == 3


@pytest.mark.parametrize("code", ["$", "¥", "ZZZ", "US", "USDD", None])
def test_unknown_currency_not_guessed(code):
    item = validate(receipt(detected_currency_code=code)).receipts[0]
    assert item.detected_currency_code is None
    assert "UNKNOWN_CURRENCY" in item.warnings


def test_known_code_with_ambiguous_symbol_evidence_is_unknown():
    item = validate(
        receipt(currency_evidence="$", detected_currency_code="USD")
    ).receipts[0]
    assert item.detected_currency_code is None
    assert "AMBIGUOUS_CURRENCY_SYMBOL" in item.warnings


def test_lowercase_currency_normalization_and_not_enabled_source():
    assert (
        validate(receipt(detected_currency_code=" thb "))
        .receipts[0]
        .detected_currency_code
        == "THB"
    )


def test_low_currency_confidence_does_not_authorize_conversion():
    assert (
        validate(receipt(currency_confidence=0.4)).receipts[0].detected_currency_code
        is None
    )


def test_missing_date_retained_as_reviewable_partial_item():
    item = validate(receipt(transaction_date=None)).receipts[0]
    assert item.original_amount == Decimal("12.34")
    assert item.transaction_date is None
    assert item.status == ReceiptStatus.NEEDS_REVIEW


def test_unreadable_and_malformed_items_do_not_discard_good_receipt():
    response = validate(receipt(), {"status": "UNREADABLE"}, "not an object", None)
    assert response.receipt_count == 4
    assert response.receipts[0].status == ReceiptStatus.READY
    assert all(
        item.status == ReceiptStatus.UNREADABLE for item in response.receipts[1:]
    )


@pytest.mark.parametrize("confidence", [-0.1, 1.1, "NaN", "Infinity", True, {}, []])
def test_confidence_bounds_are_rejected_not_promoted(confidence):
    item = validate(receipt(confidence=confidence)).receipts[0]
    assert item.confidence is None
    assert item.status == ReceiptStatus.NEEDS_REVIEW
    assert "INVALID_CONFIDENCE" in item.warnings


def test_ready_confidence_boundary_keeps_uncertain_merchant_text_reviewable():
    assert validate(receipt(confidence=0.94)).receipts[0].status == ReceiptStatus.NEEDS_REVIEW
    assert validate(receipt(confidence=0.95)).receipts[0].status == ReceiptStatus.READY


def test_identical_physical_position_is_deduplicated():
    item = receipt(bounding_box=[0.1, 0.1, 0.4, 0.9])
    response = validate(item, dict(item))
    assert response.receipt_count == 1
    assert "DUPLICATE_RECEIPT_REMOVED" in response.warnings


def test_same_payment_at_distinct_positions_is_preserved():
    response = validate(
        receipt(bounding_box=[0, 0, 0.4, 1]), receipt(bounding_box=[0.5, 0, 1, 1])
    )
    assert response.receipt_count == 2
    assert all(item.status == ReceiptStatus.READY for item in response.receipts)


def test_duplicate_without_spatial_evidence_warns_both():
    response = validate(receipt(), receipt())
    assert response.receipt_count == 2
    assert all(
        "POSSIBLE_DUPLICATE_RECEIPT" in item.warnings for item in response.receipts
    )


def test_no_hallucinated_fx_or_ocr_fields_in_contract():
    response = validate(
        receipt(exchange_rate="150", converted_amount="1851", raw_text="CARD SECRET")
    )
    serialized = response.model_dump_json()
    assert "exchange_rate" not in serialized
    assert "converted_amount" not in serialized
    assert "CARD SECRET" not in serialized
    assert "exchange_rate" not in str(_RECEIPT_ANALYSIS_SCHEMA)
    amount_schema = _RECEIPT_ANALYSIS_SCHEMA["properties"]["receipts"]["items"][
        "properties"
    ]["original_amount"]
    assert amount_schema["type"] == "STRING"


def test_category_reuses_existing_and_preserves_safe_new_suggestion():
    existing = validate(
        receipt(category_name="食費", category_source="EXISTING"),
        categories=["食費"],
    ).receipts[0]
    assert existing.category_name == "食費"
    assert existing.category_source == "EXISTING"

    suggested = validate(
        receipt(
            category_name="반려동물",
            category_source="NEW",
            category_reason="반려동물 용품이 확인됨",
        )
    ).receipts[0]
    assert suggested.category_name == "반려동물"
    assert suggested.category_source == "NEW"
    assert "CATEGORY_NEW_SUGGESTION" in suggested.warnings


def test_no_category_candidates_uses_stable_fallback_instead_of_blank():
    response = validate_response(
        {"receipts": [receipt(category_name=None)]},
        ReceiptAnalysisOptions(),
        ocr_engine="vision",
        used_ai=True,
    )
    item = response.receipts[0]
    assert item.category_name == "기타"
    assert item.category_source == "FALLBACK"
    assert "CATEGORY_FALLBACK_USED" in item.warnings


@pytest.mark.parametrize(
    "result", [None, [], "json", {"amount": "123"}, {"receipts": "bad"}]
)
def test_malformed_envelope_rejected_for_safe_fallback(result):
    with pytest.raises(ValueError):
        validate_response(
            result, ReceiptAnalysisOptions(), ocr_engine="vision", used_ai=True
        )


def test_target_currency_never_enters_prompt_or_source_detection():
    first = build_receipt_vision_prompt(ReceiptAnalysisOptions(currency_code="JPY"))
    second = build_receipt_vision_prompt(ReceiptAnalysisOptions(currency_code="USD"))
    assert first == second


def upload(contents=b"image"):
    return UploadFile(
        filename="receipts.jpg",
        file=io.BytesIO(contents),
        headers=Headers({"content-type": "image/jpeg"}),
    )


def service(
    vision_result=None, vision_error=None, text_result=None, doc=None
) -> tuple[ReceiptAnalysisService, Any, AsyncMock]:
    ocr = OCRService()
    ocr.extract_document_from_upload = AsyncMock(
        return_value=doc
        or OCRDocument(
            [
                OCRLine("Cafe", [[0, 0], [100, 0], [100, 20], [0, 20]], 0.99),
                OCRLine(
                    "TOTAL USD 12.34", [[0, 30], [100, 30], [100, 50], [0, 50]], 0.99
                ),
                OCRLine("2026-09-15", [[0, 60], [100, 60], [100, 80], [0, 80]], 0.99),
            ]
        )
    )
    provider = AsyncMock()
    provider.call_with_image.return_value = vision_result or {"receipts": [receipt()]}
    provider.call_with_image.side_effect = vision_error
    provider.call.return_value = text_result or {"receipts": [receipt()]}
    return ReceiptAnalysisService(ocr, provider), ocr, provider


def run(svc, mode=None, **kwargs):
    return asyncio.run(
        svc.analyze(
            upload(),
            ReceiptAnalysisOptions(
                analysis_mode=mode, category_candidates=["Food"], **kwargs
            ),
        )
    )


def test_vision_success_does_not_run_ocr():
    svc, ocr, provider = service()
    response = run(svc, ReceiptAnalysisMode.VISION_ONLY)
    assert response.receipt_count == 1
    assert response.used_ai
    provider.call_with_image.assert_awaited_once()
    ocr.extract_document_from_upload.assert_not_awaited()


def test_vision_debug_trace_preserves_provider_and_validation_stages(tmp_path, monkeypatch):
    class MetadataProvider:
        async def call_with_image_with_metadata(self, **_kwargs):
            return StructuredGenerationResult(
                data={"receipts": [receipt()]},
                input_tokens=321,
                output_tokens=123,
                provider="openai",
                model="gpt-5.6-luna",
                status="completed",
                latency_ms=456,
            )

    monkeypatch.setenv("RECEIPT_DEBUG_TRACE_DIR", str(tmp_path))
    svc = ReceiptAnalysisService(
        OCRService(), cast(TextGenerationProvider, MetadataProvider())
    )
    response = asyncio.run(
        svc.analyze(
            upload(),
            ReceiptAnalysisOptions(analysis_mode=ReceiptAnalysisMode.VISION_ONLY),
            trace_id="receipt-trace-test",
        )
    )
    payload = json.loads((tmp_path / "receipt-trace-test.json").read_text("utf-8"))
    assert response.analysis_trace_id == "receipt-trace-test"
    assert payload["input"]["sha256"]
    assert payload["providerCallCount"] == 1
    assert payload["providerAttempts"][0]["model"] == "gpt-5.6-luna"
    assert payload["providerAttempts"][0]["candidateCount"] == 1
    assert payload["validation"]["candidateCount"] == 1
    assert payload["request"]["imageDetail"] == "high"
    assert response.runtime_identity is not None
    assert payload["runtimeIdentity"]["run_id"] == response.runtime_identity.run_id
    assert payload["runtimeIdentity"]["source_fingerprint"] == response.runtime_identity.source_fingerprint
    assert payload["runtimeIdentity"]["process_id"] == response.runtime_identity.process_id


def test_incomplete_bounded_receipt_is_recovered_from_an_automatic_source_crop():
    class RecoveryProvider:
        def __init__(self):
            self.calls = []

        async def call_with_image(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return {
                    "receipts": [
                        receipt(
                            title="LAWSON 船堀店",
                            store_name="LAWSON",
                            original_amount=None,
                            purchase_total=None,
                            detected_currency_code="JPY",
                            transaction_date="2026-09-19",
                            bounding_box=[0.75, 0.1, 1.0, 0.95],
                            status="NEEDS_REVIEW",
                        )
                    ]
                }
            return {
                "receipts": [
                    receipt(
                        title="LAWSON 船堀店",
                        store_name="LAWSON",
                        original_amount=None,
                        purchase_total="1680",
                        detected_currency_code="JPY",
                        transaction_date="2026-09-19",
                        status="READY",
                    )
                ]
            }

    encoded = io.BytesIO()
    Image.new("RGB", (1200, 1600), "white").save(encoded, format="JPEG")
    image = UploadFile(
        filename="multi.jpg",
        file=io.BytesIO(encoded.getvalue()),
        headers=Headers({"content-type": "image/jpeg"}),
    )
    provider = RecoveryProvider()
    response = asyncio.run(
        ReceiptAnalysisService(
            OCRService(), cast(TextGenerationProvider, provider)
        ).analyze(
            image,
            ReceiptAnalysisOptions(
                analysis_mode=ReceiptAnalysisMode.VISION_ONLY,
                category_candidates=["Food"],
            ),
        )
    )
    assert len(provider.calls) == 2
    assert len(provider.calls[1]["image_bytes"]) < len(encoded.getvalue())
    assert provider.calls[1]["schema"] == _RECEIPT_ANALYSIS_SCHEMA
    assert response.receipt_count == 1
    assert response.receipts[0].purchase_total == Decimal("1680")
    assert response.receipts[0].book_amount == Decimal("1680")
    assert "REGION_RECOVERY_USED" in response.receipts[0].warnings
    assert "VISION_REGION_RECOVERY_USED" in response.warnings


def test_identical_identity_strings_from_crop_still_require_source_review():
    class SameWrongIdentityProvider:
        def __init__(self):
            self.calls = 0

        async def call_with_image(self, **_kwargs):
            self.calls += 1
            common = receipt(
                title="AEON フードスタイル船橋店",
                store_name="AEON",
                branch_name="フードスタイル船橋店",
                merchant_evidence="AEON" if self.calls == 1 else "株式会社イオンフードスタイル",
                branch_evidence="フードスタイル船橋店",
                purchase_total="6238",
                detected_currency_code="JPY",
                transaction_date="2026-09-21",
                bounding_box=[0.54, 0.30, 0.79, 0.94],
            )
            return {"receipts": [common]}

    encoded = io.BytesIO()
    Image.new("RGB", (1600, 1200), "white").save(encoded, format="JPEG")
    provider = SameWrongIdentityProvider()
    response = asyncio.run(
        ReceiptAnalysisService(
            OCRService(), cast(TextGenerationProvider, provider)
        ).analyze(
            upload(encoded.getvalue()),
            ReceiptAnalysisOptions(
                analysis_mode=ReceiptAnalysisMode.VISION_ONLY,
                category_candidates=["Food"],
            ),
        )
    )

    assert provider.calls == 2
    item = response.receipts[0]
    assert item.status == ReceiptStatus.NEEDS_REVIEW
    assert item.identity_verification == "SOURCE_REGION_MODEL_REREAD"
    assert item.identity_source_box == pytest.approx([0.54, 0.30, 0.79, 0.4792])
    assert "IDENTITY_SOURCE_REVIEW_REQUIRED" in item.warnings
    assert "REGION_RECOVERY_USED" in item.warnings


def test_two_incomplete_regions_recover_in_parallel_without_rereading_healthy_candidate():
    class ParallelRecoveryProvider:
        def __init__(self):
            self.calls = 0
            self.active_recoveries = 0
            self.max_active_recoveries = 0

        async def call_with_image(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return {
                    "receipts": [
                        receipt(
                            title="A",
                            store_name="A",
                            purchase_total=None,
                            original_amount=None,
                            bounding_box=[0.02, 0.02, 0.31, 0.98],
                            status="NEEDS_REVIEW",
                        ),
                        receipt(
                            title="B",
                            store_name="B",
                            purchase_total=None,
                            original_amount=None,
                            bounding_box=[0.35, 0.02, 0.64, 0.98],
                            status="NEEDS_REVIEW",
                        ),
                        receipt(
                            title="Healthy",
                            store_name="Healthy",
                            merchant_evidence="Healthy",
                            purchase_total="30.00",
                            original_amount=None,
                            bounding_box=[0.68, 0.02, 0.98, 0.98],
                        ),
                    ]
                }
            self.active_recoveries += 1
            self.max_active_recoveries = max(
                self.max_active_recoveries, self.active_recoveries
            )
            try:
                await asyncio.sleep(0.05)
                amount = "10.00" if self.calls == 2 else "20.00"
                return {"receipts": [receipt(purchase_total=amount, original_amount=None)]}
            finally:
                self.active_recoveries -= 1

    encoded = io.BytesIO()
    Image.new("RGB", (1600, 1200), "white").save(encoded, format="JPEG")
    provider = ParallelRecoveryProvider()
    response = asyncio.run(
        ReceiptAnalysisService(
            OCRService(), cast(TextGenerationProvider, provider)
        ).analyze(
            upload(encoded.getvalue()),
            ReceiptAnalysisOptions(
                analysis_mode=ReceiptAnalysisMode.VISION_ONLY,
                category_candidates=["Food"],
            ),
        )
    )

    assert provider.calls == 3
    assert provider.max_active_recoveries == 2
    assert response.receipt_count == 3
    assert response.receipts[2].purchase_total == Decimal("30.00")


def test_region_crop_accepts_camera_original_larger_than_the_ocr_decode_limit():
    encoded = io.BytesIO()
    width, height = 3024, 4032
    Image.new("RGB", (width, height), "white").save(encoded, format="JPEG")
    raw = receipt(
        title="LAWSON 船堀店",
        store_name="LAWSON",
        branch_name="船堀店",
        merchant_evidence="LAWSON",
        branch_evidence="船堀店",
        purchase_total="1680",
        original_amount=None,
        detected_currency_code="JPY",
        transaction_date="2026-09-19",
        bounding_box=[0.81, 0.49, 1.0, 0.93],
    )

    prepared = object.__new__(ReceiptAnalysisService)._prepare_recovery_regions(
        [raw], encoded.getvalue(), "prompt", 4
    )

    assert width * height > settings.OCR_MAX_IMAGE_PIXELS
    assert width * height <= settings.RECEIPT_VISION_MAX_IMAGE_PIXELS
    assert len(prepared) == 1
    assert prepared[0].plan.kind == "IDENTITY_REGION_RECOVERY"


def test_shared_receipt_provider_limit_caps_concurrent_photo_calls_at_three():
    class ConcurrencyProvider:
        def __init__(self):
            self.active = 0
            self.maximum = 0

        async def call_with_image(self, **_kwargs):
            self.active += 1
            self.maximum = max(self.maximum, self.active)
            try:
                await asyncio.sleep(0.04)
                return {"receipts": [receipt()]}
            finally:
                self.active -= 1

    async def exercise() -> int:
        provider = ConcurrencyProvider()
        services = [
            ReceiptAnalysisService(OCRService(), cast(TextGenerationProvider, provider))
            for _ in range(7)
        ]
        await asyncio.gather(*(
            service.analyze(
                upload(),
                ReceiptAnalysisOptions(analysis_mode=ReceiptAnalysisMode.VISION_ONLY),
            )
            for service in services
        ))
        return provider.maximum

    assert asyncio.run(exercise()) == 3


def test_recovery_failure_preserves_other_candidates_and_releases_shared_permit():
    class PartiallyFailingProvider:
        def __init__(self):
            self.calls = 0

        async def call_with_image(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return {
                    "receipts": [
                        receipt(
                            title="Broken",
                            store_name="Broken",
                            purchase_total=None,
                            original_amount=None,
                            bounding_box=[0.02, 0.02, 0.48, 0.98],
                            status="NEEDS_REVIEW",
                        ),
                        receipt(
                            title="Healthy",
                            store_name="Healthy",
                            purchase_total="30.00",
                            original_amount=None,
                            bounding_box=[0.52, 0.02, 0.98, 0.98],
                        ),
                    ]
                }
            raise TimeoutError("synthetic crop timeout")

    encoded = io.BytesIO()
    Image.new("RGB", (1400, 1000), "white").save(encoded, format="JPEG")
    provider = PartiallyFailingProvider()
    response = asyncio.run(
        ReceiptAnalysisService(
            OCRService(), cast(TextGenerationProvider, provider)
        ).analyze(
            upload(encoded.getvalue()),
            ReceiptAnalysisOptions(analysis_mode=ReceiptAnalysisMode.VISION_ONLY),
        )
    )

    assert response.receipt_count == 2
    assert response.receipts[1].purchase_total == Decimal("30.00")
    assert "REGION_RECOVERY_TIMEOUT" in response.receipts[0].warnings

    follow_up, _, follow_up_provider = service()
    assert run(follow_up, ReceiptAnalysisMode.VISION_ONLY).receipt_count == 1
    follow_up_provider.call_with_image.assert_awaited_once()


def test_total_deadline_skips_new_recovery_when_remaining_budget_is_too_small(monkeypatch):
    class SlowInitialProvider:
        def __init__(self):
            self.calls = 0

        async def call_with_image(self, **_kwargs):
            self.calls += 1
            await asyncio.sleep(0.03)
            return {
                "receipts": [
                    receipt(
                        purchase_total=None,
                        original_amount=None,
                        bounding_box=[0.05, 0.05, 0.95, 0.95],
                        status="NEEDS_REVIEW",
                    )
                ]
            }

    monkeypatch.setattr(
        settings, "RECEIPT_ANALYSIS_TOTAL_TIMEOUT_SECONDS", 0.08, raising=False
    )
    monkeypatch.setattr(
        settings, "RECEIPT_VISION_RECOVERY_MIN_REMAINING_SECONDS", 0.06, raising=False
    )
    encoded = io.BytesIO()
    Image.new("RGB", (1200, 1200), "white").save(encoded, format="JPEG")
    provider = SlowInitialProvider()
    started = time.perf_counter()
    response = asyncio.run(
        ReceiptAnalysisService(
            OCRService(), cast(TextGenerationProvider, provider)
        ).analyze(
            upload(encoded.getvalue()),
            ReceiptAnalysisOptions(analysis_mode=ReceiptAnalysisMode.VISION_ONLY),
        )
    )

    assert time.perf_counter() - started < 0.08
    assert provider.calls == 1
    assert response.receipt_count == 1
    assert "REGION_RECOVERY_SKIPPED_TIME_BUDGET" in response.receipts[0].warnings


def test_recovery_plan_keeps_financial_completion_when_identity_also_needs_review():
    raw = receipt(
        title="LAWSON 船堀店",
        store_name="LAWSON",
        branch_name="船堀店",
        merchant_evidence="LAWSON",
        branch_evidence="船堀店",
        original_amount=None,
        purchase_total=None,
        detected_currency_code=None,
        transaction_date=None,
        bounding_box=[0.75, 0.1, 1.0, 0.95],
    )

    plan = ReceiptAnalysisService._recovery_plan(raw)

    assert plan is not None
    assert plan.identity_needed is True
    assert plan.financial_needed is True
    assert plan.kind == "INCOMPLETE_REGION_RECOVERY"


def test_same_receipt_financial_recovery_survives_identity_conflict():
    target = receipt(
        title="LAWSON 船橋店",
        store_name="LAWSON",
        branch_name="船橋店",
        merchant_evidence="LAWSON",
        branch_evidence="船橋店",
        purchase_total=None,
        original_amount=None,
        detected_currency_code=None,
        transaction_date=None,
    )
    recovery = receipt(
        title="LAWSON 船堀店",
        store_name="LAWSON",
        branch_name="船堀店",
        merchant_evidence="LAWSON",
        branch_evidence="船堀店",
        purchase_total="1680",
        original_amount="1680",
        detected_currency_code="JPY",
        transaction_date="2026-09-19",
    )

    changed = ReceiptAnalysisService._merge_missing_receipt_fields(
        target,
        recovery,
        same_receipt_region=True,
        merge_identity=True,
        merge_financial=True,
    )

    assert changed is True
    assert target["branch_name"] == "船橋店"
    assert target["identity_verification"] == "SOURCE_REGION_CONFLICT"
    assert target["purchase_total"] == "1680"
    assert target["original_amount"] == "1680"
    assert target["detected_currency_code"] == "JPY"
    assert target["transaction_date"] == "2026-09-19"
    assert target["financial_recovery_provenance"] == "SAME_RECEIPT_SOURCE_REGION"


def test_unproven_recovery_does_not_merge_finance_from_another_receipt():
    target = receipt(
        title="LAWSON",
        store_name="LAWSON",
        merchant_evidence="LAWSON",
        purchase_total=None,
        original_amount=None,
    )
    recovery = receipt(
        title="Other shop",
        store_name="Other shop",
        merchant_evidence="Other shop",
        purchase_total="9999",
        original_amount="9999",
    )

    changed = ReceiptAnalysisService._merge_missing_receipt_fields(
        target,
        recovery,
        same_receipt_region=False,
        merge_identity=True,
        merge_financial=True,
    )

    assert changed is True
    assert target["purchase_total"] is None
    assert target["original_amount"] is None
    assert target["identity_verification"] == "SOURCE_REGION_CONFLICT"


def test_model_evidence_string_agreement_is_not_image_verification():
    item = receipt(
        title="LAWSON 船堀店",
        store_name="LAWSON",
        branch_name="船堀店",
        merchant_evidence="LAWSON",
        branch_evidence="船堀店",
    )

    ReceiptAnalysisService._annotate_identity_verification(item)

    assert item["identity_verification"] == "MODEL_TEXT_SELF_CONSISTENT"


def test_visibly_truncated_merchant_identity_is_recovered_from_source_crop():
    target = receipt(
        title="LAWS... 船堀店",
        store_name="LAWS...",
        bounding_box=[0.75, 0.1, 1.0, 0.95],
    )
    recovery = receipt(
        title="LAWSON 船堀店", store_name="LAWSON", merchant_evidence="LAWSON"
    )
    assert ReceiptAnalysisService._needs_region_recovery(target) is True
    assert ReceiptAnalysisService._merge_missing_receipt_fields(
        target, recovery, same_receipt_region=True,
        merge_identity=True, merge_financial=False,
    ) is True
    assert target["title"] == "LAWSON 船堀店"
    assert target["store_name"] == "LAWSON"


def test_merchant_identity_is_replaced_by_literal_source_crop_without_confidence_ranking():
    target = receipt(
        title="Wrong merchant 船堀店",
        store_name="Wrong merchant",
        branch_name="船堀店",
        confidence=0.72,
        bounding_box=[0.75, 0.1, 1.0, 0.95],
    )
    recovery = receipt(
        title="LAWSON 船堀店",
        store_name="LAWSON",
        branch_name="船堀店",
        merchant_evidence="LAWSON",
        branch_evidence="船堀店",
        confidence=0.95,
    )

    assert ReceiptAnalysisService._needs_region_recovery(target) is True
    assert ReceiptAnalysisService._merge_missing_receipt_fields(
        target, recovery, same_receipt_region=True,
        merge_identity=True, merge_financial=False,
    ) is True
    assert target["title"] == "LAWSON 船堀店"
    assert target["store_name"] == "LAWSON"
    assert target["confidence"] == 0.72


def test_high_confidence_wrong_branch_is_replaced_by_literal_crop_evidence_only():
    target = receipt(
        title="どらっぐ ぱぱす 船橋店",
        store_name="どらっぐ ぱぱす",
        branch_name="船橋店",
        merchant_evidence="どらっぐ ぱぱす",
        branch_evidence="判読不能",
        confidence=0.99,
        bounding_box=[0.05, 0.05, 0.45, 0.95],
        purchase_total="7089",
        payment_breakdown=[
            {"payment_type": "LOYALTY_POINTS", "amount": "2069", "evidence": "ポイント支払", "duplicate_group": None},
            {"payment_type": "CREDIT_CARD", "amount": "5020", "evidence": "クレジット", "duplicate_group": None},
        ],
    )
    recovery = receipt(
        title="どらっぐ ぱぱす 船堀店",
        store_name="どらっぐ ぱぱす",
        branch_name="船堀店",
        merchant_evidence="どらっぐ ぱぱす",
        branch_evidence="船堀店",
        confidence=0.70,
        purchase_total="9999",
        payment_breakdown=[],
    )

    assert ReceiptAnalysisService._needs_region_recovery(target) is True
    assert ReceiptAnalysisService._merge_missing_receipt_fields(
        target, recovery, same_receipt_region=True,
        merge_identity=True, merge_financial=False,
    ) is True
    assert target["title"] == "どらっぐ ぱぱす 船堀店"
    assert target["branch_name"] == "船堀店"
    assert target["purchase_total"] == "7089"
    assert target["payment_breakdown"][1]["amount"] == "5020"


def test_validation_preserves_identity_evidence_and_source_region():
    raw = receipt(
        title="LAWSON 船堀店",
        store_name="LAWSON",
        branch_name="船堀店",
        merchant_evidence="LAWSON",
        branch_evidence="船堀店",
        identity_source_box=[0.8, 0.3, 1.0, 0.55],
        identity_verification="SOURCE_REGION_CONFIRMED",
    )

    item = validate(raw).receipts[0]

    assert item.merchant_evidence == "LAWSON"
    assert item.branch_evidence == "船堀店"
    assert item.identity_source_box == [0.8, 0.3, 1.0, 0.55]
    assert item.identity_verification == "SOURCE_REGION_CONFIRMED"


def test_source_crop_cannot_truncate_a_supported_complete_branch_literal():
    target = receipt(
        title="ROYAL KITCHENS! 千葉県千葉市美浜区中瀬2-1",
        store_name="ROYAL KITCHENS!",
        branch_name="千葉県千葉市美浜区中瀬2-1",
        merchant_evidence="ROYAL KITCHENS!",
        branch_evidence="千葉県千葉市美浜区中瀬2-1",
    )
    recovery = receipt(
        title="ROYAL KITCHENS! 千葉市美浜区中瀬2-1",
        store_name="ROYAL KITCHENS!",
        branch_name="千葉市美浜区中瀬2-1",
        merchant_evidence="ROYAL KITCHENS!",
        branch_evidence="千葉市美浜区中瀬2-1",
    )

    assert ReceiptAnalysisService._merge_missing_receipt_fields(
        target, recovery, same_receipt_region=True,
        merge_identity=True, merge_financial=False,
    ) is True
    assert target["title"] == "ROYAL KITCHENS! 千葉県千葉市美浜区中瀬2-1"
    assert target["branch_name"] == "千葉県千葉市美浜区中瀬2-1"
    assert target["branch_evidence"] == "千葉県千葉市美浜区中瀬2-1"
    assert target["identity_verification"] == "SOURCE_REGION_CONFLICT"
    assert target["status"] == "NEEDS_REVIEW"
    assert "IDENTITY_REGION_CONFLICT" in target["warnings"]


def test_two_source_supported_but_conflicting_branch_observations_do_not_rank_by_order():
    target = receipt(
        title="LAWSON 船堀店",
        store_name="LAWSON",
        branch_name="船堀店",
        merchant_evidence="LAWSON",
        branch_evidence="船堀店",
        confidence=0.51,
    )
    recovery = receipt(
        title="LAWSON 船橋店",
        store_name="LAWSON",
        branch_name="船橋店",
        merchant_evidence="LAWSON",
        branch_evidence="船橋店",
        confidence=0.99,
    )

    assert ReceiptAnalysisService._merge_missing_receipt_fields(
        target, recovery, same_receipt_region=True,
        merge_identity=True, merge_financial=False,
    ) is True
    assert target["title"] == "LAWSON 船堀店"
    assert target["branch_name"] == "船堀店"
    assert target["identity_verification"] == "SOURCE_REGION_CONFLICT"
    assert "IDENTITY_REGION_CONFLICT" in target["warnings"]


def test_identity_recovery_without_literal_evidence_is_rejected_and_financial_facts_stay_unchanged():
    target = receipt(
        title="LAWSON 船橋店", store_name="LAWSON", branch_name="船橋店",
        merchant_evidence="LAWSON", branch_evidence="船橋店", confidence=0.99,
        bounding_box=[0.05, 0.05, 0.45, 0.95], purchase_total="1680",
    )
    recovery = receipt(
        title="LAWSON 船堀店", store_name="LAWSON", branch_name="船堀店",
        merchant_evidence="different header", branch_evidence="different branch",
        confidence=1.0, purchase_total="1",
    )

    assert ReceiptAnalysisService._merge_missing_receipt_fields(
        target, recovery, same_receipt_region=True,
        merge_identity=True, merge_financial=False,
    ) is True
    assert target["title"] == "LAWSON 船橋店"
    assert target["purchase_total"] == "1680"
    assert target["identity_verification"] == "SOURCE_REGION_CONFLICT"
    assert target["status"] == "NEEDS_REVIEW"


def test_default_mode_is_vision_first(monkeypatch):
    monkeypatch.setattr(settings, "RECEIPT_ANALYSIS_MODE", "VISION_FIRST")
    svc, ocr, provider = service()
    run(svc)
    provider.call_with_image.assert_awaited_once()
    ocr.extract_document_from_upload.assert_not_awaited()


def test_invalid_configured_mode_fails_closed_to_vision_only(monkeypatch):
    monkeypatch.setattr(settings, "RECEIPT_ANALYSIS_MODE", "vision-with-typo")
    svc, ocr, provider = service(vision_error=RuntimeError("provider unavailable"))

    response = run(svc)

    provider.call_with_image.assert_awaited_once()
    ocr.extract_document_from_upload.assert_not_awaited()
    assert response.receipt_count == 0
    assert response.ocr_engine == "vision"
    assert "ANALYSIS_UNAVAILABLE" in response.warnings


def test_partial_low_confidence_vision_preserved_without_whole_image_fallback():
    svc, ocr, _ = service(
        vision_result={
            "receipts": [receipt(), {"status": "UNREADABLE"}, receipt(confidence=0.2)]
        }
    )
    response = run(svc, ReceiptAnalysisMode.VISION_FIRST)
    assert response.receipt_count == 3
    ocr.extract_document_from_upload.assert_not_awaited()


def test_vision_failure_falls_back_to_spatial_ocr_ai():
    svc, ocr, provider = service(vision_error=RuntimeError("private provider response"))
    response = run(svc, ReceiptAnalysisMode.VISION_FIRST, currency_code="JPY")
    assert response.receipts[0].detected_currency_code == "USD"
    assert "OCR_FALLBACK_USED" in response.warnings
    assert (
        ocr.extract_document_from_upload.call_args.kwargs["ocr_language"]
        == settings.OCR_LANGUAGE
    )
    prompt = json.loads(provider.call.call_args.kwargs["data"])
    assert len(prompt["spatial_lines"]) == 3
    assert prompt["spatial_lines"][0]["bounding_box"] is not None


def test_malformed_vision_falls_back_without_request_500():
    svc, _, _ = service(vision_result={"bad": []})
    assert run(svc, ReceiptAnalysisMode.VISION_FIRST).receipt_count == 1


def test_vision_only_failure_returns_diagnostic_without_ocr():
    svc, ocr, _ = service(vision_error=ValueError("SECRET"))
    response = run(svc, ReceiptAnalysisMode.VISION_ONLY)
    assert response.receipt_count == 0
    assert "VISION_UNAVAILABLE" in response.warnings
    ocr.extract_document_from_upload.assert_not_awaited()


def test_ocr_only_keeps_single_receipt_as_unconfirmed_draft():
    svc, _, provider = service()
    response = run(svc, ReceiptAnalysisMode.OCR_ONLY)
    assert response.receipts[0].original_amount == Decimal("12.34")
    assert response.receipts[0].status == ReceiptStatus.NEEDS_REVIEW
    provider.call.assert_not_awaited()
    provider.call_with_image.assert_not_awaited()


def test_ocr_only_never_selects_largest_total_from_multiple_receipts():
    doc = OCRDocument([OCRLine("TOTAL USD 12.34"), OCRLine("TOTAL JPY 1,280")])
    svc, _, _ = service(doc=doc)
    item = run(svc, ReceiptAnalysisMode.OCR_ONLY).receipts[0]
    assert item.original_amount is None
    assert item.detected_currency_code is None
    assert "OCR_MULTIPLE_OR_UNKNOWN_TOTALS" in item.warnings


def test_ocr_ai_without_spatial_context_cannot_confirm_amount():
    svc, _, _ = service(doc=OCRDocument([OCRLine("TOTAL USD 12.34")]))
    item = run(svc, ReceiptAnalysisMode.OCR_WITH_AI).receipts[0]
    assert item.original_amount is None
    assert "OCR_RECEIPT_BOUNDARIES_UNVERIFIED" in item.warnings


def test_paddle_v3_keeps_boxes_and_repeated_text():
    value = [
        {
            "rec_texts": ["TOTAL", "TOTAL"],
            "rec_scores": [0.9, 0.8],
            "rec_polys": [
                [[0, 0], [100, 0], [100, 20], [0, 20]],
                [[200, 0], [300, 0], [300, 20], [200, 20]],
            ],
        }
    ]
    doc = OCRDocument(OCRService()._collect_lines(value))
    assert doc.text == "TOTAL\nTOTAL"
    assert doc.lines[0].bounding_box != doc.lines[1].bounding_box
    assert doc.lines[1].confidence == 0.8


def test_paddle_v2_keeps_spatial_information():
    lines = OCRService()._collect_lines(
        [[[[0, 0], [10, 0], [10, 10], [0, 10]], ("Coffee", 0.91)]]
    )
    assert len(lines) == 1
    assert lines[0].bounding_box == [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]


def test_ocr_total_is_not_cash_change_subtotal_or_tax():
    candidates = extract_receipt_candidates(
        "Cafe\nSUBTOTAL USD 10.00\nTax 2.34\nTOTAL USD 12.34\nCash received 20.00\nChange 7.66"
    )
    assert candidates["original_amount"] == "12.34"


def test_production_logs_exclude_provider_payload(caplog):
    svc, _, _ = service(vision_error=ValueError("CARD 4111111111111111 phone SECRET"))
    run(svc, ReceiptAnalysisMode.VISION_FIRST)
    assert "4111111111111111" not in caplog.text
    assert "SECRET" not in caplog.text


def test_malformed_numeric_fields_cannot_discard_other_receipts():
    response = validate(
        receipt(),
        receipt(
            original_amount=10**5000,
            bounding_box=[0, 0, 10**5000, 1],
            currency_confidence="NaN",
        ),
    )
    assert response.receipt_count == 2
    assert response.receipts[0].status == ReceiptStatus.READY
    assert response.receipts[1].original_amount is None
    assert response.receipts[1].detected_currency_code is None


def test_named_month_in_foreign_language_retains_valid_vision_iso_date():
    item = validate(
        receipt(source_date="15 septembre 2026", transaction_date="2026-09-15")
    ).receipts[0]
    assert item.transaction_date is not None
    assert item.transaction_date.isoformat() == "2026-09-15"


def test_ambiguous_numeric_date_with_surrounding_words_remains_unknown():
    item = validate(
        receipt(source_date="Date: 03/04/2026", transaction_date="2026-03-04")
    ).receipts[0]
    assert item.transaction_date is None


def test_text_lengths_match_registration_contract():
    item = validate(
        receipt(title="t" * 200, store_name="s" * 200, memo="m" * 1000)
    ).receipts[0]
    assert item.title is not None and len(item.title) == 100
    assert item.store_name is not None and len(item.store_name) == 100
    assert item.memo is not None and len(item.memo) == 500


def test_receipt_limit_matches_backend_and_reports_truncation():
    response = validate(*(receipt(store_name=f"Store {index}") for index in range(31)))
    assert response.receipt_count == 30
    assert "RECEIPT_LIMIT_REACHED" in response.warnings
