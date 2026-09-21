import asyncio
import io
import json
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import UploadFile
from starlette.datastructures import Headers

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
        "confidence": 0.92,
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


def test_single_receipt_decimal_json_contract():
    response = validate(receipt())
    assert response.receipt_count == 1
    assert response.receipts[0].status == ReceiptStatus.READY
    assert (
        json.loads(response.model_dump_json())["receipts"][0]["original_amount"]
        == "12.34"
    )


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


def test_category_must_be_exact_existing_candidate():
    item = validate(receipt(category_name="식비")).receipts[0]
    assert item.category_name is None
    assert "CATEGORY_NOT_IN_ACCOUNT_BOOK" in item.warnings
    assert (
        validate(receipt(category_name="食費"), categories=["食費"])
        .receipts[0]
        .category_name
        == "食費"
    )


def test_no_category_candidates_does_not_invent_category():
    response = validate_response(
        {"receipts": [receipt()]},
        ReceiptAnalysisOptions(),
        ocr_engine="vision",
        used_ai=True,
    )
    assert response.receipts[0].category_name is None


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


def test_default_mode_is_vision_first(monkeypatch):
    monkeypatch.setattr(settings, "RECEIPT_ANALYSIS_MODE", "VISION_FIRST")
    svc, ocr, provider = service()
    run(svc)
    provider.call_with_image.assert_awaited_once()
    ocr.extract_document_from_upload.assert_not_awaited()


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
