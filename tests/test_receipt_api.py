from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.api.dependencies import get_receipt_analysis_service
from app.api.v1.receipt import _parse_options, router
from app.features.receipt.validation import validate_response
from app.schemas.receipt import ReceiptAnalysisOptions


def test_multipart_endpoint_returns_batch_and_decimal_strings():
    options = ReceiptAnalysisOptions(category_candidates=["Food"])
    response = validate_response(
        {
            "receipts": [
                {
                    "title": "Cafe",
                    "original_amount": "12.34",
                    "detected_currency_code": "USD",
                    "transaction_date": "2026-09-15",
                    "confidence": 0.95,
                },
                {"status": "UNREADABLE"},
            ]
        },
        options,
        ocr_engine="vision",
        used_ai=True,
    )
    fake_service = AsyncMock()
    fake_service.analyze.return_value = response
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_receipt_analysis_service] = lambda: fake_service
    with TestClient(app) as client:
        result = client.post(
            "/account-book/receipts/analyze",
            files={"file": ("receipt.jpg", b"test", "image/jpeg")},
            data={
                "options": '{"category_candidates":["Food"],"analysis_mode":"VISION_FIRST"}',
                "trace_id": "test-trace-123",
            },
        )
    assert result.status_code == 200
    assert result.json()["receipt_count"] == 2
    assert result.json()["receipts"][0]["original_amount"] == "12.34"
    assert result.json()["receipts"][1]["status"] == "UNREADABLE"
    assert fake_service.analyze.call_args.kwargs["options"].category_candidates == [
        "Food"
    ]
    assert fake_service.analyze.call_args.kwargs["trace_id"] == "test-trace-123"


@pytest.mark.parametrize(
    "options",
    ["not json", "[]", '{"analysis_mode":"UNKNOWN"}', '{"category_candidates":[1]}'],
)
def test_invalid_options_are_client_error(options):
    with pytest.raises(HTTPException) as error:
        _parse_options(options)
    assert error.value.status_code == 400


def test_old_target_currency_option_is_accepted_without_influencing_source():
    assert _parse_options('{"currency_code":"JPY"}').category_candidates == []


def test_runtime_identity_endpoint_returns_process_start_snapshot_without_provider_call():
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        first = client.get("/account-book/receipts/runtime-identity")
        second = client.get("/account-book/receipts/runtime-identity")
    assert first.status_code == 200
    assert first.json()["run_id"] == second.json()["run_id"]
    assert first.json()["source_fingerprint"] == second.json()["source_fingerprint"]
    assert first.json()["started_at"] == second.json()["started_at"]
    assert first.json()["process_id"] > 0
    assert first.json()["provider_call_count"] == second.json()["provider_call_count"]


def test_runtime_identity_uses_the_process_start_fingerprint(monkeypatch):
    from app.features.receipt import runtime_identity

    before = runtime_identity.get_receipt_runtime_identity()
    monkeypatch.setattr(runtime_identity, "receipt_source_fingerprint", lambda *_: "f" * 64)
    after = runtime_identity.get_receipt_runtime_identity()
    assert after.source_fingerprint == before.source_fingerprint
    assert after.started_at == before.started_at


def test_receipt_schema_survives_both_provider_adapters():
    from app.ai.providers.gemini.configs import sanitize_gemini_response_schema
    from app.ai.providers.openai.schema import normalize_openai_response_schema
    from app.features.receipt.service import _RECEIPT_ANALYSIS_SCHEMA

    gemini_schema = sanitize_gemini_response_schema(_RECEIPT_ANALYSIS_SCHEMA)
    assert gemini_schema is not None
    item = gemini_schema["properties"]["receipts"]["items"]
    assert "title" in item["properties"]
    assert set(item["required"]) == set(item["properties"])
    assert item["properties"]["original_amount"]["type"] == "STRING"
    openai_schema = normalize_openai_response_schema(_RECEIPT_ANALYSIS_SCHEMA)
    assert openai_schema is not None
    assert openai_schema["properties"]["receipts"]["items"]["properties"][
        "original_amount"
    ]["type"] == ["string", "null"]
