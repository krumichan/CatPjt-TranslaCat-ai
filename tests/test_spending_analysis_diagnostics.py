import logging
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

import app.features.spending_analysis.service as module
from app.schemas.spending_analysis import SpendingAnalysisRequest
from tests.test_spending_analysis import request_data
from tests.test_spending_coaching import coaching


PRIVATE = "synthetic-private-comment-and-token"


@pytest.fixture(autouse=True)
def capture_service_logs(caplog):
    # 앱의 propagate=False 설정에도 실제 서비스 logger에서 같은 로그를 검사한다.
    module.logger.addHandler(caplog.handler)
    caplog.set_level(logging.INFO, logger=module.__name__)
    try:
        yield
    finally:
        module.logger.removeHandler(caplog.handler)


def request() -> SpendingAnalysisRequest:
    data = request_data()
    data["comment"] = PRIVATE
    data["transactions"][0]["memo"] = PRIVATE
    return SpendingAnalysisRequest.model_validate(data)


def assert_safe_logs(caplog, stage: str, code: str) -> None:
    assert f"stage={stage}" in caplog.text
    assert f"code={code}" in caplog.text
    assert "elapsed_ms=" in caplog.text
    assert PRIVATE not in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


@pytest.mark.asyncio
@pytest.mark.parametrize("exception,status,code", [
    (RuntimeError(PRIVATE), 503, "UNAVAILABLE"),
    (TimeoutError(PRIVATE), 504, "TIMEOUT"),
    (HTTPException(status_code=429, detail=PRIVATE), 429, "UPSTREAM_HTTP"),
])
async def test_generation_failures_keep_public_status_and_redact_exception_messages(
    caplog, exception, status, code
):
    # 준비: 실제 공급자 대신 민감한 메시지가 포함된 예외를 주입한다.
    provider = AsyncMock()
    provider.call.side_effect = exception
    caplog.set_level(logging.INFO, logger=module.__name__)

    # 실행
    with pytest.raises(HTTPException) as caught:
        await module.SpendingAnalysisService(provider).analyze(request())

    # 검증: 원래 응답과 단일 호출을 유지하고 메시지·입력·traceback은 기록하지 않는다.
    assert caught.value.status_code == status
    provider.call.assert_awaited_once()
    assert_safe_logs(caplog, "GENERATE", code)
    assert "schema_enum_values=" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("review", [False, True])
async def test_draft_and_review_schema_failures_log_only_fixed_validation_types(caplog, review):
    # 준비: 공급자 원문에는 비밀 표식을, summary에는 잘못된 타입을 넣는다.
    invalid = {"policyVersion": "purchase-context-v4", "summary": {"private": PRIVATE}, "findings": []}
    provider = AsyncMock()
    provider.call.side_effect = [coaching(), invalid] if review else [invalid]

    # 실행
    with pytest.raises(HTTPException) as caught:
        await module.SpendingAnalysisService(provider).analyze(request())

    # 검증
    assert caught.value.status_code == 502
    assert provider.call.await_count == (2 if review else 1)
    assert_safe_logs(caplog, "VALIDATE_REVIEW" if review else "VALIDATE_DRAFT", "SCHEMA_VALIDATION")
    assert "string_type" in caplog.text


@pytest.mark.asyncio
async def test_semantic_reviewer_failure_is_distinguished_without_another_generation(caplog):
    # 준비
    provider = AsyncMock()
    provider.call.side_effect = [coaching(), RuntimeError(PRIVATE)]

    # 실행
    with pytest.raises(HTTPException) as caught:
        await module.SpendingAnalysisService(provider).analyze(request())

    # 검증
    assert caught.value.status_code == 503
    assert provider.call.await_count == 2
    assert_safe_logs(caplog, "REVIEW", "UNAVAILABLE")


@pytest.mark.asyncio
async def test_grounding_rejection_logs_fixed_reason_and_preserves_502(caplog):
    # 준비
    invalid = deepcopy(coaching())
    invalid["findings"][0]["evidenceRefs"] = ["tx-999"]
    provider = AsyncMock()
    provider.call.return_value = invalid

    # 실행
    with pytest.raises(HTTPException) as caught:
        await module.SpendingAnalysisService(provider, semantic_review=False).analyze(request())

    # 검증
    assert caught.value.status_code == 502
    provider.call.assert_awaited_once()
    assert_safe_logs(caplog, "FINALIZE", "UNKNOWN_PURCHASE_REFERENCE")


@pytest.mark.asyncio
async def test_context_preparation_failure_keeps_original_exception_without_provider_call(caplog, monkeypatch):
    # 준비: 기존 try 바깥 예외를 같은 타입으로 유지한다.
    provider = AsyncMock()
    error = ValueError(PRIVATE)

    def fail(*_args):
        raise error

    monkeypatch.setattr(module, "prepare_coaching_payload", fail)

    # 실행
    with pytest.raises(ValueError) as caught:
        await module.SpendingAnalysisService(provider).analyze(request())

    # 검증
    assert caught.value is error
    provider.call.assert_not_awaited()
    assert_safe_logs(caplog, "PREPARE_CONTEXT", "CONTEXT_PREPARATION")
