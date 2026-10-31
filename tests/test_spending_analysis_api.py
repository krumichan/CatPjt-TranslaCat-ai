import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api.v1.spending_analysis import get_spending_analysis_service
from app.core.config import settings
from app.main import app
from app.schemas.spending_analysis import SpendingAnalysisResponse

_CASE = json.loads((Path(__file__).parent / "fixtures" / "account-book-analysis-eval.json").read_text(encoding="utf-8"))["cases"][0]


def request_data():
    return deepcopy(_CASE["request"])


def response_data():
    return deepcopy(_CASE["response"])


@pytest.fixture
def api_client(monkeypatch):
    # 준비: 애플리케이션 공통 API key 경계를 실제로 통과하되 Provider는 가짜로 고정한다.
    monkeypatch.setattr(settings, "SERVER_API_KEY", "synthetic-spending-analysis-api-key")
    service = AsyncMock()
    service.analyze.return_value = SpendingAnalysisResponse.model_validate(response_data())
    app.dependency_overrides[get_spending_analysis_service] = lambda: service
    client = TestClient(app)
    try:
        yield client, service
    finally:
        client.close()
        app.dependency_overrides.pop(get_spending_analysis_service, None)


def test_new_route_rejects_unauthenticated_internal_call(api_client):
    # 준비
    client, service = api_client

    # 실행
    response = client.post("/api/v1/account-book/spending-analysis", json=request_data())

    # 검증
    assert response.status_code == 401
    service.analyze.assert_not_awaited()


def test_new_route_forwards_authorized_calendar_snapshot(api_client):
    # 준비
    client, service = api_client
    data = request_data()
    data["comment"] = "구독 중 한 건은 가족용입니다."

    # 실행
    response = client.post(
        "/api/v1/account-book/spending-analysis", json=data,
        headers={"X-API-KEY": "synthetic-spending-analysis-api-key"},
    )

    # 검증: 새 라우터 등록·인증·schema alias가 BE 계약과 일치한다.
    assert response.status_code == 200
    assert response.json()["findings"][0]["evidenceTransactionRefs"] == ["tx-2", "tx-3"]
    service.analyze.assert_awaited_once()
    assert service.analyze.call_args.args[0].comment == data["comment"]
    assert service.analyze.call_args.args[0].transactions[0].amount == "100000"


@pytest.mark.parametrize("change", [
    {"year": 1999}, {"year": 9999}, {"comment": "x" * 2001}, {"periodType": "WEEKLY"},
])
def test_new_route_invalid_input_never_reaches_paid_service(api_client, change):
    # 준비
    client, service = api_client

    # 실행
    response = client.post(
        "/api/v1/account-book/spending-analysis", json={**request_data(), **change},
        headers={"X-API-KEY": "synthetic-spending-analysis-api-key"},
    )

    # 검증
    assert response.status_code == 422
    service.analyze.assert_not_awaited()


def test_startup_prepares_local_sdk_without_a_generation_call(monkeypatch):
    # 준비: 실제 초기화가 준비 객체 접근까지만 진행되는지 확인한다.
    import app.api.dependencies as dependencies

    create = AsyncMock()
    provider = SimpleNamespace(ready=True, client=SimpleNamespace(responses=SimpleNamespace(create=create)))
    monkeypatch.setattr(dependencies, "_model_execution_provider", provider)

    # 실행
    dependencies.prepare_spending_analysis_provider()

    # 검증
    create.assert_not_called()
    provider.ready = False
    provider.client = None
    dependencies.prepare_spending_analysis_provider()
