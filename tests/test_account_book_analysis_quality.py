from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi import HTTPException

from app.ai.ports import TextGenerationProvider
from app.features.spending_analysis.service import SpendingAnalysisService
from app.schemas.spending_analysis import (
    SpendingAnalysisRequest,
    SpendingAnalysisResponse,
)


_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "account-book-analysis-eval.json"
_CORPUS = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
_CASES = {item["id"]: item for item in _CORPUS["cases"]}


def _request(case_id: str) -> dict[str, Any]:
    case = _CASES[case_id]
    if "inheritsRequestFrom" not in case:
        return deepcopy(case["request"])
    inherited = _request(case["inheritsRequestFrom"])
    inherited.update(deepcopy(case["requestOverrides"]))
    return inherited


def _expected_corpus_purchase_units(
    case_id: str, source: dict[str, Any]
) -> list[dict[str, Any]]:
    # 이 코퍼스의 일반 메모는 원문을 유지하고, 목적이 같은 두 헤드폰만 묶는다.
    # 운영 코드의 묶음 함수를 기대값 생성에 사용하지 않는다.
    units = [
        {
            "refId": item["refId"],
            "date": item["date"],
            "title": item["title"],
            "storeName": item["storeName"],
            "category": item["category"],
            "memo": item["memo"] or "",
            "amount": item["amount"],
            "transactionCount": 1,
            "dates": [item["date"]],
            "purposeKnown": bool(item["memo"]),
        }
        for item in source["transactions"]
    ]
    if case_id in {
        "repeated-headphones-purpose-unknown",
        "headphone-gift-comment-reanalysis",
    }:
        return [{
            **units[0],
            "amount": "53600",
            "transactionCount": 2,
            "dates": ["2026-10-05", "2026-10-07"],
        }]
    return units


class _RecordedProvider:
    """Only deterministic supplied outputs are returned; no real provider is used."""

    def __init__(self, *responses: dict[str, Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def call(self, *, type_name: str, data: str, schema: dict | None = None) -> Any:
        self.calls.append({"type_name": type_name, "data": json.loads(data), "schema": schema})
        response = deepcopy(self.responses[len(self.calls) - 1])
        for finding in response.get("findings", []):
            finding.setdefault("reviewDisposition", "REVIEW")
            finding.setdefault("reviewPattern", "REPEAT_PURCHASE" if len(finding.get("evidenceTransactionRefs", [])) > 1 else "ITEM_REVIEW")
        return response


@pytest.mark.asyncio
@pytest.mark.parametrize("case_id", list(_CASES))
async def test_quality_corpus_preserves_purchase_level_evidence_and_context(case_id: str) -> None:
    # 준비: 바람직한 예시 응답은 수동 품질 기준이며 실제 모델의 의미 판단을 입증하지 않는다.
    case = _CASES[case_id]
    request = SpendingAnalysisRequest.model_validate(_request(case_id))
    original_request = request.model_dump(mode="json")
    expected_response = SpendingAnalysisResponse.model_validate(case["response"])
    provider = _RecordedProvider(case["response"])
    service = SpendingAnalysisService(cast(TextGenerationProvider, provider))

    # 실행
    response = await service.analyze(request)

    # 검증: 기존 기록의 문구·근거는 유지하고 새 선택 필드의 기본값만 같은 스키마로 비교한다.
    assert response.model_dump(mode="json") == expected_response.model_dump(mode="json")
    assert response.policyVersion == "legacy-v2"
    assert response.savings is None
    assert len(provider.calls) == 1
    assert provider.calls[0]["type_name"] == "ACCOUNT_BOOK_SPENDING_ANALYSIS"

    # 검증: 구매 묶음이어도 상품·메모·모든 구매일·건수·정확한 금액을 보존한다.
    sent = provider.calls[0]["data"]
    metadata = original_request.keys() - {"transactions", "historicalTransactions"}
    assert {key: sent[key] for key in metadata} == {
        key: original_request[key] for key in metadata
    }
    assert sent["transactions"] == _expected_corpus_purchase_units(case_id, original_request)
    assert sent["historicalTransactions"] == original_request["historicalTransactions"] == []
    assert request.model_dump(mode="json") == original_request
    expected_limit = 8 if request.periodType == "YEARLY" else 4
    assert provider.calls[0]["schema"]["properties"]["findings"]["maxItems"] == expected_limit
    assert provider.calls[0]["schema"]["properties"]["policyVersion"]["const"] == "purchase-context-v5"
    assert all(isinstance(item["amount"], str) for item in sent["transactions"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "defect",
    ["unknown-reference", "repeated-reference", "empty-evidence", "category-only", "unobserved-item", "repeated-finding"],
)
async def test_ungrounded_recommendations_fail_without_automatic_generation_retry(defect: str) -> None:
    # 준비: 실제 거래 대신 없는 참조·모호한 카테고리·다른 물품을 제시한 응답을 만든다.
    request = SpendingAnalysisRequest.model_validate(_request("rent-largest-subscription-duplicate"))
    invalid = deepcopy(_CASES["rent-largest-subscription-duplicate"]["response"])
    finding = invalid["findings"][0]
    if defect == "unknown-reference":
        finding["evidenceTransactionRefs"] = ["tx-404"]
    elif defect == "repeated-reference":
        finding["evidenceTransactionRefs"] = ["tx-2", "tx-2"]
    elif defect == "empty-evidence":
        finding["evidenceTransactionRefs"] = []
    elif defect == "category-only":
        finding["title"] = "생활 카테고리 지출 검토"
        finding["reason"] = "생활 카테고리의 지출 비중이 높으므로 불필요한 소비를 줄이고 다음 달에는 전체 생활비 예산을 더 낮게 설정하는 편이 좋겠습니다."
    elif defect == "unobserved-item":
        finding["title"] = "고급 커피 머신 구매 검토"
        finding["reason"] = "고급 커피 머신을 여러 번 구매한 것으로 보이므로 반복 구매의 필요성을 점검하고 같은 물건을 다시 구매하기 전에 보유 물품을 확인하는 편이 좋겠습니다."
    else:
        invalid["findings"].append(deepcopy(finding))
    provider = _RecordedProvider(invalid)
    service = SpendingAnalysisService(cast(TextGenerationProvider, provider))

    # 실행
    with pytest.raises(HTTPException) as failure:
        await service.analyze(request)

    # 검증: 품질 실패를 새 유료 호출로 숨기거나 다른 거래로 자동 보정하지 않는다.
    assert failure.value.status_code == 502
    assert failure.value.detail == {"code": "SPENDING_ANALYSIS_INVALID_RESPONSE"}
    assert len(provider.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("invented_field", ["savingsAmount", "evidenceAmount", "evidenceTransactions"])
async def test_model_cannot_supply_unverified_money_or_transaction_objects(invented_field: str) -> None:
    # 준비: 근거 금액과 상세 거래는 BE가 원본에서 구성하며 모델 응답에는 허용하지 않는다.
    request = SpendingAnalysisRequest.model_validate(_request("rent-largest-subscription-duplicate"))
    invalid = deepcopy(_CASES["rent-largest-subscription-duplicate"]["response"])
    invalid["findings"][0][invented_field] = [] if invented_field == "evidenceTransactions" else "999999"
    provider = _RecordedProvider(invalid)
    service = SpendingAnalysisService(cast(TextGenerationProvider, provider))

    # 실행
    with pytest.raises(HTTPException) as failure:
        await service.analyze(request)

    # 검증
    assert failure.value.status_code == 502
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_comment_reanalysis_sends_new_purpose_with_the_same_purchases() -> None:
    # 준비: 두 선물이라는 설명이 추가된 시나리오와 검토용 예시 응답을 준비한다.
    before = SpendingAnalysisRequest.model_validate(_request("repeated-headphones-purpose-unknown"))
    after = SpendingAnalysisRequest.model_validate(_request("headphone-gift-comment-reanalysis"))
    provider = _RecordedProvider(
        _CASES["repeated-headphones-purpose-unknown"]["response"],
        _CASES["headphone-gift-comment-reanalysis"]["response"],
    )
    service = SpendingAnalysisService(cast(TextGenerationProvider, provider))

    # 실행: 명시적인 재분석 두 번만 수행한다.
    first = await service.analyze(before)
    second = await service.analyze(after)

    # 검증: 이는 새 문맥 전달·응답 보존 검사이며 실제 모델의 해석 품질 검사는 아니다.
    assert len(provider.calls) == 2
    assert provider.calls[0]["data"]["transactions"] == provider.calls[1]["data"]["transactions"]
    assert provider.calls[0]["data"]["comment"] is None
    assert provider.calls[1]["data"]["comment"] == after.comment
    assert first.findings and not second.findings
    assert first.summary != second.summary


@pytest.mark.asyncio
async def test_partial_current_year_retains_comparison_and_coverage_qualifications() -> None:
    # 준비: 진행 중인 연도·부분 표본을 전년도 전체 금액과 잘못 비교하지 않도록 문맥을 보존한다.
    case = _CASES["incomplete-year-not-comparable"]
    request = SpendingAnalysisRequest.model_validate(_request(case["id"]))
    provider = _RecordedProvider(case["response"])
    service = SpendingAnalysisService(cast(TextGenerationProvider, provider))

    # 실행
    await service.analyze(request)

    # 검증: 모델 입력에는 완전한 기간 합계와 표본의 금액을 구분하는 필드가 함께 간다.
    sent = provider.calls[0]["data"]
    assert sent["periodType"] == "YEARLY"
    assert sent["previousPeriod"]["isComparable"] is False
    assert sent["coverage"]["isPartial"] is True
    assert sent["coverage"]["analyzedTransactionCount"] == 2
    assert sent["coverage"]["totalTransactionCount"] == 3
    assert sent["totalExpense"] == "14000"
    assert [item["amount"] for item in sent["transactions"]] == ["6000", "7000"]


@pytest.mark.asyncio
async def test_large_decimal_expense_is_not_rounded_in_provider_payload() -> None:
    # 준비
    case = _CASES["large-decimal-money-retained"]
    request = SpendingAnalysisRequest.model_validate(_request(case["id"]))
    provider = _RecordedProvider(case["response"])
    service = SpendingAnalysisService(cast(TextGenerationProvider, provider))

    # 실행
    await service.analyze(request)

    # 검증: JavaScript 부동소수점 정밀도보다 큰 금액도 문자열 그대로 보존한다.
    sent = provider.calls[0]["data"]
    assert sent["currencyCode"] == "KWD"
    assert sent["decimalPlaces"] == 3
    assert sent["totalExpense"] == "18014398509481986.250"
    assert all(item["amount"] == "9007199254740993.125" for item in sent["transactions"])
