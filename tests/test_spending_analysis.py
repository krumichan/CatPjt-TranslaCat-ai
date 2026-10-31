import asyncio
import json
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.features.spending_analysis.service import TASK_NAME, SpendingAnalysisService
from app.schemas.spending_analysis import SpendingAnalysisRequest


def request_data():
    return {
        "periodType": "MONTHLY", "year": 2026, "month": 10, "locale": "ko",
        "currencyCode": "JPY", "decimalPlaces": 0,
        "periodStart": "2026-10-01", "periodEndExclusive": "2026-11-01",
        "totalExpense": "12000", "transactionCount": 2,
        "coverage": {"analyzedTransactionCount": 2, "totalTransactionCount": 2, "isPartial": False, "reason": None},
        "previousPeriod": {"periodStart": "2026-09-01", "periodEndExclusive": "2026-10-01", "totalExpense": "4000", "transactionCount": 1, "isComparable": False},
        "comment": None,
        "transactions": [
            {"refId": "tx-1", "date": "2026-10-03", "title": "무선 이어폰", "storeName": "Amazon", "category": "쇼핑", "memo": "이어폰 구매", "amount": "6000", "sourceType": None},
            {"refId": "tx-2", "date": "2026-10-05", "title": "무선 이어폰", "storeName": "전자매장", "category": "쇼핑", "memo": "이어폰 구매", "amount": "6000", "sourceType": None},
        ],
    }


def response_data():
    return {
        "summary": "짧은 간격으로 같은 품목이 기록되어 구매 목적을 확인할 만한 거래가 있습니다.",
        "findings": [{
            "title": "짧은 간격의 무선 이어폰 구매 확인",
            "reason": "무선 이어폰 구매가 짧은 간격으로 두 번 기록되어 있습니다. 같은 사람이 사용할 물건이었다면 용도가 겹칠 수 있지만 선물이나 교체 구매인지는 확인이 필요합니다.",
            "action": "두 거래의 구매 목적과 반품 가능 여부를 확인한 뒤 다음 구매 전에 보유 물품을 살펴보세요.",
            "contextQuestion": "두 번째 이어폰은 선물이나 고장 교체용이었나요?", "uncertainty": "거래 기록만으로 구매 목적을 확정할 수 없습니다.",
            "evidenceTransactionRefs": ["tx-1", "tx-2"],
            "reviewDisposition": "REVIEW", "reviewPattern": "REPEAT_PURCHASE",
        }],
    }


@pytest.mark.asyncio
async def test_one_structured_call_preserves_exact_money_and_comment():
    # 준비
    data = request_data()
    data["comment"] = "두 번째 이어폰은 어머니 선물입니다."
    data["transactions"][0]["amount"] = "9007199254740993.125"
    data["totalExpense"] = "9007199254746993.125"
    provider = AsyncMock()
    provider.call.return_value = response_data()
    service = SpendingAnalysisService(provider)

    # 실행
    result = await service.analyze(SpendingAnalysisRequest.model_validate(data))

    # 검증: 실제 AI 호출과 금액 변경 없이 재분석 문맥을 전달한다.
    provider.call.assert_awaited_once()
    call = provider.call.call_args.kwargs
    assert call["type_name"] == TASK_NAME
    assert json.loads(call["data"])["comment"] == data["comment"]
    assert json.loads(call["data"])["transactions"][0]["amount"] == "9007199254740993.125"
    assert call["schema"]["additionalProperties"] is False
    assert result.findings[0].evidenceTransactionRefs == ["tx-1", "tx-2"]


@pytest.mark.parametrize("change", [
    {"month": None}, {"month": 13}, {"locale": "unsupported"},
    {"periodEndExclusive": "2026-12-01"}, {"totalExpense": "11999"},
    {"transactionCount": 3}, {"comment": "x" * 2001},
])
def test_inconsistent_or_unbounded_input_fails_before_ai(change):
    # 준비
    data = {**request_data(), **change}

    # 실행 / 검증
    with pytest.raises(ValidationError):
        SpendingAnalysisRequest.model_validate(data)


def test_yearly_leap_period_and_explicit_sample_coverage():
    # 준비
    data = request_data()
    data.update(periodType="YEARLY", year=2024, month=None, periodStart="2024-01-01", periodEndExclusive="2025-01-01", totalExpense="20000", transactionCount=5, previousPeriod=None)
    data["transactions"][0]["date"] = "2024-02-29"
    data["transactions"][1]["date"] = "2024-12-31"
    data["coverage"].update(totalTransactionCount=5, isPartial=True, reason="연간 거래 중 월별 표본만 포함했습니다.")

    # 실행
    result = SpendingAnalysisRequest.model_validate(data)

    # 검증
    assert result.coverage.isPartial
    assert result.transactions[0].date.isoformat() == "2024-02-29"
    data["coverage"]["reason"] = None
    with pytest.raises(ValidationError):
        SpendingAnalysisRequest.model_validate(data)


@pytest.mark.asyncio
async def test_two_character_original_purchase_names_are_valid_anchors():
    # 준비: 한일 거래명은 두 글자도 자연스러운 상품·서비스 이름이다.
    data = request_data()
    for item in data["transactions"]:
        item.update(title="택시", storeName="우버", category="교통")
    output = response_data()
    output["findings"][0].update(
        title="택시 이용 목적 확인",
        reason="택시 거래가 비슷한 시기에 두 번 기록되어 이용 목적을 살펴볼 수 있습니다. 필수 이동이었는지 확인하기 전에는 불필요한 소비로 단정하기 어렵습니다.",
        action="당시 이동의 필요성을 확인하고 같은 상황에서 대체 교통수단이 가능한지 살펴보세요.",
    )
    provider = AsyncMock()
    provider.call.return_value = output

    # 실행
    result = await SpendingAnalysisService(provider).analyze(SpendingAnalysisRequest.model_validate(data))

    # 검증
    assert result.findings[0].title == "택시 이용 목적 확인"


@pytest.mark.parametrize("kind", ["unknown_ref", "duplicate_ref", "duplicate_finding", "generic", "invented_field"])
@pytest.mark.asyncio
async def test_invalid_model_evidence_is_rejected_without_paid_repair(kind):
    # 준비
    output = response_data()
    if kind == "unknown_ref":
        output["findings"][0]["evidenceTransactionRefs"] = ["tx-999"]
    elif kind == "duplicate_ref":
        output["findings"][0]["evidenceTransactionRefs"] = ["tx-1", "tx-1"]
    elif kind == "duplicate_finding":
        output["findings"].append(deepcopy(output["findings"][0]))
    elif kind == "generic":
        output["findings"][0].update(title="쇼핑을 줄여주세요", reason="쇼핑 항목의 금액이 높으니 앞으로 지출을 조금 줄이면 좋겠습니다. 카테고리별로 예산을 정해서 소비해 보세요.")
    else:
        output["findings"][0]["estimatedSavings"] = "100000"
    provider = AsyncMock()
    provider.call.return_value = output

    # 실행
    with pytest.raises(HTTPException) as error:
        await SpendingAnalysisService(provider).analyze(SpendingAnalysisRequest.model_validate(request_data()))

    # 검증
    assert error.value.status_code == 502
    assert error.value.detail == {"code": "SPENDING_ANALYSIS_INVALID_RESPONSE"}
    provider.call.assert_awaited_once()


@pytest.mark.asyncio
async def test_empty_period_uses_no_provider_and_no_supported_concern_is_valid():
    # 준비
    data = request_data()
    data.update(totalExpense="0", transactionCount=0, transactions=[])
    data["coverage"].update(analyzedTransactionCount=0, totalTransactionCount=0)
    provider = AsyncMock()

    # 실행
    result = await SpendingAnalysisService(provider).analyze(SpendingAnalysisRequest.model_validate(data))

    # 검증
    assert result.findings == []
    provider.call.assert_not_awaited()
    provider.call.return_value = {"summary": "현재 제공된 거래만으로 불필요한 지출을 확정할 근거는 없습니다.", "findings": []}
    assert (await SpendingAnalysisService(provider).analyze(SpendingAnalysisRequest.model_validate(request_data()))).findings == []


@pytest.mark.asyncio
async def test_timeout_and_provider_failure_do_not_retry_or_expose_content():
    # 준비
    provider = AsyncMock()
    provider.call.side_effect = RuntimeError("private comment and provider body")
    request = SpendingAnalysisRequest.model_validate(request_data())

    # 실행 / 검증: 실패 원문은 외부 오류에 포함하지 않는다.
    with pytest.raises(HTTPException) as error:
        await SpendingAnalysisService(provider).analyze(request)
    assert error.value.status_code == 503
    assert "private" not in str(error.value.detail)
    provider.call.assert_awaited_once()

    # 준비: 오래 걸리는 단일 호출을 deadline으로 취소한다.
    async def blocking_call(**_):
        await asyncio.Event().wait()

    provider = AsyncMock()
    provider.call.side_effect = blocking_call

    # 실행 / 검증
    with pytest.raises(HTTPException) as timeout:
        await SpendingAnalysisService(provider, timeout_seconds=0.02).analyze(request)
    assert timeout.value.status_code == 504
    provider.call.assert_awaited_once()


@pytest.mark.asyncio
async def test_deadline_bounds_queue_wait_before_another_provider_call():
    # 준비
    entered = asyncio.Event()
    release = asyncio.Event()

    async def blocking_call(**_):
        entered.set()
        await release.wait()
        return response_data()

    provider = AsyncMock()
    provider.call.side_effect = blocking_call
    service = SpendingAnalysisService(provider, timeout_seconds=2, max_concurrency=1)
    request = SpendingAnalysisRequest.model_validate(request_data())

    # 실행: 먼저 시작한 긴 호출 동안 짧은 deadline의 대기 요청을 취소한다.
    first = asyncio.create_task(service.analyze(request))
    await asyncio.wait_for(entered.wait(), timeout=1)
    service.timeout_seconds = 0.02
    with pytest.raises(HTTPException) as queued:
        await service.analyze(request)

    # 검증: 두 번째 요청이 대기 중 timeout이므로 Provider 호출을 늘리지 않는다.
    assert queued.value.status_code == 504
    provider.call.assert_awaited_once()
    release.set()
    assert (await first).findings


@pytest.mark.asyncio
async def test_explained_gift_and_unsupported_essential_costs_are_not_review_cards():
    # 준비: 모델이 확인 메모를 채우더라도 설명된 구매·근거 없는 필수 지출을 제외한다.
    data = request_data()
    data["comment"] = "이어폰 두 번째는 어머니 선물이며 진료는 꼭 필요합니다."
    data["transactions"].extend([
        {"refId": "tx-3", "date": "2026-10-07", "title": "처방약 및 정기 진료", "storeName": "동네 병원", "category": "의료", "memo": "정기 치료", "amount": "500", "sourceType": None},
        {"refId": "tx-4", "date": "2026-10-01", "title": "월세", "storeName": "임대인", "category": "주거", "memo": None, "amount": "100000", "sourceType": "FIXED_COST"},
    ])
    data.update(transactionCount=4, totalExpense="112500")
    data["coverage"].update(analyzedTransactionCount=4, totalTransactionCount=4)
    output = response_data()
    output["findings"][0]["reviewDisposition"] = "EXPLAINED_BY_COMMENT"
    for ref, title in (("tx-3", "처방약 및 정기 진료"), ("tx-4", "월세")):
        output["findings"].append({
            "title": f"{title} 확인", "reason": f"{title} 거래가 기록되어 있습니다. 실제 중복 청구나 잘못된 금액이라는 근거는 없지만 일반적인 비용 확인을 제안하고 있습니다.",
            "action": "일반적인 비용 확인을 해보세요.", "contextQuestion": None, "uncertainty": None,
            "evidenceTransactionRefs": [ref], "reviewDisposition": "REVIEW", "reviewPattern": "ITEM_REVIEW",
        })
    provider = AsyncMock()
    provider.call.return_value = output

    # 실행
    result = await SpendingAnalysisService(provider).analyze(SpendingAnalysisRequest.model_validate(data))

    # 검증: 제외한 이어폰을 요약에서 다시 절감 대상으로 부르지 않는다.
    assert result.findings == []
    assert "무선 이어폰" not in result.summary
    assert "필수 지출" in result.summary
    provider.call.assert_awaited_once()


@pytest.mark.asyncio
async def test_recurring_discretionary_subscriptions_can_still_be_reviewed():
    # 준비: FIXED_COST 자체를 필수 지출과 동일시하지 않는다.
    data = request_data()
    for item in data["transactions"]:
        item.update(title="영상 구독", category="생활", memo="개인 스트리밍 계정", sourceType="FIXED_COST")
    output = response_data()
    output["findings"][0].update(title="영상 구독 중복 확인", reason="영상 구독이 짧은 기간에 두 번 기록되어 같은 계정의 결제인지 확인할 수 있습니다. 가족용 등 별도 용도가 있다면 꼭 불필요하다고 단정할 수는 없습니다.")
    provider = AsyncMock()
    provider.call.return_value = output

    # 실행 / 검증
    result = await SpendingAnalysisService(provider).analyze(SpendingAnalysisRequest.model_validate(data))
    assert len(result.findings) == 1
    assert "reviewDisposition" not in result.model_dump()["findings"][0]


@pytest.mark.parametrize("title,category,memo", [
    ("이어폰", "쇼핑", "병원에서 근무할 때 쓰려고 구매"),
    ("Car rent", "Travel", "Rental car for a holiday trip"),
])
@pytest.mark.asyncio
async def test_incidental_medical_location_and_car_rental_are_not_protected_costs(title, category, memo):
    # 준비: 명확한 치료비·주거비가 아닌 구매를 단어 하나로 제외하지 않는다.
    data = request_data()
    for item in data["transactions"]:
        item.update(title=title, category=category, memo=memo)
    output = response_data()
    output["findings"][0].update(title=f"{title} 구매 목적 확인", reason=f"{title} 거래가 짧은 간격으로 두 번 기록되어 있습니다. 실제 같은 용도의 중복 구매인지 각각 별도 용도가 있었는지 확인한 후 다음 소비 계획을 정할 수 있습니다.")
    provider = AsyncMock()
    provider.call.return_value = output

    # 실행 / 검증
    result = await SpendingAnalysisService(provider).analyze(SpendingAnalysisRequest.model_validate(data))
    assert len(result.findings) == 1


@pytest.mark.parametrize("token", ["refId: tx-1", "storeName", "coverage.isPartial=true"])
@pytest.mark.asyncio
async def test_internal_metadata_cannot_leak_into_visible_analysis(token):
    # 준비
    output = response_data()
    output["findings"][0]["reason"] += f" ({token})"
    provider = AsyncMock()
    provider.call.return_value = output

    # 실행 / 검증
    with pytest.raises(HTTPException) as error:
        await SpendingAnalysisService(provider).analyze(SpendingAnalysisRequest.model_validate(request_data()))
    assert error.value.status_code == 502
    provider.call.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_duplicate_charge_claim_requires_same_date_title_and_amount():
    # 준비
    data = request_data()
    for item in data["transactions"]:
        item.update(title="定期診療", category="医療", date="2026-10-03")
    output = response_data()
    output["findings"][0].update(
        title="定期診療 중복 청구 확인", reason="定期診療 결제가 같은 날짜와 같은 금액으로 두 번 기록되어 있습니다. 두 기록이 실제 같은 청구의 중복인지 영수증으로 대조할 수 있지만 치료 필요성은 판단하지 않습니다.",
        action="두 결제 영수증을 대조하고 중복 청구라면 결제처에 확인하세요.", reviewPattern="DUPLICATE_CHARGE",
    )
    provider = AsyncMock()
    provider.call.return_value = output

    # 실행 / 검증
    assert len((await SpendingAnalysisService(provider).analyze(SpendingAnalysisRequest.model_validate(data))).findings) == 1
    data["transactions"][1]["date"] = "2026-10-04"
    with pytest.raises(HTTPException):
        await SpendingAnalysisService(provider).analyze(SpendingAnalysisRequest.model_validate(data))
