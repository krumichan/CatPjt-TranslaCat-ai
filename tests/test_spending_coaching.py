from copy import deepcopy
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.features.spending_analysis.coaching import scenario, finalize_coaching, expand_draft, prepare_coaching_payload, estimate_eligible
from app.features.spending_analysis.service import SpendingAnalysisService
from app.schemas.spending_analysis import CoachingDraft, SpendingAnalysisRequest
from tests.test_spending_analysis import request_data


def coaching():
    return {"policyVersion": "recurring-choice-v6", "summary": "이번 기간에는 역할이 가까운 물건의 구매가 이어졌습니다. 기본 사용 목적은 유지하되 같은 역할의 추가 선택을 조정하면 지출을 줄일 여지가 있습니다.",
        "findings": [{"title": "무선 이어폰의 추가 선택 조정", "observation": "무선 이어폰 구매가 서로 다른 날 기록되어 같은 역할에 비용이 배정된 사실을 볼 수 있습니다.",
            "opinion": "다른 사람의 선물이나 고장 교체가 아니라면 같은 용도의 추가 선택을 줄일 여지가 있습니다.",
            "action": "다음 구매에서는 먼저 현재 사용할 제품을 정하고, 같은 상황에 쓸 다른 제품은 실제 사용 후 선택합니다.",
            "assumption": "두 물건이 같은 사용자의 같은 용도일 때 일부 추가 선택을 줄이는 가정입니다.",
            "evidenceRefs": ["tx-1", "tx-2"], "targetRefs": ["tx-1", "tx-2"], "historicalRefs": [], "estimateKind": "SKIP_EXTRA_ITEM"}]}


@pytest.mark.asyncio
async def test_semantic_review_is_one_bounded_edit_and_uses_comment():
    # 준비: 초안의 불명확한 의견을 두 번째 단계가 수정한다.
    request = request_data()
    request["comment"] = "첫 제품은 고장 교체용입니다."
    revised = coaching()
    revised["findings"] = []
    provider = AsyncMock()
    provider.call.side_effect = [coaching(), revised]

    # 실행
    result = await SpendingAnalysisService(provider).analyze(SpendingAnalysisRequest.model_validate(request))

    # 검증: 자동 재생성을 반복하지 않고 수정된 판단만 공개한다.
    assert provider.call.await_count == 2
    assert provider.call.call_args_list[1].kwargs["type_name"] == "ACCOUNT_BOOK_SPENDING_REVIEW"
    assert request["comment"] in provider.call.call_args_list[1].kwargs["data"]
    assert result.findings == []
    assert result.savings is None


def test_retrospective_scenario_uses_cited_amount_and_replacement_cost():
    # 준비 / 실행
    result = scenario(Decimal("10000"), "DRINK_REPLACEMENT", 0, "대체 음료 비용을 포함하는 가정")

    # 검증: 절약액은 생략한 구매 전액이 아니며 JPY 정수로 계산한다.
    assert result is not None
    assert result.lower == "200"
    assert result.upper == "1000"
    assert result.grossReduction == "1500"
    assert result.replacementCost == "975"
    assert result.estimate == "525"


def test_estimate_and_historical_evidence_are_separate():
    # 준비
    request = request_data()
    old = deepcopy(request["transactions"][0])
    old.update(refId="tx-100001", date="2026-09-03", amount="99000")
    request["historicalTransactions"] = [old]
    output = coaching()
    output["findings"][0]["historicalRefs"] = [old["refId"]]

    # 실행
    result = finalize_coaching(CoachingDraft.model_validate(output), SpendingAnalysisRequest.model_validate(request))

    # 검증: 과거 구매는 맥락에만 연결하고 선택 기간 예상액에는 합산하지 않는다.
    assert result.findings[0].savings.estimate == "12000"
    assert result.findings[0].historicalTransactionRefs == [old["refId"]]
    assert result.findings[0].contextQuestion is None


@pytest.mark.parametrize("change", ["unknown", "overlap", "future_history"])
def test_grounding_and_period_boundaries_are_not_delegated_to_ai(change):
    # 준비
    request = request_data()
    draft = coaching()
    if change == "unknown": draft["findings"][0]["evidenceRefs"] = ["tx-999"]
    if change == "overlap": draft["findings"].append(deepcopy(draft["findings"][0]))
    if change == "future_history": request["historicalTransactions"] = [{**request["transactions"][0], "refId": "tx-100001"}]

    # 실행 / 검증
    with pytest.raises((ValueError, ValidationError)):
        finalize_coaching(CoachingDraft.model_validate(draft), SpendingAnalysisRequest.model_validate(request))


def test_partial_coverage_never_extrapolates_a_saving():
    # 준비
    data = request_data()
    data.update(transactionCount=5, totalExpense="20000")
    data["coverage"].update(totalTransactionCount=5, isPartial=True, reason="TRANSACTION_LIMIT")

    # 실행
    result = finalize_coaching(CoachingDraft.model_validate(coaching()), SpendingAnalysisRequest.model_validate(data))

    # 검증
    assert len(result.findings) == 1
    assert result.findings[0].savings is None
    assert result.savings is None


@pytest.mark.parametrize("kind,year,quarter,half,start,end", [
    ("QUARTERLY", 2024, 1, None, "2024-01-01", "2024-04-01"),
    ("HALF_YEARLY", 2026, None, 2, "2026-07-01", "2027-01-01"),
    ("CUSTOM", 2026, None, None, "2026-01-31", "2026-02-28"),
])
def test_calendar_groupings_and_custom_month_end(kind, year, quarter, half, start, end):
    # 준비
    data = request_data()
    data.update(periodType=kind, year=year, month=None, quarter=quarter, half=half, periodStart=start,
                periodEndExclusive=end, previousPeriod=None, transactionCount=0, totalExpense="0", transactions=[])
    data["coverage"].update(analyzedTransactionCount=0, totalTransactionCount=0)

    # 실행 / 검증
    assert SpendingAnalysisRequest.model_validate(data).periodEndExclusive.isoformat() == end
    if kind == "CUSTOM":
        data["periodEndExclusive"] = "2026-02-27"
        with pytest.raises(ValidationError): SpendingAnalysisRequest.model_validate(data)


@pytest.mark.asyncio
async def test_review_failure_is_not_reported_as_a_successful_analysis():
    # 준비
    provider = AsyncMock()
    provider.call.side_effect = [coaching(), RuntimeError("private provider detail")]

    # 실행 / 검증
    with pytest.raises(HTTPException) as error:
        await SpendingAnalysisService(provider).analyze(SpendingAnalysisRequest.model_validate(request_data()))
    assert error.value.status_code == 503
    assert "private" not in str(error.value.detail)
    assert provider.call.await_count == 2


def test_supporting_context_does_not_inflate_the_savings_basis():
    # 준비: 설명에 쓰는 생수 주문은 절감 대상인 이어폰 구매와 분리한다.
    data = request_data()
    data['transactions'].append({**data['transactions'][0], 'refId':'tx-3','title':'생수 주문','memo':'집에서 쓰는 생수','amount':'20000'})
    data.update(transactionCount=3,totalExpense='32000')
    data['coverage'].update(analyzedTransactionCount=3,totalTransactionCount=3)
    draft=coaching()
    draft['findings'][0]['observation'] += ' 비교 문맥으로 생수 주문(tx-3)도 기록되어 있습니다.'
    request=SpendingAnalysisRequest.model_validate(data)

    # 실행: 본문 인용은 근거에 연결하지만 계산 대상에는 넣지 않는다.
    expanded=expand_draft(CoachingDraft.model_validate(draft),request)
    result=finalize_coaching(expanded,request)

    # 검증
    assert result.findings[0].evidenceTransactionRefs == ['tx-1','tx-2','tx-3']
    assert result.findings[0].savingsTransactionRefs == ['tx-1','tx-2']
    assert result.findings[0].savings.basisAmount == '12000'
    assert result.findings[0].savings.estimate == '12000'


def test_purchase_grouping_preserves_every_transaction_and_exact_total():
    # 준비: 실제 품목이 같은 반복 구매를 하나의 입력 묶음으로 표현한다.
    data=request_data()
    data['transactions'][1].update(storeName='Amazon')
    request=SpendingAnalysisRequest.model_validate(data)

    # 실행
    payload=prepare_coaching_payload(request)
    draft=coaching()
    draft['findings'][0].update(evidenceRefs=['tx-1'],targetRefs=['tx-1'])
    result=finalize_coaching(expand_draft(CoachingDraft.model_validate(draft),request),request)

    # 검증
    assert len(payload['transactions'])==1
    assert payload['transactions'][0]['transactionCount']==2
    assert payload['transactions'][0]['dates']==['2026-10-03','2026-10-05']
    assert payload['transactions'][0]['amount']=='12000'
    assert result.findings[0].savingsTransactionRefs==['tx-1','tx-2']


def test_valid_purchase_citations_with_korean_particles_are_rendered_in_all_fields():
    # 준비: 실제 상품을 참조로 지칭한 본문도 거래에 연결할 수 있어야 한다.
    data=request_data()
    draft=coaching()
    draft['findings'][0]['observation']='tx-1과 tx-2에서 같은 역할의 취미 물품을 선택한 기록이 있어 보유 물품과의 역할을 비교할 여지가 있습니다.'
    draft['findings'][0]['assumption']='tx-1의 구매 목적이 추가 취미 선택인 경우에만 일부를 줄이는 가정입니다.'

    # 실행
    request=SpendingAnalysisRequest.model_validate(data)
    result=finalize_coaching(expand_draft(CoachingDraft.model_validate(draft),request),request)

    # 검증: 공개 본문·가정 어느 쪽에도 내부 거래 코드를 남기지 않는다.
    assert 'tx-' not in result.model_dump_json(exclude={'findings':{'__all__':{'evidenceTransactionRefs','savingsTransactionRefs'}}})
    assert result.findings[0].savings.basisAmount=='12000'


def test_localized_source_date_is_a_concrete_anchor_for_translated_product_names():
    # 준비: 원장의 일본어 명칭을 한국어로 설명해도 날짜가 일치하면 실제 거래를 가리킨다.
    data=request_data()
    data['transactions'][0].update(title='日本語の商品名',storeName='日本語の店舗')
    data['transactions'][1].update(title='別の商品名',storeName='別の店舗')
    draft=coaching()
    draft['findings'][0]['observation']='10월 3일에 같은 역할의 물건을 사고 이후에 다른 제품을 추가로 구매한 기록이 있어, 다음 선택에서 역할을 나누어 볼 수 있습니다.'
    draft['findings'][0]['title']='같은 역할의 추가 선택 조정'

    # 실행 / 검증
    result=finalize_coaching(CoachingDraft.model_validate(draft),SpendingAnalysisRequest.model_validate(data))
    assert len(result.findings)==1


def test_known_soft_drinks_are_not_rejected_because_the_merchant_is_amazon():
    # 준비: 구입처가 아니라 실제 음료 종류로 대체 가능 여부를 구분한다.
    data=request_data()
    data['transactions'][0].update(title='Amazon コカ・コーラ ゼロ',memo='コカ・コーラ ゼロ 24本')
    data['transactions'][1].update(title='Amazon 天然水',memo='天然水 24本')
    request=SpendingAnalysisRequest.model_validate(data)

    # 실행 / 검증
    assert estimate_eligible(request.transactions[0],'DRINK_REPLACEMENT')
    assert not estimate_eligible(request.transactions[1],'DRINK_REPLACEMENT')
