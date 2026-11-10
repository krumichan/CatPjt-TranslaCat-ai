from copy import deepcopy
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from app.features.spending_analysis.coaching import prepare_coaching_payload
from app.features.spending_analysis.service import SpendingAnalysisService
from app.schemas.spending_analysis import SpendingAnalysisRequest
from tests.test_spending_analysis import request_data
from tests.test_spending_coaching import coaching


def with_transactions(rows, *, yearly=False):
    data = request_data()
    data.update(
        transactions=rows, previousPeriod=None, transactionCount=len(rows),
        totalExpense=str(sum(Decimal(row["amount"]) for row in rows)),
    )
    data["coverage"].update(analyzedTransactionCount=len(rows), totalTransactionCount=len(rows))
    if yearly:
        data.update(periodType="YEARLY", month=None, periodStart="2026-01-01", periodEndExclusive="2027-01-01")
    return SpendingAnalysisRequest.model_validate(data)


def supported_finding(title, refs, **changes):
    return {
        **coaching()["findings"][0], "title": title,
        "observation": f"기록에서 동일한 소비 목적의 거래를 여러 날에 볼 수 있습니다. 구입 상황에 따라 대체할 수 있습니다. ({refs[0]})",
        "action": "다음에 같은 목적의 지출이 생기면 현재 필요한 부분을 유지하고 대안을 이용해 보세요.",
        "evidenceRefs": refs, "targetRefs": [], "historicalRefs": [], "estimateKind": "NONE",
        **changes,
    }


@pytest.mark.asyncio
async def test_yearly_omits_a_question_and_one_off_without_losing_a_recurring_coffee_finding():
    # 준비: 이름으로 음료를 확인할 수 있는 두 날짜와, 같은 날의 일회성 지출을 섞는다.
    rows = [
        dict(refId="tx-1", date="2026-10-01", title="아메리카노", storeName="카페 A",
             category="식비", memo=None, amount="3000"),
        dict(refId="tx-2", date="2026-10-03", title="아메리카노", storeName="카페 A",
             category="식비", memo=None, amount="3000"),
        dict(refId="tx-3", date="2026-10-20", title="한 번 산 전시 기념품", storeName="전시장",
             category="취미", memo="전시 한정 기념품", amount="1000"),
        dict(refId="tx-4", date="2026-10-20", title="같은 주문의 기념품 봉투", storeName="전시장",
             category="취미", memo="같은 주문의 봉투", amount="100"),
    ]
    request = with_transactions(rows, yearly=True)
    good = supported_finding(
        "반복 아메리카노 구매는 가능한 날에 집에서 준비",
        ["tx-1"], targetRefs=["tx-1"], estimateKind="DRINK_REPLACEMENT",
    )
    question = {**good, "title": "커피가 필요한가요?", "action": "커피 대신 물을 마시는 편이 나을까요?"}
    once = supported_finding("전시 기념품을 줄이기", ["tx-3", "tx-4"])
    draft = {**coaching(), "findings": [question, once, good]}
    provider = AsyncMock()
    provider.call.side_effect = [draft, draft]

    # 실행
    result = await SpendingAnalysisService(provider).analyze(request)

    # 검증: 묶인 두 거래는 반복으로 인정하고, 질문이나 단발 구매는 결과·합계에 넣지 않는다.
    payload = prepare_coaching_payload(request)
    assert payload["purchasePatterns"]["PREPARED_DRINK"]["transactions"] == ["tx-1"]
    assert [finding.title for finding in result.findings] == [good["title"]]
    assert result.findings[0].evidenceTransactionRefs == ["tx-1", "tx-2"]
    assert result.savings and result.savings.basisAmount == "6000"
    assert "?" not in result.summary and "전시 기념품" not in result.summary


@pytest.mark.asyncio
@pytest.mark.parametrize("difference", [None, "quantity", "product"])
async def test_essential_alternative_requires_an_observed_lower_price_for_the_same_need(difference):
    # 준비: 같은 처방을 지키는 대안과 수량이 다른 거래를 구분한다.
    common = {"title": "정기 처방 조제", "category": "의료", "memo": "같은 처방 30정"}
    rows = [
        {**common, "refId": "tx-1", "date": "2026-10-01", "storeName": "약국 A", "amount": "6000"},
        {**common, "refId": "tx-2", "date": "2026-10-23", "storeName": "약국 B", "amount": "4000",
         "title": "다른 약 조제" if difference == "product" else common["title"],
         "memo": "같은 처방 10정" if difference == "quantity" else common["memo"]},
    ]
    request = with_transactions(rows)
    draft = {**coaching(), "findings": [
        supported_finding("같은 처방을 유지하는 조제 비용 비교", ["tx-1", "tx-2"], essentialAlternativeRef="tx-2"),
    ]}
    provider = AsyncMock()
    provider.call.side_effect = [draft, draft]

    # 실행 / 검증
    result = await SpendingAnalysisService(provider).analyze(request)
    assert len(result.findings) == (1 if difference is None else 0)
    assert result.savings is None


@pytest.mark.asyncio
async def test_monthly_analysis_can_return_dozen_distinct_evidenced_changes():
    # 준비: 월별 분석에도 서로 다른 반복 지출이 24개 있는 가상 원장.
    rows, findings = [], []
    for index in range(1, 25):
        refs = [f"tx-{index * 2 - 1}", f"tx-{index * 2}"]
        title = f"반복 기호 지출 {index}"
        for day, ref in zip((2, 20), refs, strict=True):
            rows.append(dict(refId=ref, date=f"2026-10-{day:02}", title=title, storeName=None,
                             category="기호품", memo=f"{title}의 추가 구매", amount="1000"))
        findings.append(supported_finding(title, refs[:1]))
    request = with_transactions(rows)
    draft = {**coaching(), "findings": findings}
    provider = AsyncMock()
    provider.call.side_effect = [draft, deepcopy(draft)]

    # 실행 / 검증: 현재 월이라고 제안 수를 임의로 줄이지 않는다.
    result = await SpendingAnalysisService(provider).analyze(request)
    assert len(result.findings) == 24
    assert result.findings[-1].evidenceTransactionRefs == ["tx-47", "tx-48"]
