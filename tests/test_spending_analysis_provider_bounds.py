from copy import deepcopy
import json
from unittest.mock import AsyncMock

import pytest

from app.features.spending_analysis.service import SpendingAnalysisService
from app.schemas.spending_analysis import SpendingAnalysisRequest
from tests.test_spending_analysis_breadth import half_year_request, wide_review


@pytest.mark.asyncio
@pytest.mark.parametrize("overflow_stage", ["draft", "review"])
async def test_half_year_keeps_verified_references_beyond_model_requested_list_sizes(overflow_stage):
    # 준비: 공급자가 여러 달의 음료 구매와 이전 비교 기록을 한 제안에 인용한다.
    data = half_year_request().model_dump()
    repeated_purchases = data["transactions"][9:30]
    for index, item in enumerate(repeated_purchases, 1):
        item.update(
            title="テイクアウトコーヒー", storeName=f"試験売店 {index}", category="食費",
            memo="自宅から持参できる日の追加ドリンク",
        )
    data["historicalTransactions"] = [
        {**repeated_purchases[0], "refId": f"tx-{1000 + index}", "date": "2026-06-03",
         "storeName": f"過去の試験売店 {index}"}
        for index in range(1, 32)
    ]
    request = SpendingAnalysisRequest.model_validate(data)
    current_refs = [item["refId"] for item in repeated_purchases]
    historical_refs = [item["refId"] for item in data["historicalTransactions"]]
    wide = wide_review()
    finding = wide["findings"][1]
    finding.update(
        observation="持参できる日の追加ドリンクとしてテイクアウトコーヒーを購入した記録が複数あります。(tx-10)",
        evidenceRefs=current_refs, targetRefs=current_refs, historicalRefs=historical_refs,
        estimateKind="DRINK_REPLACEMENT",
    )
    wide["findings"] = [finding]
    short = deepcopy(wide)
    short["findings"][0].update(
        evidenceRefs=current_refs[:1], targetRefs=current_refs[:1], historicalRefs=[],
    )
    provider = AsyncMock()
    provider.call.side_effect = [wide if overflow_stage == "draft" else short, wide]

    # 실행
    result = await SpendingAnalysisService(provider).analyze(request)

    # 검증: 검토에 실제 제공한 근거는 모두 연결하고, 현재 기간의 대상만 절약액에 포함한다.
    assert provider.call.await_count == 2
    assert result.findings[0].evidenceTransactionRefs == current_refs
    assert result.findings[0].historicalTransactionRefs == historical_refs
    assert result.findings[0].savingsTransactionRefs == current_refs
    assert result.findings[0].savings and result.findings[0].savings.basisAmount == "63000"
    for call in provider.call.await_args_list:
        fields = call.kwargs["schema"]["$defs"]["CoachingFinding"]["properties"]
        assert fields["evidenceRefs"]["maxItems"] == 20
        assert fields["targetRefs"]["maxItems"] == 20
        assert fields["historicalRefs"]["maxItems"] == 30


@pytest.mark.asyncio
async def test_half_year_excludes_an_unsubstantiated_ninth_finding_without_losing_eight_valid_ones():
    # 준비: 실제 기록이 한 번뿐인 아홉 번째 제안을 두 응답에 모두 포함한다.
    ordered = wide_review()
    last = deepcopy(ordered["findings"][0])
    last.update(
        observation="品目不明のカード明細が一日分だけあります。この一回の記録では同じ選択の繰り返しは分かりません。 (tx-10)",
        evidenceRefs=["tx-10"], targetRefs=[], estimateKind="NONE",
    )
    ordered["findings"].append(last)
    provider = AsyncMock()
    provider.call.side_effect = [ordered, ordered]

    # 실행
    result = await SpendingAnalysisService(provider).analyze(half_year_request())

    # 검증: 기간별 상한 대신 실제 반복 구매 근거를 적용한다.
    review = json.loads(provider.call.await_args_list[1].kwargs["data"])
    assert len(review["draft"]["findings"]) == 9
    assert len(result.findings) == 8
    assert result.findings[-1].evidenceTransactionRefs == ["tx-8", "tx-591"]
