import json
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from app.features.spending_analysis.coaching import prepare_coaching_payload
from app.features.spending_analysis.service import SpendingAnalysisService
from app.schemas.spending_analysis import SpendingAnalysisRequest
from tests.test_spending_analysis import request_data
from tests.test_spending_coaching import coaching


PURCHASES = [
    (7, "ショッピング", "季節装飾の暖炉ランプ", "同じ棚にツリーとランタンがある。暖炉の演出は飾りの追加分。"),
    (7, "食費", "コンビニのカフェラテ", "自宅で使えるコーヒー器具あり。移動中の追加ドリンク。"),
    (8, "食費", "配達注文の追加デザート", "夕食とは別料金の追加デザート。夕食の量はそのままで足りる。"),
    (8, "生活", "小分けの洗濯洗剤", "自宅の洗剤容器は詰め替え対応。持ち運び用ではない。"),
    (9, "趣味", "同シリーズの追加くじ", "欲しかった景品は入手済み。別の景品目当てで追加した。"),
    (10, "通信", "動画サービスの広告なしオプション", "基本プラン契約に加えて広告なしを追加。対象番組は基本でも視聴可。"),
    (10, "交通", "週末の短距離タクシー", "荷物なし、急ぎの予定なし。同じ区間のバスも利用できる。"),
    (11, "ショッピング", "色違いのスマホケース", "同じ機種のケースが使用可能。故障交換ではなく色違いの予備。"),
]


def half_year_request() -> SpendingAnalysisRequest:
    data = request_data()
    transactions = [
        {"refId": f"tx-{index}", "date": f"2026-{month:02}-05", "title": title,
         "storeName": None, "category": category, "memo": memo, "amount": str(index * 1000)}
        for index, (month, category, title, memo) in enumerate(PURCHASES, 1)
    ]
    # 같은 구매 묶음의 금액을 첫 달에 몰아넣는 회귀와 긴 기간의 참조 한도를 함께 잡는다.
    transactions.append({**transactions[1], "refId": "tx-9", "date": "2026-08-05"})
    for index in range(10, 585):
        transactions.append({
            "refId": f"tx-{index}", "date": f"2026-{7 + index % 5:02}-10",
            "title": f"品目不明のカード明細 {index}", "storeName": None,
            "category": "その他", "memo": None, "amount": "3000",
        })
    for index, item in enumerate((row for row in transactions[:8] if row["refId"] != "tx-2"), 585):
        transactions.append({**item, "refId": f"tx-{index}", "date": item["date"][:8] + "18"})
    data.update(
        periodType="HALF_YEARLY", month=None, half=2, locale="ja",
        periodStart="2026-07-01", periodEndExclusive="2027-01-01", previousPeriod=None,
        transactions=transactions, transactionCount=591,
        totalExpense=str(sum(Decimal(item["amount"]) for item in transactions)),
        coverage={"analyzedTransactionCount": 591, "totalTransactionCount": 591,
                  "isPartial": False, "reason": None},
    )
    return SpendingAnalysisRequest.model_validate(data)


def wide_review() -> dict:
    draft = coaching()
    draft["summary"] = (
        "日常支出に加えて、飲み物、洗濯用品、動画視聴や移動の選び方にも見直せる余地がありました。"
        "選んだ買い物の用途と次回の代替手段を取引ごとに整理しました。"
    )
    draft["findings"] = []
    for index, (_, _, title, _) in enumerate(PURCHASES, 1):
        ref = f"tx-{index}"
        can_estimate = index in (1, 8)
        draft["findings"].append({
            "title": "ケースの買い足しを控える" if index == 8 else f"{title}を選ぶ場面を整理",
            "observation": (
                f"購入メモにはその品目の用途と、以前から使える代替手段の両方が記録されています。"
                f"次の判断は今回の購入記録に結びつきます。 ({ref})"
            ),
            "opinion": "目的が既存の選択肢で満たせるなら、今回の追加分は選ばずに済む可能性があります。",
            "action": "次回はメモに記録された既存の選択肢を使い、足りる場合に追加の購入を控えます。",
            "assumption": "必要な用途が既存の選択肢だけで満たせる場合の想定です。",
            "evidenceRefs": [ref],
            "historicalRefs": [],
            "targetRefs": [ref] if can_estimate else [],
            "estimateKind": "SKIP_EXTRA_ITEM" if can_estimate else "NONE",
        })
    return draft


@pytest.mark.asyncio
async def test_half_year_review_can_add_missed_choices_from_other_areas_and_months():
    # 준비: 초안은 장식 한 건에 집중하고 검토 결과는 빠진 일곱 소비 습관을 제안한다.
    request = half_year_request()
    reviewed = wide_review()
    reviewed["findings"][0]["observation"] = "７月の暖炉ランプは、すでにツリーとランタンがある同じ棚への装飾追加でした。(tx-1)"
    draft = {**reviewed, "findings": reviewed["findings"][:1]}
    provider = AsyncMock()
    provider.call.side_effect = [draft, reviewed]

    # 실행
    result = await SpendingAnalysisService(provider).analyze(request)

    # 검증: 모든 구매 묶음을 전달하면서 상한과 기간별 거래 분포를 함께 보낸다.
    generation, review = [call.kwargs for call in provider.call.await_args_list]
    source = json.loads(generation["data"])
    inspected = json.loads(review["data"])
    assert sum(unit["transactionCount"] for unit in source["transactions"]) == 591
    assert inspected["currentTransactions"] == source["transactions"]
    areas = {area["name"]: area for area in inspected["spendingAreas"]}
    assert areas["식비"]["months"] == [
        {"month": "2026-07", "amount": "2000", "transactionCount": 1},
        {"month": "2026-08", "amount": "8000", "transactionCount": 3},
    ]
    assert areas["쇼핑"]["detailRefsNotCitedByDraft"] == ["tx-8"]
    assert areas["식비"]["detailRefsNotCitedByDraft"] == ["tx-2", "tx-3"]
    assert areas["その他"]["detailRefsNotCitedByDraft"] == []
    for call in (generation, review):
        schema = call["schema"]
        assert schema["properties"]["findings"]["maxItems"] == 40
        items = schema["$defs"]["CoachingFinding"]["properties"]["evidenceRefs"]["items"]
        assert "pattern" in items and "enum" not in items

    # 검증: 기존 응답의 근거 연결과 중복 없는 합계가 넓어진 결과에도 적용된다.
    assert len(result.findings) == 8
    assert result.findings[0].reason.endswith("(取引日 2026-07-05)")
    assert result.findings[1].evidenceTransactionRefs == ["tx-2", "tx-9"]
    assert result.findings[-1].evidenceTransactionRefs == ["tx-8", "tx-591"]
    assert all("tx-" not in finding.reason for finding in result.findings)
    assert result.savings and result.savings.estimate == "18000"


@pytest.mark.asyncio
@pytest.mark.parametrize("hidden_history", [False, True])
async def test_large_request_with_pattern_schema_rejects_evidence_the_model_did_not_receive(hidden_history):
    # 준비: 형식만 맞는 미등록 ID 또는 검토에 보이지 않은 과거 근거를 제안한다.
    request = half_year_request()
    invalid = wide_review()
    provider = AsyncMock()
    if hidden_history:
        data = request.model_dump()
        data["historicalTransactions"] = [{
            **data["transactions"][3],
            "refId": "tx-9999",
            "date": "2026-06-10",
        }]
        request = SpendingAnalysisRequest.model_validate(data)
        draft = {**invalid, "findings": invalid["findings"][:1]}
        invalid["findings"][-1]["historicalRefs"] = ["tx-9999"]
        provider.call.side_effect = [draft, invalid]
    else:
        invalid["findings"][-1]["evidenceRefs"] = ["tx-9999"]
        provider.call.return_value = invalid

    # 실행 / 검증
    with pytest.raises(HTTPException) as caught:
        await SpendingAnalysisService(provider, semantic_review=hidden_history).analyze(request)
    assert caught.value.detail == {"code": "SPENDING_ANALYSIS_INVALID_RESPONSE"}
    assert provider.call.await_count == (2 if hidden_history else 1)


@pytest.mark.parametrize("end", ["2026-02-28", "2026-04-30", "2026-07-31"])
def test_custom_period_length_does_not_limit_number_of_supported_findings(end):
    # 준비
    data = request_data()
    data.update(periodType="CUSTOM", month=None, periodStart="2026-01-31", periodEndExclusive=end,
                transactionCount=0, totalExpense="0", transactions=[], previousPeriod=None)
    data["coverage"].update(analyzedTransactionCount=0, totalTransactionCount=0)
    request = SpendingAnalysisRequest.model_validate(data)

    # 실행 / 검증
    assert prepare_coaching_payload(request)["maxFindings"] == 40
