from __future__ import annotations

from calendar import monthrange
from collections import defaultdict
from datetime import date
from decimal import Decimal, ROUND_HALF_UP, localcontext
import re
import unicodedata

from app.schemas.spending_analysis import (
    MAX_ANALYSIS_FINDINGS,
    MIN_OBSERVATION_LENGTH,
    TRANSACTION_REF_PATTERN,
    AnalysisTransaction,
    CoachingDraft,
    SavingsScenario,
    SpendingAnalysisFinding,
    SpendingAnalysisRequest,
    SpendingAnalysisResponse,
)

POLICY_VERSION = "purchase-context-v5"
MAX_SCHEMA_ENUM_VALUES = 1000
PURCHASE_REF = r"(?<![A-Za-z0-9_-])tx-\d+(?![A-Za-z0-9_-])"
SCENARIOS = {
    "SMALLER_SELECTION": (500, 1500, 0, 3000),
    "DRINK_REPLACEMENT": (1000, 2000, 5000, 8000),
    "SKIP_EXTRA_ITEM": (10000, 10000, 0, 0),
}


def purchase_note(transaction: AnalysisTransaction) -> str:
    # 주문 식별자·청구일 반복을 줄이되 상품·치료·운임 등 의미 있는 기록은 보존한다.
    memo = unicodedata.normalize("NFKC", transaction.memo or "").strip()
    if "商品:" in memo:
        return re.sub(r"\s*\[ASIN:[^]]+\]", "", memo.split("商品:", 1)[1]).strip()
    if re.search(r"가져오기 ID:|EPOS 이용내역|추가 EPOS 명세", memo):
        if "SF 차감액" in memo:
            return "정기권 밖 추가운임 포함 실제 이용 차감액. 충전액 제외."
        if "물품 구매 표시" in memo:
            return "교통카드 물품 구매. 점포와 상품 미확인."
        return "카드 가맹점 명세. 개별 상품·구매 목적 미확인."
    return re.sub(r"注文番号[:：]\s*[^\s/]+", "", memo).strip()


def has_item_detail(transaction: AnalysisTransaction) -> bool:
    memo = purchase_note(transaction)
    return bool(memo) and not any(term in memo for term in ("미확인", "실제 이용 차감액", "食品・飲料", "日用品・医薬品"))


def has_purchase_context(transaction: AnalysisTransaction, comment: str | None) -> bool:
    if has_item_detail(transaction):
        return True
    normalized = unicodedata.normalize("NFKC", comment or "").casefold()
    return any(anchor and len(anchor) >= 2 and unicodedata.normalize("NFKC", anchor).casefold() in normalized
               for anchor in (transaction.title, transaction.storeName, transaction.date.isoformat()))


def protected_cost(transaction: AnalysisTransaction) -> bool:
    text = unicodedata.normalize("NFKC", " ".join(filter(None, [transaction.title, transaction.category, purchase_note(transaction)]))).casefold()
    category = (transaction.category or "").casefold()
    if category in ("의료", "医療", "医療費", "medical", "주거", "住居", "housing"):
        return True
    return bool(re.search(r"처방|정기 치료|의사 처방|診療|処方|월세|임대료|家賃|정기권|교통 이용|jetbrains|openai|anthropic|coursera|업무용|仕事用", text))


def scenario(amount: Decimal, kind: str, decimals: int, assumption: str) -> SavingsScenario | None:
    if kind not in SCENARIOS or amount <= 0:
        return None
    low, high, replacement_low, replacement_high = SCENARIOS[kind]
    with localcontext() as context:
        context.prec = 80
        unit = Decimal(1).scaleb(-decimals)
        rounded = lambda value: value.quantize(unit, rounding=ROUND_HALF_UP)
        gross = rounded(amount * Decimal(low + high) / 20000)
        replacement = rounded(gross * Decimal(replacement_low + replacement_high) / 20000)
        minimum = rounded(amount * low / 10000 * (1 - Decimal(replacement_high) / 10000))
        maximum = rounded(amount * high / 10000 * (1 - Decimal(replacement_low) / 10000))
        estimate = min(maximum, max(minimum, gross - replacement))
    return SavingsScenario(basisAmount=str(amount), lower=str(minimum), estimate=str(estimate), upper=str(maximum),
        grossReduction=str(gross), replacementCost=str(replacement), reductionLowBps=low, reductionHighBps=high,
        replacementLowBps=replacement_low, replacementHighBps=replacement_high, kind=kind, assumption=assumption)


def combine_scenarios(findings: list[SpendingAnalysisFinding]) -> SavingsScenario | None:
    values = [finding.savings for finding in findings if finding.savings]
    if not values:
        return None
    def total(field: str) -> str:
        with localcontext() as context:
            context.prec = 80
            return str(sum((Decimal(getattr(value, field)) for value in values), Decimal(0)))
    return SavingsScenario(basisAmount=total("basisAmount"), lower=total("lower"), estimate=total("estimate"), upper=total("upper"),
        grossReduction=total("grossReduction"), replacementCost=total("replacementCost"),
        reductionLowBps=0, reductionHighBps=0, replacementLowBps=0, replacementHighBps=0,
        kind="COMBINED", assumption="DISJOINT_CITED_PURCHASES_ONLY")


def finding_limit(request: SpendingAnalysisRequest) -> int:
    start, end = request.periodStart, request.periodEndExclusive
    months = (end.year - start.year) * 12 + end.month - start.month
    anniversary = date(end.year, end.month, min(start.day, monthrange(end.year, end.month)[1]))
    if end < anniversary:
        months -= 1
    if months < 3:
        return 4
    if months < 6:
        return 6
    return MAX_ANALYSIS_FINDINGS


def _category_name(transaction: AnalysisTransaction) -> str:
    name = unicodedata.normalize("NFKC", transaction.category or "").strip()
    return {"食費": "식비", "ショッピング": "쇼핑", "生活": "생활"}.get(name, name or "미분류")


def _spending_areas(request: SpendingAnalysisRequest, units: list[dict]) -> list[dict]:
    # 구매 묶음이 여러 달에 걸쳐도 월별 금액과 건수는 개별 거래의 실제 날짜로 집계한다.
    months: dict[str, dict[str, list[AnalysisTransaction]]] = defaultdict(lambda: defaultdict(list))
    source = {transaction.refId: transaction for transaction in request.transactions}
    for transaction in request.transactions:
        months[_category_name(transaction)][transaction.date.isoformat()[:7]].append(transaction)

    # 세부 구매 맥락이 있는 거래를 찾는 색인이다. 절감 가능하다는 판정은 아니다.
    details: dict[str, list[str]] = defaultdict(list)
    for unit in units:
        transaction = source[unit["refId"]]
        if has_purchase_context(transaction, request.comment) and not protected_cost(transaction):
            details[_category_name(transaction)].append(transaction.refId)

    with localcontext() as context:
        context.prec = 80
        areas = []
        for name, area_months in months.items():
            monthly = []
            for month, items in sorted(area_months.items()):
                amount = sum((Decimal(item.amount) for item in items), Decimal(0))
                monthly.append({"month": month, "amount": str(amount), "transactionCount": len(items)})
            total = sum((Decimal(row["amount"]) for row in monthly), Decimal(0))
            areas.append({
                "name": name,
                "amount": str(total),
                "months": monthly,
                "detailRefs": details[name],
            })
        return sorted(areas, key=lambda area: (Decimal(area["amount"]).copy_negate(), area["name"]))


def prepare_coaching_payload(request: SpendingAnalysisRequest, max_findings: int | None = None) -> dict:
    payload = request.model_dump(mode="json")
    # 거래 원문 대신 동일 금액·날짜·실제 구매 정보를 제공한다. 거래를 샘플링하지 않는다.
    for name in ("transactions", "historicalTransactions"):
        payload[name] = purchase_units(getattr(request, name))
    # 과거는 모든 구매 묶음의 금액·날짜를 남기고, 중복된 긴 상품 광고 문구는 축약한다.
    # 현재 기간의 원문과 근거 연결은 축약하지 않는다.
    for unit in payload["historicalTransactions"]:
        memo = unit["memo"]
        unit["memo"] = memo[:240]
        unit["detailTruncated"] = len(memo) > 240
    patterns: dict[str, dict[str, list[str]]] = {}
    roles = {
        "SEASON_DECOR": r"クリスマス|Christmas|スノーグローブ|LED 暖炉ランプ",
        "STAR_LIGHTING": r"星空|プラネタリウム",
        "DESK_ACCESSORY": r"マウスパッド|マウスマット|デスクマット",
        "COLLECTIBLE": r"가챠|ガチャ|一番くじ|フィギュア|ぬいぐるみ",
        "PHONE_CASE": r"スマホケース|iPhoneケース",
        "PREPARED_DRINK": r"COFFEE|CAFF|コーヒー|ラテ|カフェラ|ブレンド|麦茶",
    }
    for role, expression in roles.items():
        related = {name: [unit["refId"] for unit in payload[name]
                         if re.search(expression, unit["title"] + " " + unit["memo"], re.I)]
                   for name in ("transactions", "historicalTransactions")}
        if related["transactions"] and sum(len(refs) for refs in related.values()) > 1:
            patterns[role] = related
    payload["purchasePatterns"] = patterns
    payload["spendingAreas"] = _spending_areas(request, payload["transactions"])
    limit = finding_limit(request)
    payload["maxFindings"] = min(limit, max_findings) if max_findings is not None else limit
    payload["estimateScenarios"] = SCENARIOS
    payload["historyMeaning"] = "이전 조언은 실제 실행 기록이 아니다. 이후 지출·코멘트만으로 실천 성공/실패를 확정하지 않는다."
    return payload


def purchase_units(transactions: list[AnalysisTransaction]) -> list[dict]:
    groups: dict[tuple, list[AnalysisTransaction]] = defaultdict(list)
    for item in transactions:
        note = purchase_note(item)
        merchant = unicodedata.normalize("NFKC", item.storeName or item.title).casefold().strip()
        # 명세용 반복 행은 한 묶음, 실제 품목은 내용이 같은 구매끼리만 묶는다.
        key = (merchant, item.category, note, item.title if has_item_detail(item) else "")
        groups[key].append(item)
    units = []
    for items in groups.values():
        first = items[0]
        with localcontext() as context:
            context.prec = 80
            total = sum((Decimal(item.amount) for item in items), Decimal(0))
        units.append({"refId": first.refId, "date": min(item.date for item in items).isoformat(),
            "title": first.title, "storeName": first.storeName, "category": first.category, "memo": purchase_note(first),
            "amount": str(total), "transactionCount": len(items), "dates": sorted({item.date.isoformat() for item in items}),
            "purposeKnown": has_item_detail(first)})
    return units


def expand_draft(
    draft: CoachingDraft,
    request: SpendingAnalysisRequest,
    visible_history_refs: set[str] | None = None,
) -> CoachingDraft:
    def mapping(items: list[AnalysisTransaction]) -> dict[str, list[str]]:
        result = {}
        groups: dict[tuple, list[AnalysisTransaction]] = defaultdict(list)
        for item in items:
            note = purchase_note(item)
            merchant = unicodedata.normalize("NFKC", item.storeName or item.title).casefold().strip()
            groups[(merchant, item.category, note, item.title if has_item_detail(item) else "")].append(item)
        for grouped in groups.values():
            result[grouped[0].refId] = [item.refId for item in grouped]
        return result
    current, historical = mapping(request.transactions), mapping(request.historicalTransactions)
    if visible_history_refs is not None:
        historical = {ref: rows for ref, rows in historical.items() if ref in visible_history_refs}
    findings = []
    for item in draft.findings:
        evidence_units, historical_units = list(item.evidenceRefs), list(item.historicalRefs)
        # 본문에만 적힌 유효한 구매 인용도 열어볼 수 있게 연결한다. 절감 대상에는 추가하지 않는다.
        narrative = " ".join((item.title, item.observation, item.opinion, item.action, item.assumption))
        for ref in re.findall(PURCHASE_REF, narrative):
            if ref in current and ref not in evidence_units:
                evidence_units.append(ref)
            elif ref in historical and ref not in historical_units:
                historical_units.append(ref)
            elif ref not in current and ref not in historical:
                raise ValueError("Unknown narrative purchase unit")
        if any(ref not in current for ref in evidence_units + item.targetRefs) or any(ref not in historical for ref in historical_units):
            raise ValueError("Unknown purchase unit")
        # 모델에는 짧은 묶음 참조만, 공개 결과에는 원본 거래 전체를 연결한다.
        values = item.model_dump()
        values["evidenceRefs"] = [ref for unit in evidence_units for ref in current[unit]]
        values["historicalRefs"] = [ref for unit in historical_units for ref in historical[unit]]
        findings.append(item.model_copy(update={"evidenceRefs": values["evidenceRefs"], "historicalRefs": values["historicalRefs"],
            "targetRefs": [ref for unit in item.targetRefs for ref in current[unit]]}))
    return draft.model_copy(update={"findings": findings})


def coaching_schema(payload: dict, *, review: bool = False) -> dict:
    schema = CoachingDraft.model_json_schema()
    fields = schema["$defs"]["CoachingFinding"]["properties"]
    current = [unit["refId"] for unit in payload["currentTransactions" if review else "transactions"]]
    history = [unit["refId"] for unit in payload["historicalEvidence" if review else "historicalTransactions"]]
    # 긴 기간도 근거를 샘플링하지 않는다. 공급자 enum 한도를 넘으면 형식만 지정하고
    # 실제 구매 참조의 존재 여부와 현재/과거 구분은 기존 서버 검증에서 확인한다.
    enum_count = 2 * len(current) + max(1, len(history)) + len(fields["estimateKind"]["enum"])
    if enum_count <= MAX_SCHEMA_ENUM_VALUES:
        fields["evidenceRefs"]["items"]["enum"] = current
        fields["targetRefs"]["items"]["enum"] = current
        fields["historicalRefs"]["items"]["enum"] = history or ["NO_HISTORICAL_EVIDENCE"]
    else:
        for field in ("evidenceRefs", "targetRefs", "historicalRefs"):
            fields[field]["items"]["pattern"] = TRANSACTION_REF_PATTERN
    if not history:
        fields["historicalRefs"]["maxItems"] = 0
    schema["properties"]["findings"]["maxItems"] = payload["maxFindings"]
    return schema


def estimate_eligible(transaction: AnalysisTransaction, kind: str, comment: str | None = None) -> bool:
    if protected_cost(transaction) or not has_purchase_context(transaction, comment):
        return False
    text = transaction.title + " " + purchase_note(transaction)
    if kind == "DRINK_REPLACEMENT":
        if re.search(r"天然水|生水|생수|ミネラルウォーター|炭酸水|탄산수|bottled water|mineral water|sparkling water|弁当|トースト|サンド|ボロネーゼ|食事|飲食|定食|パスタ|ラーメン", text, re.I):
            return False
        return bool(re.search(r"ラテ|コーヒー|ココア|モンスター|麦茶|ブレンド|カフェラ|オーレ|茶|コーラ|ジュース|커피|음료|콜라|주스|coffee|latte|cocoa|cola|soda", text, re.I))
    return True


def finalize_coaching(draft: CoachingDraft, request: SpendingAnalysisRequest) -> SpendingAnalysisResponse:
    source = {item.refId: item for item in request.transactions}
    historical = {item.refId: item for item in request.historicalTransactions}
    all_source = source | historical
    findings: list[SpendingAnalysisFinding] = []
    used: set[str] = set()

    def display(text: str, allowed: set[str], *, min_length: int = 0) -> str:
        def replace(match: re.Match[str]) -> str:
            ref = match.group(0)
            if ref not in allowed:
                raise ValueError("Narrative cites unrelated evidence")
            return all_source[ref].date.isoformat()
        # 연결된 원문을 먼저 검증한 후, 본문의 순수 참조 목록은 거래 버튼으로 대신한다.
        for match in re.finditer(PURCHASE_REF, text):
            replace(match)
        without_citations = re.sub(
            r"\((?:(?:예|근거|거래)[:：]?\s*)?tx-\d+(?:\s*[,、~–]\s*tx-\d+)*\)",
            "",
            text,
        )
        rendered = re.sub(PURCHASE_REF, replace, without_citations)
        if len(rendered.strip()) < min_length:
            # 인용을 빼면 유효한 짧은 관찰이 사라질 수 있어 확인한 거래일은 남긴다.
            date_label = {"ko": "거래일", "ja": "取引日", "en": "purchase date"}[request.locale]
            rendered = re.sub(PURCHASE_REF, lambda match: f"{date_label} {replace(match)}", text)
        names = {"ko": "선택 구매 조정 가정", "ja": "選択的購入の調整仮定", "en": "hypothetical purchase adjustment"}
        return re.sub(r"\b(?:SMALLER_SELECTION|DRINK_REPLACEMENT|SKIP_EXTRA_ITEM)\b", names[request.locale], rendered)

    # 실제 거래 참조·현재와 과거의 경계·합산 중복은 의미 검토와 별도로 검사한다.
    for item in draft.findings:
        refs, history_refs = item.evidenceRefs, item.historicalRefs
        if len(set(refs)) != len(refs) or any(ref not in source for ref in refs):
            raise ValueError("Unknown or duplicate current evidence")
        if len(set(history_refs)) != len(history_refs) or any(ref not in historical for ref in history_refs):
            raise ValueError("Unknown or duplicate historical evidence")
        targets = item.targetRefs
        if len(set(targets)) != len(targets) or not set(targets).issubset(refs):
            raise ValueError("Savings targets must belong to current evidence")
        if used.intersection(targets):
            raise ValueError("Overlapping recommendation scope")
        used.update(targets)
        transactions = [source[ref] for ref in refs]
        if any(protected_cost(transaction) for transaction in transactions):
            continue
        if not any(has_purchase_context(transaction, request.comment) for transaction in transactions):
            continue
        narrative = " ".join((item.title, item.observation, item.opinion, item.action, item.assumption))
        anchors = [value for transaction in transactions for value in (transaction.title, transaction.storeName, transaction.date.isoformat()) if value]
        # 점포 접미사·상품명이 붙은 제목에서도 브랜드 본문을 인식한다. 의미 판단은 별도 검토가 맡는다.
        anchors += [token for transaction in transactions for value in (transaction.title, transaction.storeName or "")
                    for token in re.findall(r"[A-Za-z][A-Za-z0-9]{3,}", value)]
        # 번역된 점포명도 실제 거래일로 연결할 수 있다. ISO 날짜만 강제하지 않는다.
        for transaction in transactions:
            date = transaction.date
            anchors.extend((f"{date.month}월{date.day}일", f"{date.month}月{date.day}日", f"{date.month}/{date.day}"))
        normalize_anchor = lambda text: re.sub(r"\s+", "", unicodedata.normalize("NFKC", text).casefold())
        cites_purchase = bool(set(re.findall(PURCHASE_REF, narrative)).intersection(refs))
        if not cites_purchase and not any(normalize_anchor(anchor) in normalize_anchor(narrative) for anchor in anchors):
            raise ValueError("Concrete purchase anchor is missing")
        if "?" in narrative or "？" in narrative:
            raise ValueError("Coaching must not end in questions")
        target_transactions = [source[ref] for ref in targets]
        with localcontext() as context:
            context.prec = 80
            amount = sum((Decimal(transaction.amount) for transaction in target_transactions), Decimal(0))
        # 표본·혼합/불명 거래의 금액을 전체 개선 여지처럼 확대하지 않는다.
        can_estimate = bool(targets) and not request.coverage.isPartial and all(estimate_eligible(transaction, item.estimateKind, request.comment) for transaction in target_transactions)
        allowed = set(refs + history_refs)
        saving = scenario(amount, item.estimateKind, request.decimalPlaces, display(item.assumption, allowed)) if can_estimate else None
        reason = display(item.observation, allowed, min_length=MIN_OBSERVATION_LENGTH)
        findings.append(SpendingAnalysisFinding(title=display(item.title, allowed), reason=reason,
            opinion=display(item.opinion, allowed), action=display(item.action, allowed), contextQuestion=None,
            uncertainty=display(item.assumption, allowed), evidenceTransactionRefs=refs,
            historicalTransactionRefs=history_refs, savings=saving, savingsTransactionRefs=targets if saving else []))
    summary = display(draft.summary, set(all_source))
    return SpendingAnalysisResponse(summary=summary, findings=findings, policyVersion=POLICY_VERSION, savings=combine_scenarios(findings))


def review_payload(draft: CoachingDraft, payload: dict) -> dict:
    # 초안이 인용하지 않은 구매도 검토에서 다시 살펴볼 수 있게 영역별로 표시한다.
    cited = {ref for finding in draft.findings for ref in finding.evidenceRefs}
    areas = []
    for area in payload["spendingAreas"]:
        areas.append({
            "name": area["name"],
            "amount": area["amount"],
            "months": area["months"],
            "detailRefsNotCitedByDraft": [ref for ref in area["detailRefs"] if ref not in cited],
        })

    # 초안의 과거 근거와 알려진 유사 구매 문맥을 함께 제공한다.
    historical_refs = {ref for finding in draft.findings for ref in finding.historicalRefs}
    for pattern in payload["purchasePatterns"].values():
        historical_refs.update(pattern["historicalTransactions"])
    return {
        "draft": draft.model_dump(mode="json"),
        "comment": payload.get("comment"),
        "locale": payload["locale"],
        "coverage": payload["coverage"],
        "periodStart": payload["periodStart"],
        "periodEndExclusive": payload["periodEndExclusive"],
        "spendingAreas": areas,
        "previousAdvice": payload.get("previousAdvice"),
        "currentTransactions": payload["transactions"],
        "historicalEvidence": [
            item for item in payload["historicalTransactions"] if item["refId"] in historical_refs
        ],
        "maxFindings": payload["maxFindings"],
        "purchasePatterns": payload["purchasePatterns"],
    }
