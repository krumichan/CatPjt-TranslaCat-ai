from __future__ import annotations

import asyncio
import re
import unicodedata
import json
from decimal import Decimal

from fastapi import HTTPException
from pydantic import ValidationError

from app.ai.ports import TextGenerationProvider
from app.schemas.spending_analysis import (
    AnalysisTransaction,
    SpendingAnalysisDraft,
    SpendingAnalysisFinding,
    SpendingAnalysisRequest,
    SpendingAnalysisResponse,
    CoachingDraft,
)
from app.features.spending_analysis.coaching import coaching_schema, expand_draft, finalize_coaching, prepare_coaching_payload, review_payload

TASK_NAME = "ACCOUNT_BOOK_SPENDING_ANALYSIS"
_ANONYMOUS_REF = r"(?<![A-Za-z0-9_])tx-\d+(?![A-Za-z0-9_])"
_INTERNAL_CLASSIFICATION = r"(?<![A-Za-z0-9_])(?:EXPLAINED_BY_COMMENT|NECESSARY|UNSUPPORTED|REVIEW|REPEAT_PURCHASE|REPEATED_MEALS|DUPLICATE_CHARGE|ITEM_REVIEW|FIXED_COST)(?![A-Za-z0-9_])"
_PRIVATE_ENUM = r"(?<![A-Za-z0-9_])(?:EXPLAINED_BY_COMMENT|REPEAT_PURCHASE|REPEATED_MEALS|DUPLICATE_CHARGE|ITEM_REVIEW|FIXED_COST)(?![A-Za-z0-9_])"
_INTERNAL_FIELD = r"(?<![A-Za-z0-9_])(?:refIds?|storeName|sourceType|reviewDisposition|reviewPattern)(?![A-Za-z0-9_])|coverage\."


def _normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _essential_expense(transaction: AnalysisTransaction) -> bool:
    title = _normalized(transaction.title)
    category = _normalized(transaction.category or "")
    memo = _normalized(transaction.memo or "")
    # 점포명·단순 근무 장소·렌터카·모든 고정 구독을 필수 치료/주거비와 혼동하지 않는다.
    medical_category = any(term in category for term in ("의료", "医療")) or bool(re.search(r"\b(?:medical|healthcare)\b", category))
    treatment_title = any(term in title for term in ("진료", "처방", "치료", "의약품", "診療", "処方", "治療")) or bool(re.search(r"\b(?:prescription|medical bill|hospital bill)\b", title))
    treatment_memo = any(term in memo for term in ("의사 처방", "정기 치료", "처방약", "医師の処方", "処方薬")) or bool(re.search(r"\b(?:prescribed medication|prescribed medicine)\b", memo))
    housing_title = any(term in title for term in ("월세", "임대료", "家賃", "住宅ローン")) or bool(re.search(r"\b(?:monthly rent|house rent|apartment rent|mortgage payment)\b", title))
    housing_category = category in ("주거", "住居", "housing", "housing costs") and title in ("rent", "mortgage")
    return medical_category or treatment_title or treatment_memo or housing_title or housing_category


def _duplicate_charge(transactions: list[AnalysisTransaction]) -> bool:
    if len(transactions) < 2:
        return False
    first = transactions[0]
    return all(
        item.date == first.date and _normalized(item.title) == _normalized(first.title)
        and Decimal(item.amount) == Decimal(first.amount)
        for item in transactions[1:]
    )


def _display_citations(text: str | None, allowed: set[str], source: dict[str, AnalysisTransaction], locale: str) -> str | None:
    if text is None:
        return None

    # 모델이 본문에도 인용한 익명 참조를 같은 근거 거래의 실제 날짜로 표시한다.
    # 허용된 근거 밖의 참조를 연결하거나 새로운 사실을 만들지는 않는다.
    def replace(match: re.Match[str]) -> str:
        ref = match.group(0)
        if ref not in allowed or ref not in source:
            raise ValueError("Narrative citation must be part of the finding evidence")
        return source[ref].date.isoformat()

    rendered = re.sub(_ANONYMOUS_REF, replace, text)
    label = {"ko": "거래", "ja": "取引", "en": "transactions"}[locale]
    return re.sub(r"\brefIds?\s*[:=]\s*", label + ": ", rendered)


def _has_purchase_context(transaction: AnalysisTransaction, comment: str | None) -> bool:
    if transaction.memo and transaction.memo.strip():
        return True
    title = _normalized(transaction.title)
    if re.search(r"충동|미사용|중복 구매|衝動|未使用|\b(?:unused|impulse|duplicate purchase)\b", title):
        return True
    context = _normalized(comment or "")
    anchors = (transaction.title, transaction.storeName, transaction.date.isoformat())
    return any(anchor and len(_normalized(anchor)) >= 2 and _normalized(anchor) in context for anchor in anchors)


def finalize_review(draft: SpendingAnalysisDraft, request: SpendingAnalysisRequest) -> SpendingAnalysisResponse:
    public = [SpendingAnalysisFinding.model_validate(item.model_dump(exclude={"reviewDisposition", "reviewPattern"})) for item in draft.findings]
    validate_grounding(SpendingAnalysisResponse(summary=draft.summary, findings=public), request)
    source = {item.refId: item for item in request.transactions}
    findings = []
    insufficient_detail = False

    # 상황으로 설명된 구매·필수 지출·근거 없는 확인 메모는 개선 카드로 노출하지 않는다.
    for candidate, finding in zip(draft.findings, public, strict=True):
        if candidate.reviewDisposition == "EXPLAINED_BY_COMMENT" and not request.comment:
            raise ValueError("A comment resolution requires an actual user comment")
        if candidate.reviewDisposition != "REVIEW":
            continue
        evidence = [source[ref] for ref in candidate.evidenceTransactionRefs]
        if candidate.reviewPattern in ("REPEAT_PURCHASE", "REPEATED_MEALS") and len(evidence) < 2:
            raise ValueError("Repeated-purchase patterns require multiple source transactions")
        if candidate.reviewPattern == "DUPLICATE_CHARGE" and not _duplicate_charge(evidence):
            raise ValueError("Duplicate-charge claims require matching date, title and amount")
        if any(_essential_expense(item) for item in evidence) and candidate.reviewPattern != "DUPLICATE_CHARGE":
            continue
        if candidate.reviewPattern == "ITEM_REVIEW" and len(evidence) == 1 and not _has_purchase_context(evidence[0], request.comment):
            # 한 건의 금액과 점포명, 품목 불명만으로 환불·절감 후보를 만들지 않는다.
            insufficient_detail = True
            continue
        fields = finding.model_dump()
        for field in ("title", "reason", "action", "contextQuestion", "uncertainty"):
            fields[field] = _display_citations(fields[field], set(candidate.evidenceTransactionRefs), source, request.locale)
        findings.append(SpendingAnalysisFinding.model_validate(fields))

    summary = draft.summary
    if "EXPLAINED_BY_COMMENT" in summary and not request.comment:
        raise ValueError("A comment resolution requires an actual user comment")
    internal_summary = bool(re.search(_INTERNAL_CLASSIFICATION, summary))
    if len(findings) != len(draft.findings) or internal_summary:
        # 제외한 후보가 요약에 다시 '절감 대상'으로 남지 않도록 정직한 범위 설명을 쓴다.
        contextual_summaries = {
            "ko": f"추가 설명과 거래 근거를 반영해 검토할 항목 {len(findings)}개를 정리했습니다. 설명된 구매와 일반적인 필수 지출은 절감 대상으로 분류하지 않았습니다.",
            "ja": f"補足と取引根拠を反映し、確認する項目を{len(findings)}件に整理しました。説明済みの購入や通常の必要支出は削減対象にしていません。",
            "en": f"After considering your context and the transaction evidence, {len(findings)} items remain worth reviewing. Explained purchases and routine essential costs were excluded.",
        }
        record_summaries = {
            "ko": f"제공된 거래 근거를 바탕으로 검토할 항목 {len(findings)}개를 정리했습니다. 근거가 부족한 권고나 일반적인 필수 지출은 절감 대상으로 분류하지 않았습니다.",
            "ja": f"取引の根拠に基づき、確認する項目を{len(findings)}件に整理しました。裏付けのない提案や通常の必要支出は削減対象にしていません。",
            "en": f"The supplied transaction evidence identifies {len(findings)} items worth reviewing. Unsupported suggestions and routine essential costs were excluded.",
        }
        summary = (contextual_summaries if request.comment else record_summaries)[request.locale]
        if request.coverage.isPartial:
            summaries = {
                "ko": "전체 거래 중 일부 표본을 살펴본 분석입니다. ",
                "ja": "全取引の一部サンプルを確認した分析です。",
                "en": "This analysis reviews a sample of the complete period. ",
            }
            summary = summaries[request.locale] + summary

    if insufficient_detail:
        notices = {
            "ko": " 일부 단일 거래는 품목·구매 목적 정보가 부족해 불필요한 지출로 판단하지 않았습니다. 해당 거래의 영수증 메모나 추가 설명을 넣으면 더 구체적으로 살펴볼 수 있습니다.",
            "ja": " 単独の取引には品目・購入目的が不明なものがあり、不要な支出とは判断していません。レシートのメモや補足があれば具体的に確認できます。",
            "en": " Some individual purchases lack item or purpose details, so they were not judged unnecessary. Add receipt notes or context about those purchases for a more specific review.",
        }
        summary += notices[request.locale]

    # 익명 참조는 연결 배열 안에서만 사용하고 사용자에게 내부 필드명을 노출하지 않는다.
    summary = _display_citations(summary, set(source), source, request.locale)
    assert summary is not None
    text = " ".join([summary, *(" ".join(filter(None, (item.title, item.reason, item.action, item.contextQuestion, item.uncertainty))) for item in findings)])
    if re.search(f"{_ANONYMOUS_REF}|{_INTERNAL_FIELD}|{_PRIVATE_ENUM}", text):
        raise ValueError("Internal identifiers must not appear in user-facing analysis")
    return SpendingAnalysisResponse(summary=summary, findings=findings)


def validate_grounding(
    response: SpendingAnalysisResponse, request: SpendingAnalysisRequest
) -> None:
    evidence = {item.refId: item for item in request.transactions}
    seen_groups: set[frozenset[str]] = set()
    for finding in response.findings:
        refs = finding.evidenceTransactionRefs
        if len(set(refs)) != len(refs) or any(ref not in evidence for ref in refs):
            raise ValueError("Unknown or duplicate evidence reference")
        group = frozenset(refs)
        if group in seen_groups:
            raise ValueError("Duplicate finding evidence group")
        seen_groups.add(group)

        # 카테고리 순위만 설명한 결과는 배제하고 실제 거래의 이름·점포·날짜를 요구한다.
        narrative = _normalized(f"{finding.title} {finding.reason}")
        anchors = [
            anchor
            for ref in refs
            for anchor in (evidence[ref].title, evidence[ref].storeName, evidence[ref].date.isoformat())
            if anchor and len(_normalized(anchor)) >= 2
        ]
        if not any(_normalized(anchor) in narrative for anchor in anchors):
            raise ValueError("Finding must identify its concrete source transaction")


class SpendingAnalysisService:
    def __init__(
        self,
        provider: TextGenerationProvider,
        *,
        timeout_seconds: float = 240.0,
        max_concurrency: int = 2,
        semantic_review: bool = True,
        max_findings: int = 2,
    ) -> None:
        self.provider = provider
        self.timeout_seconds = timeout_seconds
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self.semantic_review = semantic_review
        self.max_findings = max_findings

    async def analyze(self, request: SpendingAnalysisRequest) -> SpendingAnalysisResponse:
        # 빈 기간은 외부로 보내지 않는다. 표본 범위와 금액은 입력 schema에서 검증한다.
        if not request.transactions:
            summaries = {
                "ko": "선택한 기간에 분석할 지출 거래가 없습니다.",
                "ja": "選択した期間に分析できる支出取引がありません。",
                "en": "There are no expense transactions to analyze in this period.",
            }
            return SpendingAnalysisResponse(summary=summaries[request.locale], findings=[])
        context = prepare_coaching_payload(request, self.max_findings)
        payload = json.dumps(context, ensure_ascii=False)
        if len(payload) > 2_000_000:
            raise HTTPException(status_code=413, detail={"code": "SPENDING_ANALYSIS_INPUT_TOO_LARGE"})

        # 대기를 포함한 단일 deadline이며 외부 호출·검증 실패를 자동 재생성하지 않는다.
        try:
            async with asyncio.timeout(self.timeout_seconds), self._semaphore:
                result = await self.provider.call(
                    type_name=TASK_NAME,
                    data=payload,
                    schema=coaching_schema(context),
                )
                # 기존 공급자 응답/fixture는 공개 v2 형식을 읽을 수 있게 유지한다.
                # 실제 v3 호출의 Structured Outputs는 CoachingDraft 계약을 사용한다.
                if "policyVersion" not in result:
                    return finalize_review(SpendingAnalysisDraft.model_validate(result), request)
                draft = CoachingDraft.model_validate(result)
                if self.semantic_review:
                    review_context = review_payload(draft, context)
                    revised = await self.provider.call(type_name="ACCOUNT_BOOK_SPENDING_REVIEW",
                        data=json.dumps(review_context, ensure_ascii=False), schema=coaching_schema(review_context, review=True))
                    draft = CoachingDraft.model_validate(revised)
                if len(draft.findings) > self.max_findings:
                    raise ValueError("Too many coaching findings")
                return finalize_coaching(expand_draft(draft, request), request)
        except TimeoutError as exc:
            raise HTTPException(
                status_code=504, detail={"code": "SPENDING_ANALYSIS_TIMEOUT"}
            ) from exc
        except (ValidationError, ValueError) as exc:
            raise HTTPException(
                status_code=502, detail={"code": "SPENDING_ANALYSIS_INVALID_RESPONSE"}
            ) from exc
        except HTTPException:
            raise
        except Exception as exc:
            # 입력·코멘트·외부 응답 원문을 오류 본문이나 로그에 남기지 않는다.
            raise HTTPException(
                status_code=503, detail={"code": "SPENDING_ANALYSIS_UNAVAILABLE"}
            ) from exc
