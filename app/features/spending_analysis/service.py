from __future__ import annotations

import asyncio
import re
import unicodedata
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
)

TASK_NAME = "ACCOUNT_BOOK_SPENDING_ANALYSIS"


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


def finalize_review(draft: SpendingAnalysisDraft, request: SpendingAnalysisRequest) -> SpendingAnalysisResponse:
    public = [SpendingAnalysisFinding.model_validate(item.model_dump(exclude={"reviewDisposition", "reviewPattern"})) for item in draft.findings]
    validate_grounding(SpendingAnalysisResponse(summary=draft.summary, findings=public), request)
    source = {item.refId: item for item in request.transactions}
    findings = []

    # 상황으로 설명된 구매·필수 지출·근거 없는 확인 메모는 개선 카드로 노출하지 않는다.
    for candidate, finding in zip(draft.findings, public, strict=True):
        if candidate.reviewDisposition != "REVIEW":
            continue
        evidence = [source[ref] for ref in candidate.evidenceTransactionRefs]
        if candidate.reviewPattern in ("REPEAT_PURCHASE", "REPEATED_MEALS") and len(evidence) < 2:
            raise ValueError("Repeated-purchase patterns require multiple source transactions")
        if candidate.reviewPattern == "DUPLICATE_CHARGE" and not _duplicate_charge(evidence):
            raise ValueError("Duplicate-charge claims require matching date, title and amount")
        if any(_essential_expense(item) for item in evidence) and candidate.reviewPattern != "DUPLICATE_CHARGE":
            continue
        findings.append(finding)

    summary = draft.summary
    if len(findings) != len(draft.findings):
        # 제외한 후보가 요약에 다시 '절감 대상'으로 남지 않도록 정직한 범위 설명을 쓴다.
        summaries = {
            "ko": f"추가 설명과 거래 근거를 반영해 검토할 항목 {len(findings)}개를 정리했습니다. 설명된 구매와 일반적인 필수 지출은 절감 대상으로 분류하지 않았습니다.",
            "ja": f"補足と取引根拠を反映し、確認する項目を{len(findings)}件に整理しました。説明済みの購入や通常の必要支出は削減対象にしていません。",
            "en": f"After considering your context and the transaction evidence, {len(findings)} items remain worth reviewing. Explained purchases and routine essential costs were excluded.",
        }
        summary = summaries[request.locale]
        if request.coverage.isPartial:
            summaries = {
                "ko": "전체 거래 중 일부 표본을 살펴본 분석입니다. ",
                "ja": "全取引の一部サンプルを確認した分析です。",
                "en": "This analysis reviews a sample of the complete period. ",
            }
            summary = summaries[request.locale] + summary

    # 익명 참조는 연결 배열 안에서만 사용하고 사용자에게 내부 필드명을 노출하지 않는다.
    text = " ".join([summary, *(" ".join(filter(None, (item.title, item.reason, item.action, item.contextQuestion, item.uncertainty))) for item in findings)])
    if re.search(r"\btx-\d+\b|\brefIds?\b|\bstoreName\b|\bsourceType\b|\breviewDisposition\b|\breviewPattern\b|\bFIXED_COST\b|coverage\.", text):
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
        timeout_seconds: float = 70.0,
        max_concurrency: int = 2,
    ) -> None:
        self.provider = provider
        self.timeout_seconds = timeout_seconds
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def analyze(self, request: SpendingAnalysisRequest) -> SpendingAnalysisResponse:
        # 빈 기간은 외부로 보내지 않는다. 표본 범위와 금액은 입력 schema에서 검증한다.
        if not request.transactions:
            summaries = {
                "ko": "선택한 기간에 분석할 지출 거래가 없습니다.",
                "ja": "選択した期間に分析できる支出取引がありません。",
                "en": "There are no expense transactions to analyze in this period.",
            }
            return SpendingAnalysisResponse(summary=summaries[request.locale], findings=[])
        payload = request.model_dump_json()
        if len(payload) > 350_000:
            raise HTTPException(status_code=413, detail={"code": "SPENDING_ANALYSIS_INPUT_TOO_LARGE"})

        # 대기를 포함한 단일 deadline이며 외부 호출·검증 실패를 자동 재생성하지 않는다.
        try:
            async with asyncio.timeout(self.timeout_seconds), self._semaphore:
                result = await self.provider.call(
                    type_name=TASK_NAME,
                    data=payload,
                    schema=SpendingAnalysisDraft.model_json_schema(),
                )
            return finalize_review(SpendingAnalysisDraft.model_validate(result), request)
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
