from __future__ import annotations

import asyncio
import unicodedata

from fastapi import HTTPException
from pydantic import ValidationError

from app.ai.ports import TextGenerationProvider
from app.schemas.spending_analysis import SpendingAnalysisRequest, SpendingAnalysisResponse

TASK_NAME = "ACCOUNT_BOOK_SPENDING_ANALYSIS"


def _normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


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
                    schema=SpendingAnalysisResponse.model_json_schema(),
                )
            response = SpendingAnalysisResponse.model_validate(result)
            validate_grounding(response, request)
            return response
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
