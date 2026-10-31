from typing import Annotated

from fastapi import APIRouter, Depends

from app.features.spending_analysis.service import SpendingAnalysisService
from app.schemas.spending_analysis import SpendingAnalysisRequest, SpendingAnalysisResponse

router = APIRouter(prefix="/account-book", tags=["Spending Analysis"])


def get_spending_analysis_service() -> SpendingAnalysisService:
    # 이 route를 불러올 때 음성·OCR 공통 의존성을 추가 초기화하지 않는다.
    from app.api.dependencies import get_spending_analysis_service as get_service

    return get_service()


@router.post("/spending-analysis", response_model=SpendingAnalysisResponse)
async def analyze_spending(
    request: SpendingAnalysisRequest,
    service: Annotated[SpendingAnalysisService, Depends(get_spending_analysis_service)],
) -> SpendingAnalysisResponse:
    return await service.analyze(request)
