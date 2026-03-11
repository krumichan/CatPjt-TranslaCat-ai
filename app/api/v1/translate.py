from fastapi import APIRouter, Depends
from app.schemas.translation import SingleTranslationRequest, BatchTranslationRequest
from app.api.dependencies import get_gemini_service
from app.services.gemini_service import GeminiService

router = APIRouter(
    prefix="/translate",
    tags=["Translation"]
)

@router.post("/single")
async def translate_single(
    request: SingleTranslationRequest,
    service: GeminiService = Depends(get_gemini_service)
):
    """
    단일 문장을 받아 지정된 규칙에 따라 번역합니다.

    이 엔드포인트는 한 번에 하나의 문장만 처리하며, 즉각적인 응답이<br/>
    필요한 채팅이나 간단한 UI 텍스트 번역에 적합합니다.

    Args:
    - request (SingleTranslationRequest): 번역할 텍스트와 프롬프트 타입이 포함된 객체.
    - service (GeminiService): Gemini 모델과 통신하는 서비스 계층 (의존성 주입).

    Returns:
    - dict: {"translated": "번역된 결과 문자열"}
    """
    translated_text = await service.call(type_name=request.type, data=request.text)
    return {"translated": translated_text}

@router.post("/batch")
async def translate_batch(
    request: BatchTranslationRequest, 
    service: GeminiService = Depends(get_gemini_service)
):
    """
    여러 문장의 리스트를 받아 병렬로 배치 번역을 수행합니다.

    내부적으로 5개씩 묶어(Chunking) 병렬로 처리하며, 특정 문장 번역에 실패할 경우<br/>
    해당 문장만 개별적으로 재시도하여 데이터 누락을 최소화합니다.<br/>
    대용량 웹 소설 데이터 처리에 최적화되어 있습니다.

    Args:
    - request (BatchTranslationRequest): 번역할 문장 리스트와 프롬프트 타입이 포함된 객체.
    - service (GeminiService): 배치 처리 및 재시도 로직이 포함된 서비스 계층.

    Returns:
    - dict: {"translated": ["번역결과1", "번역결과2", ...]} <br/>
    (원본 리스트와 동일한 순서 보장)
    """
    results = await service.translate_batch(request.texts, request.type)
    return {"translated": results}