from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from app.api.dependencies import get_stt_service
from app.services.stt_service import STTService

router = APIRouter(
    prefix="/stt",
    tags=["Speech to Text"]
)

@router.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    service: STTService = Depends(get_stt_service)
):
    """
    업로드된 오디오 파일을 받아서 텍스트로 변환합니다.

    동작 흐름:
    1. 클라이언트가 보낸 파일을 시스템 임시 디렉토리에 저장합니다.
    2. STTService(Whisper)를 호출하여 음성을 분석합니다.
    3. 분석이 끝나면 결과를 반환하고, 생성했던 임시 파일은 즉시 삭제합니다.

    Args:
    - file (UploadFile): 업로드된 MP3, WAV 등의 오디오 파일.
    - service (STTService): STT 처리를 담당하는 싱글톤 서비스 인스턴스.

    Returns:
    - dict: {"text": "인식된 문장"} 형태의 JSON 데이터.
    """
    try:
        text = await service.transcribe_file(file)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))