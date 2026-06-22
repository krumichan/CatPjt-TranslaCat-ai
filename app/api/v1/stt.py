from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.dependencies import get_stt_service
from app.services.stt_service import STTService

router = APIRouter(
    prefix="/stt",
    tags=["Speech to Text"],
)


@router.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    service: STTService = Depends(get_stt_service),
):
    try:
        text = await service.transcribe_file(file)
        return {"text": text}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
