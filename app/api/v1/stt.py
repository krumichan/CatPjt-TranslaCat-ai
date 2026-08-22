from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.dependencies import get_stt_service
from app.features.speech_to_text import (
    SpeechRuntimeClosed,
    SpeechRuntimeNotReady,
    SpeechRuntimeQueueFull,
)
from app.services.stt_service import AudioFileTooLarge, EmptyAudioFile, STTService

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
    except EmptyAudioFile as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except AudioFileTooLarge as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except (
        SpeechRuntimeClosed,
        SpeechRuntimeNotReady,
        SpeechRuntimeQueueFull,
    ) as exc:
        raise HTTPException(
            status_code=503,
            detail="음성 인식 Runtime이 요청을 처리할 수 없습니다.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="음성 인식 처리에 실패했습니다.",
        ) from exc
