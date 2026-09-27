import asyncio
import base64
import json
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import ValidationError

from app.ai.ports import SpeechSynthesisProvider
from app.api.dependencies import get_speech_execution_provider
from app.common.provider_failure import provider_failure_origin
from app.common.provider_retry import retry_after_details
from app.schemas.speech_execution import SpeechSynthesisCommand, SpeechSynthesisResponse

router = APIRouter(prefix="/speech", tags=["Internal Speech Execution"])


@router.post("/synthesize", response_model=SpeechSynthesisResponse)
async def synthesize(
    request: Request,
    provider: Annotated[SpeechSynthesisProvider, Depends(get_speech_execution_provider)],
) -> SpeechSynthesisResponse:
    # 본문 크기를 먼저 제한하고 검증 오류에는 발화 원문을 포함하지 않는다.
    raw = bytearray()
    async for chunk in request.stream():
        if len(raw) + len(chunk) > 128_000:
            raise HTTPException(
                413,
                detail={"code": "SPEECH_REQUEST_TOO_LARGE", "retryable": False},
            )
        raw.extend(chunk)
    try:
        command = SpeechSynthesisCommand.model_validate(json.loads(raw))
    except (ValidationError, ValueError, TypeError, UnicodeDecodeError, RecursionError):
        raise HTTPException(
            422,
            detail={"code": "SPEECH_REQUEST_INVALID", "retryable": False},
        ) from None

    # 학습 프롬프트나 합격 판정 없이 명시된 발화를 한 번만 실행한다.
    try:
        async with asyncio.timeout(command.remaining_milliseconds / 1000.0):
            result = await provider.synthesize_speech(
                text=command.text,
                voice=command.voice,
                language=command.language,
                speed=command.speed,
            )
    except TimeoutError as error:
        raise HTTPException(
            504,
            detail={
                "code": "PROVIDER_TIMEOUT",
                "retryable": True,
                **provider_failure_origin(error),
            },
        ) from None
    except ValueError as error:
        raise HTTPException(
            422,
            detail={
                "code": "SPEECH_REQUEST_INVALID",
                "retryable": False,
                **provider_failure_origin(error),
            },
        ) from None
    except Exception as error:
        status = getattr(error, "status_code", None)
        retryable = status == 429 or isinstance(status, int) and status >= 500
        raise HTTPException(
            503 if retryable else 502,
            detail={
                "code": "PROVIDER_UNAVAILABLE" if retryable else "PROVIDER_EXECUTION_FAILED",
                "retryable": retryable,
                **retry_after_details(error),
                **provider_failure_origin(error),
            },
        ) from None

    # 오디오 저장·접근권한은 호출자가 소유한다. 실행 계층은 URL을 내려받지 않는다.
    if not result.audio_bytes or len(result.audio_bytes) > 20_000_000:
        raise HTTPException(502, detail={"code": "SPEECH_RESPONSE_INVALID", "retryable": False})
    return SpeechSynthesisResponse(
        audioBase64=base64.b64encode(result.audio_bytes).decode("ascii"),
        contentType=result.content_type,
        durationSeconds=result.duration_seconds,
        provider=result.provider,
        model=result.model,
    )
