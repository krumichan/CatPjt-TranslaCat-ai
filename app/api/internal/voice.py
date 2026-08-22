from __future__ import annotations

import asyncio
import json
import secrets
from typing import Annotated, Any
from uuid import uuid4

from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.api.dependencies import (
    get_voice_stream_service,
    get_voice_translation_service,
)
from app.core.config import settings
from app.features.voice_translation.errors import VoicePipelineException
from app.features.voice_translation.stream import (
    VoiceChannelStreamContext,
    VoiceStreamApplicationService,
)
from app.features.voice_translation.translation import VoiceTranslationService
from app.schemas.voice_translation import (
    SUPPORTED_VOICE_LANGUAGES,
    VoiceChannel,
    VoiceErrorCode,
    VoicePipelineFailedEvent,
    VoiceReadinessResponse,
    VoiceStage,
    VoiceStreamClose,
    VoiceStreamFlush,
    VoiceStreamOpen,
    VoiceTranslationRetryRequest,
    VoiceTranslationRetryResponse,
)

router = APIRouter(prefix="/voice", tags=["Internal Voice Translation V2"])


class _StreamOpenTimeout(Exception):
    pass


def verify_internal_service_api_key(
    x_api_key: Annotated[str | None, Header(alias="X-API-KEY")] = None,
) -> None:
    if not _is_authorized(x_api_key):
        raise HTTPException(
            status_code=401,
            detail={
                "code": VoiceErrorCode.UNAUTHORIZED.value,
                "stage": VoiceStage.AUTH.value,
                "retryable": False,
                "message": "인증되지 않은 Internal Service 요청입니다.",
            },
        )


@router.websocket("/streams")
async def stream_voice_translation(
    websocket: WebSocket,
    service: VoiceStreamApplicationService = Depends(get_voice_stream_service),
) -> None:
    if not _is_authorized(websocket.headers.get("X-API-KEY")):
        await websocket.close(
            code=4401,
            reason=VoiceErrorCode.UNAUTHORIZED.value,
        )
        return

    await websocket.accept()
    context: VoiceChannelStreamContext | None = None
    sender_task: asyncio.Task[None] | None = None
    open_payload: dict[str, Any] = {}
    try:
        stream_open = await _receive_stream_open(websocket, open_payload)
        context = await service.open_stream(stream_open)
        sender_task = asyncio.create_task(
            _send_events(websocket, context),
            name=f"voice-event-sender-{stream_open.session_id}",
        )
        await _receive_stream_frames(websocket, context, sender_task)
    except WebSocketDisconnect:
        pass
    except asyncio.CancelledError:
        raise
    except _StreamOpenTimeout:
        await _send_pre_open_failure(
            websocket,
            open_payload,
            VoicePipelineException(
                code=VoiceErrorCode.INVALID_STREAM_OPEN,
                stage=VoiceStage.STREAM,
                message="STREAM_OPEN 대기 시간이 초과되었습니다.",
                retryable=False,
            ),
        )
        await websocket.close(code=4408, reason="STREAM_OPEN_TIMEOUT")
    except VoicePipelineException as exc:
        if context is not None:
            await context.emit_failure(exc)
            await context.close("PIPELINE_ERROR")
            if sender_task is not None:
                await _wait_for_sender(sender_task)
        else:
            await _send_pre_open_failure(websocket, open_payload, exc)
            await websocket.close(code=4400, reason=exc.code.value)
    finally:
        if context is not None:
            await service.release_stream(context)
        if sender_task is not None and not sender_task.done():
            sender_task.cancel()
            await asyncio.gather(sender_task, return_exceptions=True)


@router.post(
    "/translation/retry",
    response_model=VoiceTranslationRetryResponse,
    response_model_by_alias=True,
    dependencies=[Depends(verify_internal_service_api_key)],
)
async def retry_voice_translation(
    request: VoiceTranslationRetryRequest,
    service: VoiceTranslationService = Depends(get_voice_translation_service),
) -> VoiceTranslationRetryResponse:
    try:
        return await service.retry(request)
    except VoicePipelineException as exc:
        if exc.code == VoiceErrorCode.TRANSLATION_TIMEOUT:
            status_code = 504
        elif exc.code == VoiceErrorCode.INVALID_EVENT_SCHEMA:
            status_code = 409
        else:
            status_code = 502
        raise HTTPException(
            status_code=status_code,
            detail=exc.as_error().model_dump(by_alias=True, mode="json"),
        ) from exc


@router.get(
    "/readiness",
    response_model=VoiceReadinessResponse,
    response_model_by_alias=True,
    dependencies=[Depends(verify_internal_service_api_key)],
)
async def voice_readiness(
    service: VoiceStreamApplicationService = Depends(get_voice_stream_service),
):
    payload = VoiceReadinessResponse(
        ready=service.ready,
        accepting_streams=service.accepting_streams,
        stt_model=service.stt_provider.model_version,
        translation_model=settings.AI_VOICE_TRANSLATION_MODEL_NAME,
        active_streams=service.active_stream_count,
    )
    if not payload.ready:
        return JSONResponse(
            status_code=503,
            content=payload.model_dump(by_alias=True, mode="json"),
        )
    return payload


async def _receive_stream_open(
    websocket: WebSocket,
    open_payload: dict[str, Any],
) -> VoiceStreamOpen:
    try:
        first_message = await asyncio.wait_for(
            websocket.receive(),
            timeout=settings.AI_VOICE_STREAM_OPEN_TIMEOUT_SECONDS,
        )
    except (TimeoutError, asyncio.TimeoutError) as exc:
        raise _StreamOpenTimeout from exc

    if first_message.get("type") == "websocket.disconnect":
        disconnect_code = first_message.get("code")
        raise WebSocketDisconnect(
            code=disconnect_code if isinstance(disconnect_code, int) else 1000
        )
    text = first_message.get("text")
    if not isinstance(text, str):
        raise VoicePipelineException(
            code=VoiceErrorCode.INVALID_STREAM_OPEN,
            stage=VoiceStage.STREAM,
            message="첫 Frame은 STREAM_OPEN JSON Text여야 합니다.",
            retryable=False,
        )

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            open_payload.update(parsed)
        return VoiceStreamOpen.model_validate(parsed)
    except ValidationError as exc:
        audio_format_error = any(
            error.get("loc", ())[:1] in {("audioFormat",), ("audio_format",)}
            for error in exc.errors()
        )
        raise VoicePipelineException(
            code=(
                VoiceErrorCode.UNSUPPORTED_AUDIO_FORMAT
                if audio_format_error
                else VoiceErrorCode.INVALID_STREAM_OPEN
            ),
            stage=VoiceStage.STREAM,
            message="STREAM_OPEN Schema 또는 지원 범위가 올바르지 않습니다.",
            retryable=False,
        ) from exc
    except (json.JSONDecodeError, TypeError) as exc:
        raise VoicePipelineException(
            code=VoiceErrorCode.INVALID_STREAM_OPEN,
            stage=VoiceStage.STREAM,
            message="STREAM_OPEN JSON이 올바르지 않습니다.",
            retryable=False,
        ) from exc


async def _receive_stream_frames(
    websocket: WebSocket,
    context: VoiceChannelStreamContext,
    sender_task: asyncio.Task[None],
) -> None:
    while True:
        message = await websocket.receive()
        if message.get("type") == "websocket.disconnect":
            return

        audio_bytes = message.get("bytes")
        if isinstance(audio_bytes, bytes):
            await context.feed_audio(audio_bytes)
            continue

        control_text = message.get("text")
        if not isinstance(control_text, str):
            raise VoicePipelineException(
                code=VoiceErrorCode.INVALID_EVENT_SCHEMA,
                stage=VoiceStage.STREAM,
                message="Control Frame은 JSON Text여야 합니다.",
                retryable=False,
            )
        if await _handle_stream_control(control_text, context, sender_task):
            return


async def _handle_stream_control(
    control_text: str,
    context: VoiceChannelStreamContext,
    sender_task: asyncio.Task[None],
) -> bool:
    try:
        control = json.loads(control_text)
        control_type = control.get("type") if isinstance(control, dict) else None
        if control_type == "STREAM_OPEN":
            raise VoicePipelineException(
                code=VoiceErrorCode.INVALID_STREAM_OPEN,
                stage=VoiceStage.STREAM,
                message="동일 Connection에서 STREAM_OPEN은 한 번만 허용됩니다.",
                retryable=False,
            )
        if control_type == "STREAM_FLUSH":
            flush = VoiceStreamFlush.model_validate(control)
            await context.flush(flush.reason)
            return False
        if control_type == "STREAM_CLOSE":
            close = VoiceStreamClose.model_validate(control)
            await context.close(close.reason)
            await _wait_for_sender(sender_task)
            return True
    except VoicePipelineException:
        raise
    except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
        raise VoicePipelineException(
            code=VoiceErrorCode.INVALID_EVENT_SCHEMA,
            stage=VoiceStage.STREAM,
            message="지원하지 않는 Voice Control Schema입니다.",
            retryable=False,
        ) from exc

    raise VoicePipelineException(
        code=VoiceErrorCode.INVALID_EVENT_SCHEMA,
        stage=VoiceStage.STREAM,
        message="지원하지 않는 Voice Control Type입니다.",
        retryable=False,
    )


async def _send_events(
    websocket: WebSocket,
    context: VoiceChannelStreamContext,
) -> None:
    while True:
        event = await context.next_event()
        payload = event.model_dump(by_alias=True, mode="json", exclude_none=True)
        await websocket.send_json(payload)
        if payload.get("type") == "STREAM_CLOSED":
            return


async def _wait_for_sender(sender_task: asyncio.Task[None]) -> None:
    try:
        await asyncio.wait_for(
            asyncio.shield(sender_task),
            timeout=settings.AI_VOICE_SHUTDOWN_GRACE_SECONDS,
        )
    except (TimeoutError, asyncio.TimeoutError):
        sender_task.cancel()
        await asyncio.gather(sender_task, return_exceptions=True)


async def _send_pre_open_failure(
    websocket: WebSocket,
    payload: dict[str, Any],
    error: VoicePipelineException,
) -> None:
    channel_value = payload.get("channel")
    try:
        channel = VoiceChannel(channel_value)
    except (TypeError, ValueError):
        channel = VoiceChannel.SELF
    session_id = str(payload.get("sessionId") or "unknown")[:100] or "unknown"
    target_value = payload.get("targetLanguage")
    target_language = (
        target_value
        if isinstance(target_value, str) and target_value in SUPPORTED_VOICE_LANGUAGES
        else None
    )
    event = VoicePipelineFailedEvent(
        event_id=str(uuid4()),
        session_id=session_id,
        channel=channel,
        target_language=target_language,
        error=error.as_error(),
    )
    await websocket.send_json(
        event.model_dump(by_alias=True, mode="json", exclude_none=True)
    )


def _is_authorized(provided_api_key: str | None) -> bool:
    expected_api_key = settings.SERVER_API_KEY
    if not expected_api_key or not provided_api_key:
        return False
    return secrets.compare_digest(provided_api_key, expected_api_key)
