import asyncio
import base64
import binascii
import io
import json
import wave
from typing import Annotated, overload

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ValidationError

from app.api.dependencies import get_speaking_speech_runtime, get_speech_runtime
from app.features.speech_execution.audio import AudioDecodeError, AudioDecoder
from app.features.speech_execution.errors import provider_failure
from app.features.speech_to_text import FasterWhisperRuntime, InferencePriority
from app.features.voice_translation.speech_detector import SileroSpeechEvidenceGuard
from app.schemas.speech_transcription import (
    AudioDecodeCommand,
    AudioDecodeResponse,
    SpeechTranscriptionCommand,
    SpeechTranscriptionResponse,
    TranscriptionSegment,
)

router = APIRouter(prefix="/speech", tags=["Internal Speech Execution"])
_evidence_runtime = SileroSpeechEvidenceGuard(enabled=True)


def get_speech_evidence_runtime() -> SileroSpeechEvidenceGuard:
    return _evidence_runtime


@overload
async def _command(
    request: Request, schema: type[SpeechTranscriptionCommand]
) -> SpeechTranscriptionCommand: ...


@overload
async def _command(request: Request, schema: type[AudioDecodeCommand]) -> AudioDecodeCommand: ...


async def _command(request: Request, schema: type[BaseModel]) -> BaseModel:
    # 프레임과 prompt의 입력 크기를 먼저 제한하고 오류에는 원문을 노출하지 않는다.
    data = bytearray()
    async for chunk in request.stream():
        if len(data) + len(chunk) > 14_100_000:
            raise HTTPException(
                413, detail={"code": "SPEECH_REQUEST_TOO_LARGE", "retryable": False}
            )
        data.extend(chunk)
    try:
        return schema.model_validate(json.loads(data))
    except (ValidationError, ValueError, TypeError, UnicodeDecodeError, RecursionError):
        raise HTTPException(
            422, detail={"code": "SPEECH_REQUEST_INVALID", "retryable": False}
        ) from None


def _bytes(encoded: str) -> bytes:
    try:
        value = base64.b64decode(encoded, validate=True)
        if not value or len(value) > 10 * 1024 * 1024:
            raise ValueError
        return value
    except (ValueError, binascii.Error):
        raise HTTPException(
            422, detail={"code": "AUDIO_BYTES_INVALID", "retryable": False}
        ) from None


@router.post("/normalize", response_model=AudioDecodeResponse)
async def normalize(request: Request) -> AudioDecodeResponse:
    command = await _command(request, AudioDecodeCommand)
    data = _bytes(command.audio_base64)
    try:
        async with asyncio.timeout(command.remaining_milliseconds / 1000.0):
            result = await asyncio.to_thread(AudioDecoder().decode, data)
    except TimeoutError:
        raise HTTPException(
            504, detail={"code": "AUDIO_DECODE_TIMEOUT", "retryable": True}
        ) from None
    except AudioDecodeError as error:
        raise HTTPException(422, detail={"code": str(error), "retryable": False}) from None
    return AudioDecodeResponse(
        audioBase64=base64.b64encode(result.wav_bytes).decode("ascii"),
        durationSeconds=result.duration_seconds,
        sourceFormat=result.source_format,
        rms=result.rms,
        peak=result.peak,
        silenceRatio=result.silence_ratio,
        sampleRate=result.sample_rate,
        channels=result.channels,
    )


@router.post("/evidence")
async def evidence(
    request: Request,
    runtime: Annotated[SileroSpeechEvidenceGuard, Depends(get_speech_evidence_runtime)],
) -> dict[str, bool]:
    command = await _command(request, AudioDecodeCommand)
    data = _bytes(command.audio_base64)
    try:
        # 형식이 확인된 PCM만 기술적 음성 탐지기에 전달한다. STT 재호출은 LL이 결정한다.
        with wave.open(io.BytesIO(data), "rb") as reader:
            if (
                reader.getnchannels() != 1
                or reader.getsampwidth() != 2
                or reader.getframerate() != 16000
                or reader.getcomptype() != "NONE"
            ):
                raise ValueError
            frames = reader.getnframes()
            pcm = reader.readframes(frames)
            if len(pcm) != frames * 2:
                raise ValueError
    except (ValueError, EOFError, wave.Error):
        raise HTTPException(
            422, detail={"code": "AUDIO_FORMAT_INVALID", "retryable": False}
        ) from None
    try:
        async with asyncio.timeout(command.remaining_milliseconds / 1000.0):
            if not runtime.ready:
                await runtime.warm_up()
            return {"hasSpeech": await runtime.has_speech(pcm)}
    except TimeoutError:
        raise HTTPException(
            504, detail={"code": "SPEECH_EVIDENCE_TIMEOUT", "retryable": True}
        ) from None
    except Exception:
        raise HTTPException(
            503, detail={"code": "SPEECH_EVIDENCE_UNAVAILABLE", "retryable": True}
        ) from None


@router.post("/transcribe", response_model=SpeechTranscriptionResponse)
async def transcribe(
    request: Request,
    shared: Annotated[FasterWhisperRuntime, Depends(get_speech_runtime)],
    accurate: Annotated[FasterWhisperRuntime, Depends(get_speaking_speech_runtime)],
) -> SpeechTranscriptionResponse:
    command = await _command(request, SpeechTranscriptionCommand)
    data = _bytes(command.audio_base64)
    runtime = shared if command.runtime == "shared" else accurate

    # 호출자가 명시한 기술 옵션만 전달한다. 발화 언어·학습 일치도·점수는 여기서 판정하지 않는다.
    try:
        async with asyncio.timeout(command.remaining_milliseconds / 1000.0):
            if not runtime.ready:
                await runtime.warm_up()
            result = await runtime.transcribe(
                io.BytesIO(data),
                options={
                    "beam_size": command.beam_size,
                    "language": command.language,
                    "initial_prompt": command.initial_prompt,
                    "vad_filter": command.vad_filter,
                    "vad_parameters": {"min_silence_duration_ms": command.min_silence_duration_ms},
                    "condition_on_previous_text": command.condition_on_previous_text,
                },
                priority=InferencePriority.STANDARD,
            )
    except Exception as error:
        raise provider_failure(error) from None

    # 실제 전사와 기술적 근거를 그대로 전달하고 transcript를 로그에 남기지 않는다.
    return SpeechTranscriptionResponse(
        text=result.text,
        language=result.language,
        languageProbability=result.language_probability,
        durationSeconds=result.duration_seconds,
        provider=result.provider,
        model=result.model,
        modelVersion=result.model_version,
        segments=[
            TranscriptionSegment(
                startSeconds=segment.start_seconds,
                endSeconds=segment.end_seconds,
                text=segment.text,
                avgLogprob=segment.avg_logprob,
                noSpeechProbability=segment.no_speech_probability,
            )
            for segment in result.segments
        ],
    )
