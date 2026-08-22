from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.alias_generators import to_camel


SUPPORTED_VOICE_LANGUAGES = frozenset({"ko", "ja", "en"})


class VoiceCamelCaseModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        str_strip_whitespace=True,
        extra="forbid",
    )


class VoiceChannel(str, Enum):
    SELF = "SELF"
    REMOTE = "REMOTE"


class VoiceMode(str, Enum):
    MIC = "MIC"
    MEDIA = "MEDIA"
    MEETING = "MEETING"


class VoiceSourceLanguageMode(str, Enum):
    AUTO = "AUTO"
    MANUAL = "MANUAL"


class VoiceErrorCode(str, Enum):
    UNAUTHORIZED = "VOICE_AI_UNAUTHORIZED"
    INVALID_STREAM_OPEN = "VOICE_AI_INVALID_STREAM_OPEN"
    UNSUPPORTED_AUDIO_FORMAT = "VOICE_AI_UNSUPPORTED_AUDIO_FORMAT"
    INVALID_AUDIO_FRAME = "VOICE_AI_INVALID_AUDIO_FRAME"
    BACKPRESSURE = "VOICE_AI_BACKPRESSURE"
    MODEL_NOT_READY = "VOICE_AI_MODEL_NOT_READY"
    STT_TIMEOUT = "VOICE_AI_STT_TIMEOUT"
    STT_FAILED = "VOICE_AI_STT_FAILED"
    LANGUAGE_UNDETERMINED = "VOICE_AI_LANGUAGE_UNDETERMINED"
    TRANSLATION_TIMEOUT = "VOICE_AI_TRANSLATION_TIMEOUT"
    TRANSLATION_FAILED = "VOICE_AI_TRANSLATION_FAILED"
    READING_WARNING = "VOICE_AI_READING_WARNING"
    INVALID_EVENT_SCHEMA = "VOICE_AI_INVALID_EVENT_SCHEMA"
    INTERNAL_ERROR = "VOICE_AI_INTERNAL_ERROR"


class VoiceStage(str, Enum):
    AUTH = "AUTH"
    STREAM = "STREAM"
    AUDIO = "AUDIO"
    VAD = "VAD"
    STT = "STT"
    LANGUAGE = "LANGUAGE"
    TRANSLATION = "TRANSLATION"
    READING = "READING"
    RUNTIME = "RUNTIME"


class VoiceAudioFormat(VoiceCamelCaseModel):
    encoding: Literal["PCM_S16LE"] = "PCM_S16LE"
    sample_rate: Literal[16000] = 16000
    channels: Literal[1] = 1
    frame_duration_ms: int = Field(default=100, ge=20, le=200)


class VoiceStreamPolicy(VoiceCamelCaseModel):
    endpointing_silence_ms: int = Field(default=300, ge=250, le=350)
    min_utterance_duration_ms: int = Field(default=250, ge=100, le=1000)
    max_utterance_duration_ms: int = Field(default=10_000, ge=1000, le=10_000)
    language_lock_confidence: float = Field(default=0.80, ge=0, le=1)
    language_switch_confidence: float = Field(default=0.85, ge=0, le=1)
    language_switch_consecutive_count: int = Field(default=3, ge=1, le=10)

    @model_validator(mode="after")
    def validate_utterance_range(self) -> "VoiceStreamPolicy":
        if self.min_utterance_duration_ms >= self.max_utterance_duration_ms:
            raise ValueError(
                "minUtteranceDurationMs must be smaller than maxUtteranceDurationMs"
            )
        return self


class VoiceStreamOpen(VoiceCamelCaseModel):
    type: Literal["STREAM_OPEN"]
    request_id: str = Field(..., min_length=1, max_length=100)
    session_id: str = Field(..., min_length=1, max_length=100)
    channel: VoiceChannel
    mode: VoiceMode
    source_language_mode: VoiceSourceLanguageMode = VoiceSourceLanguageMode.AUTO
    manual_source_language: str | None = Field(default=None, min_length=2, max_length=3)
    last_locked_language: str | None = Field(default=None, min_length=2, max_length=3)
    target_language: str = Field(..., min_length=2, max_length=3)
    audio_format: VoiceAudioFormat
    policy: VoiceStreamPolicy = Field(default_factory=VoiceStreamPolicy)

    @model_validator(mode="after")
    def validate_languages(self) -> "VoiceStreamOpen":
        if self.target_language not in SUPPORTED_VOICE_LANGUAGES:
            raise ValueError("targetLanguage must be one of ko, ja, en")

        if self.source_language_mode == VoiceSourceLanguageMode.MANUAL:
            if self.manual_source_language not in SUPPORTED_VOICE_LANGUAGES:
                raise ValueError(
                    "manualSourceLanguage must be one of ko, ja, en in MANUAL mode"
                )
        elif self.manual_source_language is not None:
            raise ValueError("manualSourceLanguage is only allowed in MANUAL mode")
        if (
            self.last_locked_language is not None
            and self.last_locked_language not in SUPPORTED_VOICE_LANGUAGES
        ):
            raise ValueError("lastLockedLanguage must be one of ko, ja, en")
        return self


class VoiceStreamFlush(VoiceCamelCaseModel):
    type: Literal["STREAM_FLUSH"]
    reason: str = Field(default="REQUESTED", min_length=1, max_length=100)


class VoiceStreamClose(VoiceCamelCaseModel):
    type: Literal["STREAM_CLOSE"]
    reason: str = Field(default="REQUESTED", min_length=1, max_length=100)


class VoiceReadingToken(VoiceCamelCaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        str_strip_whitespace=False,
        extra="forbid",
    )

    surface: str = Field(..., min_length=1, max_length=500)
    reading: str = Field(..., min_length=1, max_length=500)


class VoiceWarning(VoiceCamelCaseModel):
    code: VoiceErrorCode
    stage: VoiceStage
    message: str = Field(..., min_length=1, max_length=500)


class VoiceError(VoiceCamelCaseModel):
    code: VoiceErrorCode
    stage: VoiceStage
    message: str = Field(..., min_length=1, max_length=500)
    retryable: bool


class VoiceLatency(VoiceCamelCaseModel):
    endpointing_ms: int = Field(default=0, ge=0)
    speech_evidence_ms: int = Field(default=0, ge=0)
    partial_inference_ms: int | None = Field(default=None, ge=0)
    stt_finalize_ms: int = Field(default=0, ge=0)
    translation_ms: int = Field(default=0, ge=0)
    ai_total_after_speech_ms: int = Field(default=0, ge=0)


class VoiceModelMetadata(VoiceCamelCaseModel):
    stt_version: str = Field(..., min_length=1, max_length=200)
    translation_version: str = Field(..., min_length=1, max_length=200)
    prompt_version: str = Field(..., min_length=1, max_length=100)


class VoiceEventBase(VoiceCamelCaseModel):
    event_id: str = Field(..., min_length=1, max_length=100)
    session_id: str = Field(..., min_length=1, max_length=100)
    channel: VoiceChannel


class VoiceUtteranceEventBase(VoiceEventBase):
    utterance_key: str = Field(..., min_length=1, max_length=120)
    utterance_sequence: int = Field(..., ge=1)


class VoiceStreamReadyEvent(VoiceEventBase):
    type: Literal["STREAM_READY"] = "STREAM_READY"
    request_id: str = Field(..., min_length=1, max_length=100)
    model_ready: bool = True


class VoiceSpeechStartedEvent(VoiceUtteranceEventBase):
    type: Literal["SPEECH_STARTED"] = "SPEECH_STARTED"
    started_at_offset_ms: int = Field(..., ge=0)


class VoiceTranscriptPartialEvent(VoiceUtteranceEventBase):
    type: Literal["TRANSCRIPT_PARTIAL"] = "TRANSCRIPT_PARTIAL"
    revision: int = Field(..., ge=1)
    source_text: str = Field(..., min_length=1, max_length=10_000)
    detected_language: str | None = Field(default=None, min_length=2, max_length=3)
    language_confidence: float | None = Field(default=None, ge=0, le=1)
    started_at_offset_ms: int = Field(..., ge=0)
    ended_at_offset_ms: int = Field(..., ge=0)


class VoiceFinalEventBase(VoiceUtteranceEventBase):
    source_text: str = Field(..., min_length=1, max_length=10_000)
    detected_language: str | None = Field(default=None, min_length=2, max_length=3)
    language_confidence: float | None = Field(default=None, ge=0, le=1)
    locked_language: str | None = Field(default=None, min_length=2, max_length=3)
    started_at_offset_ms: int = Field(..., ge=0)
    ended_at_offset_ms: int = Field(..., ge=0)
    speech_duration_ms: int = Field(..., ge=0)
    no_speech_probability: float | None = Field(default=None, ge=0, le=1)


class VoiceTranscriptFinalEvent(VoiceFinalEventBase):
    type: Literal["TRANSCRIPT_FINAL"] = "TRANSCRIPT_FINAL"


class VoicePipelineCompletedEvent(VoiceFinalEventBase):
    type: Literal["VOICE_PIPELINE_COMPLETED"] = "VOICE_PIPELINE_COMPLETED"
    target_language: str = Field(..., min_length=2, max_length=3)
    translated_text: str = Field(..., min_length=1, max_length=10_000)
    translation_skipped: bool
    source_reading_tokens: list[VoiceReadingToken] = Field(default_factory=list)
    warnings: list[VoiceWarning] = Field(default_factory=list)
    latency: VoiceLatency
    model: VoiceModelMetadata


class VoicePipelineFailedEvent(VoiceEventBase):
    type: Literal["VOICE_PIPELINE_FAILED"] = "VOICE_PIPELINE_FAILED"
    utterance_key: str | None = Field(default=None, max_length=120)
    utterance_sequence: int | None = Field(default=None, ge=1)
    source_text: str | None = Field(default=None, max_length=10_000)
    detected_language: str | None = Field(default=None, min_length=2, max_length=3)
    language_confidence: float | None = Field(default=None, ge=0, le=1)
    locked_language: str | None = Field(default=None, min_length=2, max_length=3)
    target_language: str | None = Field(default=None, min_length=2, max_length=3)
    error: VoiceError


class VoiceNoSpeechEvent(VoiceEventBase):
    type: Literal["NO_SPEECH"] = "NO_SPEECH"
    utterance_key: str | None = Field(default=None, max_length=120)
    utterance_sequence: int | None = Field(default=None, ge=1)
    started_at_offset_ms: int = Field(..., ge=0)
    ended_at_offset_ms: int = Field(..., ge=0)
    speech_duration_ms: Literal[0] = 0


class VoiceBackpressureEvent(VoiceEventBase):
    type: Literal["BACKPRESSURE"] = "BACKPRESSURE"
    utterance_key: str | None = Field(default=None, max_length=120)
    utterance_sequence: int | None = Field(default=None, ge=1)
    error: VoiceError
    buffered_audio_ms: int = Field(..., ge=0)
    retry_after_ms: int = Field(..., ge=0)


class VoiceStreamClosedEvent(VoiceEventBase):
    type: Literal["STREAM_CLOSED"] = "STREAM_CLOSED"
    reason: str = Field(..., min_length=1, max_length=100)


class VoiceTranslationProviderPayload(VoiceCamelCaseModel):
    translated_text: str = Field(..., min_length=1, max_length=10_000)
    source_reading_tokens: list[VoiceReadingToken] = Field(
        default_factory=list,
        max_length=2000,
    )


class VoiceTranslationRetryRequest(VoiceCamelCaseModel):
    request_id: str = Field(..., min_length=1, max_length=100)
    session_id: str = Field(..., min_length=1, max_length=100)
    segment_id: int = Field(..., ge=1)
    source_text: str = Field(..., min_length=1, max_length=10_000)
    source_language: str = Field(..., min_length=2, max_length=3)
    target_language: str = Field(..., min_length=2, max_length=3)

    @model_validator(mode="after")
    def validate_supported_languages(self) -> "VoiceTranslationRetryRequest":
        if self.source_language not in SUPPORTED_VOICE_LANGUAGES:
            raise ValueError("sourceLanguage must be one of ko, ja, en")
        if self.target_language not in SUPPORTED_VOICE_LANGUAGES:
            raise ValueError("targetLanguage must be one of ko, ja, en")
        return self


class VoiceTranslationRetryResponse(VoiceCamelCaseModel):
    request_id: str
    session_id: str
    segment_id: int
    source_text: str
    source_language: str
    target_language: str
    translated_text: str
    translation_skipped: bool
    source_reading_tokens: list[VoiceReadingToken] = Field(default_factory=list)
    warnings: list[VoiceWarning] = Field(default_factory=list)
    latency_ms: int = Field(default=0, ge=0)
    model: VoiceModelMetadata


class VoiceReadinessResponse(VoiceCamelCaseModel):
    ready: bool
    accepting_streams: bool
    stt_model: str
    translation_model: str
    active_streams: int = Field(default=0, ge=0)
