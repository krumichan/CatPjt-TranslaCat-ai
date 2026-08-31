from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class CamelCaseModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        str_strip_whitespace=True,
        extra="forbid",
    )


class ListeningDifficulty(str, Enum):
    EASY = "EASY"
    MY_LEVEL = "MY_LEVEL"
    CHALLENGE = "CHALLENGE"


class ListeningTaskType(str, Enum):
    DICTATION = "DICTATION"
    INTERPRETATION = "INTERPRETATION"
    REPEAT_AFTER_AUDIO = "REPEAT_AFTER_AUDIO"


class EvaluationPurpose(str, Enum):
    OFFICIAL = "OFFICIAL"
    PRACTICE = "PRACTICE"


class ListeningAssistanceType(str, Enum):
    REPLAY = "REPLAY"
    SLOW_PLAYBACK = "SLOW_PLAYBACK"
    TOPIC_HINT = "TOPIC_HINT"
    KEYWORD_HINT = "KEYWORD_HINT"
    SHOW_ANSWER = "SHOW_ANSWER"


class ListeningAssistanceLevel(str, Enum):
    INDEPENDENT = "INDEPENDENT"
    ASSISTED = "ASSISTED"
    GUIDED = "GUIDED"


class ListeningTaskStatus(str, Enum):
    EVALUATED = "EVALUATED"
    NOT_EVALUABLE = "NOT_EVALUABLE"
    NOT_SELECTED = "NOT_SELECTED"


class MetricEvaluationState(str, Enum):
    EVALUATED = "EVALUATED"
    NOT_EVALUABLE = "NOT_EVALUABLE"


class ListeningStage(str, Enum):
    GENERATION = "GENERATION"
    TTS = "TTS"
    DICTATION = "DICTATION"
    INTERPRETATION = "INTERPRETATION"
    AUDIO_VALIDATION = "AUDIO_VALIDATION"
    STT = "STT"
    ALIGNMENT = "ALIGNMENT"
    PRONUNCIATION = "PRONUNCIATION"
    EXPLANATION = "EXPLANATION"


class ListeningErrorCode(str, Enum):
    INVALID_REQUEST = "INVALID_REQUEST"
    INVALID_RESPONSE_SCHEMA = "INVALID_RESPONSE_SCHEMA"
    GENERATION_FAILED = "GENERATION_FAILED"
    DUPLICATE_CONTENT = "DUPLICATE_CONTENT"
    CONTENT_DIVERSITY_EXHAUSTED = "CONTENT_DIVERSITY_EXHAUSTED"
    UNSAFE_CONTENT = "UNSAFE_CONTENT"
    TTS_FAILED = "TTS_FAILED"
    AUDIO_DECODE_FAILED = "AUDIO_DECODE_FAILED"
    AUDIO_TOO_SHORT = "AUDIO_TOO_SHORT"
    AUDIO_TOO_LONG = "AUDIO_TOO_LONG"
    AUDIO_TOO_LARGE = "AUDIO_TOO_LARGE"
    SILENCE = "SILENCE"
    NON_SPEECH = "NON_SPEECH"
    LANGUAGE_MISMATCH = "LANGUAGE_MISMATCH"
    ALIGNMENT_INSUFFICIENT = "ALIGNMENT_INSUFFICIENT"
    LOW_AUDIO_QUALITY = "LOW_AUDIO_QUALITY"
    STT_FAILED = "STT_FAILED"
    INTERPRETATION_FAILED = "INTERPRETATION_FAILED"
    EXPLANATION_FAILED = "EXPLANATION_FAILED"
    ANSWER_REVEALED = "ANSWER_REVEALED"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    PROVIDER_RATE_LIMITED = "PROVIDER_RATE_LIMITED"
    PROVIDER_TIMEOUT = "PROVIDER_TIMEOUT"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    MANUAL_RETRY_LIMIT_EXCEEDED = "MANUAL_RETRY_LIMIT_EXCEEDED"


class AssistanceUsage(CamelCaseModel):
    type: ListeningAssistanceType
    count: int = Field(default=1, ge=1, le=100)


class StageUsage(CamelCaseModel):
    latency_ms: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    audio_seconds: float = Field(default=0, ge=0)
    tts_characters: int = Field(default=0, ge=0)
    tts_audio_seconds: float = Field(default=0, ge=0)
    provider: str | None = None
    model: str | None = None
    provider_version: str | None = None
    prompt_version: str | None = None
    evaluation_version: str | None = None


class ListeningUsage(CamelCaseModel):
    generation: StageUsage | None = None
    tts: StageUsage | None = None
    dictation: StageUsage | None = None
    interpretation: StageUsage | None = None
    stt: StageUsage | None = None
    alignment: StageUsage | None = None
    pronunciation: StageUsage | None = None
    explanation: StageUsage | None = None


class ListeningError(CamelCaseModel):
    code: ListeningErrorCode
    failed_stage: ListeningStage
    message: str
    retryable: bool
    details: dict[str, Any] = Field(default_factory=dict)


class SafetyResult(CamelCaseModel):
    passed: bool
    categories: list[str] = Field(default_factory=list, max_length=30)
