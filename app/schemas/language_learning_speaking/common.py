from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class CamelCaseModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        str_strip_whitespace=True,
        extra="forbid",
    )


class SpeakingPracticeMode(str, Enum):
    READ_ALOUD = "READ_ALOUD"
    GUIDED = "GUIDED"
    FREE = "FREE"


class ConversationStartMode(str, Enum):
    AI_FIRST = "AI_FIRST"
    USER_FIRST = "USER_FIRST"
    TOPIC_RECOMMENDED = "TOPIC_RECOMMENDED"


class CorrectionMode(str, Enum):
    CONVERSATION = "CONVERSATION"
    COACHING = "COACHING"


class AssistanceType(str, Enum):
    REPLAY = "REPLAY"
    SLOW_PLAYBACK = "SLOW_PLAYBACK"
    SHOW_QUESTION = "SHOW_QUESTION"
    HINT = "HINT"
    TRANSLATION = "TRANSLATION"
    SAMPLE_ANSWER = "SAMPLE_ANSWER"


class AssistanceLevel(str, Enum):
    NONE = "NONE"
    ASSISTED = "ASSISTED"
    GUIDED = "GUIDED"


class SpeakingMetricType(str, Enum):
    GRAMMAR = "GRAMMAR"
    VOCABULARY = "VOCABULARY"
    NATURALNESS = "NATURALNESS"
    MEANING = "MEANING"
    EXPRESSIVENESS = "EXPRESSIVENESS"
    FLUENCY = "FLUENCY"
    PRONUNCIATION = "PRONUNCIATION"
    INTERACTION = "INTERACTION"


class MetricEvaluationState(str, Enum):
    EVALUATED = "EVALUATED"
    NOT_EVALUABLE = "NOT_EVALUABLE"


class SpeakingEvaluationStatus(str, Enum):
    EVALUATED = "EVALUATED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class SpeakingStage(str, Enum):
    AUDIO_VALIDATION = "AUDIO_VALIDATION"
    STT = "STT"
    CONVERSATION = "CONVERSATION"
    TTS = "TTS"
    EVALUATION = "EVALUATION"


class SpeakingErrorCode(str, Enum):
    INVALID_AUDIO = "INVALID_AUDIO"
    SILENCE_DETECTED = "SILENCE_DETECTED"
    UNSUPPORTED_AUDIO_FORMAT = "UNSUPPORTED_AUDIO_FORMAT"
    AUDIO_TOO_SHORT = "AUDIO_TOO_SHORT"
    AUDIO_TOO_LONG = "AUDIO_TOO_LONG"
    AUDIO_TOO_LARGE = "AUDIO_TOO_LARGE"
    STT_FAILED = "STT_FAILED"
    CONVERSATION_GENERATION_FAILED = "CONVERSATION_GENERATION_FAILED"
    TTS_FAILED = "TTS_FAILED"
    EVALUATION_FAILED = "EVALUATION_FAILED"
    PROVIDER_RATE_LIMITED = "PROVIDER_RATE_LIMITED"
    PROVIDER_TIMEOUT = "PROVIDER_TIMEOUT"
    UNSAFE_TOPIC = "UNSAFE_TOPIC"
    INVALID_RESPONSE_SCHEMA = "INVALID_RESPONSE_SCHEMA"
    MANUAL_RETRY_LIMIT_EXCEEDED = "MANUAL_RETRY_LIMIT_EXCEEDED"


class AssistanceUsage(CamelCaseModel):
    type: AssistanceType
    count: int = Field(default=1, ge=1, le=100)


class ConversationMessage(CamelCaseModel):
    role: str = Field(..., pattern="^(USER|ASSISTANT)$")
    text: str = Field(..., min_length=1, max_length=4000)
    turn_id: str | None = Field(default=None, max_length=100)


class StageUsage(CamelCaseModel):
    latency_ms: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    audio_seconds: float = Field(default=0, ge=0)
    tts_characters: int = Field(default=0, ge=0)
    tts_audio_seconds: float = Field(default=0, ge=0)
    provider: str | None = None
    model: str | None = None
    prompt_version: str | None = None
    evaluation_version: str | None = None


class SpeakingUsage(CamelCaseModel):
    stt: StageUsage | None = None
    conversation: StageUsage | None = None
    tts: StageUsage | None = None
    evaluation: StageUsage | None = None


class SpeakingError(CamelCaseModel):
    code: SpeakingErrorCode
    stage: SpeakingStage
    message: str
    retryable: bool
