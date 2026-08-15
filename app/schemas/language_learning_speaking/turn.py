from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from app.schemas.language_learning import LearningProfileSummary, SelectedKeyword
from app.schemas.language_learning_speaking.common import (
    AssistanceLevel,
    AssistanceUsage,
    CamelCaseModel,
    ConversationMessage,
    ConversationStartMode,
    CorrectionMode,
    SpeakingError,
    SpeakingUsage,
)


class SessionPolicySnapshot(CamelCaseModel):
    max_session_minutes: int = Field(default=10, ge=1, le=10)
    max_turns: int = Field(default=20, ge=1, le=20)
    min_valid_audio_seconds: float = Field(default=1, ge=0.1, le=10)
    max_turn_audio_seconds: float = Field(default=60, ge=1, le=60)
    max_audio_file_bytes: int = Field(
        default=10 * 1024 * 1024,
        ge=1024,
        le=10 * 1024 * 1024,
    )
    automatic_retry_limit_per_stage: int = Field(default=2, ge=0, le=2)
    manual_retry_limit_per_stage: int = Field(default=1, ge=0, le=1)


class SpeakingSessionContext(CamelCaseModel):
    request_id: str = Field(..., min_length=1, max_length=100)
    idempotency_key: str = Field(..., min_length=1, max_length=200)
    session_id: str = Field(..., min_length=1, max_length=100)
    turn_index: int = Field(..., ge=0, le=20)
    origin_language: str = Field(..., min_length=2, max_length=20)
    learning_language: str = Field(..., min_length=2, max_length=20)
    topic: str = Field(..., min_length=1, max_length=500)
    category: str | None = Field(default=None, max_length=100)
    goal: str | None = Field(default=None, max_length=1000)
    persona: str | None = Field(default=None, max_length=1000)
    conversation_start_mode: ConversationStartMode
    topic_recommended_start_mode: ConversationStartMode | None = None
    correction_mode: CorrectionMode = CorrectionMode.CONVERSATION
    target_level: str | None = Field(default=None, max_length=50)
    learning_profile_summary: LearningProfileSummary | None = None
    selected_keywords: list[SelectedKeyword] = Field(
        default_factory=list,
        max_length=20,
    )
    focus_signals: list[str] = Field(default_factory=list, max_length=30)
    conversation_history: list[ConversationMessage] = Field(
        default_factory=list,
        max_length=100,
    )
    assistance_usage: list[AssistanceUsage] = Field(
        default_factory=list,
        max_length=20,
    )
    session_summary: str | None = Field(default=None, max_length=4000)
    session_elapsed_seconds: float = Field(default=0, ge=0, le=600)
    session_policy_snapshot: SessionPolicySnapshot = Field(
        default_factory=SessionPolicySnapshot
    )
    audio_reference: str | None = Field(default=None, max_length=500)
    audio_format: str | None = Field(default=None, max_length=100)
    duration_seconds: float | None = Field(default=None, ge=0, le=60)
    voice: str = Field(default="Kore", min_length=1, max_length=100)
    playback_speed: str = Field(default="NORMAL", pattern="^(SLOW|NORMAL)$")
    manual_retry_attempt: int = Field(default=0, ge=0, le=1)

    @model_validator(mode="after")
    def validate_recommended_mode(self) -> "SpeakingSessionContext":
        if (
            self.conversation_start_mode == ConversationStartMode.TOPIC_RECOMMENDED
            and self.topic_recommended_start_mode is None
        ):
            raise ValueError(
                "TOPIC_RECOMMENDED에는 topicRecommendedStartMode가 필요합니다."
            )
        return self


class AudioQualitySignals(CamelCaseModel):
    rms: float = Field(..., ge=0)
    peak: float = Field(..., ge=0)
    silence_ratio: float = Field(..., ge=0, le=1)
    sample_rate: int = Field(..., ge=1)
    channels: int = Field(..., ge=1)


class SttSegment(CamelCaseModel):
    start_ms: int = Field(..., ge=0)
    end_ms: int = Field(..., ge=0)
    text: str
    confidence: float = Field(..., ge=0, le=1)

    @model_validator(mode="after")
    def validate_timestamps(self) -> "SttSegment":
        if self.end_ms < self.start_ms:
            raise ValueError("STT Segment endMs는 startMs 이상이어야 합니다.")
        return self


class SttAnalysisMetadata(CamelCaseModel):
    provider: str
    model: str
    model_version: str | None = None
    detected_language: str | None = None
    requested_language: str
    low_confidence_threshold: float = Field(..., ge=0, le=1)
    audio_duration: float = Field(..., ge=0)
    audio_quality_signals: AudioQualitySignals
    normalization_version: str
    stt_hint_version: str


class TranscriptResult(CamelCaseModel):
    text: str
    language: str
    confidence: float = Field(..., ge=0, le=1)
    is_low_confidence: bool = False
    segments: list[SttSegment] = Field(default_factory=list)
    metadata: SttAnalysisMetadata


class CoachingCorrection(CamelCaseModel):
    original: str = Field(..., min_length=1, max_length=2000)
    improved: str = Field(..., min_length=1, max_length=2000)
    explanation: str = Field(..., min_length=1, max_length=2000)
    improvement_link: str | None = Field(default=None, max_length=100)


class ConversationPayload(CamelCaseModel):
    assistant_text: str = Field(..., min_length=1, max_length=4000)
    intent: str = Field(..., min_length=1, max_length=200)
    difficulty: str = Field(..., min_length=1, max_length=100)
    should_end: bool = False
    end_reason: str | None = Field(default=None, max_length=500)
    hint: str | None = Field(default=None, max_length=1000)
    coaching_corrections: list[CoachingCorrection] = Field(default_factory=list, max_length=10)
    session_summary: str | None = Field(default=None, max_length=4000)


class AssistantAudio(CamelCaseModel):
    audio_reference: str
    content_type: str = "audio/wav"
    voice: str
    cache_key: str
    duration_seconds: float | None = Field(default=None, ge=0)
    status: str = Field(default="READY", pattern="^(READY|FAILED)$")


class AssistantTurn(CamelCaseModel):
    text: str
    voice: str
    audio: AssistantAudio | None = None
    tts_error: SpeakingError | None = None


class ConversationResult(CamelCaseModel):
    intent: str
    difficulty: str
    should_end: bool
    end_reason: str | None = None
    hint: str | None = None
    coaching_corrections: list[CoachingCorrection] = Field(default_factory=list)
    session_summary: str | None = None
    assistance_level: AssistanceLevel = AssistanceLevel.NONE


class SessionStartRequest(SpeakingSessionContext):
    turn_index: int = 0


class SessionStartResponse(CamelCaseModel):
    request_id: str
    session_id: str
    resolved_start_mode: ConversationStartMode
    assistant: AssistantTurn | None
    conversation: ConversationResult | None
    usage: SpeakingUsage


class SttRequestContext(CamelCaseModel):
    request_id: str = Field(..., min_length=1, max_length=100)
    idempotency_key: str = Field(..., min_length=1, max_length=200)
    session_id: str = Field(..., min_length=1, max_length=100)
    turn_index: int = Field(..., ge=1, le=20)
    learning_language: str = Field(..., min_length=2, max_length=20)
    phrase_hints: list[str] = Field(default_factory=list, max_length=30)
    audio_reference: str | None = Field(default=None, max_length=500)
    audio_format: str | None = Field(default=None, max_length=100)
    duration_seconds: float | None = Field(default=None, ge=0, le=60)
    session_policy_snapshot: SessionPolicySnapshot = Field(
        default_factory=SessionPolicySnapshot
    )
    manual_retry_attempt: int = Field(default=0, ge=0, le=1)


class SttResponse(CamelCaseModel):
    request_id: str
    session_id: str
    turn_index: int
    transcript: TranscriptResult
    usage: SpeakingUsage


class ConversationGenerationRequest(SpeakingSessionContext):
    transcript: TranscriptResult | None = None
    is_initial_turn: bool = False


class ConversationGenerationResponse(CamelCaseModel):
    request_id: str
    session_id: str
    turn_index: int
    assistant_text: str
    conversation: ConversationResult
    usage: SpeakingUsage


class TtsRequest(CamelCaseModel):
    request_id: str = Field(..., min_length=1, max_length=100)
    idempotency_key: str = Field(..., min_length=1, max_length=200)
    session_id: str = Field(..., min_length=1, max_length=100)
    text: str = Field(..., min_length=1, max_length=4000)
    learning_language: str = Field(..., min_length=2, max_length=20)
    voice: str = Field(default="Kore", min_length=1, max_length=100)
    playback_speed: str = Field(default="NORMAL", pattern="^(SLOW|NORMAL)$")
    automatic_retry_limit: int = Field(default=2, ge=0, le=2)
    manual_retry_attempt: int = Field(default=0, ge=0, le=1)


class TtsResponse(CamelCaseModel):
    request_id: str
    session_id: str
    audio: AssistantAudio
    usage: SpeakingUsage


class TurnProcessResponse(CamelCaseModel):
    request_id: str
    session_id: str
    turn_index: int
    status: str = Field(pattern="^(READY|PARTIAL_FAILURE)$")
    transcript: TranscriptResult | None = None
    assistant: AssistantTurn | None = None
    conversation: ConversationResult | None = None
    failed_stage: str | None = None
    error: SpeakingError | None = None
    usage: SpeakingUsage
    idempotent_replay: bool = False
    internal_metadata: dict[str, Any] = Field(default_factory=dict)
