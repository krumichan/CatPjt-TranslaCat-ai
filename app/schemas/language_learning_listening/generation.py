from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import Field, model_validator

from app.schemas.language_learning import SelectedKeyword
from app.schemas.language_learning_quality import (
    DiversityContext,
    DiversityMetadata,
    DiversitySummary,
    LanguageComplexityContext,
)
from app.schemas.language_learning_listening.common import (
    CamelCaseModel,
    ListeningDifficulty,
    ListeningLearningMode,
    ListeningError,
    ListeningUsage,
    SafetyResult,
)


class ListeningTopic(CamelCaseModel):
    id: int | str
    title: str = Field(..., min_length=1, max_length=200)


class ListeningUserContext(CamelCaseModel):
    origin_language: str = Field(..., min_length=2, max_length=20)
    learning_language: str = Field(..., min_length=2, max_length=20)
    level: ListeningDifficulty = ListeningDifficulty.MY_LEVEL
    profile_focus: list[str] = Field(default_factory=list, max_length=30)


class ListeningSetContext(CamelCaseModel):
    learning_date: date
    learning_mode: ListeningLearningMode = ListeningLearningMode.DICTATION
    topic: ListeningTopic
    selected_keywords: list[SelectedKeyword] = Field(
        default_factory=list, max_length=20
    )
    item_count: int = Field(default=5, ge=1, le=30)
    difficulty: ListeningDifficulty = ListeningDifficulty.MY_LEVEL


class ListeningGenerationConstraints(CamelCaseModel):
    audio_seconds_min: float | None = Field(default=None, ge=1, le=30)
    audio_seconds_max: float | None = Field(default=None, ge=1, le=30)
    recent_content_hashes: list[str] = Field(default_factory=list, max_length=200)
    recent_similarity_summaries: list[str] = Field(default_factory=list, max_length=200)

    @model_validator(mode="after")
    def validate_range(self) -> "ListeningGenerationConstraints":
        if (
            self.audio_seconds_min is not None
            and self.audio_seconds_max is not None
            and self.audio_seconds_min > self.audio_seconds_max
        ):
            raise ValueError("audioSecondsMin은 audioSecondsMax 이하여야 합니다.")
        return self


class ListeningSetGenerationRequest(CamelCaseModel):
    request_id: str = Field(..., min_length=1, max_length=100)
    idempotency_key: str = Field(..., min_length=1, max_length=200)
    user_context: ListeningUserContext
    set_context: ListeningSetContext
    constraints: ListeningGenerationConstraints = Field(
        default_factory=ListeningGenerationConstraints
    )
    policy_version: str = Field(default="listening", min_length=1, max_length=100)
    model_config_version: str = Field(
        default="listening-model-config", min_length=1, max_length=100
    )
    manual_retry_attempt: int = Field(default=0, ge=0, le=1)
    language_complexity: LanguageComplexityContext | None = None
    diversity_context: DiversityContext = Field(default_factory=DiversityContext)
    content_diversity_policy_version: str | None = Field(default=None, max_length=100)


ComprehensionFocus = Literal[
    "GIST",
    "DETAIL",
    "INTENT",
    "INFERENCE",
    "NEXT_ACTION",
]


class ListeningChoiceOption(CamelCaseModel):
    key: str = Field(..., min_length=1, max_length=12)
    text: str = Field(..., min_length=1, max_length=1000)


class GeneratedListeningItemPayload(CamelCaseModel):
    item_index: int = Field(..., ge=1, le=30)
    source_text: str = Field(..., min_length=1, max_length=4000)
    reference_meanings: list[str] = Field(..., min_length=2, max_length=3)
    key_meaning_units: list[str] = Field(..., min_length=1, max_length=30)
    target_keywords: list[str] = Field(default_factory=list, max_length=20)
    estimated_audio_seconds: float = Field(..., ge=1, le=60)
    safety: SafetyResult
    language_complexity_band: int | None = Field(default=None, ge=1, le=5)
    diversity_metadata: DiversityMetadata | None = None
    question: str | None = Field(default=None, max_length=2000)
    options: list[ListeningChoiceOption] = Field(default_factory=list, max_length=4)
    correct_option_key: str | None = Field(default=None, max_length=12)
    comprehension_focus: ComprehensionFocus | None = None
    summary_key_points: list[str] = Field(default_factory=list, max_length=8)


class ListeningGenerationPayload(CamelCaseModel):
    items: list[GeneratedListeningItemPayload] = Field(..., min_length=1, max_length=30)


class ListeningItem(CamelCaseModel):
    item_index: int
    source_text: str
    normalized_source_text: str
    reference_meanings: list[str]
    key_meaning_units: list[str]
    target_keywords: list[str]
    estimated_audio_seconds: float
    content_hash: str
    similarity_key: str
    safety: SafetyResult
    language_complexity_band: int | None = Field(default=None, ge=1, le=5)
    diversity_metadata: DiversityMetadata | None = None
    question: str | None = None
    options: list[ListeningChoiceOption] = Field(default_factory=list)
    correct_option_key: str | None = None
    comprehension_focus: ComprehensionFocus | None = None
    summary_key_points: list[str] = Field(default_factory=list)


class ListeningSetGenerationResponse(CamelCaseModel):
    request_id: str
    generation_version: str
    policy_version: str
    model_config_version: str
    items: list[ListeningItem]
    usage: ListeningUsage
    content_diversity_policy_version: str | None = None
    language_complexity_policy_version: str | None = None
    diversity_summary: DiversitySummary | None = None


class ListeningVoiceSnapshot(CamelCaseModel):
    locale: str = Field(..., min_length=2, max_length=30)
    voice_key: str = Field(..., min_length=1, max_length=100)
    version: str = Field(default="current", min_length=1, max_length=100)
    accent: str = Field(default="STANDARD", pattern="^STANDARD$")


class ListeningTtsRequest(CamelCaseModel):
    request_id: str = Field(..., min_length=1, max_length=100)
    idempotency_key: str = Field(..., min_length=1, max_length=200)
    item_id: int | str
    source_text: str = Field(..., min_length=1, max_length=4000)
    content_hash: str = Field(..., min_length=1, max_length=128)
    generation_version: str = Field(..., min_length=1, max_length=100)
    learning_language: str = Field(..., min_length=2, max_length=20)
    voice: ListeningVoiceSnapshot
    playback_speed: str = Field(default="NORMAL", pattern="^(NORMAL|SLOW)$")
    policy_version: str = Field(default="listening", min_length=1, max_length=100)
    model_config_version: str = Field(
        default="listening-model-config", min_length=1, max_length=100
    )
    automatic_retry_limit: int = Field(default=2, ge=0, le=2)
    manual_retry_attempt: int = Field(default=0, ge=0, le=1)


class ListeningAudio(CamelCaseModel):
    audio_reference: str
    duration_ms: int = Field(..., ge=0)
    format: str
    sample_rate: int = Field(default=24000, ge=1)
    channels: int = Field(default=1, ge=1)
    voice: ListeningVoiceSnapshot
    text_hash: str
    checksum: str
    cache_key: str
    tts_version: str


class ListeningTtsResponse(CamelCaseModel):
    request_id: str
    item_id: int | str
    status: str = Field(pattern="^(READY|FAILED)$")
    source_text: str
    content_hash: str
    generation_version: str
    audio: ListeningAudio | None = None
    error: ListeningError | None = None
    usage: ListeningUsage

    @model_validator(mode="after")
    def validate_status_payload(self) -> "ListeningTtsResponse":
        if self.status == "READY" and self.audio is None:
            raise ValueError("READY TTS 응답에는 audio가 필요합니다.")
        if self.status == "FAILED" and self.error is None:
            raise ValueError("FAILED TTS 응답에는 error가 필요합니다.")
        return self
