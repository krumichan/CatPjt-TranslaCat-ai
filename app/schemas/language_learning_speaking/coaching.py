from __future__ import annotations

from pydantic import Field, model_validator

from app.schemas.language_learning_speaking.common import (
    AssistanceUsage,
    CamelCaseModel,
    SpeakingCoachingContentStatus,
    SpeakingCoachingObservationKind,
    SpeakingResultKind,
    SpeakingUsage,
)
from app.schemas.language_learning_speaking.evaluation import SpeakingEvaluationRequest


class SpeakingCoachingRequest(SpeakingEvaluationRequest):
    result_kind: SpeakingResultKind = SpeakingResultKind.SESSION_COACHING
    result_policy_version: str = Field(
        default="free-session-coaching-v1", min_length=1, max_length=100
    )
    source_snapshot_hash: str = Field(..., min_length=16, max_length=128)

    @model_validator(mode="after")
    def validate_coaching_policy(self) -> "SpeakingCoachingRequest":
        if self.result_kind != SpeakingResultKind.SESSION_COACHING:
            raise ValueError("Coaching request requires SESSION_COACHING resultKind")
        if self.practice_mode.value != "FREE" or self.evaluation_scope != "SESSION":
            raise ValueError("Session coaching is available only for FREE sessions")
        return self


class SpeakingCoachingDraftItem(CamelCaseModel):
    observation_id: str = Field(..., min_length=1, max_length=100)
    kind: SpeakingCoachingObservationKind
    turn_id: str = Field(..., min_length=1, max_length=100)
    source_excerpt: str = Field(..., min_length=1, max_length=500)
    message: str = Field(..., min_length=1, max_length=1500)
    suggested_expression: str | None = Field(default=None, max_length=1000)


class SpeakingCoachingDraftPayload(CamelCaseModel):
    content_status: SpeakingCoachingContentStatus
    limitation_reasons: list[str] = Field(default_factory=list, max_length=10)
    items: list[SpeakingCoachingDraftItem] = Field(default_factory=list, max_length=3)

    @model_validator(mode="after")
    def validate_content_shape(self) -> "SpeakingCoachingDraftPayload":
        ids = [item.observation_id for item in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("Coaching observationId must be unique")
        if self.content_status == SpeakingCoachingContentStatus.GROUNDED and not self.items:
            raise ValueError("GROUNDED coaching requires at least one item")
        if self.content_status == SpeakingCoachingContentStatus.NO_USABLE_EVIDENCE and self.items:
            raise ValueError("NO_USABLE_EVIDENCE cannot contain coaching items")
        return self


class SpeakingCoachingEvidence(CamelCaseModel):
    turn_id: str
    turn_index: int
    recording_revision: int | None = None
    transcript_excerpt: str
    transcript_hash: str
    reference_assistant_turn_id: str | None = None
    assistance_usage: list[AssistanceUsage] = Field(default_factory=list)
    source_provenance: str = "AUTOMATIC_SPEECH_RECOGNITION"
    verbatim_accuracy_verified: bool = False


class SpeakingCoachingItem(CamelCaseModel):
    observation_id: str
    kind: SpeakingCoachingObservationKind
    evidence: SpeakingCoachingEvidence
    message: str
    suggested_expression: str | None = None
    suggestion_is_learner_evidence: bool = False


class SpeakingCoachingResponse(CamelCaseModel):
    request_id: str
    session_id: str
    result_kind: SpeakingResultKind = SpeakingResultKind.SESSION_COACHING
    result_policy_version: str
    schema_version: str
    source_snapshot_hash: str
    content_status: SpeakingCoachingContentStatus
    limitation_reasons: list[str] = Field(default_factory=list)
    items: list[SpeakingCoachingItem] = Field(default_factory=list, max_length=3)
    prompt_version: str
    usage: SpeakingUsage
