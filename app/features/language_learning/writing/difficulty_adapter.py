"""Writing-owned adapters for the service-neutral difficulty contracts."""
from __future__ import annotations

from dataclasses import dataclass, field

from app.features.language_learning.difficulty.contracts import (
    DifficultyAcceptanceDecision,
    DifficultyTarget,
    DifficultyValidationResult,
    SemanticAssessment,
)
from app.features.language_learning.quality import LANGUAGE_COMPLEXITY_POLICY_VERSION
from app.features.language_learning.writing.assessment_contract import (
    TaskReview,
    recovered_acceptance,
)
from app.features.language_learning.writing.difficulty_spec import (
    WritingDifficultySpec,
    build_difficulty_spec,
    measure_draft,
)
from app.features.language_learning.writing.generation_contract import (
    WritingDraft,
    WritingSlot,
    deterministic_draft_reason,
)
from app.schemas.language_learning import DailyWritingGenerationRequest


def writing_difficulty_target(slot: WritingSlot) -> DifficultyTarget[int]:
    return DifficultyTarget(
        service="writing",
        scale="language-production-band",
        value=slot.target_band,
        policy_version=LANGUAGE_COMPLEXITY_POLICY_VERSION,
    )


@dataclass(frozen=True)
class WritingDifficultySpecAdapter:
    target: DifficultyTarget[int]
    spec: WritingDifficultySpec

    @property
    def version(self) -> str:
        return self.spec.version

    def generation_payload(self) -> dict:
        return self.spec.generation_payload()


def writing_difficulty_spec(
    request: DailyWritingGenerationRequest,
    slot: WritingSlot,
) -> WritingDifficultySpecAdapter:
    return WritingDifficultySpecAdapter(
        writing_difficulty_target(slot),
        build_difficulty_spec(request, slot),
    )


def validate_writing_candidate(
    request: DailyWritingGenerationRequest,
    draft: WritingDraft,
    slot: WritingSlot,
) -> DifficultyValidationResult:
    measurements = measure_draft(draft).log_fields()
    reason = deterministic_draft_reason(request, draft, slot=slot)
    if reason is None:
        return DifficultyValidationResult.accept(measurements=measurements)
    return DifficultyValidationResult.reject(reason, measurements=measurements)


@dataclass(frozen=True, kw_only=True)
class WritingSemanticAssessment(SemanticAssessment[int]):
    """Normalized Common assessment with opaque Writing-owned policy state."""

    review: TaskReview = field(repr=False, compare=False)


def normalize_writing_assessment(review: TaskReview) -> WritingSemanticAssessment:
    evidence = list(review.difficulty_evidence_segment_ids)
    for check in review.checks:
        evidence.extend(check.evidence_segment_ids)
    return WritingSemanticAssessment(
        status=review.verdict,
        difficulty_status=review.difficulty_status,
        observed_target=review.estimated_band,
        alternative_target=review.alternative_band,
        issue_codes=tuple(review.issues),
        confidence=review.confidence,
        difficulty_confidence=review.difficulty_confidence,
        evidence_refs=tuple(dict.fromkeys(evidence)),
        review=review,
    )


@dataclass(frozen=True)
class WritingAcceptanceContext:
    slot: WritingSlot
    allow_adjacent_recheck: bool


@dataclass(frozen=True)
class WritingDifficultyAcceptancePolicy:
    """Stateless adapter; the actual decision table remains Writing-owned."""

    def decide(
        self,
        *,
        target: DifficultyTarget[int],
        assessment: WritingSemanticAssessment,
        context: WritingAcceptanceContext,
        adjudicated: bool,
    ) -> DifficultyAcceptanceDecision:
        if target.value != context.slot.target_band:
            raise ValueError("Writing target and slot must agree")
        result = recovered_acceptance(
            assessment.review,
            context.slot,
            adjudicated=adjudicated,
            allow_adjacent_recheck=context.allow_adjacent_recheck,
        )
        return DifficultyAcceptanceDecision(result.action, result.reason)
