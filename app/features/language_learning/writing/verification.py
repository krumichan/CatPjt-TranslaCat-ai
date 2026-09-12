"""Writing reference: one semantic review; bounded uncertainty/adjacent recheck.

No Nano gate, confidence threshold or routine separate note review. Infrastructure
and protocol failures may retry the SAME stage once; valid judgments are never
re-rolled for a higher confidence. No unverified candidate is published on failure.
"""
from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from app.ai.ports import TextGenerationProvider
from app.features.language_learning.writing.assessment_contract import (
    NoteReview, TaskReview, RecoveredTaskReview, binding_failure, build_review_schema,
)
from app.features.language_learning.difficulty.contracts import (
    DifficultyAcceptanceDecision,
    DifficultyValidationResult,
)
from app.features.language_learning.difficulty.orchestration import orchestrate_candidate
from app.features.language_learning.writing.difficulty_adapter import (
    WritingAcceptanceContext,
    WritingDifficultyAcceptancePolicy,
    WritingSemanticAssessment,
    normalize_writing_assessment,
    writing_difficulty_target,
)
from app.features.language_learning.writing.difficulty_spec import build_difficulty_spec, difficulty_spec_reason
from app.features.language_learning.writing.difficulty_planning import ADJACENT_RECHECKS_PER_SLOT
from app.features.language_learning.writing.generation_contract import (
    WritingDraft, WritingSlot, contains_control_character, note_needs_localization, review_content_hash,
)
from app.features.language_learning.writing.note_localization import (
    NOTE_LOCALIZATION_TASK, LocalizedWritingNote, build_note_localization_prompt,
)
from app.features.language_learning.writing.review_diagnostics import (
    ReviewFailure,
    WritingProviderConfigurationError as WritingProviderConfigurationError,
)
from app.features.language_learning.writing.review_runtime import ReviewContext, ReviewRuntime, emit_event
from app.features.language_learning.writing.source_recovery import SourceRecoveryEvidence
from app.features.language_learning.writing.verification_prompts import build_writing_review_prompt
from app.schemas.language_learning import DailyWritingGenerationRequest

MINI_QUALITY_TASK = "LANGUAGE_LEARNING_WRITING_TASK_VERIFICATION"
MINI_DIFFICULTY_TASK = "LANGUAGE_LEARNING_WRITING_DIFFICULTY_VERIFICATION"
MINI_NOTE_TASK = "LANGUAGE_LEARNING_WRITING_NOTE_VERIFICATION"


@dataclass(frozen=True)
class VerificationDecision:
    accepted: bool
    reason: str
    content_hash: str
    estimated_band: int | None = None
    adjudicated: bool = False
    localized_focus_reason: str | None = None
    confidence: float | None = None  # diagnostics ONLY
    difficulty_confidence: float | None = None  # diagnostics ONLY
    difficulty_status: str | None = None
    primary_estimated_band: int | None = None
    adjudication_reason: str | None = None


class WritingCandidateVerifier:
    def __init__(self, provider: TextGenerationProvider, timeout_seconds: float, *,
                 deadline: float | None = None) -> None:
        self.runtime = ReviewRuntime(provider, timeout_seconds, deadline=deadline)
        self.calls = self.runtime.metrics.calls
        # A verifier is constructed per generate() request. Slots have unique
        # local order even when one request contains several Writing items.
        self.adjacent_rechecks: dict[int, int] = {}

    async def _assess(self, request: DailyWritingGenerationRequest, draft: WritingDraft,
                      context: ReviewContext, *, adjudicator: bool = False,
                      source_recovery: SourceRecoveryEvidence | None = None) -> TaskReview | None:
        model = RecoveredTaskReview if source_recovery is not None else TaskReview
        schema = build_review_schema(model, draft)
        recovery_hash = source_recovery.binding_hash(request, draft) if source_recovery is not None else None
        if recovery_hash is not None:
            schema["properties"]["recoveryHash"]["enum"] = [recovery_hash]

        def inspect(result: TaskReview) -> ReviewFailure | None:
            failure = binding_failure(result, draft, context.candidate_id, context.content_hash)
            if failure is not None:
                return failure
            if recovery_hash is not None and (
                not isinstance(result, RecoveredTaskReview) or result.recovery_hash != recovery_hash
            ):
                return ReviewFailure("VERIFIER_RECOVERY_HASH_MISMATCH", "PROTOCOL")
            return None

        return await self.runtime.review(
            task=MINI_DIFFICULTY_TASK if adjudicator else MINI_QUALITY_TASK,
            prompt=build_writing_review_prompt(request, draft, candidate_id=context.candidate_id,
                                              content_hash=context.content_hash, source_recovery=source_recovery),
            model=model, schema=schema, context=context, inspect=inspect,
            # Exactly one adjudication call. Primary protocol/dependency recovery
            # may repeat once, but a valid UNSURE goes to acceptance, not retries.
            max_attempts=1 if adjudicator else 2, retry_semantic=False,
        )

    async def verify(
        self, request: DailyWritingGenerationRequest, draft: WritingDraft, slot: WritingSlot,
        candidate_id: str, *, generation_attempt: int = 1,
        validation: DifficultyValidationResult | None = None,
        source_recovery: SourceRecoveryEvidence | None = None,
    ) -> VerificationDecision:
        # An approval must refer to the exact snapshot seen by the reviewer.
        draft = draft.model_copy(deep=True)
        content_hash = review_content_hash(request, draft)
        # Legacy/direct callers did not run deterministic validation here. The
        # production generator passes its already-computed validation explicitly.
        effective_validation = (
            validation if validation is not None else DifficultyValidationResult.accept()
        )
        context_holder = [
            ReviewContext(
                request.request_id, candidate_id, content_hash, slot.order, generation_attempt,
            )
        ]
        policy = WritingDifficultyAcceptancePolicy()
        target = writing_difficulty_target(slot)

        async def assess_primary() -> WritingSemanticAssessment | None:
            review = await self._assess(
                request, draft, context_holder[0], source_recovery=source_recovery,
            )
            return None if review is None else normalize_writing_assessment(review)

        def policy_context(
            _assessment: WritingSemanticAssessment, _adjudicated: bool,
        ) -> WritingAcceptanceContext:
            return WritingAcceptanceContext(
                slot,
                self.adjacent_rechecks.get(slot.order, 0) < ADJACENT_RECHECKS_PER_SLOT,
            )

        def before_adjudication(decision: DifficultyAcceptanceDecision) -> None:
            if decision.reason == "ADJACENT_BAND_RECHECK":
                # Reserve before awaiting; even a failed/cancelled review consumes
                # the extra allowance, and siblings cannot start fresh re-votes.
                used = self.adjacent_rechecks.get(slot.order, 0)
                self.adjacent_rechecks[slot.order] = used + 1
            primary = assessment_holder[0]
            emit_event("writing.adjudication.requested", **context_holder[0].fields(), reason=decision.reason,
                       target_band=slot.target_band,
                       primary_estimated_band=(primary.review.estimated_band if primary else None),
                       adjacent_rechecks_used=self.adjacent_rechecks.get(slot.order, 0),
                       adjacent_recheck_limit=ADJACENT_RECHECKS_PER_SLOT,
                       confidence_used_for_acceptance=False)

        assessment_holder: list[WritingSemanticAssessment | None] = [None]

        async def primary_with_capture() -> WritingSemanticAssessment | None:
            assessment_holder[0] = await assess_primary()
            return assessment_holder[0]

        async def adjudicate() -> WritingSemanticAssessment | None:
            # A fresh correlation ID also prevents replaying the first verdict as
            # the adjudicator's response. No prior decisions appear in the prompt.
            context_holder[0] = ReviewContext(
                request.request_id, uuid4().hex, content_hash, slot.order, generation_attempt,
            )
            review = await self._assess(
                request, draft, context_holder[0], adjudicator=True,
                source_recovery=source_recovery,
            )
            return None if review is None else normalize_writing_assessment(review)

        result = await orchestrate_candidate(
            target=target,
            validation=effective_validation,
            assessor=primary_with_capture,
            policy=policy,
            context_factory=policy_context,
            deterministic_rejection=lambda checked: DifficultyAcceptanceDecision(
                "REJECT", checked.primary_issue or "SEMANTIC_UNRESOLVED",
            ),
            is_adjudication_requested=lambda decision: decision.action == "ADJUDICATE",
            unresolved_decision=DifficultyAcceptanceDecision("REJECT", "SEMANTIC_UNRESOLVED"),
            adjudicator=adjudicate,
            before_adjudication=before_adjudication,
        )
        if result.assessment is None:
            return VerificationDecision(
                False,
                result.decision.reason,
                content_hash,
                adjudicated=result.adjudicated,
            )
        quality = result.assessment.review
        primary_band = (
            result.primary_assessment.review.estimated_band
            if result.primary_assessment is not None
            else None
        )
        acceptance = result.decision
        adjudicated = result.adjudicated
        adjudication_reason = result.adjudication_reason

        def decision(accepted: bool, reason: str, final: WritingDraft = draft) -> VerificationDecision:
            return VerificationDecision(
                accepted, reason, review_content_hash(request, final), quality.estimated_band,
                adjudicated=adjudicated,
                localized_focus_reason=final.focus_reason if accepted and final is not draft else None,
                confidence=quality.confidence, difficulty_confidence=quality.difficulty_confidence,
                difficulty_status=quality.difficulty_status, primary_estimated_band=primary_band,
                adjudication_reason=adjudication_reason,
            )

        emit_event("writing.acceptance.decided", **context_holder[0].fields(), action=acceptance.action,
                   reason=acceptance.reason, target_band=slot.target_band,
                   primary_estimated_band=primary_band, adjudication_reason=adjudication_reason,
                   adjacent_rechecks_used=self.adjacent_rechecks.get(slot.order, 0),
                   estimated_band=quality.estimated_band, difficulty_status=quality.difficulty_status,
                   issue_codes=list(quality.issues), confidence=quality.confidence,
                   difficulty_confidence=quality.difficulty_confidence, confidence_used_for_acceptance=False,
                   source_preservation=(quality.source_preservation.status
                                        if isinstance(quality, RecoveredTaskReview) else None))
        if acceptance.action == "REJECT":
            return decision(False, acceptance.reason)
        if acceptance.action == "LOCALIZE_NOTE" or note_needs_localization(request, draft):
            # Only a language-only note defect is repairable. The main task passed;
            # all task fields remain immutable. The replacement gets its own audit.
            localized = await self._localize_note(request, draft, slot, context_holder[0])
            if isinstance(localized, str):
                return decision(False, localized)
            return decision(True, "VERIFIED_NOTE_LOCALIZED", localized)
        if acceptance.action != "ACCEPT":
            return decision(False, "SEMANTIC_UNRESOLVED")
        return decision(True, "VERIFIED")

    async def _localize_note(
        self, request: DailyWritingGenerationRequest, draft: WritingDraft,
        slot: WritingSlot, context: ReviewContext,
    ) -> WritingDraft | str:
        emit_event("writing.note.localization", **context.fields(), action="PROPOSE_ONCE", field="focusReason")
        proposal = await self.runtime.review(
            task=NOTE_LOCALIZATION_TASK,
            prompt=build_note_localization_prompt(request, draft, candidate_id=context.candidate_id,
                                                 content_hash=context.content_hash),
            model=LocalizedWritingNote, context=context, max_attempts=1, retry_semantic=False,
            inspect=lambda result: (
                ReviewFailure("VERIFIER_IDENTITY_MISMATCH", "PROTOCOL") if result.candidate_id != context.candidate_id else
                ReviewFailure("VERIFIER_CONTENT_HASH_MISMATCH", "PROTOCOL") if result.content_hash != context.content_hash else None
            ),
        )
        if proposal is None:
            return "NOTE_LOCALIZATION_UNAVAILABLE"
        if proposal.focus_reason == draft.focus_reason:
            return "NOTE_LOCALIZATION_UNCHANGED"
        localized = WritingDraft.model_validate({
            **draft.model_dump(mode="json", by_alias=True), "focusReason": proposal.focus_reason,
        })
        if contains_control_character(localized.focus_reason):
            return "NOTE_LOCALIZATION_CONTROL_CHARACTER"
        if note_needs_localization(request, localized):
            return "NOTE_LOCALIZATION_SCRIPT_MISMATCH"
        reason = difficulty_spec_reason(localized, build_difficulty_spec(request, slot))
        if reason is not None:
            return reason
        final_hash = review_content_hash(request, localized)
        note_context = ReviewContext(request.request_id, uuid4().hex, final_hash, context.slot, context.generation_attempt)
        note = await self.runtime.review(
            task=MINI_NOTE_TASK,
            prompt=build_writing_review_prompt(request, localized, candidate_id=note_context.candidate_id,
                                              content_hash=final_hash, note_only=True),
            model=NoteReview, schema=build_review_schema(NoteReview, localized), context=note_context,
            inspect=lambda result: binding_failure(result, localized, note_context.candidate_id, final_hash),
            retry_semantic=False,
        )
        if note is None or note.verdict == "UNSURE":
            return "NOTE_VERIFIER_UNCERTAIN"
        if note.verdict == "REJECT":
            return "NOTE_" + note.issues[0]
        emit_event("writing.note.localization.approved", **note_context.fields())
        return localized
