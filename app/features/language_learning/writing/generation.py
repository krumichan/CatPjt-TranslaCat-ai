"""Bounded content generation -> observed spec checks -> semantic review -> publication."""
from __future__ import annotations

import asyncio
import logging
import math
import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from uuid import uuid4

from fastapi import HTTPException
from pydantic import ValidationError

from app.ai.ports import TextGenerationProvider
from app.features.language_learning.quality import (
    CONTENT_DIVERSITY_POLICY_VERSION,
    LANGUAGE_COMPLEXITY_POLICY_VERSION,
    DiversityCandidate,
    DiversityValidator,
    DiversityValidationStats,
)
from app.features.language_learning.writing.difficulty_spec import (
    WRITING_DIFFICULTY_SPEC_VERSION, measure_draft,
)
from app.features.language_learning.writing.difficulty_planning import (
    WRITING_DIFFICULTY_CONTROL_VERSION, DifficultyRecoveryState, production_blueprint,
)
from app.features.language_learning.writing.generation_contract import (
    CANDIDATES_PER_SLOT,
    MAX_GENERATION_ATTEMPTS,
    SERVER_OWNED_FIELDS,
    WRITING_VERIFICATION_POLICY_VERSION,
    WritingDraft,
    WritingSlot,
    build_candidate_schema,
    deterministic_draft_reason,
    parse_draft,
    plan_slots,
    review_content_hash,
    script_rejection_field,
    validate_final_items,
)
from app.features.language_learning.writing.prompts import (
    DAILY_WRITING_GENERATION_PROMPT_VERSION,
    build_daily_writing_generation_prompt,
)
from app.features.language_learning.writing.verification import (
    VerificationDecision,
    WritingCandidateVerifier,
    WritingProviderConfigurationError,
)
from app.features.language_learning.writing.review_diagnostics import (
    ReviewFailure, WritingVerificationUnavailableError, classify_exception,
)
from app.features.language_learning.writing.review_runtime import emit_event, safe_identifier
from app.features.language_learning.writing.source_language import SOURCE_LANGUAGE_POLICY_VERSION, script_statistics
from app.features.language_learning.writing.source_recovery import (
    SourceRecoveryEvidence, localize_sources_once,
)
from app.schemas.language_learning import DailyWritingGenerationRequest, DailyWritingGenerationResponse, DailyWritingItem, DailyWritingType
from app.schemas.language_learning_quality import DiversityHistoryEntry, DiversityMetadata, DiversitySummary, GenerationSourceType

logger = logging.getLogger(__name__)
GENERATION_TASK = "LANGUAGE_LEARNING_DAILY_WRITING_GENERATION"
# A generator changing only notes/metadata must not get fresh votes on an already
# definitely rejected task. Note-related issues are deliberately NOT in this set.
_TASK_FINAL_REJECTIONS = frozenset({
    "VERIFIED_BAND_MISMATCH", "QUALITY_TASK_TYPE", "QUALITY_MISSING_FACTS",
    "QUALITY_CONTRADICTORY_GUIDANCE", "QUALITY_AMBIGUOUS_TASK", "QUALITY_BACKGROUND_KNOWLEDGE",
    "QUALITY_UNNATURAL_LANGUAGE", "QUALITY_NOT_LANGUAGE_TASK", "QUALITY_ORIGIN_LANGUAGE",
})


@dataclass
class GenerationStats:
    generations: int = 0
    generation_elapsed_ms: float = 0.0
    candidates: int = 0
    rejections: Counter[str] = field(default_factory=Counter)
    diversity: DiversityValidationStats = field(default_factory=DiversityValidationStats)
    fallback_used: bool = False
    generator_failures: Counter[str] = field(default_factory=Counter)
    generator_backoff_ms: float = 0.0
    source_recovery_batches: int = 0
    source_recovery_proposals: int = 0
    source_recovery_accepted: int = 0
    difficulty_recovery_rounds: int = 0
    difficulty_mismatches: Counter[str] = field(default_factory=Counter)


@dataclass(frozen=True)
class VerifiedDraft:
    draft: WritingDraft
    review: VerificationDecision
    metadata: DiversityMetadata


class VerifiedWritingGenerator:
    def __init__(
        self,
        provider: TextGenerationProvider,
        *,
        generation_timeout_seconds: float,
        verification_timeout_seconds: float,
        total_timeout_seconds: float,
        max_retries: int,
    ) -> None:
        if any(not math.isfinite(value) or value <= 0 for value in (
            generation_timeout_seconds, verification_timeout_seconds, total_timeout_seconds,
        )):
            raise ValueError("Writing generation timeouts must be positive")
        if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 0:
            raise ValueError("Writing generation max_retries must be a nonnegative integer")
        self.provider = provider
        self.generation_timeout_seconds = generation_timeout_seconds
        self.total_timeout_seconds = total_timeout_seconds
        self.max_attempts = min(MAX_GENERATION_ATTEMPTS, max_retries + 1)
        self.verification_timeout_seconds = verification_timeout_seconds

    async def generate(self, request: DailyWritingGenerationRequest) -> DailyWritingGenerationResponse:
        # All mutable state is request-local. Never share candidates or verdicts between users.
        request = request.model_copy(deep=True)
        stats = GenerationStats()
        started = time.monotonic()
        verifier = WritingCandidateVerifier(self.provider, self.verification_timeout_seconds,
                                            deadline=started + self.total_timeout_seconds)
        failure_code = None
        outcome = "FAILED"
        returned_count = 0
        try:
            response = await asyncio.wait_for(self._generate(request, stats, verifier), timeout=self.total_timeout_seconds)
            outcome = "SUCCEEDED"
            returned_count = len(response.items)
            return response
        except asyncio.CancelledError:
            outcome = "CANCELLED"
            raise
        except TimeoutError as exc:
            failure_code = "WRITING_GENERATION_DEADLINE_EXCEEDED"
            raise HTTPException(status_code=504, detail={
                "code": "WRITING_GENERATION_DEADLINE_EXCEEDED",
                "retryable": True,
                "message": "Writing 생성·검증의 전체 제한 시간을 초과했습니다. 검증되지 않은 문항은 제공하지 않습니다.",
            }) from exc
        except WritingVerificationUnavailableError as exc:
            failure_code = "WRITING_VERIFICATION_UNAVAILABLE"
            raise HTTPException(status_code=503, detail={
                "code": failure_code, "retryable": exc.failure.retryable,
                "message": "Writing 검증 서비스가 유효한 판정을 반환하지 못했습니다. 미검증 문제는 제공하지 않습니다.",
                "task": exc.task, "failureCode": exc.failure.code, "attempts": exc.attempts,
            }) from exc
        except WritingProviderConfigurationError as exc:
            failure_code = "WRITING_PROVIDER_CONFIGURATION_ERROR"
            raise HTTPException(status_code=502, detail={
                "code": "WRITING_PROVIDER_CONFIGURATION_ERROR", "retryable": False,
                "message": "Writing 생성·검증 Provider 설정을 확인해야 합니다.",
            }) from exc
        except HTTPException as exc:
            failure_code = exc.detail.get("code") if isinstance(exc.detail, dict) else "HTTP_ERROR"
            raise
        except Exception:
            failure_code = "WRITING_INTERNAL_ERROR"
            raise
        finally:
            emit_event(
                "writing.generation.finished", request_id=request.request_id,
                policy=WRITING_VERIFICATION_POLICY_VERSION, difficulty_spec_version=WRITING_DIFFICULTY_SPEC_VERSION,
                source_language_policy_version=SOURCE_LANGUAGE_POLICY_VERSION,
                difficulty_control_policy_version=WRITING_DIFFICULTY_CONTROL_VERSION,
                difficulty_recovery_rounds=stats.difficulty_recovery_rounds,
                difficulty_mismatches=dict(stats.difficulty_mismatches),
                adjacent_band_rechecks=sum(verifier.adjacent_rechecks.values()),
                generator_failures=dict(stats.generator_failures), generator_backoff_ms=round(stats.generator_backoff_ms, 2),
                source_recovery_batches=stats.source_recovery_batches,
                source_recovery_proposals=stats.source_recovery_proposals,
                source_recovery_accepted=stats.source_recovery_accepted,
                outcome=outcome, returned_items=returned_count,
                failure_code=failure_code, duration_ms=round((time.monotonic() - started) * 1000, 2),
                generation_calls=stats.generations, generation_elapsed_ms=round(stats.generation_elapsed_ms, 2),
                candidates=stats.candidates, rejections=dict(stats.rejections), history_fallback=stats.fallback_used,
                **verifier.runtime.metrics.summary(),
            )
            logger.info(
                "Writing verified generation finished. request_id=%s policy=%s generation_calls=%d candidates=%d review_calls=%s rejections=%s history_fallback=%s outcome=%s returned_items=%d",
                safe_identifier(request.request_id), WRITING_VERIFICATION_POLICY_VERSION, stats.generations,
                stats.candidates, dict(verifier.calls), dict(stats.rejections), stats.fallback_used, outcome, returned_count,
            )

    async def _generate(self, request: DailyWritingGenerationRequest, stats: GenerationStats,
                        verifier: WritingCandidateVerifier) -> DailyWritingGenerationResponse:
        items: list[DailyWritingItem] = []
        accepted: list[DiversityCandidate] = []
        for slot in plan_slots(request):
            verified = await self._generate_slot(request, slot, items, accepted, stats, verifier)
            # Only the server assembles metadata, and only AFTER independent approval.
            if not verified.review.accepted or verified.review.content_hash != review_content_hash(request, verified.draft):
                raise ValueError("Writing verification evidence no longer matches its content")
            payload = verified.draft.model_dump(mode="json", by_alias=True)
            payload.update({
                "order": slot.order,
                "difficulty": slot.difficulty.value,
                "languageComplexityBand": slot.target_band,
                "diversityMetadata": verified.metadata.model_dump(mode="json", by_alias=True),
            })
            item = DailyWritingItem.model_validate(payload)
            items.append(item)
            accepted.append(DiversityCandidate(item.origin_text, verified.metadata))

        validate_final_items(request, items)
        return DailyWritingGenerationResponse(
            request_id=request.request_id,
            prompt_version=DAILY_WRITING_GENERATION_PROMPT_VERSION,
            items=items,
            content_diversity_policy_version=CONTENT_DIVERSITY_POLICY_VERSION,
            language_complexity_policy_version=LANGUAGE_COMPLEXITY_POLICY_VERSION,
            diversity_summary=DiversitySummary(
                policy_version=CONTENT_DIVERSITY_POLICY_VERSION,
                candidate_count=stats.candidates,
                accepted_count=len(items),
                rejected_exact=stats.diversity.rejected_exact,
                rejected_similarity=stats.diversity.rejected_similarity,
                rejected_structural=stats.diversity.rejected_structural,
                rejected_background_knowledge=stats.diversity.rejected_background_knowledge,
                fallback_used=stats.fallback_used,
            ),
        )

    async def _generate_slot(
        self,
        request: DailyWritingGenerationRequest,
        slot: WritingSlot,
        items: list[DailyWritingItem],
        accepted: list[DiversityCandidate],
        stats: GenerationStats,
        verifier: WritingCandidateVerifier,
    ) -> VerifiedDraft:
        seen: set[str] = set()
        rejected_tasks: set[str] = set()
        fallback: VerifiedDraft | None = None
        feedback: Counter[str] = Counter()
        strict_diversity = DiversityValidator(request.diversity_context)
        relaxed_diversity = DiversityValidator(request.diversity_context, relaxed_history=True)
        attempt_limit = self.max_attempts
        recovery_used = False
        recovery_mode = False
        source_exhausted = False
        attempts_done = 0
        difficulty_control = DifficultyRecoveryState(slot.target_band)
        for attempt in range(1, self.max_attempts + 1):
            if attempt > attempt_limit:
                break
            attempts_done = attempt
            candidate_request = self._request_with_accepted_history(request, items)
            drafts, batch_rejections = await self._candidate_batch(
                candidate_request, slot, attempt, feedback, stats, source_recovery_mode=recovery_mode,
                attempt_limit=attempt_limit, deadline=verifier.runtime.deadline,
                difficulty_recovery=difficulty_control.payload(),
            )
            feedback.update(batch_rejections)
            stats.rejections.update(batch_rejections)
            source_only = []
            source_exhausted = False
            for draft in drafts:
                reason = self._check_draft(request, draft, slot, attempt)
                if reason is not None:
                    self._reject(reason, feedback, stats)
                    if reason == "ORIGIN_TEXT_SCRIPT_MISMATCH":
                        source_only.append(draft)
                    continue
                verified, strict = await self._review_candidate(
                    request, draft, slot, attempt, seen, accepted, stats, feedback, verifier,
                    strict_diversity, relaxed_diversity, rejected_tasks=rejected_tasks,
                    difficulty_control=difficulty_control,
                )
                if verified is not None:
                    if strict:
                        return verified
                    if fallback is None:
                        fallback = verified

            attempt_limit = self._difficulty_recovery_limit(
                difficulty_control, attempt, attempt_limit, request, slot,
            )

            # Preserve valid siblings first. Repair is only for a whole parsed batch
            # whose sole deterministic defect is its source script, not other modes,
            # malformed output, a note problem or any semantic/difficulty rejection.
            all_source_only = bool(drafts) and len(source_only) == len(drafts) and not batch_rejections
            if request.writing_type != DailyWritingType.TRANSLATION or not all_source_only:
                continue
            source_exhausted = True
            if recovery_used:
                break
            recovery_used = recovery_mode = True
            # Recovery policies may shorten, never extend, an already selected
            # final round (including a preceding difficulty recovery).
            attempt_limit = min(attempt_limit, attempt + 1)
            emit_event("writing.source.recovery.selected", request_id=request.request_id,
                       slot=slot.order, generation_attempt=attempt, remaining_generation_rounds=attempt_limit - attempt,
                       candidate_count=len(source_only), action="FIELD_ONLY_REPAIR_ONCE")
            stats.source_recovery_batches += 1
            proposals = await localize_sources_once(
                request, source_only, slot, verifier.runtime, generation_attempt=attempt,
            )
            stats.source_recovery_proposals += len(proposals)
            for proposal in proposals:
                reason = self._check_draft(request, proposal.draft, slot, attempt, recovered=True)
                if reason is not None:
                    self._reject("SOURCE_RECOVERY_" + reason, feedback, stats)
                    continue
                source_exhausted = False
                verified, strict = await self._review_candidate(
                    request, proposal.draft, slot, attempt, seen, accepted, stats, feedback, verifier,
                    strict_diversity, relaxed_diversity, source_recovery=proposal.evidence, rejected_tasks=rejected_tasks,
                    difficulty_control=difficulty_control,
                )
                if verified is not None:
                    stats.source_recovery_accepted += 1
                    if strict:
                        return verified
                    if fallback is None:
                        fallback = verified

            attempt_limit = self._difficulty_recovery_limit(
                difficulty_control, attempt, attempt_limit, request, slot,
            )

        if fallback is not None:
            stats.fallback_used = True
            return fallback
        reasons = dict(feedback)
        code = ("WRITING_SOURCE_LANGUAGE_EXHAUSTED" if source_exhausted else
                "CONTENT_DIVERSITY_EXHAUSTED" if reasons and all(
                    key.startswith("DIVERSITY_") or key == "REPEATED_REJECTED_CANDIDATE" for key in reasons
                ) else "WRITING_GENERATION_VALIDATION_EXHAUSTED")
        raise HTTPException(status_code=422, detail={
            "code": code, "retryable": False,
            "message": "Writing 생성·검증 기준을 만족하는 문항이 부족합니다.",
            "slotOrder": slot.order, "attempts": attempts_done,
            "sourceRecoveryUsed": recovery_used, "rejectionCounts": reasons,
            "targetBand": slot.target_band,
            "observedBandCounts": {str(band): count for band, count in sorted(difficulty_control.observed_bands.items())},
            "difficultyRecoveryUsed": difficulty_control.direction is not None,
        })

    @staticmethod
    def _difficulty_recovery_limit(control: DifficultyRecoveryState, attempt: int, limit: int,
                                   request: DailyWritingGenerationRequest, slot: WritingSlot) -> int:
        # Two final mismatches in one direction select exactly one targeted round,
        # including when those mismatches were found after source localization.
        if not control.select_after_round(attempt, limit):
            return limit
        emit_event("writing.difficulty.recovery.selected", request_id=request.request_id,
                   slot=slot.order, generation_attempt=attempt, target_band=slot.target_band,
                   direction=control.direction, observed_band_counts=dict(control.observed_bands),
                   remaining_generation_rounds=1, policy=WRITING_DIFFICULTY_CONTROL_VERSION)
        return min(limit, attempt + 1)

    @staticmethod
    def _reject(reason: str, feedback: Counter[str], stats: GenerationStats) -> None:
        feedback[reason] += 1
        stats.rejections[reason] += 1

    def _check_draft(self, request: DailyWritingGenerationRequest, draft: WritingDraft,
                     slot: WritingSlot, attempt: int, *, recovered: bool = False) -> str | None:
        reason = deterministic_draft_reason(request, draft, slot=slot)
        emit_event("writing.spec.checked", request_id=request.request_id, slot=slot.order,
                   generation_attempt=attempt, policy=WRITING_DIFFICULTY_SPEC_VERSION,
                   target_band=slot.target_band, outcome="PASS" if reason is None else "REJECT", reason=reason,
                   recovered_source=recovered, measurements=measure_draft(draft).log_fields(),
                   source_script=script_statistics(request.origin_language, draft.origin_text))
        if reason is not None:
            logger.info(
                "Writing candidate rejected before review. request_id=%s slot=%d "
                "attempt=%d/%d reason=%s field=%s origin_language=%s writing_type=%s",
                safe_identifier(request.request_id), slot.order, attempt, self.max_attempts, reason,
                script_rejection_field(reason), request.origin_language, request.writing_type.value,
            )
        return reason

    async def _review_candidate(
        self, request: DailyWritingGenerationRequest, draft: WritingDraft, slot: WritingSlot,
        attempt: int, seen: set[str], accepted: list[DiversityCandidate], stats: GenerationStats,
        feedback: Counter[str], verifier: WritingCandidateVerifier,
        strict_diversity: DiversityValidator, relaxed_diversity: DiversityValidator,
        *, source_recovery: SourceRecoveryEvidence | None = None, rejected_tasks: set[str],
        difficulty_control: DifficultyRecoveryState,
    ) -> tuple[VerifiedDraft | None, bool]:
        fingerprint = review_content_hash(request, draft)
        task_key = hashlib.sha256(json.dumps(draft.task_content(), sort_keys=True,
                                               ensure_ascii=False).encode("utf-8")).hexdigest()
        if task_key in rejected_tasks:
            self._reject("REPEATED_REJECTED_TASK", feedback, stats)
            return None, False
        seen_key = (fingerprint if source_recovery is None else
                    fingerprint + ":" + source_recovery.binding_hash(request, draft))
        if seen_key in seen:
            self._reject("REPEATED_REJECTED_CANDIDATE", feedback, stats)
            return None, False
        seen.add(seen_key)
        # Recompute all duplicate/hash decisions on the NEW source, never the old
        # Japanese/English proposal. A repair may be an exact existing question.
        candidate = DiversityCandidate(draft.origin_text, draft.to_diversity_metadata())
        decision = strict_diversity.validate(candidate, accepted)
        stats.diversity.record(decision)
        relaxed = None
        if not decision.accepted:
            self._reject("DIVERSITY_" + str(decision.reason), feedback, stats)
            relaxed = relaxed_diversity.validate(candidate, accepted)
            if not relaxed.accepted:
                return None, False
        candidate_id = uuid4().hex
        review_started = time.monotonic()
        extra = {"source_recovery": source_recovery} if source_recovery is not None else {}
        verdict = await verifier.verify(request, draft, slot, candidate_id, generation_attempt=attempt, **extra)
        emit_event(
            "writing.candidate.reviewed", request_id=request.request_id, candidate_id=candidate_id,
            content_hash=fingerprint, slot=slot.order, generation_attempt=attempt,
            duration_ms=round((time.monotonic() - review_started) * 1000, 2),
            target_band=slot.target_band, estimated_band=verdict.estimated_band,
            accepted=verdict.accepted, reason=verdict.reason, confidence=verdict.confidence,
            difficulty_confidence=verdict.difficulty_confidence, confidence_used_for_acceptance=False,
            adjudicated=verdict.adjudicated, recovered_source=source_recovery is not None,
            primary_estimated_band=verdict.primary_estimated_band,
            adjudication_reason=verdict.adjudication_reason,
        )
        logger.info(
            "Writing independent review. request_id=%s slot=%d attempt=%d/%d target_band=%d "
            "estimated_band=%s accepted=%s reason=%s adjudicated=%s content_hash=%s",
            safe_identifier(request.request_id), slot.order, attempt, self.max_attempts, slot.target_band,
            verdict.estimated_band, verdict.accepted, verdict.reason, verdict.adjudicated, fingerprint[:12],
        )
        if not verdict.accepted:
            if difficulty_control.record(estimated_band=verdict.estimated_band,
                                         difficulty_status=verdict.difficulty_status,
                                         reason=verdict.reason, generation_attempt=attempt):
                stats.difficulty_mismatches[f"{slot.target_band}->{verdict.estimated_band}"] += 1
            if verdict.reason in _TASK_FINAL_REJECTIONS:
                rejected_tasks.add(task_key)
            self._reject(verdict.reason, feedback, stats)
            return None, False
        reviewed_draft = draft
        if verdict.localized_focus_reason is not None:
            reviewed_draft = WritingDraft.model_validate({
                **draft.model_dump(mode="json", by_alias=True), "focusReason": verdict.localized_focus_reason,
            })
        if verdict.content_hash != review_content_hash(request, reviewed_draft):
            raise ValueError("Writing verification evidence no longer matches its content")
        if deterministic_draft_reason(request, reviewed_draft, slot=slot) is not None:
            raise ValueError("Reviewed Writing candidate violates the final content contract")
        if decision.accepted:
            metadata = decision.metadata
        elif relaxed is not None:
            metadata = relaxed.metadata
        else:
            raise AssertionError("Missing verified diversity decision")
        return VerifiedDraft(reviewed_draft, verdict, metadata), decision.accepted

    async def _candidate_batch(
        self,
        request: DailyWritingGenerationRequest,
        slot: WritingSlot,
        attempt: int,
        feedback: Counter[str],
        stats: GenerationStats,
        *, source_recovery_mode: bool = False, attempt_limit: int | None = None,
        deadline: float | None = None,
        difficulty_recovery: dict | None = None,
    ) -> tuple[list[WritingDraft], Counter[str]]:
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            raise TimeoutError("Writing request deadline expired before generation")
        timeout = min(self.generation_timeout_seconds, remaining) if remaining is not None else self.generation_timeout_seconds
        stats.generations += 1
        if difficulty_recovery is not None:
            stats.difficulty_recovery_rounds += 1
        blueprint = production_blueprint(slot, attempt)
        prompt = build_daily_writing_generation_prompt(
            request, slot=slot, retry_feedback=dict(feedback), source_recovery_mode=source_recovery_mode,
            generation_attempt=attempt, difficulty_recovery=difficulty_recovery,
        )
        started = time.monotonic()
        outcome = "ERROR"
        failure: ReviewFailure | None = None
        raw = None
        try:
            raw = await asyncio.wait_for(self.provider.call(
                type_name=GENERATION_TASK, data=prompt, schema=build_candidate_schema(request=request, slot=slot),
            ), timeout=timeout)
            outcome = "RESPONSE_RECEIVED"
        except asyncio.CancelledError:
            outcome = "CANCELLED"
            raise
        except Exception as exc:
            failure = classify_exception(exc)
            outcome = "UNAVAILABLE" if failure.category == "DEPENDENCY" else "INVALID"
            if failure.category == "CONFIGURATION":
                raise WritingProviderConfigurationError("Writing generator configuration failure") from exc
            if failure.category == "INTERNAL":
                raise
        finally:
            elapsed = (time.monotonic() - started) * 1000
            stats.generation_elapsed_ms += elapsed
            failure_fields = failure.fields() if failure is not None else {}
            if failure is not None:
                failure_fields["failure_code"] = failure.code.replace("VERIFIER_", "GENERATOR_", 1)
                stats.generator_failures[failure_fields["failure_code"]] += 1
            emit_event("writing.generator.finished", request_id=request.request_id,
                       task=GENERATION_TASK, slot=slot.order, generation_attempt=attempt,
                       duration_ms=round(elapsed, 2), outcome=outcome,
                       production_blueprint=blueprint["blueprintId"],
                       difficulty_recovery_direction=(difficulty_recovery["direction"] if difficulty_recovery else None),
                       **failure_fields)
        if failure is not None:
            code = failure.code.replace("VERIFIER_", "GENERATOR_", 1)
            if failure.category == "DEPENDENCY":
                limit = self.max_attempts if attempt_limit is None else attempt_limit
                delay = max(0.25, failure.retry_after_seconds or 0.0)
                remaining = None if deadline is None else deadline - time.monotonic()
                if attempt >= limit or delay > 5.0 or (remaining is not None and remaining <= delay + 0.05):
                    raise HTTPException(status_code=503, detail={
                        "code": "WRITING_GENERATION_UNAVAILABLE", "retryable": True,
                        "failureCode": code, "attempts": attempt,
                        "message": "Writing 생성 서비스 호출에 실패했습니다. 잠시 후 다시 시도해 주세요.",
                    })
                waiting = time.monotonic()
                try:
                    await asyncio.sleep(delay)
                finally:
                    elapsed = (time.monotonic() - waiting) * 1000
                    stats.generator_backoff_ms += elapsed
                    emit_event("writing.generator.retry_wait", request_id=request.request_id,
                               slot=slot.order, generation_attempt=attempt, duration_ms=round(elapsed, 2))
            return [], Counter({code: 1})

        if not isinstance(raw, dict) or set(raw) != {"items"} or not isinstance(raw.get("items"), list):
            return [], Counter({"GENERATOR_BATCH_SCHEMA": 1})
        raw_items = raw["items"]
        if not 1 <= len(raw_items) <= CANDIDATES_PER_SLOT:
            return [], Counter({"GENERATOR_CANDIDATE_COUNT": 1})
        stats.candidates += len(raw_items)
        drafts = []
        rejected: Counter[str] = Counter()
        for item in raw_items:
            if not isinstance(item, dict):
                rejected["CANDIDATE_SCHEMA"] += 1
                continue
            if SERVER_OWNED_FIELDS.intersection(item):
                logger.info("Writing generator control fields discarded. request_id=%s slot=%d attempt=%d", safe_identifier(request.request_id), slot.order, attempt)
            try:
                drafts.append(parse_draft(item))
            except ValidationError:
                # Do not log raw model/user text or let one invalid sibling lose the batch.
                rejected["CANDIDATE_SCHEMA"] += 1
        return drafts, rejected

    @staticmethod
    def _request_with_accepted_history(
        request: DailyWritingGenerationRequest, accepted: list[DailyWritingItem],
    ) -> DailyWritingGenerationRequest:
        entries = list(request.diversity_context.current_session)
        entries.extend(DiversityHistoryEntry(
            source_type=GenerationSourceType.WRITING, content=item.origin_text,
            content_hash=item.diversity_metadata.content_hash,
            scenario_category=item.diversity_metadata.scenario_category,
            communicative_intent=item.diversity_metadata.communicative_intent,
            task_archetype=item.diversity_metadata.task_archetype,
            grammar_focus_codes=item.diversity_metadata.grammar_focus_codes,
            semantic_summary=item.diversity_metadata.semantic_summary, age_days=0,
        ) for item in accepted if item.diversity_metadata is not None)
        # The prompt context has its own documented cap; the acceptance validator
        # still compares against every accepted candidate in this request.
        return request.model_copy(deep=True, update={"diversity_context": request.diversity_context.model_copy(
            deep=True, update={"current_session": entries[-40:]},
        )})
