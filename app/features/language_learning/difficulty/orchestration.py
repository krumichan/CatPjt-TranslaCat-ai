"""Candidate-level difficulty orchestration with service-owned policy callbacks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Generic, TypeVar

from app.features.language_learning.difficulty.contracts import (
    DifficultyAcceptanceDecision,
    DifficultyTarget,
    DifficultyValidationResult,
)
from app.features.language_learning.difficulty.policy import DifficultyAcceptancePolicy

TargetT = TypeVar("TargetT")
AssessmentT = TypeVar("AssessmentT")
ContextT = TypeVar("ContextT")


@dataclass(frozen=True)
class CandidateDifficultyResult(Generic[AssessmentT]):
    validation: DifficultyValidationResult
    primary_assessment: AssessmentT | None
    assessment: AssessmentT | None
    decision: DifficultyAcceptanceDecision
    adjudicated: bool = False
    adjudication_reason: str | None = None


async def orchestrate_candidate(
    *,
    target: DifficultyTarget[TargetT],
    validation: DifficultyValidationResult,
    assessor: Callable[[], Awaitable[AssessmentT | None]],
    policy: DifficultyAcceptancePolicy[TargetT, AssessmentT, ContextT],
    context_factory: Callable[[AssessmentT, bool], ContextT],
    deterministic_rejection: Callable[
        [DifficultyValidationResult], DifficultyAcceptanceDecision
    ],
    is_adjudication_requested: Callable[[DifficultyAcceptanceDecision], bool],
    unresolved_decision: DifficultyAcceptanceDecision,
    adjudicator: Callable[[], Awaitable[AssessmentT | None]] | None = None,
    before_adjudication: Callable[[DifficultyAcceptanceDecision], None] | None = None,
) -> CandidateDifficultyResult[AssessmentT]:
    """Run one candidate only; slot loops, retries, diversity and recovery stay outside."""
    if not validation.passed:
        return CandidateDifficultyResult(
            validation,
            None,
            None,
            deterministic_rejection(validation),
        )
    primary = await assessor()
    if primary is None:
        return CandidateDifficultyResult(
            validation, None, None, unresolved_decision,
        )
    decision = policy.decide(
        target=target,
        assessment=primary,
        context=context_factory(primary, False),
        adjudicated=False,
    )
    if not is_adjudication_requested(decision):
        return CandidateDifficultyResult(validation, primary, primary, decision)
    if adjudicator is None:
        return CandidateDifficultyResult(
            validation, primary, primary, unresolved_decision, False, decision.reason,
        )
    if before_adjudication is not None:
        before_adjudication(decision)
    final = await adjudicator()
    if final is None:
        return CandidateDifficultyResult(
            validation, primary, None, unresolved_decision, True, decision.reason,
        )
    final_decision = policy.decide(
        target=target,
        assessment=final,
        context=context_factory(final, True),
        adjudicated=True,
    )
    if is_adjudication_requested(final_decision):
        final_decision = unresolved_decision
    return CandidateDifficultyResult(
        validation, primary, final, final_decision, True, decision.reason,
    )
