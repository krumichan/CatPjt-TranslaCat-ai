from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from app.features.language_learning.difficulty.contracts import (
    DifficultyAcceptanceDecision,
    DifficultyTarget,
    DifficultyValidationResult,
)
from app.features.language_learning.difficulty.orchestration import orchestrate_candidate


@dataclass(frozen=True)
class Assessment:
    observed: int


@dataclass(frozen=True)
class Context:
    allow_recheck: bool


@dataclass(frozen=True)
class OpaquePolicy:
    recheck_action: str = "SERVICE_RECHECK"

    def decide(self, *, target, assessment, context, adjudicated):
        if assessment.observed == target.value:
            return DifficultyAcceptanceDecision("SERVICE_ACCEPT", "EXACT")
        if not adjudicated and context.allow_recheck:
            return DifficultyAcceptanceDecision(self.recheck_action, "RECHECK")
        return DifficultyAcceptanceDecision("SERVICE_DENY", "MISMATCH")


def invoke(**updates):
    values = {
        "target": DifficultyTarget("x", "custom", 3, "v1"),
        "validation": DifficultyValidationResult.accept(),
        "assessor": lambda: value(Assessment(3)),
        "policy": OpaquePolicy(),
        "context_factory": lambda _assessment, _adjudicated: Context(True),
        "deterministic_rejection": lambda checked: DifficultyAcceptanceDecision(
            "SERVICE_DENY", checked.primary_issue or "INVALID",
        ),
        "is_adjudication_requested": lambda decision: decision.action == "SERVICE_RECHECK",
        "unresolved_decision": DifficultyAcceptanceDecision("SERVICE_DENY", "UNRESOLVED"),
    }
    values.update(updates)
    return orchestrate_candidate(**values)


def test_deterministic_rejection_uses_service_factory_and_short_circuits_assessor():
    calls = []

    async def assess():
        calls.append("assess")
        return Assessment(3)

    result = asyncio.run(invoke(
        validation=DifficultyValidationResult.reject("SERVICE_RULE"),
        assessor=assess,
    ))
    assert result.decision == DifficultyAcceptanceDecision("SERVICE_DENY", "SERVICE_RULE")
    assert not calls


def test_custom_opaque_action_requests_one_adjudication_after_reservation():
    trace = []

    async def primary():
        trace.append("primary")
        return Assessment(2)

    async def adjudicator():
        trace.append("adjudicator")
        return Assessment(3)

    result = asyncio.run(invoke(
        assessor=primary,
        adjudicator=adjudicator,
        before_adjudication=lambda _decision: trace.append("reserved"),
    ))
    assert trace == ["primary", "reserved", "adjudicator"]
    assert result.adjudicated
    assert result.decision == DifficultyAcceptanceDecision("SERVICE_ACCEPT", "EXACT")
    assert result.adjudication_reason == "RECHECK"


def test_policy_can_disable_adjudication_without_common_defaults():
    result = asyncio.run(invoke(
        assessor=lambda: value(Assessment(2)),
        context_factory=lambda _assessment, _adjudicated: Context(False),
    ))
    assert result.decision == DifficultyAcceptanceDecision("SERVICE_DENY", "MISMATCH")
    assert not result.adjudicated


def test_missing_adjudicator_fails_closed_without_claiming_adjudication():
    result = asyncio.run(invoke(assessor=lambda: value(Assessment(2))))
    assert result.decision == DifficultyAcceptanceDecision("SERVICE_DENY", "UNRESOLVED")
    assert result.assessment == Assessment(2)
    assert not result.adjudicated
    assert result.adjudication_reason == "RECHECK"


def test_missing_primary_or_adjudication_result_fails_closed():
    no_primary = asyncio.run(invoke(assessor=lambda: value(None)))
    assert no_primary.assessment is None
    assert no_primary.decision.reason == "UNRESOLVED"
    assert not no_primary.adjudicated

    no_final = asyncio.run(invoke(
        assessor=lambda: value(Assessment(2)),
        adjudicator=lambda: value(None),
    ))
    assert no_final.assessment is None
    assert no_final.decision.reason == "UNRESOLVED"
    assert no_final.adjudicated


def test_second_adjudication_request_fails_closed_and_never_calls_twice():
    calls = 0

    async def adjudicator():
        nonlocal calls
        calls += 1
        return Assessment(2)

    class AlwaysRecheckPolicy:
        def decide(self, **_kwargs):
            return DifficultyAcceptanceDecision("SERVICE_RECHECK", "AGAIN")

    result = asyncio.run(invoke(
        assessor=lambda: value(Assessment(2)),
        adjudicator=adjudicator,
        policy=AlwaysRecheckPolicy(),
    ))
    assert calls == 1
    assert result.adjudicated
    assert result.decision == DifficultyAcceptanceDecision("SERVICE_DENY", "UNRESOLVED")


def test_adjudication_cancellation_propagates_and_cleans_up():
    entered = asyncio.Event()
    cleaned = asyncio.Event()

    async def adjudicator():
        entered.set()
        try:
            await asyncio.Future()
        finally:
            cleaned.set()

    async def scenario():
        task = asyncio.create_task(invoke(
            assessor=lambda: value(Assessment(2)),
            adjudicator=adjudicator,
        ))
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cleaned.is_set()

    asyncio.run(scenario())


async def value(result):
    return result
