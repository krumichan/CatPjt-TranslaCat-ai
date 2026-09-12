from __future__ import annotations

import asyncio
from copy import deepcopy
from typing import Any

import pytest
from pydantic import BaseModel, Field

from app.features.language_learning.writing.assessment_contract import (
    RecoveredTaskReview,
    TaskReview,
    recovered_acceptance,
)
from app.features.language_learning.difficulty.contracts import SemanticAssessment
from app.features.language_learning.writing.difficulty_adapter import (
    WritingAcceptanceContext,
    WritingDifficultyAcceptancePolicy,
    normalize_writing_assessment,
    validate_writing_candidate,
    writing_difficulty_spec,
    writing_difficulty_target,
)
from app.features.language_learning.writing.difficulty_spec import (
    build_difficulty_spec,
    measure_draft,
)
from app.features.language_learning.writing.generation import GENERATION_TASK
from app.features.language_learning.writing.generation_contract import (
    deterministic_draft_reason,
    parse_draft,
    plan_slots,
)
from app.features.language_learning.writing.verification import (
    MINI_QUALITY_TASK,
    WritingCandidateVerifier,
)
from app.features.language_learning.writing.review_diagnostics import ReviewFailure
from app.features.language_learning.writing.review_runtime import (
    ReviewContext,
    ReviewMetrics,
    ReviewRuntime,
)
from tests.test_language_learning_writing_verified_generation import draft, request, service
from tests.writing_generation_fakes import (
    WritingPipelineProvider,
    reject_issues,
    task_review,
)


class IdentityEcho(BaseModel):
    candidate_id: str = Field(alias="candidateId")
    content_hash: str = Field(alias="contentHash")


class SchemaCapturingProvider:
    def __init__(self) -> None:
        self.schemas: list[dict[str, Any]] = []

    async def call(
        self,
        type_name: str,
        data: Any,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        assert schema is not None
        self.schemas.append(deepcopy(schema))
        return {"candidateId": "candidate-1", "contentHash": "a" * 64}

    async def call_with_image(
        self,
        type_name: str,
        prompt: str,
        image_bytes: bytes,
        mime_type: str,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise AssertionError("Unexpected image request")


@pytest.mark.parametrize("mode", ["TRANSLATION", "GUIDED", "FREE"])
@pytest.mark.parametrize("base,difficulty", [(1, "REVIEW"), (3, "NORMAL"), (5, "CHALLENGE")])
def test_target_and_spec_adapter_are_exact_views_of_writing_policy(mode, base, difficulty):
    req = request(mode=mode, band=base, difficulty=difficulty)
    slot = plan_slots(req)[0]
    target = writing_difficulty_target(slot)
    adapted = writing_difficulty_spec(req, slot)
    direct = build_difficulty_spec(req, slot)
    assert target.value == slot.target_band
    assert adapted.target == target
    assert adapted.version == direct.version
    assert adapted.generation_payload() == direct.generation_payload()


def test_validation_adapter_preserves_reason_and_measurements():
    req = request()
    slot = plan_slots(req)[0]
    candidate = parse_draft(draft())
    result = validate_writing_candidate(req, candidate, slot)
    assert result.primary_issue == deterministic_draft_reason(req, candidate, slot=slot)
    assert dict(result.measurements) == measure_draft(candidate).log_fields()


@pytest.mark.parametrize("observed", range(1, 6))
def test_acceptance_adapter_delegates_to_the_existing_writing_table(observed):
    req = request(band=3, difficulty="NORMAL")
    slot = plan_slots(req)[0]
    data = {"candidateId": "c", "contentHash": "a" * 64, "writingType": "TRANSLATION"}
    review = TaskReview.model_validate(task_review(data, observed))
    direct = recovered_acceptance(review, slot)
    normalized = normalize_writing_assessment(review)
    adapted = WritingDifficultyAcceptancePolicy().decide(
        target=writing_difficulty_target(slot),
        assessment=normalized,
        context=WritingAcceptanceContext(slot, True),
        adjudicated=False,
    )
    assert isinstance(normalized, SemanticAssessment)
    assert normalized.observed_target == review.estimated_band
    assert (adapted.action, adapted.reason) == (direct.action, direct.reason)


@pytest.mark.parametrize("status", ["PASS", "FAIL", "UNSURE"])
@pytest.mark.parametrize("adjudicated", [False, True])
def test_recovered_source_acceptance_adapter_matches_existing_table(status, adjudicated):
    req = request(band=3, difficulty="NORMAL")
    slot = plan_slots(req)[0]
    data = {"candidateId": "c", "contentHash": "a" * 64, "writingType": "TRANSLATION"}
    raw = task_review(data, 3)
    raw.update(
        recoveryHash="b" * 64,
        sourcePreservation={
            "status": status,
            "issues": ["MEANING_CHANGED"] if status == "FAIL" else [],
            "evidenceSegmentIds": [] if status == "UNSURE" else ["S0", "O1"],
        },
    )
    review = RecoveredTaskReview.model_validate(raw)
    direct = recovered_acceptance(review, slot, adjudicated=adjudicated)
    adapted = WritingDifficultyAcceptancePolicy().decide(
        target=writing_difficulty_target(slot),
        assessment=normalize_writing_assessment(review),
        context=WritingAcceptanceContext(slot, True),
        adjudicated=adjudicated,
    )
    assert (adapted.action, adapted.reason) == (direct.action, direct.reason)


@pytest.mark.parametrize("observed", [2, 4])
def test_adjacent_adjudication_adapter_matches_existing_table(observed):
    req = request(band=3, difficulty="NORMAL")
    slot = plan_slots(req)[0]
    data = {"candidateId": "c", "contentHash": "a" * 64, "writingType": "TRANSLATION"}
    review = TaskReview.model_validate(task_review(data, observed))
    direct = recovered_acceptance(review, slot, allow_adjacent_recheck=True)
    adapted = WritingDifficultyAcceptancePolicy().decide(
        target=writing_difficulty_target(slot),
        assessment=normalize_writing_assessment(review),
        context=WritingAcceptanceContext(slot, True),
        adjudicated=False,
    )
    assert (adapted.action, adapted.reason) == (direct.action, direct.reason)


@pytest.mark.parametrize("issue", ["NOTE_ORIGIN_LANGUAGE", "ANSWER_LEAK"])
def test_note_and_hard_issue_actions_match_existing_table(issue):
    req = request(band=3, difficulty="NORMAL")
    slot = plan_slots(req)[0]
    data = {"candidateId": "c", "contentHash": "a" * 64, "writingType": "TRANSLATION"}
    raw = task_review(data, 3)
    raw = reject_issues(issue)(raw, data)
    review = TaskReview.model_validate(raw)
    direct = recovered_acceptance(review, slot)
    adapted = WritingDifficultyAcceptancePolicy().decide(
        target=writing_difficulty_target(slot),
        assessment=normalize_writing_assessment(review),
        context=WritingAcceptanceContext(slot, True),
        adjudicated=False,
    )
    assert (adapted.action, adapted.reason) == (direct.action, direct.reason)


@pytest.mark.parametrize("confidence", [None, 0.0, 0.25, 0.73, 1.0])
def test_confidence_does_not_change_acceptance_adapter_decision(confidence):
    req = request(band=3, difficulty="NORMAL")
    slot = plan_slots(req)[0]
    data = {"candidateId": "c", "contentHash": "a" * 64, "writingType": "TRANSLATION"}
    raw = task_review(data, 3)
    raw["confidence"] = confidence
    raw["difficultyConfidence"] = confidence
    review = TaskReview.model_validate(raw)
    adapted = WritingDifficultyAcceptancePolicy().decide(
        target=writing_difficulty_target(slot),
        assessment=normalize_writing_assessment(review),
        context=WritingAcceptanceContext(slot, True),
        adjudicated=False,
    )
    assert adapted.action == "ACCEPT"


def test_legacy_direct_verify_still_reviews_a_deterministically_invalid_draft():
    req = request(band=3, difficulty="NORMAL")
    slot = plan_slots(req)[0]
    raw = draft()
    raw["providedFacts"] = ["legacy direct-call input"]
    candidate = parse_draft(raw)
    assert deterministic_draft_reason(req, candidate, slot=slot) is not None
    provider = WritingPipelineProvider([], {raw["originText"]: 3})
    result = asyncio.run(
        WritingCandidateVerifier(provider, 1).verify(req, candidate, slot, "legacy-candidate")
    )
    assert result.accepted
    assert provider.counts == {MINI_QUALITY_TASK: 1}


def test_writing_facade_keeps_type_names_and_counter_missing_key_semantics():
    context = ReviewContext("r", "c", "a" * 64)
    failure = ReviewFailure("VERIFIER_TEST", "PROTOCOL")
    metrics = ReviewMetrics()
    assert type(context).__name__ == "ReviewContext" and repr(context).startswith("ReviewContext(")
    assert type(failure).__name__ == "ReviewFailure" and repr(failure).startswith("ReviewFailure(")
    assert type(metrics).__name__ == "ReviewMetrics" and repr(metrics).startswith("ReviewMetrics(")
    assert metrics.elapsed_ms["missing"] == 0
    assert "missing" not in metrics.summary()["review_elapsed_ms"]


def test_writing_review_schema_keeps_identity_enum_binding():
    provider = SchemaCapturingProvider()
    result = asyncio.run(
        ReviewRuntime(provider, 1).review(
            task=MINI_QUALITY_TASK,
            prompt="review",
            model=IdentityEcho,
            context=ReviewContext("request-1", "candidate-1", "a" * 64),
            inspect=lambda _result: None,
            max_attempts=1,
        )
    )

    assert result is not None
    assert len(provider.schemas) == 1
    properties = provider.schemas[0]["properties"]
    assert properties["candidateId"]["enum"] == ["candidate-1"]
    assert properties["contentHash"]["enum"] == ["a" * 64]


def test_writing_review_keeps_one_or_two_attempt_limit():
    provider = SchemaCapturingProvider()
    with pytest.raises(
        ValueError,
        match="Review stage permits one initial call and at most one retry",
    ):
        asyncio.run(
            ReviewRuntime(provider, 1).review(
                task=MINI_QUALITY_TASK,
                prompt="review",
                model=IdentityEcho,
                context=ReviewContext("request-1", "candidate-1", "a" * 64),
                inspect=lambda _result: None,
                max_attempts=3,
            )
        )
    assert provider.schemas == []


def test_common_wiring_keeps_normal_path_at_one_generation_and_one_primary_mini():
    item = draft()
    provider = WritingPipelineProvider([{"items": [item]}], {item["originText"]: 3})
    result = asyncio.run(service(provider, generation_max_retries=0).generate_daily(request()))
    assert len(result.items) == 1
    assert provider.counts == {GENERATION_TASK: 1, MINI_QUALITY_TASK: 1}
