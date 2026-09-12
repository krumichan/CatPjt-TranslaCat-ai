"""Typed semantic evidence and deterministic acceptance for the Writing reference.

Self-reported confidence is diagnostic only. An explicit UNSURE is not a PASS.
Evidence IDs are bound to actual fields; no ID can prove a semantic judgment true.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import Field, StrictStr, model_validator

from app.features.language_learning.writing.generation_contract import WritingDraft, WritingSlot
from app.features.language_learning.writing.review_diagnostics import ReviewFailure
from app.features.language_learning.writing.review_segments import writing_segments
from app.schemas.language_learning import CamelCaseModel

Confidence = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
Band = Annotated[int, Field(strict=True, ge=1, le=5)]
SegmentId = Annotated[StrictStr, Field(min_length=2, max_length=12)]
EvidenceIds = Annotated[list[SegmentId], Field(max_length=12)]
Criterion = Literal["ORIGIN_LANGUAGE", "TASK_VALIDITY", "ANSWER_LEAK", "NATURALNESS", "NOTE_QUALITY"]
Issue = Literal[
    "ORIGIN_LANGUAGE", "ANSWER_LEAK", "TASK_TYPE", "MISSING_FACTS", "CONTRADICTORY_GUIDANCE",
    "AMBIGUOUS_TASK", "BACKGROUND_KNOWLEDGE", "UNNATURAL_LANGUAGE", "NOT_LANGUAGE_TASK",
    "NOTE_ORIGIN_LANGUAGE", "UNSUPPORTED_FOCUS_REASON", "INTERNAL_CLAIM",
]

ISSUE_CRITERION = {
    "ORIGIN_LANGUAGE": "ORIGIN_LANGUAGE",
    "ANSWER_LEAK": "ANSWER_LEAK",
    "TASK_TYPE": "TASK_VALIDITY", "MISSING_FACTS": "TASK_VALIDITY",
    "CONTRADICTORY_GUIDANCE": "TASK_VALIDITY", "AMBIGUOUS_TASK": "TASK_VALIDITY",
    "BACKGROUND_KNOWLEDGE": "TASK_VALIDITY", "NOT_LANGUAGE_TASK": "TASK_VALIDITY",
    "UNNATURAL_LANGUAGE": "NATURALNESS",
    "NOTE_ORIGIN_LANGUAGE": "NOTE_QUALITY", "UNSUPPORTED_FOCUS_REASON": "NOTE_QUALITY",
    "INTERNAL_CLAIM": "NOTE_QUALITY",
}
CRITERIA = frozenset(ISSUE_CRITERION.values())


class CriterionCheck(CamelCaseModel):
    criterion: Criterion
    status: Literal["PASS", "FAIL", "UNSURE"]
    evidence_segment_ids: EvidenceIds

    @model_validator(mode="after")
    def check_evidence(self) -> "CriterionCheck":
        if len(self.evidence_segment_ids) != len(set(self.evidence_segment_ids)):
            raise ValueError("Duplicate evidence segment IDs")
        if self.status != "UNSURE" and not self.evidence_segment_ids:
            raise ValueError("Definite checks require evidence segment IDs")
        return self


class BoundReview(CamelCaseModel):
    candidate_id: Annotated[StrictStr, Field(min_length=1, max_length=100)]
    content_hash: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    confidence: Confidence | None


class TaskReview(BoundReview):
    verdict: Literal["PASS", "REJECT", "UNSURE"]
    observed_writing_type: Literal["TRANSLATION", "GUIDED", "FREE", "UNSURE"]
    difficulty_status: Literal["ASSESSED", "BORDERLINE", "UNSURE"]
    estimated_band: Band | None
    alternative_band: Band | None
    difficulty_confidence: Confidence | None
    difficulty_evidence_segment_ids: EvidenceIds
    checks: list[CriterionCheck] = Field(..., min_length=5, max_length=5)
    issues: list[Issue] = Field(..., max_length=12)

    @model_validator(mode="after")
    def check_consistency(self) -> "TaskReview":
        if {check.criterion for check in self.checks} != CRITERIA:
            raise ValueError("Every semantic criterion must occur exactly once")
        if len(self.issues) != len(set(self.issues)):
            raise ValueError("Duplicate task review issues")
        ids = self.difficulty_evidence_segment_ids
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate difficulty evidence segment IDs")
        if self.difficulty_status == "ASSESSED":
            if self.estimated_band is None or self.alternative_band is not None or not ids:
                raise ValueError("Assessed difficulty requires one band and evidence")
        elif self.difficulty_status == "BORDERLINE":
            if (self.estimated_band is None or self.alternative_band is None or not ids
                    or abs(self.estimated_band - self.alternative_band) != 1):
                raise ValueError("Borderline difficulty requires two adjacent bands and evidence")
        elif self.estimated_band is not None or self.alternative_band is not None:
            raise ValueError("Unknown difficulty must not claim a band")
        failed = {check.criterion for check in self.checks if check.status == "FAIL"}
        issue_criteria = {ISSUE_CRITERION[issue] for issue in self.issues}
        if failed != issue_criteria:
            raise ValueError("Failed criteria and explicit issues must agree")
        uncertain = (any(check.status == "UNSURE" for check in self.checks)
                     or self.observed_writing_type == "UNSURE" or self.difficulty_status != "ASSESSED")
        expected = "REJECT" if failed else "UNSURE" if uncertain else "PASS"
        if self.verdict != expected:
            raise ValueError("Verdict contradicts criterion or difficulty results")
        return self


class NoteReview(BoundReview):
    verdict: Literal["PASS", "REJECT", "UNSURE"]
    issues: list[Literal["ORIGIN_LANGUAGE", "ANSWER_LEAK", "UNSUPPORTED_FOCUS_REASON", "INTERNAL_CLAIM"]] = Field(..., max_length=4)
    evidence_segment_ids: EvidenceIds

    @model_validator(mode="after")
    def check_consistency(self) -> "NoteReview":
        if len(self.issues) != len(set(self.issues)):
            raise ValueError("Duplicate note issues")
        if len(self.evidence_segment_ids) != len(set(self.evidence_segment_ids)):
            raise ValueError("Duplicate note evidence IDs")
        if self.verdict != "UNSURE" and "N1" not in self.evidence_segment_ids:
            raise ValueError("Definite note decisions require note evidence")
        if (self.verdict == "REJECT") != bool(self.issues):
            raise ValueError("Note verdict contradicts issues")
        return self


def binding_failure(
    review: TaskReview | NoteReview, draft: WritingDraft, candidate_id: str, content_hash: str,
) -> ReviewFailure | None:
    if review.candidate_id != candidate_id:
        return ReviewFailure("VERIFIER_IDENTITY_MISMATCH", "PROTOCOL")
    if review.content_hash != content_hash:
        return ReviewFailure("VERIFIER_CONTENT_HASH_MISMATCH", "PROTOCOL")
    fields = {segment.id: segment.field for segment in writing_segments(draft)}
    groups = ([review.evidence_segment_ids] if isinstance(review, NoteReview) else
              [review.difficulty_evidence_segment_ids, *(check.evidence_segment_ids for check in review.checks)])
    if any(segment_id not in fields for ids in groups for segment_id in ids):
        return ReviewFailure("VERIFIER_EVIDENCE_SEGMENT_INVALID", "PROTOCOL")
    if isinstance(review, TaskReview):
        if "N1" in review.difficulty_evidence_segment_ids:
            return ReviewFailure("VERIFIER_EVIDENCE_SCOPE_MISMATCH", "PROTOCOL")
        for check in review.checks:
            if check.status == "UNSURE":
                continue
            ids = set(check.evidence_segment_ids)
            if check.criterion == "NOTE_QUALITY" and "N1" not in ids:
                return ReviewFailure("VERIFIER_EVIDENCE_SCOPE_MISMATCH", "PROTOCOL")
            if check.criterion != "NOTE_QUALITY" and not ids.difference({"N1"}):
                return ReviewFailure("VERIFIER_EVIDENCE_SCOPE_MISMATCH", "PROTOCOL")
            if check.criterion == "ANSWER_LEAK" and check.status == "PASS" and "N1" not in ids:
                return ReviewFailure("VERIFIER_EVIDENCE_SCOPE_MISMATCH", "PROTOCOL")
    return None


def build_review_schema(model: type[TaskReview] | type[NoteReview], draft: WritingDraft) -> dict:
    """Per-request closed evidence IDs; desired band/verdict are NEVER restricted."""
    schema = model.model_json_schema(by_alias=True)
    ids = [segment.id for segment in writing_segments(draft)]
    if issubclass(model, TaskReview):
        schema["properties"]["difficultyEvidenceSegmentIds"]["items"]["enum"] = [value for value in ids if value != "N1"]
        schema["$defs"]["CriterionCheck"]["properties"]["evidenceSegmentIds"]["items"]["enum"] = ids
    else:
        schema["properties"]["evidenceSegmentIds"]["items"]["enum"] = ids
    return schema


@dataclass(frozen=True)
class AcceptanceResult:
    action: Literal["ACCEPT", "REJECT", "ADJUDICATE", "LOCALIZE_NOTE"]
    reason: str


def assess_acceptance(review: TaskReview, slot: WritingSlot, *, adjudicated: bool = False,
                      allow_adjacent_recheck: bool = True) -> AcceptanceResult:
    """Single decision table. Confidence cannot influence any branch.

    Explicit uncertainty keeps its single final-review path. A quality-PASS
    adjacent band estimate may also get one fresh full review, if the caller's
    per-slot budget allows it. Far mismatches and quality failures are final.
    The final review must independently PASS at the exact target; never average
    votes or silently promote/demote a band. Confidence is never a gate.
    """
    hard_issues = [issue for issue in review.issues if issue != "NOTE_ORIGIN_LANGUAGE"]
    if hard_issues:
        return AcceptanceResult("REJECT", "QUALITY_" + hard_issues[0])
    if review.observed_writing_type not in {slot.writing_type.value, "UNSURE"}:
        return AcceptanceResult("REJECT", "QUALITY_TASK_TYPE")
    if review.difficulty_status == "ASSESSED" and review.estimated_band != slot.target_band:
        if (not adjudicated and allow_adjacent_recheck and review.verdict == "PASS"
                and review.estimated_band is not None
                and abs(review.estimated_band - slot.target_band) == 1):
            return AcceptanceResult("ADJUDICATE", "ADJACENT_BAND_RECHECK")
        return AcceptanceResult("REJECT", "VERIFIED_BAND_MISMATCH")
    if review.difficulty_status == "BORDERLINE" and slot.target_band not in {review.estimated_band, review.alternative_band}:
        return AcceptanceResult("REJECT", "VERIFIED_BAND_MISMATCH")
    uncertain = (review.observed_writing_type == "UNSURE" or review.difficulty_status != "ASSESSED"
                 or any(check.status == "UNSURE" for check in review.checks))
    if uncertain:
        return AcceptanceResult("REJECT", "SEMANTIC_UNRESOLVED") if adjudicated else AcceptanceResult("ADJUDICATE", "EXPLICIT_UNCERTAINTY")
    if review.issues == ["NOTE_ORIGIN_LANGUAGE"]:
        return AcceptanceResult("LOCALIZE_NOTE", "NOTE_ORIGIN_LANGUAGE")
    if review.verdict == "PASS":
        return AcceptanceResult("ACCEPT", "VERIFIED")
    return AcceptanceResult("REJECT", "SEMANTIC_UNRESOLVED")


class SourceMeaningCheck(CamelCaseModel):
    status: Literal["PASS", "FAIL", "UNSURE"]
    issues: list[Literal["MEANING_CHANGED", "FACT_CHANGED", "POLARITY_CHANGED", "REGISTER_CHANGED",
                         "UNSUPPORTED_ADDITION"]] = Field(..., max_length=5)
    evidence_segment_ids: list[Literal["S0", "O1"]] = Field(..., max_length=2)

    @model_validator(mode="after")
    def check_preservation(self) -> "SourceMeaningCheck":
        if len(self.issues) != len(set(self.issues)):
            raise ValueError("Duplicate preservation issues")
        if (self.status == "FAIL") != bool(self.issues):
            raise ValueError("Preservation status and issues disagree")
        if len(self.evidence_segment_ids) != len(set(self.evidence_segment_ids)):
            raise ValueError("Duplicate preservation evidence IDs")
        if self.status != "UNSURE" and set(self.evidence_segment_ids) != {"S0", "O1"}:
            raise ValueError("Preservation requires original and localized source evidence")
        return self


class RecoveredTaskReview(TaskReview):
    """Extra evidence ONLY for repaired sources; ordinary V4 contract is unchanged."""
    recovery_hash: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{64}$")]
    source_preservation: SourceMeaningCheck


def recovered_acceptance(review: TaskReview, slot: WritingSlot, *, adjudicated: bool = False,
                         allow_adjacent_recheck: bool = True) -> AcceptanceResult:
    base = assess_acceptance(review, slot, adjudicated=adjudicated,
                             allow_adjacent_recheck=allow_adjacent_recheck)
    if base.action == "REJECT":
        return base
    if isinstance(review, RecoveredTaskReview):
        if review.source_preservation.status == "FAIL":
            return AcceptanceResult("REJECT", "SOURCE_MEANING_" + review.source_preservation.issues[0])
        if review.source_preservation.status == "UNSURE":
            return (AcceptanceResult("REJECT", "SOURCE_MEANING_UNRESOLVED") if adjudicated else
                    AcceptanceResult("ADJUDICATE", "SOURCE_MEANING_UNSURE"))
    return base
