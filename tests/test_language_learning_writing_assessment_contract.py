"""Closed evidence references, consistency and acceptance decision table."""
from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from app.features.language_learning.writing.assessment_contract import (
    CRITERIA, TaskReview, assess_acceptance, binding_failure, build_review_schema,
)
from app.features.language_learning.writing.generation_contract import parse_draft, plan_slots, review_content_hash
from app.features.language_learning.writing.review_segments import writing_segments
from tests.test_language_learning_writing_verified_generation import draft, request
from tests.writing_generation_fakes import task_review, reject_issues, unsure_review


def bound(mode="TRANSLATION"):
    req, item = request(mode=mode), parse_draft(draft(mode=mode))
    data = {"candidateId": "candidate", "contentHash": review_content_hash(req, item), "writingType": mode}
    return req, item, data, task_review(data, 3)


@pytest.mark.parametrize("mode", ["TRANSLATION", "GUIDED", "FREE"])
def test_ids_are_server_owned_stable_and_cover_all_visible_fields(mode):
    _, item, _, _ = bound(mode)
    parts = writing_segments(item)
    assert parts[0].id == "O1" and parts[-1].id == "N1"
    assert len({p.id for p in parts}) == len(parts)
    assert len(parts) == 2 + len(item.provided_facts) + len(item.required_intents) + len(item.response_constraints)
    assert writing_segments(item) == parts
    item.focus_reason += " 변경"
    assert writing_segments(item)[-1].text != parts[-1].text


@pytest.mark.parametrize("mode", ["TRANSLATION", "GUIDED", "FREE"])
def test_schema_references_only_this_candidate_segments_and_no_desired_verdict(mode):
    _, item, _, _ = bound(mode)
    schema = build_review_schema(TaskReview, item)
    all_ids = {s.id for s in writing_segments(item)}
    assert set(schema["properties"]["difficultyEvidenceSegmentIds"]["items"]["enum"]) == all_ids - {"N1"}
    assert set(schema["$defs"]["CriterionCheck"]["properties"]["evidenceSegmentIds"]["items"]["enum"]) == all_ids
    assert schema["properties"]["verdict"]["enum"] == ["PASS", "REJECT", "UNSURE"]
    assert "enum" not in schema["properties"]["estimatedBand"]
    assert "quote" not in str(schema)


@pytest.mark.parametrize("criterion", sorted(CRITERIA))
def test_missing_or_duplicated_criterion_is_not_an_implicit_pass(criterion):
    _, _, _, value = bound()
    value["checks"] = [c for c in value["checks"] if c["criterion"] != criterion]
    with pytest.raises(ValidationError):
        TaskReview.model_validate(value)
    value["checks"].append(copy.deepcopy(value["checks"][0]))
    with pytest.raises(ValidationError):
        TaskReview.model_validate(value)


@pytest.mark.parametrize("criterion", sorted(CRITERIA))
def test_each_definite_criterion_must_reference_actual_evidence(criterion):
    _, item, data, value = bound()
    row = next(c for c in value["checks"] if c["criterion"] == criterion)
    row["evidenceSegmentIds"] = []
    with pytest.raises(ValidationError):
        TaskReview.model_validate(value)
    row["evidenceSegmentIds"] = ["Z1"]
    failure = binding_failure(TaskReview.model_validate(value), item, data["candidateId"], data["contentHash"])
    assert failure.code == "VERIFIER_EVIDENCE_SEGMENT_INVALID"


@pytest.mark.parametrize("criterion", sorted(CRITERIA))
def test_note_and_task_evidence_scopes_cannot_be_substituted(criterion):
    _, item, data, value = bound()
    row = next(c for c in value["checks"] if c["criterion"] == criterion)
    row["evidenceSegmentIds"] = ["O1"] if criterion in {"NOTE_QUALITY", "ANSWER_LEAK"} else ["N1"]
    failure = binding_failure(TaskReview.model_validate(value), item, data["candidateId"], data["contentHash"])
    assert failure.code == "VERIFIER_EVIDENCE_SCOPE_MISMATCH"


@pytest.mark.parametrize("kind", ["PASS_WITH_FAIL", "REJECT_WITHOUT_ISSUE", "ISSUE_WITHOUT_FAIL", "UNSURE_WITHOUT_UNCERTAINTY"])
def test_contradictory_judgments_are_protocol_failures(kind):
    _, _, _, value = bound()
    if kind == "PASS_WITH_FAIL":
        value["checks"][0]["status"] = "FAIL"
    elif kind == "REJECT_WITHOUT_ISSUE":
        value["verdict"] = "REJECT"
    elif kind == "ISSUE_WITHOUT_FAIL":
        value["issues"] = ["ANSWER_LEAK"]
    else:
        value["verdict"] = "UNSURE"
    with pytest.raises(ValidationError):
        TaskReview.model_validate(value)


@pytest.mark.parametrize("bands", [(1, 3), (3, 3), (0, 1), (5, 6), (None, 3)])
def test_borderline_means_two_real_adjacent_bands_not_arbitrary_range(bands):
    _, _, data, value = bound()
    value = unsure_review(borderline=bands)(value, data)
    with pytest.raises(ValidationError):
        TaskReview.model_validate(value)


@pytest.mark.parametrize("confidence", [0.0, 0.72, 0.73, 1.0, None])
@pytest.mark.parametrize("issue", ["ANSWER_LEAK", "AMBIGUOUS_TASK", "UNSUPPORTED_FOCUS_REASON"])
def test_hard_failure_is_never_overridden_even_by_confident_judge(confidence, issue):
    req, _, data, value = bound()
    value = reject_issues(issue)(value, data)
    value["confidence"] = confidence
    result = assess_acceptance(TaskReview.model_validate(value), plan_slots(req)[0])
    assert result.action == "REJECT" and result.reason == "QUALITY_" + issue


def test_unknown_check_is_not_logged_as_arbitrary_field_text():
    from app.features.language_learning.writing.review_diagnostics import validation_failure
    _, _, _, value = bound()
    value["checks"][0]["criterion"] = "SECRET_BODY"
    with pytest.raises(ValidationError) as exc:
        TaskReview.model_validate(value)
    result = validation_failure(exc.value)
    assert result.schema_errors == (("checks.0.criterion", "literal_error"),)
    assert "SECRET" not in str(result.fields())
