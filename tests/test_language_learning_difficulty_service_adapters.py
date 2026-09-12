from __future__ import annotations

import copy

import pytest

from app.features.language_learning.difficulty.contracts import DifficultySpec
from app.features.language_learning.listening.difficulty_adapter import (
    build_listening_difficulty_spec,
    validate_listening_candidate,
)
from app.features.language_learning.level_test.difficulty_adapter import (
    build_level_test_difficulty_spec,
    project_level_test_candidate_validation,
)
from app.features.language_learning.reading_vocabulary.reading_difficulty_adapter import (
    ReadingAcceptanceContext,
    ReadingSemanticQualityPolicy,
    build_reading_passage_difficulty_spec,
    build_reading_question_difficulty_spec,
    normalize_reading_semantic_assessment,
    project_reading_question_validation,
    validate_reading_passage,
)
from app.features.language_learning.reading_vocabulary.vocabulary_difficulty_adapter import (
    CompositionSemanticQualityPolicy,
    MeaningRelationSemanticQualityPolicy,
    UsageDistinctionSemanticQualityPolicy,
    VocabularyAcceptanceContext,
    build_vocabulary_difficulty_spec,
    normalize_vocabulary_semantic_assessment,
    project_vocabulary_validation,
    semantic_quality_policy_for_vocabulary_mode,
)
from app.features.language_learning.speaking.generation_difficulty_adapter import (
    build_speaking_generation_difficulty_spec,
    project_speaking_generation_validation,
)
from app.schemas.language_learning_listening import GeneratedListeningItemPayload
from tests import test_language_learning_listening as listening_fixtures
from tests import test_language_learning_reading_vocabulary as practice_fixtures
from tests import test_language_learning_speaking as speaking_fixtures
from tests import test_language_learning_current as current_fixtures


def _listening_candidate() -> GeneratedListeningItemPayload:
    return GeneratedListeningItemPayload.model_validate(
        copy.deepcopy(listening_fixtures.generation_payload()["items"][0])
    )


def test_listening_adapter_binds_text_duration_and_mode_target():
    request = listening_fixtures.generation_request()
    spec = build_listening_difficulty_spec(request, duration_range=(8.0, 20.0))

    assert isinstance(spec, DifficultySpec)
    assert spec.target.service == "listening"
    assert spec.target.scale == "listening-generation"
    assert spec.target.value.text_band == 3
    assert (spec.duration_min_seconds, spec.duration_max_seconds) == (8.0, 20.0)
    assert spec.target.value.mode == "DICTATION"

    other_duration_spec = build_listening_difficulty_spec(
        request,
        duration_range=(9.0, 21.0),
    )
    assert other_duration_spec.target == spec.target
    assert other_duration_spec != spec


def test_listening_validation_projection_preserves_ordered_acceptance_and_rejection():
    request = listening_fixtures.generation_request()
    candidate = _listening_candidate()
    resolver_calls = 0

    def duration_range() -> tuple[float, float]:
        nonlocal resolver_calls
        resolver_calls += 1
        return 8.0, 20.0

    spec, accepted = validate_listening_candidate(
        request,
        candidate,
        duration_range_resolver=duration_range,
        mode_payload_validator=lambda: True,
    )
    assert spec is not None
    assert accepted.passed
    assert accepted.measurements["textBand"] == 3
    assert resolver_calls == 1

    wrong_band = candidate.model_copy(update={"language_complexity_band": 4})
    rejected_spec, rejected = validate_listening_candidate(
        request,
        wrong_band,
        duration_range_resolver=duration_range,
        mode_payload_validator=lambda: True,
    )
    assert rejected_spec is None
    assert rejected.primary_issue == "LISTENING_TEXT_BAND_OR_SAFETY_REJECTED"
    assert resolver_calls == 1


def test_reading_adapter_keeps_passage_and_question_targets_distinct():
    request = practice_fixtures._request(question_count=1, easier=0, current=1, challenge=0)
    passage_spec = build_reading_passage_difficulty_spec(
        request,
        passage_id="p1",
        complexity_band=3,
    )
    question_spec = build_reading_question_difficulty_spec(
        difficulty="CURRENT",
        complexity_band=3,
        skill_tag="MAIN_IDEA",
        passage_id="p1",
    )

    assert isinstance(passage_spec, DifficultySpec)
    assert isinstance(question_spec, DifficultySpec)
    assert passage_spec.target.scale == "reading-passage-generation"
    assert passage_spec.target.value.mode == "COMPREHENSION"
    assert passage_spec.passage_id == "p1"
    assert question_spec.target.scale == "reading-question-generation"
    assert question_spec.target.value.skill_tag == "MAIN_IDEA"
    assert question_spec.passage_id == "p1"


def test_reading_identity_changes_spec_but_not_difficulty_target():
    request = practice_fixtures._request(question_count=1, easier=0, current=1, challenge=0)
    passage_p1 = build_reading_passage_difficulty_spec(
        request,
        passage_id="p1",
        complexity_band=3,
    )
    passage_p2 = build_reading_passage_difficulty_spec(
        request,
        passage_id="p2",
        complexity_band=3,
    )
    question_p1 = build_reading_question_difficulty_spec(
        difficulty="CURRENT",
        complexity_band=3,
        skill_tag="MAIN_IDEA",
        passage_id="p1",
    )
    question_p2 = build_reading_question_difficulty_spec(
        difficulty="CURRENT",
        complexity_band=3,
        skill_tag="MAIN_IDEA",
        passage_id="p2",
    )

    assert passage_p1.target == passage_p2.target
    assert passage_p1.passage_id != passage_p2.passage_id
    assert question_p1.target == question_p2.target
    assert question_p1.passage_id != question_p2.passage_id


def test_reading_validation_projection_preserves_existing_failure_text():
    request = practice_fixtures._request(question_count=1, easier=0, current=1, challenge=0)
    passage_spec = build_reading_passage_difficulty_spec(
        request,
        passage_id="p1",
        complexity_band=3,
    )
    accepted = validate_reading_passage(
        passage_spec,
        observed_passage_id="p1",
        passage_text="本文",
        language_validator=lambda: None,
    )
    assert accepted.passed

    rejected = validate_reading_passage(
        passage_spec,
        observed_passage_id="p2",
        passage_text="本文",
        language_validator=lambda: None,
    )
    assert rejected.primary_issue == "reading passageId mismatch"

    question_spec = build_reading_question_difficulty_spec(
        difficulty="CURRENT",
        complexity_band=3,
        skill_tag="MAIN_IDEA",
        passage_id="p1",
    )

    def reject_question() -> None:
        raise ValueError("candidate complexityBand mismatch")

    question_rejection = project_reading_question_validation(
        question_spec,
        reject_question,
    )
    question_acceptance = project_reading_question_validation(
        question_spec,
        lambda: None,
    )
    assert question_acceptance.passed
    assert question_rejection.primary_issue == "candidate complexityBand mismatch"


@pytest.mark.parametrize(
    ("overrides", "expected_reason"),
    [
        ({"ambiguous": True, "supported": False}, "ambiguous single-choice item"),
        ({"supported": False}, "answer is not sufficiently supported"),
        ({"mode_fit": False}, "question does not fit requested mode/skill"),
        ({"answer_leakage": True}, "semantic verifier detected answer leakage"),
        ({"distractors_plausible": False}, "distractors are too weak or unrelated"),
        ({"best_answer_key": "B"}, "answer mismatch expected=A verifier=B"),
    ],
)
def test_reading_semantic_quality_policy_preserves_rejection_precedence(
    overrides,
    expected_reason,
):
    values = {
        "best_answer_key": "A",
        "ambiguous": False,
        "supported": True,
        "mode_fit": True,
        "answer_leakage": False,
        "distractors_plausible": True,
        **overrides,
    }
    assessment = normalize_reading_semantic_assessment(**values)
    decision = ReadingSemanticQualityPolicy().decide(
        assessment=assessment,
        context=ReadingAcceptanceContext(expected_answer_key="A"),
    )

    assert decision.action == "REJECT"
    assert decision.reason == expected_reason


def test_reading_semantic_quality_is_explicitly_not_difficulty_calibrated():
    assessment = normalize_reading_semantic_assessment(
        best_answer_key="A",
        ambiguous=False,
        supported=True,
        mode_fit=True,
        answer_leakage=False,
        distractors_plausible=True,
    )

    assert assessment.difficulty_status == "NOT_ASSESSED"
    assert assessment.observed_target is None
    assert assessment.alternative_target is None


@pytest.mark.parametrize(
    ("mode", "policy_type"),
    [
        ("MEANING_RELATION", MeaningRelationSemanticQualityPolicy),
        ("USAGE_DISTINCTION", UsageDistinctionSemanticQualityPolicy),
        ("COMPOSITION", CompositionSemanticQualityPolicy),
    ],
)
def test_vocabulary_modes_bind_distinct_acceptance_policies(mode, policy_type):
    spec = build_vocabulary_difficulty_spec(
        mode=mode,
        difficulty="CURRENT",
        complexity_band=3,
        skill_tag="CONTEXT_USAGE",
        question_type="SINGLE_CHOICE",
        target_expression="語彙",
    )

    assert isinstance(spec, DifficultySpec)
    assert spec.target.service == "vocabulary"
    assert spec.target.value.mode == mode
    assert spec.question_type == "SINGLE_CHOICE"
    assert spec.target_expression == "語彙"
    assert isinstance(semantic_quality_policy_for_vocabulary_mode(mode), policy_type)


def test_vocabulary_binding_changes_spec_but_not_difficulty_target():
    first = build_vocabulary_difficulty_spec(
        mode="COMPOSITION",
        difficulty="CURRENT",
        complexity_band=3,
        skill_tag="CONTEXT_USAGE",
        question_type="SINGLE_CHOICE",
        target_expression="語彙",
    )
    second = build_vocabulary_difficulty_spec(
        mode="COMPOSITION",
        difficulty="CURRENT",
        complexity_band=3,
        skill_tag="CONTEXT_USAGE",
        question_type="ORDERING",
        target_expression="表現",
    )

    assert first.target == second.target
    assert first.question_type != second.question_type
    assert first.target_expression != second.target_expression


def test_vocabulary_validation_projection_preserves_existing_failure_text():
    spec = build_vocabulary_difficulty_spec(
        mode="MEANING_RELATION",
        difficulty="CURRENT",
        complexity_band=3,
        skill_tag="SYNONYM",
        question_type="SINGLE_CHOICE",
        target_expression="語彙",
    )

    def reject_candidate() -> None:
        raise ValueError("vocabulary canonicalKey duplicates an accepted slot")

    result = project_vocabulary_validation(spec, reject_candidate)

    assert project_vocabulary_validation(spec, lambda: None).passed
    assert result.primary_issue == "vocabulary canonicalKey duplicates an accepted slot"


@pytest.mark.parametrize(
    ("mode", "context_dependent", "expected_action", "expected_reason"),
    [
        ("MEANING_RELATION", False, "ACCEPT", "semantic verifier accepted candidate"),
        (
            "USAGE_DISTINCTION",
            False,
            "REJECT",
            "USAGE_DISTINCTION does not require context",
        ),
        ("COMPOSITION", False, "ACCEPT", "semantic verifier accepted candidate"),
    ],
)
def test_vocabulary_mode_policies_preserve_context_requirement(
    mode,
    context_dependent,
    expected_action,
    expected_reason,
):
    assessment = normalize_vocabulary_semantic_assessment(
        best_answer_key="A",
        ambiguous=False,
        supported=True,
        mode_fit=True,
        answer_leakage=False,
        context_dependent=context_dependent,
        distractors_plausible=True,
    )
    decision = semantic_quality_policy_for_vocabulary_mode(mode).decide(
        assessment=assessment,
        context=VocabularyAcceptanceContext(expected_answer_key="A"),
    )

    assert decision.action == expected_action
    assert decision.reason == expected_reason
    assert assessment.best_answer_key == "A"
    assert assessment.context_dependent is context_dependent


def test_vocabulary_semantic_quality_is_explicitly_not_difficulty_calibrated():
    assessment = normalize_vocabulary_semantic_assessment(
        best_answer_key="A",
        ambiguous=False,
        supported=True,
        mode_fit=True,
        answer_leakage=False,
        context_dependent=True,
        distractors_plausible=True,
    )

    assert assessment.difficulty_status == "NOT_ASSESSED"
    assert assessment.observed_target is None
    assert assessment.alternative_target is None


def test_speaking_generation_adapter_binds_level_and_practice_mode():
    request = speaking_fixtures.conversation_request(practiceMode="GUIDED")
    spec = build_speaking_generation_difficulty_spec(request)

    assert isinstance(spec, DifficultySpec)
    assert spec.target.service == "speaking-generation"
    assert spec.target.value.target_level == "A2"
    assert spec.target.value.practice_mode == "GUIDED"


def test_speaking_generation_validation_preserves_existing_failure_text():
    request = speaking_fixtures.conversation_request(
        practiceMode="READ_ALOUD",
        problemIndex=1,
        attemptIndex=1,
    )
    spec = build_speaking_generation_difficulty_spec(request)

    def reject_payload() -> None:
        raise ValueError("READ_ALOUD는 assistantText와 동일한 scriptText가 필요합니다.")

    result = project_speaking_generation_validation(spec, reject_payload)

    assert project_speaking_generation_validation(spec, lambda: None).passed
    assert result.primary_issue == "READ_ALOUD는 assistantText와 동일한 scriptText가 필요합니다."


def test_level_test_adapter_binds_generation_contract_without_runtime_policy():
    request = current_fixtures.level_question_request()
    spec = build_level_test_difficulty_spec(request)

    assert isinstance(spec, DifficultySpec)
    assert spec.target.service == "level-test"
    assert spec.question_number == 1
    assert spec.target.value.domain == "VOCABULARY"
    assert spec.target.value.item_type == "VOCAB_CONTEXT_CHOICE"
    assert spec.target.value.complexity_band == 2
    assert spec.target.policy_version == request.policy_version


def test_level_test_question_identity_changes_spec_but_not_difficulty_target():
    first_request = current_fixtures.level_question_request()
    second_request = first_request.model_copy(update={"question_number": 2})
    first = build_level_test_difficulty_spec(first_request)
    second = build_level_test_difficulty_spec(second_request)

    assert first.target == second.target
    assert first.question_number == 1
    assert second.question_number == 2


def test_level_test_validation_projection_preserves_existing_failure_text():
    request = current_fixtures.level_question_request()
    spec = build_level_test_difficulty_spec(request)

    def reject_candidate() -> None:
        raise ValueError("Provider가 요청과 다른 complexityBand를 반환했습니다.")

    result = project_level_test_candidate_validation(spec, reject_candidate)

    assert project_level_test_candidate_validation(spec, lambda: None).passed
    assert result.primary_issue == "Provider가 요청과 다른 complexityBand를 반환했습니다."
