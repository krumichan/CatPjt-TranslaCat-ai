from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pytest

from app.features.language_learning.reading_vocabulary.service import (
    ReadingVocabularyGenerationService,
)
from app.features.language_learning.reading_vocabulary.vocabulary_difficulty_adapter import (
    build_vocabulary_difficulty_spec,
    project_vocabulary_validation,
)
from app.features.language_learning.reading_vocabulary.vocabulary_difficulty_recipe import (
    CompositionDemand,
    MEANING_RELATION_DIFFICULTY_RECIPE_VERSION,
    MeaningRelationDemand,
    UsageDistinctionDemand,
    VOCABULARY_DIFFICULTY_RECIPE_VERSION,
    VOCABULARY_DIFFICULTY_SHADOW_RUBRIC_VERSION,
    vocabulary_blind_rubric_payload,
    vocabulary_difficulty_recipe,
    validate_vocabulary_difficulty_recipe,
)
from app.schemas.language_learning_practice import PracticeGeneratedQuestion
from tests import test_language_learning_reading_vocabulary as practice_fixtures


@pytest.mark.parametrize(
    ("band", "distance", "near_neighbor"),
    [
        (1, "BROADLY_SEPARATED", False),
        (2, "FAMILIAR_SEMANTIC_GROUP", False),
        (3, "CLOSE_WITH_ONE_DECISIVE_FEATURE", False),
        (4, "NEAR_NEIGHBORS_WITH_NUANCE", True),
        (5, "OVERLAPPING_WITH_PRECISE_BOUNDARIES", True),
    ],
)
def test_meaning_relation_bands_map_to_server_owned_semantic_demands(
    band,
    distance,
    near_neighbor,
):
    recipe = vocabulary_difficulty_recipe(
        mode="MEANING_RELATION",
        band=band,
        skill_tag="DISTINCTION",
        question_type="SINGLE_CHOICE",
    )

    assert recipe.version == MEANING_RELATION_DIFFICULTY_RECIPE_VERSION
    assert isinstance(recipe.demand, MeaningRelationDemand)
    assert recipe.demand.semantic_distance_class == distance
    assert recipe.demand.near_neighbor_required is near_neighbor
    assert "AMBIGUITY_AS_DIFFICULTY" in recipe.demand.disallowed_shortcuts
    assert recipe.generation_payload()["modeDemand"]["authority"] == (
        "SERVER_SELECTED_REQUIREMENT"
    )


def test_generation_recipe_and_frozen_measurement_rubric_are_separate_versions():
    recipe = vocabulary_difficulty_recipe(
        mode="MEANING_RELATION",
        band=1,
        skill_tag="MEANING",
        question_type="SINGLE_CHOICE",
    )
    rubric = vocabulary_blind_rubric_payload("MEANING_RELATION")

    assert recipe.version == MEANING_RELATION_DIFFICULTY_RECIPE_VERSION
    assert rubric["version"] == VOCABULARY_DIFFICULTY_SHADOW_RUBRIC_VERSION
    assert recipe.anchor != rubric["bands"][0]["anchor"]


@pytest.mark.parametrize(
    ("skill_tag", "usage_intent", "required_dependence"),
    [
        ("DISTINCTION", "CONTEXTUAL_NEAR_EXPRESSION_CHOICE", "pragmatic_dependence"),
        ("COLLOCATION", "COLLOCATION_CHOICE", "collocation_dependence"),
        ("REGISTER", "REGISTER_CHOICE", "register_dependence"),
        ("CONTEXT_USAGE", "CONTEXTUAL_USAGE_CHOICE", "pragmatic_dependence"),
    ],
)
def test_usage_intent_selects_its_own_server_demand(
    skill_tag,
    usage_intent,
    required_dependence,
):
    recipe = vocabulary_difficulty_recipe(
        mode="USAGE_DISTINCTION",
        band=4,
        skill_tag=skill_tag,
        question_type="SINGLE_CHOICE",
        usage_intent=usage_intent,
    )

    assert isinstance(recipe.demand, UsageDistinctionDemand)
    assert recipe.demand.usage_intent == usage_intent
    assert recipe.demand.context_required is True
    assert recipe.demand.minimum_relevant_cue_kinds == 2
    assert getattr(recipe.demand, required_dependence) is True
    assert "AMBIGUITY_AS_DIFFICULTY" in recipe.demand.disallowed_shortcuts


def test_usage_b5_integrates_visible_cues_without_accepting_ambiguity():
    recipe = vocabulary_difficulty_recipe(
        mode="USAGE_DISTINCTION",
        band=5,
        skill_tag="REGISTER",
        question_type="SINGLE_CHOICE",
        usage_intent="REGISTER_CHOICE",
    )

    assert isinstance(recipe.demand, UsageDistinctionDemand)
    assert recipe.demand.cue_scope == "INTEGRATED_ROLE_REGISTER_INTENT_CONTEXT"
    assert recipe.demand.minimum_relevant_cue_kinds >= 2
    assert "AMBIGUITY_AS_DIFFICULTY" in recipe.demand.disallowed_shortcuts


@pytest.mark.parametrize(
    ("question_type", "skill_tag", "subtype"),
    [
        ("ORDERING", "COMPOSITION", "ORDERING"),
        ("SINGLE_CHOICE", "COMPOSITION", "EXPRESSION_OR_CLAUSE_COMPLETION"),
        ("SINGLE_CHOICE", "COLLOCATION", "COLLOCATION_ASSEMBLY"),
        ("SINGLE_CHOICE", "CONTEXT_USAGE", "CONTEXTUAL_COMPLETION"),
    ],
)
def test_composition_subtypes_keep_distinct_cognitive_shapes(
    question_type,
    skill_tag,
    subtype,
):
    recipe = vocabulary_difficulty_recipe(
        mode="COMPOSITION",
        band=4,
        skill_tag=skill_tag,
        question_type=question_type,
    )

    assert isinstance(recipe.demand, CompositionDemand)
    assert recipe.demand.subtype == subtype
    assert recipe.demand.required_constraint_count == 3
    assert recipe.demand.single_valid_answer_required is True
    assert "AMBIGUITY_AS_DIFFICULTY" in recipe.demand.disallowed_shortcuts


def test_generator_authored_demand_cannot_replace_canonical_recipe():
    recipe = vocabulary_difficulty_recipe(
        mode="MEANING_RELATION",
        band=4,
        skill_tag="SYNONYM",
        question_type="SINGLE_CHOICE",
    )
    assert isinstance(recipe.demand, MeaningRelationDemand)
    invalid = replace(
        recipe,
        demand=replace(recipe.demand, near_neighbor_required=False),
    )

    with pytest.raises(ValueError, match="does not match server recipe"):
        validate_vocabulary_difficulty_recipe(
            mode="MEANING_RELATION",
            band=4,
            skill_tag="SYNONYM",
            question_type="SINGLE_CHOICE",
            usage_intent=None,
            recipe=invalid,
        )

    spec = build_vocabulary_difficulty_spec(
        mode="MEANING_RELATION",
        difficulty="CURRENT",
        complexity_band=4,
        skill_tag="SYNONYM",
        question_type="SINGLE_CHOICE",
        target_expression="見直す",
    )
    validation = project_vocabulary_validation(
        replace(spec, recipe=invalid),
        lambda: None,
    )
    assert validation.passed is False


def test_meaning_relation_exact_answer_text_leak_is_deterministically_rejected():
    service = ReadingVocabularyGenerationService(practice_fixtures.PipelineProvider())
    question = PracticeGeneratedQuestion.model_validate(
        {
            "order": 1,
            "questionType": "SINGLE_CHOICE",
            "difficulty": "CURRENT",
            "complexityBand": 3,
            "passageId": None,
            "passageText": None,
            "prompt": "最も適切な内容を選んでください。",
            "options": [
                {"key": "A", "text": "最も適切な内容"},
                {"key": "B", "text": "別の意味"},
                {"key": "C", "text": "反対の意味"},
                {"key": "D", "text": "無関係な意味"},
            ],
            "correctAnswer": ["A"],
            "skillTag": "MEANING",
            "evidenceText": None,
            "explanationLearning": "説明です。",
            "explanationOrigin": "설명입니다.",
            "targetExpression": "対象表現",
            "canonicalKey": "target",
            "reviewTarget": False,
            "vocabularyCandidates": [],
        }
    )

    with pytest.raises(ValueError, match="stem leaks correct option text"):
        service._validate_meaning_relation_candidate(question)


@pytest.mark.asyncio
async def test_generator_difficulty_self_report_is_replaced_by_vocabulary_slot_metadata():
    class SelfReportedDifficultyProvider(practice_fixtures.PipelineProvider):
        def _candidate(self, payload: dict, slot: dict) -> dict:
            candidate = super()._candidate(payload, slot)
            candidate.update(
                difficulty="EASIER",
                complexityBand=1,
                skillTag="REGISTER",
                questionType="ORDERING",
            )
            return candidate

    provider = SelfReportedDifficultyProvider()
    response = await ReadingVocabularyGenerationService(provider).generate(
        practice_fixtures._vocab_request(mode="MEANING_RELATION")
    )

    assert len(response.questions) == 10
    assert response.questions[0].difficulty.value == "CURRENT"
    assert response.questions[0].complexity_band == 3
    assert response.questions[0].skill_tag == "DISTINCTION"
    assert response.questions[0].question_type.value == "SINGLE_CHOICE"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "expected_calls"),
    [
        (
            "MEANING_RELATION",
            Counter(
                {
                    ReadingVocabularyGenerationService.TYPE_NAME: 5,
                    ReadingVocabularyGenerationService.VERIFICATION_TYPE_NAME: 1,
                    ReadingVocabularyGenerationService.ORIGIN_EXPLANATION_TYPE_NAME: 3,
                }
            ),
        ),
        (
            "USAGE_DISTINCTION",
            Counter(
                {
                    ReadingVocabularyGenerationService.TYPE_NAME: 10,
                    ReadingVocabularyGenerationService.PRESCREEN_TYPE_NAME: 10,
                    ReadingVocabularyGenerationService.VERIFICATION_TYPE_NAME: 10,
                    ReadingVocabularyGenerationService.ORIGIN_EXPLANATION_TYPE_NAME: 3,
                }
            ),
        ),
        (
            "COMPOSITION",
            Counter(
                {
                    ReadingVocabularyGenerationService.TYPE_NAME: 10,
                    ReadingVocabularyGenerationService.VERIFICATION_TYPE_NAME: 6,
                    ReadingVocabularyGenerationService.ORIGIN_EXPLANATION_TYPE_NAME: 3,
                }
            ),
        ),
    ],
)
async def test_vocabulary_mode_call_count_and_recipe_binding_baseline(
    mode,
    expected_calls,
):
    provider = practice_fixtures.PipelineProvider()
    response = await ReadingVocabularyGenerationService(provider).generate(
        practice_fixtures._vocab_request(mode=mode)
    )

    assert len(response.questions) == 10
    assert provider.calls == expected_calls
    for slot_batch in provider.candidate_slot_payloads:
        for slot in slot_batch:
            recipe = slot["difficultyRecipe"]
            expected_version = (
                MEANING_RELATION_DIFFICULTY_RECIPE_VERSION
                if mode == "MEANING_RELATION"
                else VOCABULARY_DIFFICULTY_RECIPE_VERSION
            )
            assert recipe["version"] == expected_version
            assert recipe["mode"] == mode
            assert recipe["band"] == slot["complexityBand"]
            assert recipe["modeDemand"]["authority"] == "SERVER_SELECTED_REQUIREMENT"
