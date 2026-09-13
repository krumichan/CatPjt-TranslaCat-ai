from __future__ import annotations

from dataclasses import replace

import pytest

from app.features.language_learning.reading_vocabulary.prompts import (
    build_reading_passage_prompt,
)
from app.features.language_learning.reading_vocabulary.reading_difficulty_adapter import (
    build_reading_passage_difficulty_spec,
    validate_reading_passage,
)
from app.features.language_learning.reading_vocabulary.reading_difficulty_recipe import (
    READING_DIFFICULTY_RECIPE_VERSION,
    READING_DIFFICULTY_SHADOW_RUBRIC_VERSION,
    READING_PASSAGE_BLUEPRINT_VERSION,
    passage_difficulty_recipe,
    question_demand_recipe,
    reading_blind_rubric_payload,
    validate_passage_demand_blueprint,
)
from tests import test_language_learning_reading_vocabulary as practice_fixtures


@pytest.mark.parametrize(
    ("band", "minimum_relations", "cross_paragraph", "linear_allowed"),
    [
        (1, 0, False, True),
        (2, 1, False, True),
        (3, 1, False, True),
        (4, 2, True, False),
        (5, 3, True, False),
    ],
)
def test_passage_bands_map_to_server_owned_discourse_blueprints(
    band,
    minimum_relations,
    cross_paragraph,
    linear_allowed,
):
    recipe = passage_difficulty_recipe(band, mode="COMPREHENSION")
    demand = recipe.passage_demand

    assert demand is not None
    assert demand.minimum_interacting_relation_categories == minimum_relations
    assert demand.cross_paragraph_dependency_required is cross_paragraph
    assert demand.local_only_linear_discourse_allowed is linear_allowed
    assert demand.external_knowledge_allowed is False
    assert demand.version == READING_PASSAGE_BLUEPRINT_VERSION
    assert recipe.generation_payload()["passageDemand"]["authority"] == (
        "SERVER_SELECTED_REQUIREMENT"
    )


def test_b3_b4_and_b5_passage_blueprints_keep_distinct_boundaries():
    b3 = passage_difficulty_recipe(3, mode="COMPREHENSION")
    b4 = passage_difficulty_recipe(4, mode="COMPREHENSION")
    b5 = passage_difficulty_recipe(5, mode="COMPREHENSION")

    assert b3.passage_demand != b4.passage_demand
    assert b4.passage_demand != b5.passage_demand
    assert b3.anchor != b4.anchor
    assert b4.anchor != b5.anchor
    assert "crossParagraphDependency" in b4.semantic_dimensions
    assert "crossParagraphDependency" not in b3.semantic_dimensions
    assert b3.passage_demand is not None
    assert b4.passage_demand is not None
    assert b5.passage_demand is not None
    assert b3.passage_demand.interpretive_requirements == ()
    assert len(b4.passage_demand.interpretive_requirements) == 1
    assert len(b5.passage_demand.interpretive_requirements) > len(
        b4.passage_demand.interpretive_requirements
    )
    assert (
        b4.passage_demand.interpretive_requirements
        != b5.passage_demand.interpretive_requirements
    )
    assert (
        b4.passage_demand.minimum_interacting_relation_categories
        < b5.passage_demand.minimum_interacting_relation_categories
    )


@pytest.mark.parametrize(
    ("mode", "expected_emphasis", "expected_interpretive_requirement"),
    [
        ("COMPREHENSION", "qualified overall judgment", "STANCE_WITH_LIMITATION"),
        (
            "STRUCTURE",
            "proposal, concession or competing position",
            "CONCESSION_WITH_QUALIFICATION",
        ),
        ("CONTEXT_INFERENCE", "indirect intent", "INDIRECT_INTENT"),
    ],
)
def test_b4_blueprint_has_shared_boundary_and_mode_specific_emphasis(
    mode,
    expected_emphasis,
    expected_interpretive_requirement,
):
    recipe = passage_difficulty_recipe(4, mode=mode)
    demand = recipe.passage_demand

    assert demand is not None
    assert demand.minimum_interacting_relation_categories == 2
    assert demand.cross_paragraph_dependency_required is True
    assert demand.local_only_linear_discourse_allowed is False
    assert expected_emphasis in " ".join(demand.mode_emphasis)
    assert demand.interpretive_requirements == (expected_interpretive_requirement,)
    assert len(demand.relation_category_options) > 2


def test_invalid_local_linear_b4_blueprint_is_rejected_deterministically():
    recipe = passage_difficulty_recipe(4, mode="STRUCTURE")
    assert recipe.passage_demand is not None
    invalid = replace(
        recipe,
        passage_demand=replace(
            recipe.passage_demand,
            minimum_interacting_relation_categories=1,
            cross_paragraph_dependency_required=False,
            local_only_linear_discourse_allowed=True,
        ),
    )

    with pytest.raises(ValueError, match="does not match server recipe"):
        validate_passage_demand_blueprint(
            mode="STRUCTURE",
            band=4,
            recipe=invalid,
        )


def test_generator_authored_b4_interpretive_requirement_cannot_replace_blueprint():
    recipe = passage_difficulty_recipe(4, mode="COMPREHENSION")
    assert recipe.passage_demand is not None
    generator_authored = replace(
        recipe,
        passage_demand=replace(
            recipe.passage_demand,
            interpretive_requirements=("INDIRECT_INTENT",),
        ),
    )

    with pytest.raises(ValueError, match="does not match server recipe"):
        validate_passage_demand_blueprint(
            mode="COMPREHENSION",
            band=4,
            recipe=generator_authored,
        )


def _passage_spec(band: int):
    request = practice_fixtures._request(mode="COMPREHENSION")
    return build_reading_passage_difficulty_spec(
        request,
        passage_id="p1",
        complexity_band=band,
    )


def test_passage_validation_rechecks_server_blueprint():
    spec = _passage_spec(4)
    assert spec.recipe.passage_demand is not None
    invalid_spec = replace(
        spec,
        recipe=replace(spec.recipe, passage_demand=None),
    )

    invalid = validate_reading_passage(
        invalid_spec,
        observed_passage_id="p1",
        passage_text="第一の判断です。\n\nしかし別の条件があります。",
        language_validator=lambda: None,
    )

    assert invalid.passed is False
    assert "blueprint is required" in invalid.primary_issue


@pytest.mark.parametrize("band", [4, 5])
def test_cross_paragraph_blueprint_rejects_one_paragraph_passage(band):
    result = validate_reading_passage(
        _passage_spec(band),
        observed_passage_id="p1",
        passage_text="一つの段落だけです。",
        language_validator=lambda: None,
    )

    assert result.passed is False
    assert "requires at least two paragraphs" in result.primary_issue


def test_b4_two_paragraph_passage_satisfies_the_structural_minimum():
    result = validate_reading_passage(
        _passage_spec(4),
        observed_passage_id="p1",
        passage_text="第一の段落です。\n\n第二の段落です。",
        language_validator=lambda: None,
    )

    assert result.passed is True
    assert result.measurements["paragraph_count"] == 2


def test_b3_one_paragraph_passage_remains_allowed():
    result = validate_reading_passage(
        _passage_spec(3),
        observed_passage_id="p1",
        passage_text="一つの段落でも有効です。",
        language_validator=lambda: None,
    )

    assert result.passed is True
    assert result.measurements["paragraph_count"] == 1


def test_semantic_discourse_quality_remains_outside_hard_validation():
    result = validate_reading_passage(
        _passage_spec(4),
        observed_passage_id="p1",
        passage_text="単純な第一段落です。\n\n単純な第二段落です。",
        language_validator=lambda: None,
    )

    assert result.passed is True


def test_passage_generation_prompt_carries_the_fixed_server_blueprint():
    request = practice_fixtures._request(mode="STRUCTURE").model_copy(
        update={"complexity_band": 4}
    )
    prompt = build_reading_passage_prompt(
        request,
        passage_id="p1",
        passage_number=1,
    )
    payload = practice_fixtures._practice_data(prompt)
    recipe = payload["difficultyRecipe"]

    assert recipe["version"] == READING_DIFFICULTY_RECIPE_VERSION
    assert recipe["passageDemand"]["version"] == READING_PASSAGE_BLUEPRINT_VERSION
    assert READING_PASSAGE_BLUEPRINT_VERSION == "reading-passage-blueprint-b4-v2"
    assert recipe["passageDemand"]["crossParagraphDependencyRequired"] is True
    assert recipe["passageDemand"]["localOnlyLinearDiscourseAllowed"] is False
    assert recipe["passageDemand"]["interpretiveRequirements"] == [
        "CONCESSION_WITH_QUALIFICATION"
    ]
    assert recipe["passageDemand"]["authority"] == "SERVER_SELECTED_REQUIREMENT"


def test_shadow_passage_measurement_rubric_remains_the_v1_instrument():
    rubric = reading_blind_rubric_payload()
    generation = passage_difficulty_recipe(4, mode="COMPREHENSION")

    assert rubric["version"] == READING_DIFFICULTY_SHADOW_RUBRIC_VERSION
    assert rubric["version"] == "reading-difficulty-recipe-v1-shadow"
    assert rubric["passageBands"][3]["anchor"] == (
        "Connected discourse using concession, indirect intent, stance or "
        "register-sensitive relations."
    )
    assert rubric["passageBands"][3]["anchor"] != generation.anchor


def test_question_recipe_v2_payload_does_not_gain_a_passage_blueprint():
    comprehension = question_demand_recipe(
        4,
        mode="COMPREHENSION",
        skill_tag="INFERENCE",
    ).generation_payload()
    structure = question_demand_recipe(
        4,
        mode="STRUCTURE",
        skill_tag="STRUCTURE",
    ).generation_payload()
    context = question_demand_recipe(
        4,
        mode="CONTEXT_INFERENCE",
        skill_tag="CONTEXT_INFERENCE",
    ).generation_payload()

    assert comprehension["questionDemand"]["kind"] == "CROSS_UNIT_INFERENCE"
    assert structure["questionDemand"]["kind"] == "DISCOURSE_STRUCTURE"
    assert "questionDemand" not in context
    assert all(
        "passageDemand" not in recipe
        for recipe in (comprehension, structure, context)
    )
