from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pytest

from app.core.config import settings
from app.features.language_learning.reading_vocabulary.reading_difficulty_adapter import (
    ReadingSemanticDifficultyPolicy,
    build_reading_passage_difficulty_spec,
    build_reading_question_difficulty_spec,
    measure_reading_passage,
    measure_reading_question,
    normalize_reading_semantic_difficulty_assessment,
    reading_passage_segments,
)
from app.features.language_learning.reading_vocabulary.reading_difficulty_recipe import (
    EvidenceScope,
    QuestionDemandKind,
    READING_DIFFICULTY_RECIPE_VERSION,
    READING_DIFFICULTY_SHADOW_RUBRIC_VERSION,
    calibrated_reading_skill,
    passage_difficulty_recipe,
    question_demand_recipe,
    reading_blind_rubric_payload,
    validate_question_demand_blueprint,
)
from app.features.language_learning.reading_vocabulary.service import (
    ReadingVocabularyGenerationService,
)
from app.features.language_learning.reading_vocabulary.prompts import (
    PRACTICE_VERIFICATION_SYSTEM_PROMPT,
    READING_STRUCTURE_MODE_FIT_CLARIFICATION,
)
from app.schemas.language_learning_practice import (
    PracticeGeneratedQuestion,
    PracticeGenerationRequest,
)
from tests import test_language_learning_reading_vocabulary as practice_fixtures


@pytest.fixture(autouse=True)
def historical_calibration_fake_uses_luna(monkeypatch):
    """Old deterministic calibration fixtures do not implement the Sol task route."""
    monkeypatch.setattr(settings, "AI_READING_GENERATION_MODEL", "LUNA")


class ReadingDifficultyProvider(practice_fixtures.PipelineProvider):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.passage_prompts: list[str] = []
        self.verification_schemas: list[dict | None] = []

    async def call(self, type_name, data, schema=None):
        if type_name == ReadingVocabularyGenerationService.PASSAGE_TYPE_NAME:
            self.passage_prompts.append(data)
        if type_name == ReadingVocabularyGenerationService.VERIFICATION_TYPE_NAME:
            self.verification_schemas.append(schema)
        result = await super().call(type_name, data, schema)
        if type_name == ReadingVocabularyGenerationService.PASSAGE_TYPE_NAME:
            payload = practice_fixtures._practice_data(data)
            passage_demand = payload["difficultyRecipe"]["passageDemand"]
            if passage_demand["crossParagraphDependencyRequired"]:
                result["passageText"] = (
                    "今日は会社で会議があります。担当者は資料を確認しました。\n\n"
                    "その後、顧客への説明方法について話し合いました。"
                )
        return result

    async def call_with_image(
        self,
        type_name: str,
        prompt: str,
        image_bytes: bytes,
        mime_type: str,
        schema: dict | None = None,
    ):
        raise AssertionError("Reading calibration must not call an image provider")

def _question(*, passage: str, evidence: str | None = "次です。") -> PracticeGeneratedQuestion:
    return PracticeGeneratedQuestion.model_validate(
        {
            "order": 1,
            "questionType": "SINGLE_CHOICE",
            "difficulty": "CURRENT",
            "complexityBand": 3,
            "passageId": "p1",
            "passageText": passage,
            "prompt": "本文の内容として正しいものを選んでください。",
            "options": [
                {"key": "A", "text": "次です。"},
                {"key": "B", "text": "前です。"},
                {"key": "C", "text": "別です。"},
                {"key": "D", "text": "同じです。"},
            ],
            "correctAnswer": ["A"],
            "skillTag": "DETAIL",
            "evidenceText": evidence,
            "explanationOrigin": "두 번째 문장이 근거입니다.",
            "explanationLearning": "二番目の文が根拠です。",
            "targetExpression": None,
            "canonicalKey": None,
            "reviewTarget": False,
            "vocabularyCandidates": [],
        }
    )


def _request_for_band(mode: str, band: int) -> PracticeGenerationRequest:
    payload = practice_fixtures._request(
        mode=mode,
        question_count=1,
        easier=0,
        current=1,
        challenge=0,
    ).model_dump(mode="json", by_alias=True)
    payload["complexityBand"] = band
    return PracticeGenerationRequest.model_validate(payload)


def test_reading_passage_and_question_recipes_are_separately_bound_and_versioned():
    request = practice_fixtures._request(
        question_count=1,
        easier=0,
        current=1,
        challenge=0,
    )
    passage = build_reading_passage_difficulty_spec(
        request,
        passage_id="p1",
        complexity_band=3,
    )
    question = build_reading_question_difficulty_spec(
        mode="COMPREHENSION",
        difficulty="CURRENT",
        complexity_band=3,
        skill_tag="DETAIL",
        passage_id="p1",
    )

    assert passage.recipe.kind == "PASSAGE"
    assert question.recipe.kind == "QUESTION_DEMAND"
    assert passage.recipe.version == READING_DIFFICULTY_RECIPE_VERSION
    assert question.recipe.version == READING_DIFFICULTY_RECIPE_VERSION
    assert passage.recipe.anchor != question.recipe.anchor
    assert "normalizedCharacterCount" in passage.recipe.deterministic_dimensions
    assert "evidenceExactMatchAndOffset" in question.recipe.deterministic_dimensions


def test_reading_measurements_are_nfc_stable_and_report_evidence_geometry():
    composed = measure_reading_passage("Caféです。\n\n次です。最後です。")
    decomposed = measure_reading_passage("Cafe\u0301です。\n\n次です。最後です。")
    assert composed == decomposed
    assert composed.paragraph_count == 2
    assert composed.surface_unit_count == 3

    passage = "最初です。\n\n次です。最後です。"
    measurements = measure_reading_question(
        _question(passage=passage),
        passage_id="p1",
        passage_text=passage,
    )
    assert measurements.option_count == 4
    assert measurements.evidence_exact_match is True
    assert measurements.evidence_offset == passage.index("次です。")
    assert measurements.evidence_paragraph == 2
    assert measurements.evidence_end_paragraph == 2
    assert measurements.evidence_paragraph_coverage == 1
    assert measurements.evidence_surface_unit == 2
    assert measurements.evidence_end_surface_unit == 2
    assert measurements.evidence_surface_unit_coverage == 1
    assert measurements.evidence_to_question_surface_unit_distance == 1
    assert measurements.passage_binding_matches is True

    cross_paragraph = measure_reading_question(
        _question(passage=passage, evidence="最初です。\n\n次です。"),
        passage_id="p1",
        passage_text=passage,
    )
    assert cross_paragraph.evidence_paragraph_coverage == 2
    assert cross_paragraph.evidence_surface_unit_coverage == 2


@pytest.mark.parametrize(
    ("band", "kind", "scope", "minimum_cues"),
    [
        (1, QuestionDemandKind.EXPLICIT_LOCAL, EvidenceScope.LOCAL, 1),
        (2, QuestionDemandKind.PARAPHRASED_LOCAL, EvidenceScope.ADJACENT_UNITS, 1),
        (3, QuestionDemandKind.LOCAL_RELATION, EvidenceScope.ADJACENT_UNITS, 2),
        (4, QuestionDemandKind.CROSS_UNIT_INFERENCE, EvidenceScope.CROSS_UNIT, 2),
        (5, QuestionDemandKind.QUALIFIED_MULTI_CUE, EvidenceScope.CROSS_UNIT, 3),
    ],
)
def test_comprehension_v2_maps_each_band_to_server_owned_question_demand(
    band: int,
    kind: QuestionDemandKind,
    scope: EvidenceScope,
    minimum_cues: int,
):
    recipe = question_demand_recipe(
        band,
        mode="COMPREHENSION",
        skill_tag="CONTENT" if band < 4 else "INFERENCE",
    )

    assert recipe.question_demand is not None
    assert recipe.question_demand.kind == kind
    assert recipe.question_demand.evidence_scope == scope
    assert recipe.question_demand.minimum_distinct_cues == minimum_cues
    assert recipe.generation_payload()["questionDemand"]["authority"] == (
        "SERVER_SELECTED_REQUIREMENT"
    )


@pytest.mark.parametrize(
    ("band", "kind", "scope"),
    [
        (1, QuestionDemandKind.PARAPHRASED_LOCAL, EvidenceScope.LOCAL),
        (2, QuestionDemandKind.DISCOURSE_STRUCTURE, EvidenceScope.ADJACENT_UNITS),
        (3, QuestionDemandKind.DISCOURSE_STRUCTURE, EvidenceScope.CROSS_PARAGRAPH),
        (4, QuestionDemandKind.DISCOURSE_STRUCTURE, EvidenceScope.CROSS_PARAGRAPH),
        (5, QuestionDemandKind.WHOLE_TEXT_SYNTHESIS, EvidenceScope.WHOLE_TEXT),
    ],
)
def test_structure_v2_maps_each_band_to_discourse_demand(
    band: int,
    kind: QuestionDemandKind,
    scope: EvidenceScope,
):
    recipe = question_demand_recipe(
        band,
        mode="STRUCTURE",
        skill_tag="GIST" if band < 3 else "STRUCTURE",
    )

    assert recipe.question_demand is not None
    assert recipe.question_demand.kind == kind
    assert recipe.question_demand.evidence_scope == scope
    assert any(
        "RETRIEVAL" in shortcut
        for shortcut in recipe.question_demand.disallowed_shortcuts
    )


def test_high_band_blueprints_reject_low_demand_skill_or_recipe_combinations():
    with pytest.raises(ValueError, match="low-demand DETAIL"):
        question_demand_recipe(5, mode="COMPREHENSION", skill_tag="DETAIL")
    with pytest.raises(ValueError, match="STRUCTURE skill"):
        question_demand_recipe(4, mode="STRUCTURE", skill_tag="GIST")

    valid = question_demand_recipe(5, mode="COMPREHENSION", skill_tag="INFERENCE")
    assert valid.question_demand is not None
    invalid = replace(
        valid,
        question_demand=replace(
            valid.question_demand,
            kind=QuestionDemandKind.EXPLICIT_LOCAL,
            evidence_scope=EvidenceScope.LOCAL,
        ),
    )
    with pytest.raises(ValueError, match="does not match server recipe"):
        validate_question_demand_blueprint(
            mode="COMPREHENSION",
            band=5,
            skill_tag="INFERENCE",
            recipe=invalid,
        )


def test_high_band_slot_mapping_removes_detail_and_content_only_structure_shortcuts():
    service = ReadingVocabularyGenerationService(ReadingDifficultyProvider())
    passages = {"p1": "本文一。", "p2": "本文二。"}

    comprehension = practice_fixtures._request().model_copy(
        update={"mode": "COMPREHENSION", "complexity_band": 4}
    )
    comprehension_slots = service._build_slots(comprehension, passages)
    challenge = next(
        slot for slot in comprehension_slots if slot.difficulty.value == "CHALLENGE"
    )
    assert challenge.complexity_band == 5
    assert challenge.skill_tag == "INFERENCE"
    assert challenge.prompt_payload()["difficultyRecipe"]["questionDemand"]["kind"] == (
        "QUALIFIED_MULTI_CUE"
    )

    structure = comprehension.model_copy(update={"mode": "STRUCTURE"})
    structure_slots = service._build_slots(structure, passages)
    assert all(
        slot.skill_tag == "STRUCTURE"
        for slot in structure_slots
        if slot.complexity_band >= 3
    )
    structure_challenge = next(
        slot for slot in structure_slots if slot.difficulty.value == "CHALLENGE"
    )
    assert structure_challenge.prompt_payload()["difficultyRecipe"]["questionDemand"][
        "kind"
    ] == "WHOLE_TEXT_SYNTHESIS"


def test_context_inference_keeps_v1_skill_cycle_and_has_no_v2_demand_override():
    service = ReadingVocabularyGenerationService(ReadingDifficultyProvider())
    request = practice_fixtures._request(mode="CONTEXT_INFERENCE")
    slots = service._build_slots(request, {"p1": "本文一。", "p2": "本文二。"})

    assert [slot.skill_tag for slot in slots] == [
        "CONTEXT_INFERENCE",
        "INFERENCE",
        "CONTEXT_INFERENCE",
        "INFERENCE",
        "CONTEXT_INFERENCE",
    ]
    recipes = [slot.prompt_payload()["difficultyRecipe"] for slot in slots]
    assert all("questionDemand" not in recipe for recipe in recipes)
    assert recipes[0]["anchor"] == (
        "Requires gist, one bounded inference, or a clear cause/condition/structure relation."
    )
    assert recipes[0]["deterministicDimensions"] == [
        "promptCharacterCount",
        "optionCountAndCharacterLengths",
        "evidenceExactMatchAndOffset",
        "evidenceParagraphAndSurfaceUnit",
        "evidenceToQuestionSurfaceUnitDistance",
        "skillTagAndPassageBinding",
    ]
    assert recipes[0]["enforcement"] == (
        "Preferred dimensions guide generation and shadow measurement only. "
        "They are not empirical hard thresholds."
    )
    assert calibrated_reading_skill(
        mode="CONTEXT_INFERENCE",
        band=5,
        planned_skill="INFERENCE",
    ) == "INFERENCE"
    context_rubric = reading_blind_rubric_payload()
    assert context_rubric["questionDemandBands"][2]["anchor"] == recipes[0]["anchor"]


def test_v2_shadow_retest_keeps_the_v1_question_measurement_rubric():
    rubric = reading_blind_rubric_payload()
    v1_context_recipe = question_demand_recipe(
        4,
        mode="CONTEXT_INFERENCE",
        skill_tag="CONTEXT_INFERENCE",
    )
    v2_comprehension_recipe = question_demand_recipe(
        4,
        mode="COMPREHENSION",
        skill_tag="INFERENCE",
    )

    assert READING_DIFFICULTY_RECIPE_VERSION == "reading-difficulty-recipe-v2"
    assert rubric["version"] == READING_DIFFICULTY_SHADOW_RUBRIC_VERSION
    assert rubric["version"] == "reading-difficulty-recipe-v1-shadow"
    assert rubric["questionDemandBands"][3]["anchor"] == v1_context_recipe.anchor
    assert rubric["questionDemandBands"][3]["anchor"] != v2_comprehension_recipe.anchor


def test_comprehension_passage_v2_adds_relations_without_changing_other_modes():
    comprehension = passage_difficulty_recipe(4, mode="COMPREHENSION")
    structure = passage_difficulty_recipe(4, mode="STRUCTURE")
    context = passage_difficulty_recipe(4, mode="CONTEXT_INFERENCE")

    assert "crossUnitDependencyForComprehension" in comprehension.semantic_dimensions
    assert "crossUnitDependencyForComprehension" not in structure.semantic_dimensions
    assert "crossUnitDependencyForComprehension" not in context.semantic_dimensions


@pytest.mark.asyncio
async def test_generator_difficulty_self_report_is_replaced_by_server_slot_metadata():
    class SelfReportedDifficultyProvider(ReadingDifficultyProvider):
        def _candidate(self, payload: dict, slot: dict) -> dict:
            candidate = super()._candidate(payload, slot)
            candidate.update(
                difficulty="EASIER",
                complexityBand=1,
                skillTag="DETAIL",
            )
            return candidate

    provider = SelfReportedDifficultyProvider()
    response = await ReadingVocabularyGenerationService(provider).generate(
        _request_for_band("COMPREHENSION", 4)
    )

    question = response.questions[0]
    assert question.difficulty.value == "CURRENT"
    assert question.complexity_band == 4
    assert question.skill_tag == "CONTENT"
    assert provider.calls[ReadingVocabularyGenerationService.TYPE_NAME] == 1


def test_quality_mini_contract_only_adds_structure_mode_fit_clarification():
    assert READING_STRUCTURE_MODE_FIT_CLARIFICATION in PRACTICE_VERIFICATION_SYSTEM_PROMPT
    assert "difficulty" not in PRACTICE_VERIFICATION_SYSTEM_PROMPT.lower()


def test_reading_difficulty_parser_supports_assessed_and_adjacent_results():
    allowed = {"passage:p1:unit:1", "question:1:prompt"}
    assessed = normalize_reading_semantic_difficulty_assessment(
        {
            "difficultyStatus": "ASSESSED",
            "observedBand": 3,
            "alternativeBand": None,
            "issueCodes": ["ONE_BOUNDED_INFERENCE"],
            "evidenceSegmentIds": ["question:1:prompt", "untrusted free text"],
            "difficultyConfidence": 0.73,
        },
        allowed_evidence_refs=allowed,
    )
    borderline = normalize_reading_semantic_difficulty_assessment(
        {
            "difficultyStatus": "BORDERLINE",
            "observedBand": 3,
            "alternativeBand": 4,
            "issueCodes": [],
            "evidenceSegmentIds": ["passage:p1:unit:1"],
            "difficultyConfidence": 0.5,
        },
        allowed_evidence_refs=allowed,
    )

    assert assessed.difficulty_status == "ASSESSED"
    assert assessed.observed_target == 3
    assert assessed.evidence_refs == ("question:1:prompt",)
    assert borderline.difficulty_status == "BORDERLINE"
    assert (borderline.observed_target, borderline.alternative_target) == (3, 4)


def test_reading_difficulty_confidence_never_changes_shadow_comparison():
    policy = ReadingSemanticDifficultyPolicy()
    actions = {
        policy.decide(
            requested_band=4,
            assessment=normalize_reading_semantic_difficulty_assessment(
                {
                    "difficultyStatus": "ASSESSED",
                    "observedBand": 2,
                    "alternativeBand": None,
                    "issueCodes": [],
                    "evidenceSegmentIds": [],
                    "difficultyConfidence": confidence,
                }
            ),
        ).action
        for confidence in (0.01, 0.99)
    }
    assert actions == {"SHADOW_MISMATCH"}


@pytest.mark.asyncio
async def test_reading_generation_binds_recipes_but_verifier_is_blind_and_call_count_is_unchanged():
    provider = ReadingDifficultyProvider()
    response = await ReadingVocabularyGenerationService(provider).generate(
        practice_fixtures._request()
    )

    service = ReadingVocabularyGenerationService
    assert len(response.questions) == 5
    assert provider.calls == Counter(
        {
            service.PASSAGE_TYPE_NAME: 2,
            service.TYPE_NAME: 3,
            service.VERIFICATION_TYPE_NAME: 1,
            service.ORIGIN_EXPLANATION_TYPE_NAME: 2,
        }
    )
    for prompt in provider.passage_prompts:
        recipe = practice_fixtures._practice_data(prompt)["difficultyRecipe"]
        assert recipe["kind"] == "PASSAGE"
        assert recipe["version"] == READING_DIFFICULTY_RECIPE_VERSION
    for slot_batch in provider.candidate_slot_payloads:
        for slot in slot_batch:
            assert slot["difficultyRecipe"]["kind"] == "QUESTION_DEMAND"
            assert slot["difficultyRecipe"]["band"] == slot["complexityBand"]

    verification = provider.verification_payloads[0]
    assert "readingDifficultyRubric" not in verification
    for question in verification["questions"]:
        assert "complexityBand" not in question
        assert "difficulty" not in question
        assert "difficultyRecipe" not in question
        assert "passageSegments" not in question
        assert "questionSegments" not in question

    schema = provider.verification_schemas[0]
    assert schema is not None
    assert "passageDifficultyAssessments" not in schema["properties"]
    assert "difficulty" not in schema["properties"]["verdicts"]["items"]["properties"]


@pytest.mark.asyncio
async def test_vocabulary_verification_schema_has_no_reading_shadow_fields():
    provider = ReadingDifficultyProvider()
    response = await ReadingVocabularyGenerationService(provider).generate(
        practice_fixtures._vocab_request()
    )

    assert len(response.questions) == 10
    schema = provider.verification_schemas[0]
    assert schema is not None
    assert "passageDifficultyAssessments" not in schema["properties"]
    verdict_properties = schema["properties"]["verdicts"]["items"]["properties"]
    assert "difficulty" not in verdict_properties
    verification = provider.verification_payloads[0]
    assert "readingDifficultyRubric" not in verification
    assert all(
        "passageSegments" not in question and "questionSegments" not in question
        for question in verification["questions"]
    )


@pytest.mark.asyncio
async def test_reading_quality_failure_still_regenerates_only_failed_slot():
    provider = ReadingDifficultyProvider(
        semantic_ambiguous_once={2},
    )
    response = await ReadingVocabularyGenerationService(provider).generate(
        practice_fixtures._request()
    )

    assert len(response.questions) == 5
    assert provider.calls[ReadingVocabularyGenerationService.VERIFICATION_TYPE_NAME] == 2
    assert provider.candidate_slot_calls[-1] == [2]
    assert provider.slot_occurrences[2] == 2
    assert all(
        provider.slot_occurrences[order] == 1
        for order in range(1, 6)
        if order != 2
    )


def test_reading_passage_segment_ids_are_stable_and_do_not_contain_raw_text():
    segments = reading_passage_segments("p1", "最初です。\n次です。")
    assert [segment["id"] for segment in segments] == [
        "passage:p1:unit:1",
        "passage:p1:unit:2",
    ]
    assert all("最初" not in segment["id"] for segment in segments)
