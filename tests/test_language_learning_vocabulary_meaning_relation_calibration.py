from __future__ import annotations

from collections import Counter

import pytest
from fastapi import HTTPException

from app.features.language_learning.reading_vocabulary.prompts import (
    build_practice_generation_prompt,
    build_practice_verification_prompt,
)
from app.features.language_learning.reading_vocabulary.service import (
    ReadingVocabularyGenerationService,
)
from app.features.language_learning.reading_vocabulary.vocabulary_difficulty_adapter import (
    VocabularyAcceptanceContext,
    normalize_vocabulary_semantic_assessment,
    semantic_quality_policy_for_vocabulary_mode,
)
from app.features.language_learning.reading_vocabulary.vocabulary_difficulty_recipe import (
    MEANING_RELATION_DIFFICULTY_RECIPE_VERSION,
    VOCABULARY_DIFFICULTY_RECIPE_VERSION,
    VOCABULARY_DIFFICULTY_SHADOW_RUBRIC_VERSION,
    MeaningRelationDemand,
    meaning_relation_skill_cycle,
    meaning_relation_skill_for_slot,
    vocabulary_blind_rubric_payload,
    vocabulary_difficulty_recipe,
)
from app.schemas.language_learning_practice import PracticeGeneratedQuestion
from tests import test_language_learning_reading_vocabulary as practice_fixtures


def _meaning_recipe(skill_tag: str, band: int):
    return vocabulary_difficulty_recipe(
        mode="MEANING_RELATION",
        band=band,
        skill_tag=skill_tag,
        question_type="SINGLE_CHOICE",
    )


@pytest.mark.parametrize("skill_tag", ["MEANING", "SYNONYM", "ANTONYM", "DISTINCTION"])
def test_b1_b2_keep_direct_or_familiar_relation_shapes(skill_tag):
    b1 = _meaning_recipe(skill_tag, 1)
    b2 = _meaning_recipe(skill_tag, 2)

    assert isinstance(b1.demand, MeaningRelationDemand)
    assert isinstance(b2.demand, MeaningRelationDemand)
    assert b1.version == MEANING_RELATION_DIFFICULTY_RECIPE_VERSION
    assert b2.version == MEANING_RELATION_DIFFICULTY_RECIPE_VERSION
    assert b1.demand.context_requirement == "OPTIONAL"
    assert b2.demand.context_requirement == "OPTIONAL_WHEN_NATURAL"
    assert "DIRECT" in b1.demand.task_shape or "BASIC" in b1.demand.task_shape
    assert "BARE_DICTIONARY_DEFINITION_AS_COMPLETE_TASK" not in (
        b2.demand.disallowed_shortcuts
    )


@pytest.mark.parametrize(
    ("skill_tag", "b3_shape", "high_shape_fragment"),
    [
        ("MEANING", "SENSE_IN_CONTEXT_MEANING", "CONTEXTUAL"),
        ("SYNONYM", "CONTEXT_SPECIFIC_SENSE_SYNONYM", "NEAR_SYNONYM"),
        ("ANTONYM", "CONTEXTUAL_SAME_DIMENSION_OPPOSITION", "OPPOSITION"),
        ("DISTINCTION", "ONE_DECISIVE_CONTEXTUAL_CONTRAST", "DISCRIMINATION"),
    ],
)
def test_b3_plus_requires_skill_specific_contextual_decision_shape(
    skill_tag,
    b3_shape,
    high_shape_fragment,
):
    b3 = _meaning_recipe(skill_tag, 3).demand
    b4 = _meaning_recipe(skill_tag, 4).demand
    b5 = _meaning_recipe(skill_tag, 5).demand

    assert isinstance(b3, MeaningRelationDemand)
    assert isinstance(b4, MeaningRelationDemand)
    assert isinstance(b5, MeaningRelationDemand)
    assert b3.task_shape == b3_shape
    assert "CONTEXT" in b3.context_requirement
    assert high_shape_fragment in b4.task_shape
    assert high_shape_fragment in b5.task_shape
    for demand in (b3, b4, b5):
        assert "BARE_DICTIONARY_DEFINITION_AS_COMPLETE_TASK" in (
            demand.disallowed_shortcuts
        )
        assert "STEM_DIRECTLY_PARAPHRASES_CORRECT_OPTION" in (
            demand.disallowed_shortcuts
        )


@pytest.mark.parametrize("skill_tag", ["MEANING", "SYNONYM", "ANTONYM", "DISTINCTION"])
@pytest.mark.parametrize("band", [4, 5])
def test_high_band_uses_semantic_neighbors_without_accepting_ambiguity(
    skill_tag,
    band,
):
    demand = _meaning_recipe(skill_tag, band).demand

    assert isinstance(demand, MeaningRelationDemand)
    assert demand.near_neighbor_required is True
    assert "SAME_SEMANTIC_NEIGHBORHOOD" in demand.distractor_requirements
    assert "PARTIALLY_PLAUSIBLE_IN_VISIBLE_CONTEXT" in (
        demand.distractor_requirements
    )
    assert "AMBIGUITY_AS_DIFFICULTY" in demand.disallowed_shortcuts
    assert "OUT_OF_NEIGHBORHOOD_DISTRACTORS" in demand.disallowed_shortcuts


def test_high_band_antonym_cannot_use_direct_lexical_pair_as_only_decision_basis():
    b1 = _meaning_recipe("ANTONYM", 1).demand
    b4 = _meaning_recipe("ANTONYM", 4).demand
    b5 = _meaning_recipe("ANTONYM", 5).demand

    assert isinstance(b1, MeaningRelationDemand)
    assert isinstance(b4, MeaningRelationDemand)
    assert isinstance(b5, MeaningRelationDemand)
    assert b1.task_shape == "DIRECT_LEXICAL_OPPOSITE"
    assert "DIRECT_LEXICAL_PAIR_AS_SOLE_DECISION_BASIS" not in (
        b1.disallowed_shortcuts
    )
    for demand in (b4, b5):
        assert "SAME_DIMENSION_OPPOSITION" in demand.decision_basis
        assert "DIRECT_LEXICAL_PAIR_AS_SOLE_DECISION_BASIS" in (
            demand.disallowed_shortcuts
        )


def test_high_band_distinction_does_not_teach_candidate_difference_in_stem():
    for band in (4, 5):
        demand = _meaning_recipe("DISTINCTION", band).demand
        assert isinstance(demand, MeaningRelationDemand)
        assert "VISIBLE_CONTEXT" in demand.decision_basis
        assert "STEM_TEACHES_CANDIDATE_DIFFERENCES_BEFORE_ASKING" in (
            demand.disallowed_shortcuts
        )


def _meaning_candidate(
    *,
    target_expression: str,
    correct_text: str,
    distractor_texts: tuple[str, str, str] = ("公開する", "停止する", "延期する"),
):
    return PracticeGeneratedQuestion.model_validate(
        {
            "order": 1,
            "questionType": "SINGLE_CHOICE",
            "difficulty": "CHALLENGE",
            "complexityBand": 5,
            "passageId": None,
            "passageText": None,
            "prompt": "運用環境への移行について最も適切な関係を選んでください。",
            "options": [
                {"key": "A", "text": correct_text},
                {"key": "B", "text": distractor_texts[0]},
                {"key": "C", "text": distractor_texts[1]},
                {"key": "D", "text": distractor_texts[2]},
            ],
            "correctAnswer": ["A"],
            "skillTag": "DISTINCTION",
            "evidenceText": None,
            "explanationLearning": "文脈上の意味関係を説明します。",
            "explanationOrigin": "문맥의 의미 관계를 설명합니다.",
            "targetExpression": target_expression,
            "canonicalKey": "synthetic-deployment",
            "reviewTarget": False,
            "vocabularyCandidates": [],
        }
    )


def test_target_expression_equal_to_correct_option_is_deterministically_rejected():
    service = ReadingVocabularyGenerationService(practice_fixtures.PipelineProvider())
    candidate = _meaning_candidate(
        target_expression="配備する",
        correct_text="配備する",
    )

    with pytest.raises(ValueError, match="option must not repeat targetExpression"):
        service._validate_meaning_relation_candidate(candidate)


def test_target_expression_equal_to_wrong_option_is_deterministically_rejected():
    service = ReadingVocabularyGenerationService(practice_fixtures.PipelineProvider())
    candidate = _meaning_candidate(
        target_expression="安全な導入",
        correct_text="安全な展開",
        distractor_texts=("安全な導入", "安全な運用", "安全な移行"),
    )

    with pytest.raises(ValueError, match="option must not repeat targetExpression"):
        service._validate_meaning_relation_candidate(candidate)


def test_target_expression_and_all_distinct_options_remain_valid():
    service = ReadingVocabularyGenerationService(practice_fixtures.PipelineProvider())
    candidate = _meaning_candidate(
        target_expression="安全な導入",
        correct_text="安全な展開",
        distractor_texts=("安全な公開", "安全な運用", "安全な移行"),
    )

    service._validate_meaning_relation_candidate(candidate)


class _OriginTaskFactProvider(practice_fixtures.PipelineProvider):
    def __init__(self):
        super().__init__()
        self.origin_payloads: list[dict] = []

    async def call(self, type_name, data, schema=None):
        if type_name in {
            ReadingVocabularyGenerationService.ORIGIN_EXPLANATION_TYPE_NAME,
            ReadingVocabularyGenerationService.ORIGIN_EXPLANATION_FALLBACK_TYPE_NAME,
        }:
            self.calls[type_name] += 1
            payload = practice_fixtures._practice_data(data)
            self.origin_payloads.append(payload)
            text_by_relation = {
                "MEANING": "이 문항은 해당 표현의 의미를 묻습니다.",
                "SYNONYM": "이 문항은 문맥상 가장 가까운 의미를 묻습니다.",
                "ANTONYM": "이 문항은 문맥상 반대되는 의미를 묻습니다.",
                "DISTINCTION": "이 문항은 가까운 표현의 결정적 차이를 묻습니다.",
            }
            return {
                "explanations": [
                    {
                        "order": question["order"],
                        "text": text_by_relation[
                            question["immutableTaskFact"]["relationKind"]
                        ],
                    }
                    for question in payload["questions"]
                ]
            }
        return await super().call(type_name, data, schema)


@pytest.mark.asyncio
async def test_origin_localization_receives_immutable_relation_task_fact():
    provider = _OriginTaskFactProvider()
    response = await ReadingVocabularyGenerationService(provider).generate(
        practice_fixtures._vocab_request(mode="MEANING_RELATION")
    )

    facts = [
        question["immutableTaskFact"]
        for payload in provider.origin_payloads
        for question in payload["questions"]
    ]
    assert [fact["relationKind"] for fact in facts] == [
        "DISTINCTION",
        "MEANING",
        "DISTINCTION",
        "DISTINCTION",
        "DISTINCTION",
        "SYNONYM",
        "DISTINCTION",
        "DISTINCTION",
        "DISTINCTION",
        "DISTINCTION",
    ]
    assert all(
        fact == {
            "authority": "APPLICATION_SELECTED_IMMUTABLE",
            "mode": "MEANING_RELATION",
            "relationKind": fact["relationKind"],
        }
        for fact in facts
    )
    assert all(
        "결정적 차이" in question.explanation_origin
        for question in response.questions
        if question.skill_tag == "DISTINCTION"
    )
    assert all(
        "가까운 의미" in question.explanation_origin
        for question in response.questions
        if question.skill_tag == "SYNONYM"
    )
    assert provider.calls == Counter(
        {
            ReadingVocabularyGenerationService.TYPE_NAME: 5,
            ReadingVocabularyGenerationService.VERIFICATION_TYPE_NAME: 1,
            ReadingVocabularyGenerationService.ORIGIN_EXPLANATION_TYPE_NAME: 3,
        }
    )


def test_usage_composition_versions_and_shadow_ruler_remain_v1():
    assert (
        vocabulary_difficulty_recipe(
            mode="USAGE_DISTINCTION",
            band=4,
            skill_tag="REGISTER",
            question_type="SINGLE_CHOICE",
            usage_intent="REGISTER_CHOICE",
        ).version
        == VOCABULARY_DIFFICULTY_RECIPE_VERSION
    )
    assert (
        vocabulary_difficulty_recipe(
            mode="COMPOSITION",
            band=4,
            skill_tag="COMPOSITION",
            question_type="ORDERING",
        ).version
        == VOCABULARY_DIFFICULTY_RECIPE_VERSION
    )
    assert vocabulary_blind_rubric_payload("MEANING_RELATION")["version"] == (
        VOCABULARY_DIFFICULTY_SHADOW_RUBRIC_VERSION
    )


def test_meaning_calibration_instruction_is_not_added_to_other_vocabulary_modes():
    service = ReadingVocabularyGenerationService(practice_fixtures.PipelineProvider())
    meaning_request = practice_fixtures._vocab_request(mode="MEANING_RELATION")
    usage_request = practice_fixtures._vocab_request(mode="USAGE_DISTINCTION")
    composition_request = practice_fixtures._vocab_request(mode="COMPOSITION")

    meaning_prompt = build_practice_generation_prompt(
        meaning_request,
        [slot.prompt_payload() for slot in service._build_slots(meaning_request, {})],
    )
    usage_prompt = build_practice_generation_prompt(
        usage_request,
        [slot.prompt_payload() for slot in service._build_slots(usage_request, {})],
    )
    composition_prompt = build_practice_generation_prompt(
        composition_request,
        [
            slot.prompt_payload()
            for slot in service._build_slots(composition_request, {})
        ],
    )

    calibration_instruction = "For MEANING_RELATION, modeDemand.taskShape"
    assert calibration_instruction in meaning_prompt
    assert calibration_instruction not in usage_prompt
    assert calibration_instruction not in composition_prompt


def test_meaning_shadow_measurement_anchors_remain_frozen_v1():
    rubric = vocabulary_blind_rubric_payload("MEANING_RELATION")

    assert rubric["version"] == VOCABULARY_DIFFICULTY_SHADOW_RUBRIC_VERSION
    assert rubric["bands"] == [
        {
            "band": 1,
            "anchor": (
                "The visible item asks for a direct familiar relation with clearly "
                "separated alternatives."
            ),
        },
        {
            "band": 2,
            "anchor": (
                "The visible item distinguishes a basic relation in a familiar "
                "semantic group."
            ),
        },
        {
            "band": 3,
            "anchor": "Close candidates require at least one visible semantic distinction.",
        },
        {
            "band": 4,
            "anchor": (
                "Near neighbors require scope, nuance, or usage implication, and the "
                "distractors remain semantically plausible."
            ),
        },
        {
            "band": 5,
            "anchor": (
                "Partly matching candidates require precise scope, nuance, and relation "
                "boundaries to determine one unambiguous answer."
            ),
        },
    ]


def test_b3_plus_final_prompt_uses_server_owned_distinction_shell():
    request = practice_fixtures._vocab_request(mode="MEANING_RELATION")
    service = ReadingVocabularyGenerationService(practice_fixtures.PipelineProvider())
    slot = next(
        slot
        for slot in service._build_slots(request, {})
        if slot.skill_tag == "DISTINCTION" and slot.complexity_band >= 3
    )

    normalized = service._normalize_meaning_relation_candidate(
        request,
        slot,
        {
            "prompt": "AI_GENERATED_QUESTION_WORDING_MUST_NOT_SURVIVE",
            "targetExpression": "導入する",
            "meaningContext": "チームは影響を確認しながら、変更を段階的に進めた。",
        },
    )

    assert normalized["prompt"].startswith(
        "チームは影響を確認しながら、変更を段階的に進めた。\n\n"
    )
    assert "意味・使われ方を区別するとき" in normalized["prompt"]
    assert "導入する" in normalized["prompt"]
    assert "AI_GENERATED_QUESTION_WORDING_MUST_NOT_SURVIVE" not in normalized["prompt"]
    assert "meaningContext" not in normalized


@pytest.mark.parametrize("band", [1, 2])
def test_b1_b2_keep_generator_owned_direct_prompt(band):
    request = practice_fixtures._request(
        domain="VOCABULARY",
        mode="MEANING_RELATION",
        question_count=1,
        easier=0,
        current=1,
        challenge=0,
        complexity_band=band,
    )
    service = ReadingVocabularyGenerationService(practice_fixtures.PipelineProvider())
    slot = service._build_slots(request, {})[0]

    normalized = service._normalize_meaning_relation_candidate(
        request,
        slot,
        {
            "prompt": "最も近い意味の表現はどれですか。",
            "targetExpression": "導入する",
            "meaningContext": None,
        },
    )

    assert normalized["prompt"] == "最も近い意味の表現はどれですか。"
    assert "meaningContext" not in normalized


def test_meaning_relation_internal_schema_isolated_from_other_modes_and_public_dto():
    service = ReadingVocabularyGenerationService(practice_fixtures.PipelineProvider())
    meaning_request = practice_fixtures._vocab_request(mode="MEANING_RELATION")
    usage_request = practice_fixtures._vocab_request(mode="USAGE_DISTINCTION")

    meaning_item = service._candidate_schema_for(meaning_request)["properties"][
        "questions"
    ]["items"]
    usage_item = service._candidate_schema_for(usage_request)["properties"]["questions"][
        "items"
    ]

    assert meaning_item["properties"]["meaningContext"] == {
        "type": ["STRING", "NULL"]
    }
    assert "meaningContext" in meaning_item["required"]
    assert "meaningContext" not in usage_item["properties"]
    assert "meaning_context" not in PracticeGeneratedQuestion.model_fields


@pytest.mark.asyncio
async def test_b3_plus_context_independent_quality_verdict_does_not_reject_candidate():
    provider = practice_fixtures.PipelineProvider(
        semantic_context_independent_rounds_by_order={1: 10}
    )
    request = practice_fixtures._request(
        domain="VOCABULARY",
        mode="MEANING_RELATION",
        question_count=1,
        easier=0,
        current=1,
        challenge=0,
        complexity_band=3,
    )

    response = await ReadingVocabularyGenerationService(provider).generate(request)

    assert len(response.questions) == 1
    assert provider.calls[ReadingVocabularyGenerationService.TYPE_NAME] == 1
    assert provider.calls[ReadingVocabularyGenerationService.VERIFICATION_TYPE_NAME] == 1


@pytest.mark.asyncio
async def test_b1_direct_relation_does_not_require_context():
    provider = practice_fixtures.PipelineProvider(
        semantic_context_independent_rounds_by_order={1: 10}
    )
    request = practice_fixtures._request(
        domain="VOCABULARY",
        mode="MEANING_RELATION",
        question_count=1,
        easier=0,
        current=1,
        challenge=0,
        complexity_band=1,
    )

    response = await ReadingVocabularyGenerationService(provider).generate(request)

    assert len(response.questions) == 1
    assert provider.calls[ReadingVocabularyGenerationService.TYPE_NAME] == 1
    assert provider.calls[ReadingVocabularyGenerationService.VERIFICATION_TYPE_NAME] == 1


@pytest.mark.parametrize(
    ("assessment_override", "expected_reason"),
    [
        ({"ambiguous": True}, "ambiguous single-choice item"),
        ({"supported": False}, "answer is not sufficiently supported"),
        ({"mode_fit": False}, "question does not fit requested mode/skill"),
        ({"answer_leakage": True}, "semantic verifier detected answer leakage"),
        ({"distractors_plausible": False}, "distractors are too weak or unrelated"),
        (
            {"best_answer_key": "B"},
            "answer mismatch expected=A verifier=B",
        ),
    ],
)
def test_meaning_relation_existing_quality_failures_remain_rejections(
    assessment_override,
    expected_reason,
):
    values = {
        "best_answer_key": "A",
        "ambiguous": False,
        "supported": True,
        "mode_fit": True,
        "answer_leakage": False,
        "context_dependent": False,
        "distractors_plausible": True,
        **assessment_override,
    }
    assessment = normalize_vocabulary_semantic_assessment(**values)

    decision = semantic_quality_policy_for_vocabulary_mode("MEANING_RELATION").decide(
        assessment=assessment,
        context=VocabularyAcceptanceContext(expected_answer_key="A"),
    )

    assert decision.action == "REJECT"
    assert decision.reason == expected_reason


@pytest.mark.parametrize("meaning_context", [None, "", "   "])
def test_b3_plus_keeps_meaning_context_as_structural_requirement(meaning_context):
    request = practice_fixtures._request(
        domain="VOCABULARY",
        mode="MEANING_RELATION",
        question_count=1,
        easier=0,
        current=1,
        challenge=0,
        complexity_band=3,
    )
    service = ReadingVocabularyGenerationService(practice_fixtures.PipelineProvider())
    slot = service._build_slots(request, {})[0]

    with pytest.raises(ValueError, match="requires (string|non-empty) meaningContext"):
        service._normalize_meaning_relation_candidate(
            request,
            slot,
            {
                "prompt": "AI prompt",
                "targetExpression": "導入する",
                "meaningContext": meaning_context,
            },
        )


@pytest.mark.parametrize(
    ("band", "expected_cycle"),
    [
        (1, ("MEANING", "SYNONYM", "ANTONYM", "DISTINCTION")),
        (2, ("MEANING", "SYNONYM", "ANTONYM", "DISTINCTION")),
        (3, ("DISTINCTION",)),
        (4, ("DISTINCTION",)),
        (5, ("DISTINCTION",)),
    ],
)
def test_band_owned_operational_skill_capability_cycle(band, expected_cycle):
    assert meaning_relation_skill_cycle(band) == expected_cycle
    assert tuple(
        meaning_relation_skill_for_slot(band=band, band_occurrence=occurrence)
        for occurrence in range(1, len(expected_cycle) + 1)
    ) == expected_cycle


def test_full_ten_slot_plan_cycles_skills_by_band_local_occurrence():
    request = practice_fixtures._vocab_request(
        mode="MEANING_RELATION",
        complexity_band=4,
    )
    slots = ReadingVocabularyGenerationService(
        practice_fixtures.PipelineProvider()
    )._build_slots(request, {})

    assert [slot.complexity_band for slot in slots] == [4, 3, 4, 5, 4, 3, 4, 5, 4, 4]
    assert [slot.skill_tag for slot in slots] == ["DISTINCTION"] * 10


@pytest.mark.asyncio
async def test_progressive_one_item_plan_matches_full_ten_slot_plan():
    base = practice_fixtures._vocab_request(
        mode="MEANING_RELATION",
        complexity_band=4,
    )
    service = ReadingVocabularyGenerationService(practice_fixtures.ProgressiveProvider())
    expected = service._build_slots(base, {})
    previous = []
    progressive_skills = []

    for global_order in range(1, 11):
        response = await service.generate(practice_fixtures._single_request(base, previous))
        question = response.questions[0]
        progressive_skills.append(question.skill_tag)
        assert question.complexity_band == expected[global_order - 1].complexity_band
        previous.append(question.model_copy(update={"order": global_order}))

    assert progressive_skills == [slot.skill_tag for slot in expected]


@pytest.mark.asyncio
async def test_fresh_retry_keeps_same_band_local_skill_for_global_slot():
    base = practice_fixtures._vocab_request(
        mode="MEANING_RELATION",
        complexity_band=4,
    )
    seeded = await ReadingVocabularyGenerationService(
        practice_fixtures.PipelineProvider()
    ).generate(base)
    previous = seeded.questions[:6]
    service = ReadingVocabularyGenerationService(practice_fixtures.PipelineProvider())

    first_attempt = practice_fixtures._single_request(
        base,
        previous,
        requestId="practice-item-7-attempt-1",
    )
    fresh_retry = practice_fixtures._single_request(
        base,
        previous,
        requestId="practice-item-7-attempt-2",
    )

    first_slot = service._build_slots(first_attempt, {})[0]
    retry_slot = service._build_slots(fresh_retry, {})[0]
    assert first_slot.complexity_band == retry_slot.complexity_band == 4
    assert first_slot.skill_tag == retry_slot.skill_tag == "DISTINCTION"


def test_b3_b4_b5_only_plan_provider_supported_distinction_slots():
    for band in (3, 4, 5):
        assert meaning_relation_skill_cycle(band) == ("DISTINCTION",)


@pytest.mark.asyncio
async def test_b4_review_target_keeps_expression_with_band_compatible_skill():
    request = practice_fixtures._request(
        domain="VOCABULARY",
        mode="MEANING_RELATION",
        question_count=1,
        easier=0,
        current=1,
        challenge=0,
        complexity_band=4,
        review_targets=[
            {
                "canonicalKey": "review-fixed",
                "expression": "慎重に進める",
                "previousQuestionTypes": [],
            }
        ],
        review_question_count=1,
    )

    response = await ReadingVocabularyGenerationService(
        practice_fixtures.PipelineProvider()
    ).generate(request)

    question = response.questions[0]
    assert question.complexity_band == 4
    assert question.skill_tag == "DISTINCTION"
    assert question.target_expression == "慎重に進める"
    assert question.canonical_key == "review-fixed"
    assert question.review_target is True


@pytest.mark.asyncio
async def test_meaning_relation_semantic_retry_budget_remains_three():
    provider = practice_fixtures.PipelineProvider(
        semantic_reject_rounds_by_order={1: 10}
    )
    request = practice_fixtures._request(
        domain="VOCABULARY",
        mode="MEANING_RELATION",
        question_count=1,
        easier=0,
        current=1,
        challenge=0,
        complexity_band=4,
    )

    with pytest.raises(HTTPException) as exc_info:
        await ReadingVocabularyGenerationService(provider).generate(request)

    assert exc_info.value.status_code == 502
    assert provider.calls[ReadingVocabularyGenerationService.TYPE_NAME] == 3
    assert provider.calls[ReadingVocabularyGenerationService.VERIFICATION_TYPE_NAME] == 3


def test_meaning_relation_weak_distractors_remain_quality_rejection():
    assessment = normalize_vocabulary_semantic_assessment(
        best_answer_key="A",
        ambiguous=False,
        supported=True,
        mode_fit=True,
        answer_leakage=False,
        context_dependent=True,
        distractors_plausible=False,
    )

    decision = semantic_quality_policy_for_vocabulary_mode("MEANING_RELATION").decide(
        assessment=assessment,
        context=VocabularyAcceptanceContext(expected_answer_key="A"),
    )

    assert decision.action == "REJECT"
    assert decision.reason == "distractors are too weak or unrelated"


def test_context_contract_clarification_is_meaning_relation_only():
    question = {
        "order": 1,
        "prompt": "文脈\n\n質問",
        "options": [],
        "skillTag": "MEANING",
    }
    meaning_prompt = build_practice_verification_prompt(
        practice_fixtures._vocab_request(mode="MEANING_RELATION"),
        [question],
    )
    usage_prompt = build_practice_verification_prompt(
        practice_fixtures._vocab_request(mode="USAGE_DISTINCTION"),
        [question],
    )

    assert "contextDependent=true only when removing" in meaning_prompt
    assert "contextDependent=true only when removing" not in usage_prompt


def test_correct_option_exactly_exposed_in_context_is_rejected():
    service = ReadingVocabularyGenerationService(practice_fixtures.PipelineProvider())
    candidate = _meaning_candidate(
        target_expression="配備する",
        correct_text="運用環境に導入する",
    ).model_copy(
        update={"prompt": "計画では運用環境に導入すると明記されています。"}
    )

    with pytest.raises(ValueError, match="stem leaks correct option text"):
        service._validate_meaning_relation_candidate(candidate)


class _LeakyOriginMetadataProvider(practice_fixtures.PipelineProvider):
    async def call(self, type_name, data, schema=None):
        if type_name in {
            ReadingVocabularyGenerationService.ORIGIN_EXPLANATION_TYPE_NAME,
            ReadingVocabularyGenerationService.ORIGIN_EXPLANATION_FALLBACK_TYPE_NAME,
        }:
            self.calls[type_name] += 1
            payload = practice_fixtures._practice_data(data)
            leaking = type_name == ReadingVocabularyGenerationService.ORIGIN_EXPLANATION_TYPE_NAME
            return {
                "explanations": [
                    {
                        "order": question["order"],
                        "text": (
                            "immutableTaskFact의 APPLICATION_SELECTED_IMMUTABLE 값에 따릅니다."
                            if leaking
                            else "이 표현의 문맥상 의미를 자연스럽게 설명합니다."
                        ),
                    }
                    for question in payload["questions"]
                ]
            }
        return await super().call(type_name, data, schema)


@pytest.mark.asyncio
async def test_origin_internal_metadata_leak_is_rejected_then_uses_existing_fallback():
    provider = _LeakyOriginMetadataProvider()
    request = practice_fixtures._request(
        domain="VOCABULARY",
        mode="MEANING_RELATION",
        question_count=1,
        easier=0,
        current=1,
        challenge=0,
        complexity_band=3,
    )

    response = await ReadingVocabularyGenerationService(provider).generate(request)

    explanation = response.questions[0].explanation_origin
    assert "immutableTaskFact" not in explanation
    assert "APPLICATION_SELECTED_IMMUTABLE" not in explanation
    assert provider.calls[ReadingVocabularyGenerationService.ORIGIN_EXPLANATION_TYPE_NAME] == 2
    assert (
        provider.calls[
            ReadingVocabularyGenerationService.ORIGIN_EXPLANATION_FALLBACK_TYPE_NAME
        ]
        == 1
    )
    assert not ReadingVocabularyGenerationService._origin_explanation_exposes_internal_metadata(
        "이 표현은 문맥상 반의어입니다."
    )


def test_structural_recipe_version_changes_but_measurement_ruler_stays_frozen():
    assert (
        MEANING_RELATION_DIFFICULTY_RECIPE_VERSION
        == "vocabulary-meaning-relation-recipe-v5"
    )
    assert (
        VOCABULARY_DIFFICULTY_SHADOW_RUBRIC_VERSION
        == "vocabulary-difficulty-shadow-rubric-v1"
    )
