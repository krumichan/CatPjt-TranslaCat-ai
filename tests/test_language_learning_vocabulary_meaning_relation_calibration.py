from __future__ import annotations

from collections import Counter

import pytest
from fastapi import HTTPException

from app.features.language_learning.reading_vocabulary.prompts import (
    PRACTICE_GENERATION_SYSTEM_PROMPT,
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
    for band in (3, 4, 5):
        demand = _meaning_recipe("DISTINCTION", band).demand
        assert isinstance(demand, MeaningRelationDemand)
        assert "VISIBLE_CONTEXT" in demand.decision_basis
        assert "TARGET_EXPRESSION_IN_CONTEXT_OR_STEM" in demand.disallowed_shortcuts
        assert (
            "TARGET_EXPRESSION_MISSING_FROM_SERVER_CORRECT_OPTION"
            in demand.disallowed_shortcuts
        )
        assert "TARGET_EXPRESSION_IN_ANY_OPTION" not in demand.disallowed_shortcuts
        if band >= 4:
            assert "STEM_TEACHES_CANDIDATE_DIFFERENCES_BEFORE_ASKING" in (
                demand.disallowed_shortcuts
            )


@pytest.mark.parametrize("band", [1, 2])
def test_low_band_meaning_recipe_disallows_target_expression_in_every_option(band):
    for skill_tag in meaning_relation_skill_cycle(band):
        demand = _meaning_recipe(skill_tag, band).demand
        assert isinstance(demand, MeaningRelationDemand)
        assert "TARGET_EXPRESSION_IN_ANY_OPTION" in demand.disallowed_shortcuts
        assert "TARGET_EXPRESSION_EQUALS_CORRECT_OPTION" not in demand.disallowed_shortcuts


def _meaning_candidate(
    *,
    target_expression: str,
    correct_text: str,
    distractor_texts: tuple[str, str, str] = ("公開する", "停止する", "延期する"),
    complexity_band: int = 5,
):
    return PracticeGeneratedQuestion.model_validate(
        {
            "order": 1,
            "questionType": "SINGLE_CHOICE",
            "difficulty": "CHALLENGE",
            "complexityBand": complexity_band,
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


def _raw_distinction_candidate(
    *,
    target_expression: str,
    distractor_texts: tuple[str, str, str],
    reserve_texts: tuple[str, ...],
) -> dict:
    return {
        "order": 1,
        "questionType": "SINGLE_CHOICE",
        "difficulty": "CURRENT",
        "complexityBand": 4,
        "passageId": None,
        "passageText": None,
        "distractors": [{"text": text} for text in distractor_texts],
        "skillTag": "DISTINCTION",
        "evidenceText": None,
        "explanationLearning": "文脈上の決定的な違いを説明します。",
        "targetExpression": target_expression,
        "canonicalKey": "reliability-candidate",
        "reviewTarget": False,
        "vocabularyCandidates": [],
        "meaningContext": "対象を分けて段階ごとに確認しながら進める計画です。",
        "reserveDistractors": [
            {"text": text} for text in reserve_texts
        ],
    }


def _normalize_distinction_candidate(raw: dict, *, previous=()):
    base = practice_fixtures._vocab_request(
        mode="MEANING_RELATION",
        complexity_band=4,
    )
    request = practice_fixtures._single_request(base, previous)
    service = ReadingVocabularyGenerationService(practice_fixtures.PipelineProvider())
    slot = service._build_slots(request, {})[0]
    normalized = service._normalize_meaning_relation_candidate(request, slot, raw)
    question = PracticeGeneratedQuestion.model_validate(
        {**normalized, "explanationOrigin": "결정적인 의미 차이를 설명합니다."}
    )
    return service, request, slot, question


def test_distinction_internal_schema_requests_surplus_distractors_only():
    request = practice_fixtures._vocab_request(mode="MEANING_RELATION")
    service = ReadingVocabularyGenerationService(practice_fixtures.PipelineProvider())
    distinction_slot = next(
        slot
        for slot in service._build_slots(request, {})
        if slot.skill_tag == "DISTINCTION"
    )
    schema = service._candidate_schema_for(request, [distinction_slot])
    question_schema = schema["properties"]["questions"]["items"]

    assert "distractors" in question_schema["properties"]
    assert "distractors" in question_schema["required"]
    assert "options" not in question_schema["properties"]
    assert "correctAnswer" not in question_schema["properties"]
    assert "prompt" not in question_schema["properties"]
    assert "reserveDistractors" in question_schema["properties"]
    assert "reserveDistractors" in question_schema["required"]
    slot_payload = distinction_slot.prompt_payload()
    assert slot_payload["reserveDistractorCount"] == 2
    assert slot_payload["primaryDistractorCount"] == 3
    assert slot_payload["answerAuthority"] == "APPLICATION_TARGET_EXPRESSION"
    assert slot_payload["targetVisibility"] == "FINAL_OPTIONS_ONLY"
    assert "reserveDistractors" not in PracticeGeneratedQuestion.model_json_schema(
        by_alias=True
    )["properties"]


def test_mixed_b2_b3_schema_keeps_one_call_compatibility_but_ignores_generator_answer():
    request = practice_fixtures._vocab_request(
        mode="MEANING_RELATION",
        complexity_band=3,
    )
    service = ReadingVocabularyGenerationService(practice_fixtures.PipelineProvider())
    slots = service._build_slots(request, {})
    low_slot = next(slot for slot in slots if slot.complexity_band == 2)
    high_slot = next(slot for slot in slots if slot.complexity_band == 3)
    question_schema = service._candidate_schema_for(
        request,
        [low_slot, high_slot],
    )["properties"]["questions"]["items"]
    raw = _raw_distinction_candidate(
        target_expression="追加の検証期間",
        distractor_texts=("予備の運用期間", "延長された監視期間", "通常の確認期間"),
        reserve_texts=("暫定運用期間", "事前確認期間"),
    )
    raw["order"] = high_slot.order
    raw["options"] = [
        {"key": key, "text": candidate["text"]}
        for key, candidate in zip(
            ("A", "B", "C"),
            raw.pop("distractors"),
            strict=True,
        )
    ]
    raw["correctAnswer"] = ["C"]

    normalized = service._normalize_meaning_relation_candidate(
        request,
        high_slot,
        raw,
    )

    assert "options" in question_schema["properties"]
    assert "correctAnswer" in question_schema["properties"]
    expected_key = ("A", "B", "C", "D")[(high_slot.order - 1) % 4]
    assert normalized["correctAnswer"] == [expected_key]
    assert next(
        option["text"]
        for option in normalized["options"]
        if option["key"] == expected_key
    ) == "追加の検証期間"


def test_high_band_target_literal_in_context_or_final_stem_is_rejected():
    raw = _raw_distinction_candidate(
        target_expression="追加の検証期間",
        distractor_texts=("予備の運用期間", "延長された監視期間", "通常の確認期間"),
        reserve_texts=("暫定運用期間", "事前確認期間"),
    )
    raw["meaningContext"] = "計画には追加の検証期間が必要だと明記されています。"

    with pytest.raises(ValueError, match="context leaks targetExpression"):
        _normalize_distinction_candidate(raw)

    raw["meaningContext"] = "計画では通常より長く結果を確認する必要があります。"
    service, _, _, question = _normalize_distinction_candidate(raw)
    leaked = question.model_copy(
        update={"prompt": f"{question.prompt} 追加の検証期間"}
    )
    with pytest.raises(ValueError, match="stem leaks targetExpression"):
        service._validate_meaning_relation_candidate(leaked)


def test_high_band_target_as_answer_validator_requires_server_correct_key():
    raw = _raw_distinction_candidate(
        target_expression="追加の検証期間",
        distractor_texts=("予備の運用期間", "延長された監視期間", "通常の確認期間"),
        reserve_texts=("暫定運用期間", "事前確認期間"),
    )
    service, _, _, question = _normalize_distinction_candidate(raw)
    wrong_key = next(
        option.key
        for option in question.options
        if option.key != question.correct_answer[0]
    )

    with pytest.raises(ValueError, match="must be the correct option"):
        service._validate_meaning_relation_candidate(
            question.model_copy(update={"correct_answer": [wrong_key]})
        )


def test_target_duplicate_distractor_is_salvaged_from_surplus_candidates(caplog):
    caplog.set_level("INFO")
    raw = _raw_distinction_candidate(
        target_expression="安全な導入",
        distractor_texts=("安全な導入", "慎重な展開", "試験的な運用"),
        reserve_texts=("限定的な稼働", "安定した移行"),
    )

    service, _, _, question = _normalize_distinction_candidate(raw)

    assert question.correct_answer == ["A"]
    assert [option.text for option in question.options] == [
        "安全な導入",
        "慎重な展開",
        "試験的な運用",
        "限定的な稼働",
    ]
    service._validate_meaning_relation_candidate(question)
    assert "target_identity_role=distractor" in caplog.text
    assert "安全な導入" not in caplog.text


def test_server_inserts_target_as_correct_without_generator_answer_authority(caplog):
    raw = _raw_distinction_candidate(
        target_expression="安全な導入",
        distractor_texts=("慎重な展開", "試験的な運用", "限定的な稼働"),
        reserve_texts=("安定した移行", "限定的な展開"),
    )

    service, _, _, question = _normalize_distinction_candidate(raw)

    assert question.options[0].text == "安全な導入"
    assert question.correct_answer == ["A"]
    service._validate_meaning_relation_candidate(question)
    assert "target_identity_role=correct" not in caplog.text


def test_duplicate_distractors_are_removed_and_replaced_by_reserve():
    raw = _raw_distinction_candidate(
        target_expression="安全な導入",
        distractor_texts=("慎重な展開", "慎重な展開", "試験的な運用"),
        reserve_texts=("限定的な稼働", "安定した移行"),
    )

    _, _, _, question = _normalize_distinction_candidate(raw)

    assert [option.text for option in question.options] == [
        "安全な導入",
        "慎重な展開",
        "試験的な運用",
        "限定的な稼働",
    ]


def test_candidate_is_rejected_when_salvage_leaves_only_two_distractors():
    raw = _raw_distinction_candidate(
        target_expression="安全な導入",
        distractor_texts=("安全な導入", "慎重な展開", "慎重な展開"),
        reserve_texts=("試験的な運用", "安全な導入"),
    )

    with pytest.raises(ValueError, match="three valid distinct distractors"):
        _normalize_distinction_candidate(raw)


class _SalvageQualityFailureProvider(practice_fixtures.PipelineProvider):
    def __init__(self, failure: str):
        super().__init__()
        self.failure = failure

    def _candidate(self, payload, slot):
        candidate = super()._candidate(payload, slot)
        if slot.get("answerAuthority") == "APPLICATION_TARGET_EXPRESSION":
            candidate["distractors"][0]["text"] = candidate["targetExpression"]
        return candidate

    async def call(self, type_name, data, schema=None):
        response = await super().call(type_name, data, schema)
        if type_name != ReadingVocabularyGenerationService.VERIFICATION_TYPE_NAME:
            return response
        verdict = response["verdicts"][0]
        if self.failure == "ambiguous":
            verdict["ambiguous"] = True
        elif self.failure == "weak_distractors":
            verdict["distractorsPlausible"] = False
        elif self.failure == "answer_key_mismatch":
            verdict["bestAnswerKey"] = next(
                key
                for key in ("A", "B", "C", "D")
                if key != verdict["bestAnswerKey"]
            )
        return response


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "quality_failure",
    ["ambiguous", "weak_distractors", "answer_key_mismatch"],
)
async def test_distractor_salvage_never_bypasses_quality_mini(quality_failure):
    provider = _SalvageQualityFailureProvider(quality_failure)
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
    assert all(
        len(question["options"]) == 4
        and "reserveDistractors" not in question
        for payload in provider.verification_payloads
        for question in payload["questions"]
    )


@pytest.mark.parametrize("band", [1, 2])
def test_b1_b2_target_expression_equal_to_correct_option_remains_rejected(band):
    service = ReadingVocabularyGenerationService(practice_fixtures.PipelineProvider())
    candidate = _meaning_candidate(
        target_expression="配備する",
        correct_text="配備する",
        complexity_band=band,
    )

    with pytest.raises(ValueError, match="option must not repeat targetExpression"):
        service._validate_meaning_relation_candidate(candidate)


@pytest.mark.parametrize("band", [1, 2])
def test_b1_b2_target_expression_equal_to_wrong_option_remains_rejected(band):
    service = ReadingVocabularyGenerationService(practice_fixtures.PipelineProvider())
    candidate = _meaning_candidate(
        target_expression="安全な導入",
        correct_text="安全な展開",
        distractor_texts=("安全な導入", "安全な運用", "安全な移行"),
        complexity_band=band,
    )

    with pytest.raises(ValueError, match="option must not repeat targetExpression"):
        service._validate_meaning_relation_candidate(candidate)


def test_high_band_target_as_answer_rejects_target_in_multiple_options():
    service = ReadingVocabularyGenerationService(practice_fixtures.PipelineProvider())
    candidate = _meaning_candidate(
        target_expression="追加の検証期間",
        correct_text="追加の検証期間",
        distractor_texts=("追加の検証期間", "臨時の対応期間", "通常の確認期間"),
    )

    with pytest.raises(ValueError, match="must appear in exactly one option"):
        service._validate_meaning_relation_candidate(candidate)


def test_b1_b2_target_expression_and_all_distinct_options_remain_valid():
    service = ReadingVocabularyGenerationService(practice_fixtures.PipelineProvider())
    candidate = _meaning_candidate(
        target_expression="追加の検証期間",
        correct_text="追加の確認期間",
        distractor_texts=("延長確認期間", "暫定対応期間", "通常運用期間"),
        complexity_band=2,
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
    all_option_instruction = "targetExpression must not appear in any answer option"
    assert all_option_instruction in meaning_prompt
    assert all_option_instruction not in usage_prompt
    assert all_option_instruction not in composition_prompt
    assert "reserveDistractors" in meaning_prompt
    assert "reserveDistractors" not in usage_prompt
    assert "reserveDistractors" not in composition_prompt
    assert "freeTargetFocus" in meaning_prompt
    assert "freeTargetFocus" not in usage_prompt
    assert "freeTargetFocus" not in composition_prompt
    assert "must not refer to an option key" in PRACTICE_GENERATION_SYSTEM_PROMPT
    assert "or position" in PRACTICE_GENERATION_SYSTEM_PROMPT
    assert "application-owned correct answer" in (
        PRACTICE_GENERATION_SYSTEM_PROMPT
    )
    assert "Do not choose or generate a separate correct alternative" in (
        PRACTICE_GENERATION_SYSTEM_PROMPT
    )


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

    raw = _raw_distinction_candidate(
        target_expression="導入する",
        distractor_texts=("慎重に運用する", "限定的に稼働する", "安定的に移行する"),
        reserve_texts=("試験的に公開する", "順次適用する"),
    )
    raw["prompt"] = "AI_GENERATED_QUESTION_WORDING_MUST_NOT_SURVIVE"
    raw["meaningContext"] = "チームは影響を確認しながら、変更を段階的に進めた。"
    normalized = service._normalize_meaning_relation_candidate(request, slot, raw)

    assert normalized["prompt"].startswith(
        "チームは影響を確認しながら、変更を段階的に進めた。\n\n"
    )
    assert "この状況を最も適切に表す表現はどれですか" in normalized["prompt"]
    assert "導入する" not in normalized["prompt"]
    assert normalized["options"][0]["text"] == "導入する"
    assert normalized["correctAnswer"] == ["A"]
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

    assert slot.target_as_answer is False
    assert "answerAuthority" not in slot.prompt_payload()
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


def test_free_target_focus_cycles_deterministically_across_full_progressive_and_retry():
    focuses = ["development", "customer service", "schedule", "deployment"]
    base = practice_fixtures._vocab_request(
        mode="MEANING_RELATION",
        complexity_band=4,
        selected_keywords=focuses,
    )
    service = ReadingVocabularyGenerationService(practice_fixtures.PipelineProvider())
    full_slots = service._build_slots(base, {})
    expected = [focuses[index % len(focuses)] for index in range(10)]

    assert [slot.free_target_focus for slot in full_slots] == expected
    previous = []
    progressive = []
    retry_focus = None
    for global_order in range(1, 11):
        request = practice_fixtures._single_request(base, previous)
        slot = service._build_slots(request, {})[0]
        progressive.append(slot.free_target_focus)
        if global_order == 7:
            retry_focus = service._build_slots(request.model_copy(), {})[0].free_target_focus
        previous.append(
            _meaning_candidate(
                target_expression=f"自由表現{global_order}",
                correct_text=f"別の表現{global_order}",
            ).model_copy(
                update={
                    "order": global_order,
                    "canonical_key": f"free-key-{global_order}",
                }
            )
        )

    assert progressive == expected
    assert retry_focus == expected[6]


@pytest.mark.asyncio
async def test_free_slot_contract_excludes_accepted_canonical_and_target_identities():
    previous = [
        _meaning_candidate(
            target_expression="段階的な導入",
            correct_text="段階的な展開",
        ).model_copy(update={"order": 1, "canonical_key": "段階的な導入"}),
        _meaning_candidate(
            target_expression="段階的なリリース",
            correct_text="限定的な公開",
        ).model_copy(update={"order": 2, "canonical_key": "段階的なリリース"}),
    ]
    base = practice_fixtures._vocab_request(
        mode="MEANING_RELATION",
        complexity_band=4,
        selected_keywords=["development", "customer service", "schedule"],
    )
    request = practice_fixtures._single_request(base, previous)
    provider = practice_fixtures.ProgressiveProvider()
    service = ReadingVocabularyGenerationService(provider)

    response = await service.generate(request)

    generation_payload = provider.generation_payloads[0]
    free_slot = generation_payload["candidateSlots"][0]
    expected_canonical_keys = {"段階的な導入", "段階的なリリース"}
    expected_target_expressions = {"段階的な導入", "段階的なリリース"}
    assert expected_canonical_keys <= set(generation_payload["excludedCanonicalKeys"])
    assert expected_target_expressions <= set(
        generation_payload["excludedTargetExpressions"]
    )
    assert expected_canonical_keys <= set(free_slot["excludedCanonicalKeys"])
    assert expected_target_expressions <= set(free_slot["excludedTargetExpressions"])
    assert free_slot["freeTargetFocus"] == "schedule"
    generated = response.questions[0]
    generated_correct = next(
        option for option in generated.options if option.key == generated.correct_answer[0]
    )
    assert generated_correct.text == generated.target_expression
    assert generated.review_target is False

    repeated = response.questions[0].model_copy(
        update={"canonical_key": "段階的な導入"}
    )
    with pytest.raises(ValueError, match="canonicalKey duplicates an accepted slot"):
        service._validate_candidate(
            request,
            repeated,
            service._build_slots(request, {})[0],
            {},
        )


def test_answer_position_rotates_by_global_slot_for_full_progressive_and_retry():
    base = practice_fixtures._vocab_request(
        mode="MEANING_RELATION",
        complexity_band=4,
    )
    service = ReadingVocabularyGenerationService(practice_fixtures.PipelineProvider())
    full_keys = []
    for slot in service._build_slots(base, {})[:4]:
        raw = _raw_distinction_candidate(
            target_expression=f"対象表現{slot.order}",
            distractor_texts=("慎重な展開", "試験的な運用", "限定的な稼働"),
            reserve_texts=("安定した移行", "限定的な展開"),
        )
        raw["order"] = slot.order
        normalized = service._normalize_meaning_relation_candidate(base, slot, raw)
        full_keys.append(normalized["correctAnswer"][0])
        assert normalized["options"][slot.order - 1]["text"] == (
            f"対象表現{slot.order}"
        )

    previous = []
    progressive_keys = []
    third_retry_key = None
    for global_order in range(1, 5):
        request = practice_fixtures._single_request(base, previous)
        slot = service._build_slots(request, {})[0]
        raw = _raw_distinction_candidate(
            target_expression=f"対象表現{global_order}",
            distractor_texts=("慎重な展開", "試験的な運用", "限定的な稼働"),
            reserve_texts=("安定した移行", "限定的な展開"),
        )
        normalized = service._normalize_meaning_relation_candidate(request, slot, raw)
        progressive_keys.append(normalized["correctAnswer"][0])
        expected_position = (global_order - 1) % 4
        assert normalized["options"][expected_position]["text"] == (
            f"対象表現{global_order}"
        )
        if global_order == 3:
            third_retry_key = service._normalize_meaning_relation_candidate(
                request, slot, raw
            )["correctAnswer"][0]
        previous.append(
            PracticeGeneratedQuestion.model_validate(
                {**normalized, "explanationOrigin": "정답을 설명합니다."}
            ).model_copy(update={"order": global_order})
        )

    assert full_keys == progressive_keys == ["A", "B", "C", "D"]
    assert third_retry_key == "C"


def test_b3_b4_b5_only_plan_provider_supported_distinction_slots():
    for band in (3, 4, 5):
        assert meaning_relation_skill_cycle(band) == ("DISTINCTION",)


@pytest.mark.asyncio
async def test_high_band_pipeline_hides_target_and_preserves_provider_stage_count():
    provider = practice_fixtures.PipelineProvider()
    request = practice_fixtures._vocab_request(
        mode="MEANING_RELATION",
        complexity_band=4,
    )

    response = await ReadingVocabularyGenerationService(provider).generate(request)

    assert len(response.questions) == 10
    for question in response.questions:
        normalized_target = ReadingVocabularyGenerationService._normalize_for_leak_check(
            question.target_expression or ""
        )
        assert normalized_target
        assert normalized_target not in (
            ReadingVocabularyGenerationService._normalize_for_leak_check(question.prompt)
        )
        matching_options = [
            option
            for option in question.options
            if ReadingVocabularyGenerationService._normalize_for_leak_check(option.text)
            == normalized_target
        ]
        assert len(matching_options) == 1
        assert matching_options[0].key == question.correct_answer[0]
    assert provider.calls[ReadingVocabularyGenerationService.TYPE_NAME] == 5
    assert provider.calls[ReadingVocabularyGenerationService.VERIFICATION_TYPE_NAME] == 1
    assert provider.calls[ReadingVocabularyGenerationService.PRESCREEN_TYPE_NAME] == 0


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
        selected_keywords=["development", "deployment"],
        review_targets=[
            {
                "canonicalKey": "review-fixed",
                "expression": "慎重に進める",
                "previousQuestionTypes": [],
            }
        ],
        review_question_count=1,
    )

    provider = practice_fixtures.PipelineProvider()
    service = ReadingVocabularyGenerationService(provider)
    slot = service._build_slots(request, {})[0]
    response = await service.generate(request)

    question = response.questions[0]
    assert question.complexity_band == 4
    assert question.skill_tag == "DISTINCTION"
    assert question.target_expression == "慎重に進める"
    assert question.canonical_key == "review-fixed"
    assert question.review_target is True
    assert slot.free_target_focus is None
    review_slot_payload = provider.candidate_slot_payloads[0][0]
    assert review_slot_payload["boundReviewTarget"] == {
        "canonicalKey": "review-fixed",
        "expression": "慎重に進める",
        "previousQuestionTypes": [],
    }
    assert "freeTargetFocus" not in review_slot_payload
    assert "excludedCanonicalKeys" not in review_slot_payload
    assert "excludedTargetExpressions" not in review_slot_payload
    normalized_target = ReadingVocabularyGenerationService._normalize_for_leak_check(
        question.target_expression
    )
    assert sum(
        ReadingVocabularyGenerationService._normalize_for_leak_check(option.text)
        == normalized_target
        for option in question.options
    ) == 1
    correct_option = next(
        option for option in question.options if option.key == question.correct_answer[0]
    )
    assert correct_option.text == question.target_expression
    assert question.target_expression not in question.prompt


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


@pytest.mark.asyncio
async def test_high_band_semantic_retry_preserves_server_owned_target_and_repairs_context():
    provider = practice_fixtures.PipelineProvider(semantic_ambiguous_once={1})
    request = practice_fixtures._request(
        domain="VOCABULARY",
        mode="MEANING_RELATION",
        question_count=1,
        easier=0,
        current=1,
        challenge=0,
        complexity_band=4,
    )

    response = await ReadingVocabularyGenerationService(provider).generate(request)

    assert len(response.questions) == 1
    assert provider.calls[ReadingVocabularyGenerationService.TYPE_NAME] == 2
    assert provider.calls[ReadingVocabularyGenerationService.VERIFICATION_TYPE_NAME] == 2
    first_slot, retry_slot = (
        payload[0] for payload in provider.candidate_slot_payloads
    )
    assert retry_slot["retryTargetExpression"] == response.questions[0].target_expression
    assert retry_slot["preserveTargetOnRetry"] is True
    assert "REPAIR_TARGET_AS_ANSWER" in retry_slot["retryFeedback"]
    assert "application-owned correct answer" in retry_slot["retryFeedback"]
    assert first_slot["answerAuthority"] == retry_slot["answerAuthority"]


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
        complexity_band=2,
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
        == "vocabulary-meaning-relation-recipe-v7"
    )
    assert (
        VOCABULARY_DIFFICULTY_SHADOW_RUBRIC_VERSION
        == "vocabulary-difficulty-shadow-rubric-v1"
    )
