from __future__ import annotations

from collections import Counter

import pytest

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
    READING_DIFFICULTY_RECIPE_VERSION,
)
from app.features.language_learning.reading_vocabulary.service import (
    ReadingVocabularyGenerationService,
)
from app.schemas.language_learning_practice import PracticeGeneratedQuestion
from tests import test_language_learning_reading_vocabulary as practice_fixtures


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
        return await super().call(type_name, data, schema)

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
    assert measurements.evidence_surface_unit == 2
    assert measurements.evidence_to_question_surface_unit_distance == 1
    assert measurements.passage_binding_matches is True


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
