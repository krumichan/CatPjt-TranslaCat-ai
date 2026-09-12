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
    def __init__(
        self,
        *,
        observed_band: int = 1,
        difficulty_confidence: float = 0.99,
        malformed_shadow: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.observed_band = observed_band
        self.difficulty_confidence = difficulty_confidence
        self.malformed_shadow = malformed_shadow
        self.passage_prompts: list[str] = []
        self.verification_schemas: list[dict | None] = []

    async def call(self, type_name, data, schema=None):
        if type_name == ReadingVocabularyGenerationService.PASSAGE_TYPE_NAME:
            self.passage_prompts.append(data)
        if type_name == ReadingVocabularyGenerationService.VERIFICATION_TYPE_NAME:
            self.verification_schemas.append(schema)
        response = await super().call(type_name, data, schema)
        if type_name != ReadingVocabularyGenerationService.VERIFICATION_TYPE_NAME:
            return response

        payload = self.verification_payloads[-1]
        if payload["domain"] != "READING":
            return response
        passage_ids = list(dict.fromkeys(question["passageId"] for question in payload["questions"]))
        response["passageDifficultyAssessments"] = [
            {
                "passageId": passage_id,
                **self._difficulty(f"passage:{passage_id}:unit:1"),
            }
            for passage_id in passage_ids
        ]
        for verdict in response["verdicts"]:
            verdict["difficulty"] = self._difficulty(
                f"question:{verdict['order']}:prompt"
            )
        if self.malformed_shadow:
            response["passageDifficultyAssessments"] = [
                {
                    "passageId": passage_id,
                    "difficultyStatus": "ASSESSED",
                    "observedBand": "not-a-band",
                }
                for passage_id in passage_ids
            ]
            for verdict in response["verdicts"]:
                verdict["difficulty"] = "not-an-assessment"
        return response

    async def call_with_image(
        self,
        type_name: str,
        prompt: str,
        image_bytes: bytes,
        mime_type: str,
        schema: dict | None = None,
    ):
        raise AssertionError("Reading calibration must not call an image provider")

    def _difficulty(self, evidence_id: str) -> dict[str, object]:
        return {
            "difficultyStatus": "ASSESSED",
            "observedBand": self.observed_band,
            "alternativeBand": None,
            "issueCodes": ["DIRECT_EVIDENCE_DEMAND"],
            "evidenceSegmentIds": [evidence_id],
            "difficultyConfidence": self.difficulty_confidence,
        }


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
    provider = ReadingDifficultyProvider(observed_band=3)
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
    assert [item["band"] for item in verification["readingDifficultyRubric"]["passageBands"]] == [
        1,
        2,
        3,
        4,
        5,
    ]
    for question in verification["questions"]:
        assert "complexityBand" not in question
        assert "difficulty" not in question
        assert "difficultyRecipe" not in question
        assert "passageSegments" in question
        assert "questionSegments" in question


@pytest.mark.asyncio
async def test_reading_difficulty_mismatch_is_shadow_only_and_logs_no_raw_content(caplog):
    provider = ReadingDifficultyProvider(observed_band=1, difficulty_confidence=0.99)
    with caplog.at_level("INFO"):
        response = await ReadingVocabularyGenerationService(provider).generate(
            practice_fixtures._request()
        )

    assert len(response.questions) == 5
    assert provider.calls[ReadingVocabularyGenerationService.VERIFICATION_TYPE_NAME] == 1
    assert provider.candidate_slot_calls == [[1, 2], [3, 4], [5]]
    assert "comparison=SHADOW_MISMATCH" in caplog.text
    assert "difficulty_confidence=0.99" in caplog.text
    assert "learning_language=ja mode=COMPREHENSION" in caplog.text
    assert "skill_tag=" in caplog.text
    assert "今日は会社で会議があります" not in caplog.text
    assert "本文の内容" not in caplog.text


@pytest.mark.asyncio
async def test_reading_passage_shadow_reject_count_is_scoped_to_its_questions(caplog):
    provider = ReadingDifficultyProvider(
        observed_band=3,
        semantic_ambiguous_once={1},
    )
    with caplog.at_level("INFO"):
        response = await ReadingVocabularyGenerationService(provider).generate(
            practice_fixtures._request()
        )

    assert len(response.questions) == 5
    assert Counter(
        question["passageId"]
        for question in provider.verification_payloads[0]["questions"]
    ) == Counter({"p1": 3, "p2": 2})
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "scope=passage passage_id=p1" in message
        and "quality_rejected_question_count=1" in message
        for message in messages
    )
    assert any(
        "scope=passage passage_id=p2" in message
        and "quality_rejected_question_count=0" in message
        for message in messages
    )


@pytest.mark.asyncio
async def test_malformed_reading_shadow_is_fail_open_without_extra_mini_retry(caplog):
    provider = ReadingDifficultyProvider(malformed_shadow=True)
    with caplog.at_level("INFO"):
        response = await ReadingVocabularyGenerationService(provider).generate(
            practice_fixtures._request()
        )

    assert len(response.questions) == 5
    assert provider.calls[ReadingVocabularyGenerationService.VERIFICATION_TYPE_NAME] == 1
    assert "difficulty_status=UNSURE" in caplog.text
    assert "difficulty_status=NOT_ASSESSED" in caplog.text

    schema = provider.verification_schemas[0]
    assert schema is not None
    verdict_schema = schema["properties"]["verdicts"]["items"]
    assert set(verdict_schema["required"]) == {
        "order",
        "bestAnswerKey",
        "ambiguous",
        "supported",
        "reason",
        "modeFit",
        "answerLeakage",
        "contextDependent",
        "distractorsPlausible",
    }
    all_json_types = {
        "OBJECT",
        "ARRAY",
        "STRING",
        "NUMBER",
        "BOOLEAN",
        "NULL",
    }
    assert set(verdict_schema["properties"]["difficulty"]["type"]) == all_json_types
    assert set(
        schema["properties"]["passageDifficultyAssessments"]["type"]
    ) == all_json_types


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


@pytest.mark.asyncio
async def test_reading_quality_failure_still_regenerates_only_failed_slot_with_difficulty_enabled():
    provider = ReadingDifficultyProvider(
        observed_band=1,
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
