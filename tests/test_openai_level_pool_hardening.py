import asyncio
import time

from app.ai.provider_pool import AiProviderPool
from app.ai.providers.openai.schema import normalize_openai_response_schema
from app.features.language_learning.level_test.normalizer import LevelTestGenerationNormalizer
from app.features.language_learning.level_test.service import LevelTestService
from app.schemas.language_learning_level_test import (
    LevelTestDomain,
    LevelTestItemType,
    LevelTestQuestionGenerationRequest,
)
from app.schemas.language_learning_quality import ScenarioCategory


def _reference_payload() -> dict:
    return {
        "sourceText": None,
        "referenceMeanings": [],
        "keyMeaningUnits": [],
        "referenceText": None,
        "translationSourceText": None,
        "emphasisText": None,
        "readingPassage": None,
        "readingQuestion": None,
        "listeningQuestion": None,
        "providedFacts": [],
        "requiredIntents": [],
        "responseConstraints": [],
    }


def _grammar_candidate(*, intent: str = "REQUEST") -> dict:
    return {
        "domain": "GRAMMAR",
        "itemType": "GRAMMAR_FORM_CHOICE",
        "complexityBand": 2,
        "instruction": "빈칸에 가장 알맞은 표현을 고르세요.",
        "instructionLanguage": "ko",
        "answerMode": "CHOICE",
        "answerLanguage": None,
        "promptText": "雨が降ったので、傘を_____。",
        "options": [
            {"key": "A", "text": "持ってきました"},
            {"key": "B", "text": "持ってきますか"},
            {"key": "C", "text": "持ってくるでしょう"},
            {"key": "D", "text": "持ってこない"},
        ],
        "internalAnswerKey": {"correctOptionKey": "A", "correctOrder": []},
        "generationPlanId": None,
        "referencePayload": _reference_payload(),
        "diversityMetadata": {
            "scenarioCategory": "DAILY_LIFE",
            "communicativeIntent": intent,
            "taskArchetype": "grammar form choice",
            "grammarFocusCodes": ["PAST_POLITE"],
            "lexicalFocusCodes": [],
            "semanticSummary": "비 때문에 우산을 가져온 상황",
            "requiresBackgroundKnowledge": False,
            "contentHash": None,
            "similarityKey": None,
        },
        "maxAnswerLength": None,
        "maxAudioSeconds": None,
    }


def test_openai_schema_turns_small_patterns_into_enums():
    normalized = normalize_openai_response_schema(
        {
            "type": "object",
            "properties": {
                "generationPlanId": {
                    "anyOf": [
                        {"type": "string", "pattern": "^[AB]$"},
                        {"type": "null"},
                    ]
                }
            },
        }
    )

    branch = normalized["properties"]["generationPlanId"]["anyOf"][0]
    assert branch == {"type": "string", "enum": ["A", "B"]}


def test_normalizer_clears_non_vocab_plan_id_and_canonicalizes_known_enums():
    data = {
        "candidates": [
            {
                "itemType": "GRAMMAR_FORM_CHOICE",
                "generationPlanId": "plan_1",
                "diversityMetadata": {
                    "scenarioCategory": "daily-life",
                    "communicativeIntent": "explain-reason",
                },
            }
        ]
    }

    normalized, stats = LevelTestGenerationNormalizer.normalize(
        data,
        origin_language="ko",
        learning_language="ja",
    )
    candidate = normalized["candidates"][0]

    assert candidate["generationPlanId"] is None
    assert candidate["diversityMetadata"]["scenarioCategory"] == "DAILY_LIFE"
    assert candidate["diversityMetadata"]["communicativeIntent"] == "EXPLAIN_REASON"
    assert stats.generation_plan_id_repairs == 1
    assert stats.enum_token_repairs == 2


def test_writing_translation_keeps_visible_prompt_as_evaluation_source():
    data = {
        "candidates": [
            {
                "itemType": "WRITING_TRANSLATION",
                "promptText": "오늘은 조금 일찍 집에 가고 싶습니다.",
                "referencePayload": {
                    "translationSourceText": "서로 다른 잘못된 숨은 원문"
                },
            }
        ]
    }

    normalized, stats = LevelTestGenerationNormalizer.normalize(
        data,
        origin_language="ko",
        learning_language="ja",
    )
    candidate = normalized["candidates"][0]

    assert candidate["referencePayload"]["translationSourceText"] == candidate["promptText"]
    assert stats.writing_translation_repairs == 1


def test_generation_schema_for_grammar_forbids_generation_plan_id():
    request = LevelTestQuestionGenerationRequest(
        request_id="schema-hardening",
        idempotency_key="schema-hardening",
        session_id="s1",
        question_number=1,
        domain=LevelTestDomain.GRAMMAR,
        item_type=LevelTestItemType.GRAMMAR_FORM_CHOICE,
        origin_language="ko",
        learning_language="ja",
        target_complexity_band=2,
        preferred_scenario_categories=[ScenarioCategory.DAILY_LIFE],
    )

    schema = LevelTestService._generation_schema(request)
    candidate = schema["$defs"]["LevelTestQuestionCandidate"]["properties"]
    diversity = schema["$defs"]["DiversityMetadata"]["properties"]

    assert candidate["domain"]["enum"] == ["GRAMMAR"]
    assert candidate["itemType"]["enum"] == ["GRAMMAR_FORM_CHOICE"]
    assert candidate["generationPlanId"] == {"type": "null"}
    assert candidate["answerMode"]["enum"] == ["CHOICE"]
    assert candidate["answerLanguage"] == {"type": "null"}
    assert diversity["scenarioCategory"]["enum"] == ["DAILY_LIFE"]


def test_candidate_parser_salvages_valid_sibling():
    good = _grammar_candidate()
    bad = _grammar_candidate(intent="NOT_A_REAL_INTENT")

    parsed, reasons, rejected = LevelTestService._parse_generation_candidates(
        {"candidates": [good, bad]}
    )

    assert [index for index, _ in parsed] == [0]
    assert rejected == 1
    assert any("communicativeIntent" in reason and reason.endswith(":enum") for reason in reasons)


def test_pool_waits_for_short_cooldown_instead_of_immediate_503():
    class Provider:
        ready = True

        async def call(self, type_name: str, data: str, schema=None):
            return "ok"

    async def run():
        pool = AiProviderPool()
        pool.dock(name="openai", provider=Provider(), max_concurrency=1, cooldown_seconds=0.01)
        pool._slots[0].activate_cooldown(0.01)
        started = time.monotonic()
        assert await pool.call(type_name="TEST", data="hello") == "ok"
        assert time.monotonic() - started >= 0.005

    asyncio.run(run())


def test_candidate_parser_reports_sentence_order_reason_code():
    bad = _grammar_candidate()
    bad["itemType"] = "GRAMMAR_SENTENCE_ORDER"
    bad["internalAnswerKey"] = {
        "correctOptionKey": None,
        "correctOrder": ["A", "B", "C"],
    }

    parsed, reasons, rejected = LevelTestService._parse_generation_candidates(
        {"candidates": [bad]}
    )

    assert parsed == []
    assert rejected == 1
    assert any(
        reason.endswith(":sentence_order_correct_order_mismatch")
        for reason in reasons
    )


def test_normalizer_repairs_missing_instruction_language_from_request():
    candidate = _grammar_candidate()
    candidate.pop("instructionLanguage")

    normalized, stats = LevelTestGenerationNormalizer.normalize(
        {"candidates": [candidate]},
        origin_language="ko",
        learning_language="ja",
    )

    assert normalized["candidates"][0]["instructionLanguage"] == "ja"
    assert stats.instruction_language_repairs == 1
