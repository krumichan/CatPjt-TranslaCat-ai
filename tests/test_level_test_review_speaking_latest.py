from __future__ import annotations

import pytest

from app.features.language_learning.level_test.policy import LEVEL_TEST_RECIPE
from app.features.language_learning.level_test.prompts import build_level_test_generation_prompt
from app.features.language_learning.level_test.service import LevelTestService
from app.features.language_learning.writing.service import _DAILY_WRITING_GENERATION_SCHEMA
from app.schemas.language_learning_level_test import (
    LevelTestDomain,
    LevelTestFeedbackDetail,
    LevelTestItemType,
    LevelTestQuestionGenerationRequest,
    LevelTestSpeakingEvaluationContext,
)


def _repeat_generation_request(question_number: int) -> LevelTestQuestionGenerationRequest:
    return LevelTestQuestionGenerationRequest.model_validate(
        {
            "requestId": f"repeat-{question_number}",
            "idempotencyKey": f"repeat-{question_number}-idem",
            "sessionId": 100,
            "questionNumber": question_number,
            "totalQuestions": 20,
            "domain": "SPEAKING",
            "itemType": "SPEAKING_REPEAT",
            "originLanguage": "ko",
            "learningLanguage": "ja",
            "targetComplexityBand": 3,
        }
    )


def _repeat_eval_request() -> LevelTestSpeakingEvaluationContext:
    return LevelTestSpeakingEvaluationContext.model_validate(
        {
            "requestId": "speaking-review",
            "idempotencyKey": "speaking-review-idem",
            "sessionId": 100,
            "itemId": 18,
            "itemType": "SPEAKING_REPEAT",
            "promptText": "音声を聞いて繰り返してください。",
            "referenceText": "明日は少し早めに出発してください。",
            "originLanguage": "ko",
            "learningLanguage": "ja",
            "complexityBand": 3,
            "maxDurationSeconds": 30,
        }
    )


def test_speaking_recipe_is_two_repeat_plus_one_guided_response():
    assert LEVEL_TEST_RECIPE[18] == (LevelTestDomain.SPEAKING, LevelTestItemType.SPEAKING_REPEAT)
    assert LEVEL_TEST_RECIPE[19] == (LevelTestDomain.SPEAKING, LevelTestItemType.SPEAKING_REPEAT)
    assert LEVEL_TEST_RECIPE[20] == (LevelTestDomain.SPEAKING, LevelTestItemType.SPEAKING_GUIDED_RESPONSE)


def test_repeat_generation_prompt_distinguishes_text_assisted_and_audio_only_modes():
    q18 = build_level_test_generation_prompt(_repeat_generation_request(18))
    q19 = build_level_test_generation_prompt(_repeat_generation_request(19))
    assert "TEXT_ASSISTED_REPEAT" in q18
    assert "AUDIO_ONLY_REPEAT" in q19
    assert "maxPlaybackCount=3" in q19
    assert "maxAudioSeconds<=20" in q18
    assert "maxAudioSeconds<=20" in q19


def test_repeat_pool_candidates_are_safe_for_audio_only_slot():
    class Candidate:
        max_audio_seconds = 20

    LevelTestService._validate_speaking_repeat_load(
        _repeat_generation_request(18), Candidate(), "明日は少し早めに出発してください。"
    )
    LevelTestService._validate_speaking_repeat_load(
        _repeat_generation_request(19), Candidate(), "明日は少し早めに出発してください。"
    )

    with pytest.raises(ValueError, match="한 문장"):
        LevelTestService._validate_speaking_repeat_load(
            _repeat_generation_request(19),
            Candidate(),
            "明日は少し早めに出発してください。到着したら連絡してください。",
        )

    class TooLongAudio:
        max_audio_seconds = 21

    with pytest.raises(ValueError, match="20초"):
        LevelTestService._validate_speaking_repeat_load(
            _repeat_generation_request(18), TooLongAudio(), "明日は少し早めに出発してください。"
        )


def test_repeat_speaking_payload_defaults_model_answer_list_without_fabricating_scores():
    raw = {
        "evaluationConfidence": 90,
        "taskResponseStatus": "partial",
        "metrics": [
            {
                "type": "pronunciation",
                "state": "evaluated",
                "score": 80,
                "confidence": 0.9,
                "summary": "발음은 대체로 명확합니다.",
                "evidence": [{"message": "문장 끝부분이 약하게 들렸습니다."}],
                "notEvaluableReason": None,
            },
            {
                "type": "fluency",
                "state": "evaluated",
                "score": 70,
                "confidence": 0.8,
                "summary": "중간에 짧은 멈춤이 있습니다.",
                "evidence": [],
                "notEvaluableReason": None,
            },
            {
                "type": "grammar",
                "state": "not_evaluable",
                "score": None,
                "confidence": 0.8,
                "summary": "반복 과제입니다.",
                "evidence": [],
                "notEvaluableReason": "반복 과제",
            },
            {
                "type": "vocabulary",
                "state": "not_evaluable",
                "score": None,
                "confidence": 0.8,
                "summary": "반복 과제입니다.",
                "evidence": [],
                "notEvaluableReason": "반복 과제",
            },
            {
                "type": "task fulfilment",
                "state": "not_evaluable",
                "score": None,
                "confidence": 0.8,
                "summary": "반복 과제입니다.",
                "evidence": [],
                "notEvaluableReason": "반복 과제",
            },
        ],
        "strengths": [],
        "improvements": [],
    }

    normalized = LevelTestService._normalize_speaking_evaluation_payload(
        _repeat_eval_request(), raw
    )

    assert normalized["evaluationConfidence"] == pytest.approx(0.9)
    assert normalized["taskResponseStatus"] == "PARTIAL"
    assert normalized["recommendedAnswers"] == []
    assert normalized["metrics"][0]["type"] == "PRONUNCIATION"
    assert normalized["metrics"][0]["evidence"] == ["문장 끝부분이 약하게 들렸습니다."]
    assert normalized["metrics"][4]["type"] == "TASK_FULFILLMENT"


def test_daily_writing_schema_requires_quality_metadata_used_by_phase35_filter():
    item_schema = _DAILY_WRITING_GENERATION_SCHEMA["properties"]["items"]["items"]
    assert "languageComplexityBand" in item_schema["required"]
    assert "diversityMetadata" in item_schema["required"]


def test_feedback_detail_uses_camel_case_contract():
    detail = LevelTestFeedbackDetail(
        category="GRAMMAR",
        severity="CORRECTION",
        original="操作が慣れていない",
        corrected="操作に慣れていない",
        explanation="助詞は「に」を使います。",
    )
    assert detail.model_dump(mode="json", by_alias=True) == {
        "category": "GRAMMAR",
        "severity": "CORRECTION",
        "original": "操作が慣れていない",
        "corrected": "操作に慣れていない",
        "explanation": "助詞は「に」を使います。",
    }
