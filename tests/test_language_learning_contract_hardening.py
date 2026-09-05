import asyncio
import copy

import pytest
from fastapi import HTTPException

from app.features.language_learning.level_test.normalizer import LevelTestGenerationNormalizer
from app.features.language_learning.level_test.service import LevelTestService
from app.features.language_learning.writing.service import (
    LanguageLearningWritingService,
    _DAILY_WRITING_V35_GENERATION_SCHEMA,
)
from app.schemas.language_learning_level_test import (
    LevelTestQuestionGenerationPayload,
    LevelTestSpeakingEvaluationContext,
)
from tests.test_language_learning_phase35 import (
    QueueProvider,
    diversity_metadata,
    level_generation_payload,
    level_question_request,
    writing_candidate_payload,
    writing_phase35_request,
    Phase35LevelTestTest as _Phase35LevelTestTest,
)


def test_daily_v35_provider_schema_requires_quality_metadata():
    item_schema = _DAILY_WRITING_V35_GENERATION_SCHEMA["properties"]["items"]["items"]
    assert "languageComplexityBand" in item_schema["required"]
    assert "diversityMetadata" in item_schema["required"]


def test_daily_v35_salvages_valid_sibling_when_one_candidate_is_malformed():
    payload = writing_candidate_payload()
    payload["items"][1].pop("diversityMetadata")
    provider = QueueProvider(plain=[payload])
    service = LanguageLearningWritingService(provider=provider)

    request = writing_phase35_request().model_copy(
        deep=True,
        update={"sentence_count": 2},
    )
    batch = asyncio.run(
        service._call_daily_candidate_batch(request, provider_attempt=0)
    )

    assert len(batch.items) == 3
    assert all(item.diversity_metadata is not None for item in batch.items)


def test_daily_v35_retries_after_an_unusable_batch_and_still_builds_set():
    unusable = copy.deepcopy(writing_candidate_payload())
    for item in unusable["items"]:
        item.pop("diversityMetadata")
    provider = QueueProvider(plain=[unusable, writing_candidate_payload()])
    service = LanguageLearningWritingService(provider=provider)

    response = asyncio.run(service.generate_daily(writing_phase35_request()))

    assert len(response.items) == 2
    assert {item.difficulty.value for item in response.items} == {"NORMAL", "CHALLENGE"}
    assert len(provider.calls) == 2


def test_vocab_context_rejects_exact_correct_answer_leak_in_prompt():
    payload = level_generation_payload()
    for candidate in payload["candidates"]:
        candidate["promptText"] = (
            "レストランに行く前にテーブルを予約しておきます。"
            "「変更したいです」という意味の表現を選んでください。"
        )
    provider = QueueProvider(structured=[copy.deepcopy(payload) for _ in range(3)])
    helper = _Phase35LevelTestTest()

    with pytest.raises(HTTPException) as raised:
        asyncio.run(helper._service(provider).generate_question(level_question_request()))

    assert raised.value.status_code == 422
    assert any("정답 표현이 직접 노출" in reason for reason in raised.value.detail["reasons"])


def test_guided_intent_enum_tokens_are_localized_for_learning_language():
    candidate = {
        "itemType": "WRITING_SHORT_PARAGRAPH",
        "instruction": "write",
        "instructionLanguage": "ko",
        "promptText": "担当者に丁寧なメールを書いてください。",
        "referencePayload": {
            "providedFacts": ["討論会に参加できない"],
            "requiredIntents": ["APOLOGIZE", "EXPLAIN_REASON", "ASK_INFORMATION"],
            "responseConstraints": ["日本語で3～4文"],
        },
    }

    normalized, stats = LevelTestGenerationNormalizer.normalize(
        {"candidates": [candidate]},
        origin_language="ko",
        learning_language="ja",
    )
    intents = normalized["candidates"][0]["referencePayload"]["requiredIntents"]

    assert intents == ["丁寧に謝罪する", "理由を説明する", "必要な情報を尋ねる"]
    assert stats.enum_token_repairs == 3


def test_unknown_internal_guidance_token_fails_closed_instead_of_reaching_ui():
    raw = level_generation_payload()["candidates"][0]
    raw.update(
        {
            "domain": "WRITING",
            "itemType": "WRITING_SHORT_PARAGRAPH",
            "complexityBand": 3,
            "instruction": "短い文章を書いてください。",
            "instructionLanguage": "ja",
            "answerMode": "TEXT",
            "answerLanguage": "ja",
            "promptText": "担当者に欠席の連絡メールを書いてください。",
            "options": [],
            "internalAnswerKey": {"correctOptionKey": None, "correctOrder": []},
            "referencePayload": {
                "providedFacts": ["会議に参加できない", "理由は家族の予定"],
                "requiredIntents": ["SOME_NEW_INTERNAL_CODE", "丁寧に謝罪する"],
                "responseConstraints": ["日本語で3～4文"],
            },
        }
    )
    normalized, _ = LevelTestGenerationNormalizer.normalize(
        {"candidates": [raw]}, origin_language="ko", learning_language="ja"
    )
    candidate = LevelTestQuestionGenerationPayload.model_validate(normalized).candidates[0]

    with pytest.raises(ValueError, match="内部 enum token|내부 enum token"):
        LevelTestService._validate_question_candidate(
            level_question_request().model_copy(
                update={
                    "domain": "WRITING",
                    "item_type": "WRITING_SHORT_PARAGRAPH",
                    "target_complexity_band": 3,
                }
            ),
            candidate,
        )


def _speaking_request() -> LevelTestSpeakingEvaluationContext:
    return LevelTestSpeakingEvaluationContext.model_validate(
        {
            "requestId": "sp-retry",
            "idempotencyKey": "sp-retry-idem",
            "sessionId": 10,
            "itemId": 18,
            "itemType": "SPEAKING_REPEAT",
            "promptText": "予定を変更せざるを得ない場合でも、相手の事情を確認してください。",
            "referenceText": "予定を変更せざるを得ない場合でも、相手の事情を確認してください。",
            "originLanguage": "ko",
            "learningLanguage": "ja",
            "complexityBand": 3,
            "maxDurationSeconds": 30,
        }
    )


def _valid_speaking_payload() -> dict:
    return {
        "evaluationConfidence": 90,
        "metrics": [
            {"type": "pronunciation", "state": "evaluated", "score": 0.8, "confidence": 90, "summary": "발음이 대체로 명료합니다.", "evidence": "STT와 음향 근거"},
            {"type": "fluency", "state": "evaluated", "score": 0.7, "confidence": 90, "summary": "일부 끊김이 있습니다.", "evidence": ["짧은 멈춤"]},
            {"type": "grammar", "state": "not-evaluable", "score": 0.9, "confidence": 80, "summary": "", "evidence": []},
            {"type": "vocabulary", "state": "not-evaluable", "score": 0.9, "confidence": 80, "summary": "", "evidence": []},
            {"type": "task fulfilment", "state": "evaluated", "score": 0.6, "confidence": 85, "summary": "원문의 일부가 누락되었습니다.", "evidence": [{"message": "후반부 누락"}]},
        ],
        "strengths": "발화가 식별 가능합니다.",
        "improvements": [{"message": "원문 후반부를 정확히 반복하세요."}],
    }


def test_speaking_schema_failure_retries_and_normalizes_safe_format_drift():
    invalid = {"evaluationConfidence": 0.8, "metrics": []}
    provider = QueueProvider(structured=[invalid, _valid_speaking_payload()])
    helper = _Phase35LevelTestTest()

    response = asyncio.run(
        helper._service(provider).evaluate_speaking(
            _speaking_request(),
            audio_bytes=b"fake",
            file_name="answer.wav",
            content_type="audio/wav",
        )
    )

    assert response.evaluable
    assert len(provider.calls) == 2
    assert len(response.metrics) == 5
    assert response.metrics[0].score == 80
    assert response.metrics[0].confidence == pytest.approx(0.9)
    grammar = next(metric for metric in response.metrics if metric.type == "GRAMMAR")
    assert grammar.score is None
    assert grammar.not_evaluable_reason
