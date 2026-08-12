import asyncio
import unittest

from fastapi import HTTPException
from pydantic import ValidationError

from app.features.language_learning.writing.policy import calculate_overall_score
from app.features.language_learning.writing.prompts import (
    build_daily_writing_generation_prompt,
    build_level_test_question_prompt,
    build_writing_evaluation_prompt,
)
from app.features.language_learning.writing.service import LanguageLearningWritingService
from app.schemas.language_learning import (
    DailyWritingGenerationRequest,
    LevelTestQuestionRequest,
    WritingEvaluationRequest,
)


class FakeProvider:
    def __init__(self, results=None, errors=None, delay: float = 0) -> None:
        self.results = list(results or [])
        self.errors = list(errors or [])
        self.delay = delay
        self.calls: list[dict] = []

    async def call(self, type_name: str, data: str, schema: dict | None = None):
        self.calls.append({"type_name": type_name, "data": data, "schema": schema})
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.errors:
            error = self.errors.pop(0)
            if error is not None:
                raise error
        if self.results:
            return self.results.pop(0)
        raise RuntimeError("No fake result configured")

    async def call_with_image(self, *args, **kwargs):
        raise NotImplementedError


def build_daily_request(**overrides) -> DailyWritingGenerationRequest:
    data = {
        "requestId": "daily-1",
        "originLanguage": "ko",
        "learningLanguage": "ja",
        "sentenceCount": 5,
        "difficultyDistribution": {
            "review": 1,
            "normal": 3,
            "challenge": 1,
        },
        "selectedKeywords": [
            {
                "key": "topic-it",
                "text": "IT",
                "source": "SYSTEM",
                "type": "TOPIC",
                "canonicalKey": "IT",
                "selectionWeight": 1.0,
            },
            {
                "key": "vocab-deploy",
                "text": "deployment",
                "source": "CUSTOM",
                "type": "VOCABULARY",
                "canonicalKey": "DEPLOY",
                "selectionWeight": 0.75,
            },
        ],
        "learningProfile": {
            "baseLevelScore": 61,
            "skillScores": {
                "meaning": 72,
                "grammar": 55,
                "vocabulary": 64,
                "naturalness": 58,
                "expression": 52,
            },
            "grammarWeaknesses": ["article"],
            "recommendedFocus": ["natural collocation"],
        },
        "generationDate": "2026-08-12",
        "snapshotId": "daily-set-snapshot-1",
    }
    data.update(overrides)
    return DailyWritingGenerationRequest.model_validate(data)


def daily_result():
    return {
        "items": [
            {
                "order": 1,
                "difficulty": "REVIEW",
                "originText": "어제 배포한 기능에 대해 동료에게 설명해 주세요.",
                "keywords": ["vocab-deploy"],
                "focusMetrics": ["GRAMMAR"],
                "focusReason": "과거형 표현 복습",
            },
            {
                "order": 2,
                "difficulty": "NORMAL",
                "originText": "IT 프로젝트의 현재 진행 상황을 상사에게 보고해 주세요.",
                "keywords": ["topic-it"],
                "focusMetrics": ["MEANING", "NATURALNESS"],
                "focusReason": "업무 상황에서 자연스러운 보고 표현",
            },
            {
                "order": 3,
                "difficulty": "NORMAL",
                "originText": "문제가 발생한 원인과 해결 방법을 간단히 설명해 주세요.",
                "keywords": ["topic-it"],
                "focusMetrics": ["VOCABULARY"],
                "focusReason": "문제 해결 관련 어휘 강화",
            },
            {
                "order": 4,
                "difficulty": "NORMAL",
                "originText": "다음 배포 일정을 고객에게 정중하게 안내해 주세요.",
                "keywords": ["vocab-deploy"],
                "focusMetrics": ["NATURALNESS", "EXPRESSION"],
                "focusReason": "정중하고 자연스러운 안내 표현",
            },
            {
                "order": 5,
                "difficulty": "CHALLENGE",
                "originText": "예상보다 배포가 늦어진 이유와 재발 방지책을 설득력 있게 설명해 주세요.",
                "keywords": ["topic-it", "vocab-deploy"],
                "focusMetrics": ["EXPRESSION", "MEANING"],
                "focusReason": "복합적인 이유와 대책을 논리적으로 표현",
            },
        ]
    }


def build_evaluation_request(**overrides) -> WritingEvaluationRequest:
    data = {
        "requestId": "eval-1",
        "context": "DAILY",
        "originLanguage": "ko",
        "learningLanguage": "ja",
        "originSentence": "고객에게 배포가 늦어진 이유를 설명해 주세요.",
        "userAnswer": "デプロイが遅れた理由をお客様に説明します。",
        "difficulty": "NORMAL",
        "keywords": [
            {
                "key": "vocab-deploy",
                "text": "deployment",
                "source": "CUSTOM",
                "type": "VOCABULARY",
                "canonicalKey": "DEPLOY",
            }
        ],
        "focusMetrics": ["MEANING", "NATURALNESS"],
    }
    data.update(overrides)
    return WritingEvaluationRequest.model_validate(data)


def evaluation_result(meaning=90, grammar=80, vocabulary=70, naturalness=60, expression=50):
    return {
        "scores": {
            "meaning": meaning,
            "grammar": grammar,
            "vocabulary": vocabulary,
            "naturalness": naturalness,
            "expression": expression,
        },
        "strengths": [
            {
                "originText": "핵심 의미를 정확하게 전달했습니다.",
                "learningText": "中心となる意味を正確に伝えています。",
            }
        ],
        "weaknesses": [
            {
                "originText": "조금 더 자연스러운 고객 응대 표현을 사용할 수 있습니다.",
                "learningText": "より自然な顧客対応表現を使うことができます。",
            }
        ],
        "corrections": [
            {
                "original": "説明します",
                "corrected": "ご説明いたします",
                "category": "NATURALNESS",
                "explanation": {
                    "originText": "고객에게는 겸양 표현이 더 자연스럽습니다.",
                    "learningText": "お客様には謙譲表現のほうが自然です。",
                },
            }
        ],
        "recommendedAnswers": [
            "デプロイが遅れた理由をご説明します。",
            "デプロイの遅延理由についてご説明いたします。",
        ],
        "explanation": {
            "originText": "의미는 잘 전달되었고 표현을 조금 더 정중하게 만들 수 있습니다.",
            "learningText": "意味は十分伝わっており、より丁寧な表現に改善できます。",
        },
        "profileSignals": {
            "strengthTags": ["meaning-clear"],
            "weaknessTags": ["politeness"],
            "grammarPatterns": [],
            "vocabularyPatterns": [],
            "naturalnessPatterns": ["customer-polite-register"],
            "expressionPatterns": ["honorific-variety"],
            "meaningPatterns": [],
            "recommendedFocus": ["customer-service-keigo"],
        },
    }


class LanguageLearningSchemaTest(unittest.TestCase):
    def test_daily_distribution_must_equal_sentence_count(self):
        with self.assertRaises(ValidationError):
            build_daily_request(
                difficultyDistribution={"review": 1, "normal": 2, "challenge": 1}
            )

    def test_keyword_source_and_type_are_separate(self):
        request = build_daily_request()
        self.assertEqual(request.selected_keywords[0].source.value, "SYSTEM")
        self.assertEqual(request.selected_keywords[0].type.value, "TOPIC")
        self.assertEqual(request.selected_keywords[1].source.value, "CUSTOM")
        self.assertEqual(request.selected_keywords[1].type.value, "VOCABULARY")


class LanguageLearningPromptTest(unittest.TestCase):
    def test_daily_prompt_contains_snapshot_and_keyword_types(self):
        prompt = build_daily_writing_generation_prompt(build_daily_request())
        self.assertIn('"snapshotId":"daily-set-snapshot-1"', prompt)
        self.assertIn('"type":"TOPIC"', prompt)
        self.assertIn('"type":"VOCABULARY"', prompt)

    def test_evaluation_prompt_requests_both_languages(self):
        prompt = build_writing_evaluation_prompt(build_evaluation_request())
        self.assertIn("Feedback originText fields MUST be written in ko", prompt)
        self.assertIn("Feedback learningText fields", prompt)
        self.assertIn("ja", prompt)

    def test_level_test_prompt_contains_previous_results(self):
        request = LevelTestQuestionRequest.model_validate(
            {
                "requestId": "level-2",
                "originLanguage": "ko",
                "learningLanguage": "ja",
                "questionNumber": 2,
                "totalQuestions": 12,
                "previousEvaluations": [
                    {
                        "questionNumber": 1,
                        "difficulty": "EASY",
                        "scores": {
                            "overall": 80,
                            "meaning": 90,
                            "grammar": 80,
                            "vocabulary": 75,
                            "naturalness": 78,
                            "expression": 70,
                        },
                    }
                ],
            }
        )
        prompt = build_level_test_question_prompt(request)
        self.assertIn('"questionNumber":1', prompt)
        self.assertIn('"overall":80', prompt)


class ScoringPolicyTest(unittest.TestCase):
    def test_weighted_score_v1(self):
        self.assertEqual(
            calculate_overall_score(
                meaning=90,
                grammar=80,
                vocabulary=70,
                naturalness=60,
                expression=50,
            ),
            75,
        )

    def test_meaning_gate_caps_very_low_meaning(self):
        score = calculate_overall_score(
            meaning=20,
            grammar=100,
            vocabulary=100,
            naturalness=100,
            expression=100,
        )
        self.assertEqual(score, 49)

    def test_meaning_gate_caps_low_meaning(self):
        score = calculate_overall_score(
            meaning=40,
            grammar=100,
            vocabulary=100,
            naturalness=100,
            expression=100,
        )
        self.assertEqual(score, 69)


class LanguageLearningWritingServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_daily_generation_returns_exact_contract(self):
        provider = FakeProvider(results=[daily_result()])
        service = LanguageLearningWritingService(
            provider=provider,
            generation_timeout_seconds=1,
            generation_max_retries=0,
        )

        response = await service.generate_daily(build_daily_request())

        self.assertEqual(len(response.items), 5)
        self.assertEqual(response.prompt_version, "daily-writing-generation-v1")
        self.assertEqual(
            provider.calls[0]["type_name"],
            "LANGUAGE_LEARNING_DAILY_WRITING_GENERATION",
        )

    async def test_invalid_daily_distribution_is_retried(self):
        invalid = daily_result()
        invalid["items"][0]["difficulty"] = "NORMAL"
        provider = FakeProvider(results=[invalid, daily_result()])
        service = LanguageLearningWritingService(
            provider=provider,
            generation_timeout_seconds=1,
            generation_max_retries=1,
        )

        response = await service.generate_daily(build_daily_request())

        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(response.items[0].difficulty.value, "REVIEW")

    async def test_evaluation_computes_overall_and_versions(self):
        provider = FakeProvider(results=[evaluation_result()])
        service = LanguageLearningWritingService(
            provider=provider,
            evaluation_timeout_seconds=1,
            evaluation_max_retries=0,
        )

        response = await service.evaluate(build_evaluation_request())

        self.assertEqual(response.scores.overall, 75)
        self.assertEqual(response.scores.expression, 50)
        self.assertEqual(response.evaluation_rubric_version, "writing-evaluation-rubric-v1")
        self.assertEqual(response.scoring_policy_version, "writing-scoring-policy-v1")
        self.assertEqual(len(response.recommended_answers), 2)

    async def test_generation_retries_three_times_then_succeeds(self):
        provider = FakeProvider(
            results=[daily_result()],
            errors=[RuntimeError("1"), RuntimeError("2"), RuntimeError("3"), None],
        )
        service = LanguageLearningWritingService(
            provider=provider,
            generation_timeout_seconds=1,
            generation_max_retries=3,
        )

        response = await service.generate_daily(build_daily_request())

        self.assertEqual(len(provider.calls), 4)
        self.assertEqual(len(response.items), 5)

    async def test_evaluation_retries_once_then_fails(self):
        provider = FakeProvider(errors=[RuntimeError("1"), RuntimeError("2")])
        service = LanguageLearningWritingService(
            provider=provider,
            evaluation_timeout_seconds=1,
            evaluation_max_retries=1,
        )

        with self.assertRaises(HTTPException) as raised:
            await service.evaluate(build_evaluation_request())

        self.assertEqual(raised.exception.status_code, 502)
        self.assertEqual(len(provider.calls), 2)

    async def test_invalid_recommended_answer_count_is_retried(self):
        invalid = evaluation_result()
        invalid["recommendedAnswers"] = ["一つだけ"]
        provider = FakeProvider(results=[invalid, evaluation_result()])
        service = LanguageLearningWritingService(
            provider=provider,
            evaluation_timeout_seconds=1,
            evaluation_max_retries=1,
        )

        response = await service.evaluate(build_evaluation_request())

        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(len(response.recommended_answers), 2)

    async def test_level_test_question_generation(self):
        provider = FakeProvider(
            results=[
                {
                    "difficulty": "EASY",
                    "originText": "어제 무엇을 했는지 설명해 주세요.",
                    "focusMetrics": ["GRAMMAR", "MEANING"],
                    "focusReason": "기초 시제와 의미 전달 확인",
                }
            ]
        )
        service = LanguageLearningWritingService(
            provider=provider,
            level_test_timeout_seconds=1,
            generation_max_retries=0,
        )
        request = LevelTestQuestionRequest.model_validate(
            {
                "requestId": "level-1",
                "originLanguage": "ko",
                "learningLanguage": "ja",
                "questionNumber": 1,
                "totalQuestions": 12,
            }
        )

        response = await service.generate_level_test_question(request)

        self.assertEqual(response.question_number, 1)
        self.assertEqual(response.difficulty.value, "EASY")
        self.assertEqual(response.prompt_version, "writing-level-test-question-v1")


if __name__ == "__main__":
    unittest.main()
