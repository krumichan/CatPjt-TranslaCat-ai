import importlib
import sys
import types
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.schemas.language_learning import (
    DailyWritingGenerationResponse,
    DailyWritingItem,
    LevelTestQuestionResponse,
    WritingEvaluationResponse,
)


class FakeWritingService:
    async def generate_daily(self, request):
        return DailyWritingGenerationResponse(
            request_id=request.request_id,
            prompt_version="daily-writing-generation-v1",
            items=[
                DailyWritingItem(
                    order=1,
                    difficulty="NORMAL",
                    origin_text="오늘 업무에 대해 설명해 주세요.",
                    keywords=[],
                    focus_metrics=["MEANING"],
                    focus_reason="기본 의미 전달 확인",
                )
            ],
        )

    async def evaluate(self, request):
        return WritingEvaluationResponse.model_validate(
            {
                "requestId": request.request_id,
                "scores": {
                    "overall": 80,
                    "meaning": 85,
                    "grammar": 80,
                    "vocabulary": 75,
                    "naturalness": 80,
                    "expression": 75,
                },
                "strengths": [],
                "weaknesses": [],
                "corrections": [],
                "recommendedAnswers": ["回答例1", "回答例2"],
                "explanation": {
                    "originText": "좋습니다.",
                    "learningText": "良いです。",
                },
                "profileSignals": {
                    "strengthTags": [],
                    "weaknessTags": [],
                    "grammarPatterns": [],
                    "vocabularyPatterns": [],
                    "naturalnessPatterns": [],
                    "expressionPatterns": [],
                    "meaningPatterns": [],
                    "recommendedFocus": [],
                },
                "evaluationRubricVersion": "writing-evaluation-rubric-v1",
                "scoringPolicyVersion": "writing-scoring-policy-v1",
                "promptVersion": "writing-evaluation-v1",
            }
        )

    async def generate_level_test_question(self, request):
        return LevelTestQuestionResponse(
            request_id=request.request_id,
            question_number=request.question_number,
            total_questions=request.total_questions,
            difficulty="EASY",
            origin_text="자기소개를 해 주세요.",
            focus_metrics=["MEANING"],
            focus_reason="기초 표현 확인",
            prompt_version="writing-level-test-question-v1",
        )


class LanguageLearningWritingApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake_dependencies = types.ModuleType("app.api.dependencies")
        cls.fake_service = FakeWritingService()
        fake_dependencies.get_language_learning_writing_service = lambda: cls.fake_service

        cls.original_dependencies = sys.modules.get("app.api.dependencies")
        cls.original_v1_package = sys.modules.get("app.api.v1")
        cls.original_module = sys.modules.pop("app.api.v1.language_learning", None)

        fake_v1_package = types.ModuleType("app.api.v1")
        fake_v1_package.__path__ = [
            str(Path(__file__).resolve().parents[1] / "app" / "api" / "v1")
        ]

        sys.modules["app.api.dependencies"] = fake_dependencies
        sys.modules["app.api.v1"] = fake_v1_package

        module = importlib.import_module("app.api.v1.language_learning")
        cls.module = module
        app = FastAPI()
        app.include_router(module.router, prefix="/api/v1")
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("app.api.v1.language_learning", None)
        if cls.original_module is not None:
            sys.modules["app.api.v1.language_learning"] = cls.original_module
        if cls.original_v1_package is None:
            sys.modules.pop("app.api.v1", None)
        else:
            sys.modules["app.api.v1"] = cls.original_v1_package
        if cls.original_dependencies is None:
            sys.modules.pop("app.api.dependencies", None)
        else:
            sys.modules["app.api.dependencies"] = cls.original_dependencies

    def test_daily_generate_contract_is_camel_case(self):
        response = self.client.post(
            "/api/v1/language-learning/writing/daily/generate",
            json={
                "requestId": "daily-api-1",
                "originLanguage": "ko",
                "learningLanguage": "ja",
                "sentenceCount": 1,
                "difficultyDistribution": {"review": 0, "normal": 1, "challenge": 0},
                "selectedKeywords": [],
                "generationDate": "2026-08-12",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["requestId"], "daily-api-1")
        self.assertEqual(response.json()["items"][0]["originText"], "오늘 업무에 대해 설명해 주세요.")

    def test_evaluate_contract_contains_policy_versions(self):
        response = self.client.post(
            "/api/v1/language-learning/writing/evaluate",
            json={
                "requestId": "eval-api-1",
                "context": "DAILY",
                "originLanguage": "ko",
                "learningLanguage": "ja",
                "originSentence": "오늘 무엇을 했나요?",
                "userAnswer": "今日は仕事をしました。",
                "difficulty": "NORMAL",
                "keywords": [],
                "focusMetrics": ["MEANING"],
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["scores"]["overall"], 80)
        self.assertEqual(body["evaluationRubricVersion"], "writing-evaluation-rubric-v1")
        self.assertEqual(body["scoringPolicyVersion"], "writing-scoring-policy-v1")

    def test_level_test_question_contract(self):
        response = self.client.post(
            "/api/v1/language-learning/writing/level-test/question",
            json={
                "requestId": "level-api-1",
                "originLanguage": "ko",
                "learningLanguage": "ja",
                "questionNumber": 1,
                "totalQuestions": 12,
                "previousEvaluations": [],
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["difficulty"], "EASY")
        self.assertEqual(response.json()["questionNumber"], 1)


if __name__ == "__main__":
    unittest.main()
