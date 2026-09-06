import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.features.language_learning.listening.dictation_service import (
    ListeningDictationService,
)
from app.features.language_learning.listening.evaluation_common import (
    build_evaluation_response,
    not_evaluable_task,
)
from app.schemas.language_learning_listening import (
    ListeningTaskType,
    ListeningUsage,
    RecommendationExplanationResponse,
)


class FakeGenerationService:
    async def generate(self, request):
        from app.schemas.language_learning_listening import (
            ListeningItem,
            ListeningSetGenerationResponse,
        )

        return ListeningSetGenerationResponse.model_validate(
            {
                "requestId": request.request_id,
                "generationVersion": "listening-generation-v1",
                "policyVersion": request.policy_version,
                "modelConfigVersion": request.model_config_version,
                "items": [
                    ListeningItem.model_validate(
                        {
                            "itemIndex": 1,
                            "sourceText": "京都へ行きたいです。",
                            "normalizedSourceText": "京都へ行きたいです",
                            "referenceMeanings": [
                                "교토에 가고 싶습니다.",
                                "교토를 방문하고 싶어요.",
                            ],
                            "keyMeaningUnits": ["교토", "가고 싶다"],
                            "targetKeywords": ["京都"],
                            "estimatedAudioSeconds": 8,
                            "contentHash": "hash",
                            "similarityKey": "similarity",
                            "safety": {"passed": True, "categories": []},
                        }
                    )
                ],
                "usage": ListeningUsage(),
            }
        )


class FakeTtsService:
    async def synthesize(self, request):
        from app.schemas.language_learning_listening import (
            ListeningError,
            ListeningTtsResponse,
        )

        return ListeningTtsResponse.model_validate(
            {
                "requestId": request.request_id,
                "itemId": request.item_id,
                "status": "FAILED",
                "sourceText": request.source_text,
                "contentHash": request.content_hash,
                "generationVersion": request.generation_version,
                "error": ListeningError.model_validate(
                    {
                        "code": "TTS_FAILED",
                        "failedStage": "TTS",
                        "message": "TTS failed",
                        "retryable": True,
                    }
                ),
                "usage": ListeningUsage(),
            }
        )


class FakeInterpretationService:
    async def evaluate(self, request):
        task = not_evaluable_task(
            request,
            ListeningTaskType.INTERPRETATION,
            reason_code="LOW_CONFIDENCE",
            confidence=0.69,
        )
        return build_evaluation_response(request, task)


class FakeRepeatService:
    def __init__(self) -> None:
        self.last_audio = None

    async def evaluate(self, request, *, audio_bytes, file_name, content_type):
        self.last_audio = (audio_bytes, file_name, content_type)
        task = not_evaluable_task(
            request,
            ListeningTaskType.REPEAT_AFTER_AUDIO,
            reason_code="SILENCE",
        )
        return build_evaluation_response(request, task)


class FakeExplanationService:
    async def explain(self, request):
        return RecommendationExplanationResponse.model_validate(
            {
                "requestId": request.request_id,
                "targetMetric": request.target_metric,
                "recommendedActivity": request.recommended_activity,
                "recommendedTask": request.recommended_task,
                "explanation": "따라 말하기로 명료도를 다듬어 보세요.",
                "ctaLabel": "따라 말하기",
                "explanationVersion": "listening-explanation-v1",
                "usage": ListeningUsage(),
            }
        )


class FakeAudioStore:
    def __init__(self) -> None:
        self.path = Path(tempfile.gettempdir()) / "translacat-listening-api.audio"
        self.path.write_bytes(b"audio")

    def get(self, reference):
        if reference != "listening-tts-ref":
            return None
        return types.SimpleNamespace(path=self.path, content_type="audio/wav")


class LanguageLearningListeningApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake_dependencies = types.ModuleType("app.api.dependencies")
        cls.generation_service = FakeGenerationService()
        cls.tts_service = FakeTtsService()
        cls.dictation_service = ListeningDictationService()
        cls.interpretation_service = FakeInterpretationService()
        cls.repeat_service = FakeRepeatService()
        cls.explanation_service = FakeExplanationService()
        cls.comprehension_service = object()
        cls.summary_service = object()
        cls.audio_store = FakeAudioStore()

        setattr(
            fake_dependencies,
            "get_language_learning_listening_generation_service",
            lambda: cls.generation_service,
        )
        setattr(
            fake_dependencies,
            "get_language_learning_listening_tts_service",
            lambda: cls.tts_service,
        )
        setattr(
            fake_dependencies,
            "get_language_learning_listening_audio_store",
            lambda: cls.audio_store,
        )
        setattr(
            fake_dependencies,
            "get_language_learning_listening_dictation_service",
            lambda: cls.dictation_service,
        )
        setattr(
            fake_dependencies,
            "get_language_learning_listening_interpretation_service",
            lambda: cls.interpretation_service,
        )
        setattr(
            fake_dependencies,
            "get_language_learning_listening_repeat_service",
            lambda: cls.repeat_service,
        )
        setattr(
            fake_dependencies,
            "get_language_learning_listening_explanation_service",
            lambda: cls.explanation_service,
        )
        setattr(
            fake_dependencies,
            "get_language_learning_listening_comprehension_service",
            lambda: cls.comprehension_service,
        )
        setattr(
            fake_dependencies,
            "get_language_learning_listening_summary_service",
            lambda: cls.summary_service,
        )

        cls.original_dependencies = sys.modules.get("app.api.dependencies")
        cls.original_v1_package = sys.modules.get("app.api.v1")
        cls.original_module = sys.modules.pop(
            "app.api.v1.language_learning_listening",
            None,
        )
        fake_v1_package = types.ModuleType("app.api.v1")
        fake_v1_package.__path__ = [
            str(Path(__file__).resolve().parents[1] / "app" / "api" / "v1")
        ]
        sys.modules["app.api.dependencies"] = fake_dependencies
        sys.modules["app.api.v1"] = fake_v1_package

        module = importlib.import_module("app.api.v1.language_learning_listening")
        cls.module = module
        app = FastAPI()
        app.include_router(module.router, prefix="/api/v1")
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("app.api.v1.language_learning_listening", None)
        if cls.original_module is not None:
            sys.modules["app.api.v1.language_learning_listening"] = cls.original_module
        if cls.original_v1_package is None:
            sys.modules.pop("app.api.v1", None)
        else:
            sys.modules["app.api.v1"] = cls.original_v1_package
        if cls.original_dependencies is None:
            sys.modules.pop("app.api.dependencies", None)
        else:
            sys.modules["app.api.dependencies"] = cls.original_dependencies

    @staticmethod
    def evaluation_base():
        return {
            "requestId": "api-evaluation-1",
            "idempotencyKey": "api-evaluation-idem-1",
            "itemId": 301,
            "attemptId": 401,
            "evaluationPurpose": "OFFICIAL",
            "answerRevealed": False,
            "assistanceUsage": [],
            "policyVersion": "listening-profile-v1",
            "modelConfigVersion": "model-v1",
        }

    def test_dictation_api_returns_all_task_camel_case_contract(self):
        payload = self.evaluation_base()
        payload.update(
            {
                "sourceText": "I can't go.",
                "answer": "i cannot go",
                "learningLanguage": "en",
                "acceptedVariants": {},
            }
        )
        response = self.client.post(
            "/api/v1/language-learning/listening/evaluate/dictation",
            json=payload,
        )
        self.assertEqual(200, response.status_code)
        body = response.json()
        self.assertEqual(5, len(body["tasks"]))
        self.assertEqual("DICTATION", body["tasks"][0]["taskType"])
        self.assertEqual(100, body["tasks"][0]["score"])
        self.assertEqual("listening-profile-v1", body["profilePolicyVersion"])

    def test_repeat_api_accepts_multipart_and_returns_structured_not_evaluable(self):
        context = self.evaluation_base()
        context.update(
            {
                "sourceText": "hello world",
                "sourceDurationSeconds": 2,
                "learningLanguage": "en",
            }
        )
        response = self.client.post(
            "/api/v1/language-learning/listening/evaluate/repeat",
            data={"context": json.dumps(context)},
            files={"audio": ("answer.wav", b"RIFFfake", "audio/wav")},
        )
        self.assertEqual(200, response.status_code)
        body = response.json()
        repeat = next(
            task for task in body["tasks"] if task["taskType"] == "REPEAT_AFTER_AUDIO"
        )
        self.assertFalse(repeat["evaluable"])
        self.assertIsNone(repeat["score"])
        self.assertEqual("SILENCE", repeat["reasonCode"])
        assert self.repeat_service.last_audio is not None
        self.assertEqual(b"RIFFfake", self.repeat_service.last_audio[0])

    def test_tts_partial_failure_keeps_text_and_provider_independent_error(self):
        response = self.client.post(
            "/api/v1/language-learning/listening/tts",
            json={
                "requestId": "tts-api-1",
                "idempotencyKey": "tts-api-idem-1",
                "itemId": 301,
                "sourceText": "hello",
                "contentHash": "generated-hash",
                "generationVersion": "listening-generation-v1",
                "learningLanguage": "en",
                "voice": {"locale": "en-US", "voiceKey": "standard-1", "version": "v1"},
                "policyVersion": "listening-v1",
                "modelConfigVersion": "model-v1",
            },
        )
        self.assertEqual(200, response.status_code)
        body = response.json()
        self.assertEqual("FAILED", body["status"])
        self.assertEqual("hello", body["sourceText"])
        self.assertEqual("TTS", body["error"]["failedStage"])
        self.assertNotIn("provider", body["error"])

    def test_explanation_response_cannot_change_be_decision_fields(self):
        response = self.client.post(
            "/api/v1/language-learning/listening/recommendations/explain",
            json={
                "requestId": "explain-api-1",
                "idempotencyKey": "explain-api-idem-1",
                "type": "RECOMMENDATION",
                "targetMetric": "PRONUNCIATION",
                "recommendedActivity": "LISTENING",
                "recommendedTask": "REPEAT_AFTER_AUDIO",
                "evidenceSummary": {
                    "sources": ["LISTENING"],
                    "count": 5,
                    "recentAverage": 72,
                },
                "originLanguage": "ko",
            },
        )
        self.assertEqual(200, response.status_code)
        body = response.json()
        self.assertEqual("LISTENING", body["recommendedActivity"])
        self.assertEqual("REPEAT_AFTER_AUDIO", body["recommendedTask"])


if __name__ == "__main__":
    unittest.main()
