import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.schemas.language_learning_speaking import (
    AssistantAudio,
    AssistantTurn,
    AssistanceResponse,
    ConversationGenerationResponse,
    ConversationResult,
    EvaluationEligibility,
    SessionStartResponse,
    SpeakingEvaluationResponse,
    SpeakingEvaluationStatus,
    SpeakingUsage,
    SttAnalysisMetadata,
    SttResponse,
    SttSegment,
    TranscriptResult,
    TtsResponse,
    TurnProcessResponse,
)


class FakeTurnService:
    async def start_session(self, request):
        return SessionStartResponse(
            requestId=request.request_id,
            sessionId=request.session_id,
            resolvedStartMode="AI_FIRST",
            assistant=AssistantTurn(
                text="こんにちは。今日は何について話しますか？",
                voice=request.voice,
                audio=AssistantAudio(
                    audioReference="tts-start",
                    contentType="audio/wav",
                    voice=request.voice,
                    cacheKey="cache-start",
                    status="READY",
                ),
            ),
            conversation=ConversationResult(
                intent="OPENING",
                difficulty="A2",
                shouldEnd=False,
                assistanceLevel="NONE",
            ),
            usage=SpeakingUsage(),
        )

    async def transcribe_audio(self, *, context, audio_bytes, file_name, content_type):
        return SttResponse(
            requestId=context.request_id,
            sessionId=context.session_id,
            turnIndex=context.turn_index,
            transcript=TranscriptResult(
                text="今日は仕事をしました。",
                language="ja",
                confidence=0.91,
                segments=[
                    SttSegment(
                        startMs=0,
                        endMs=1200,
                        text="今日は仕事をしました。",
                        confidence=0.91,
                    )
                ],
                metadata=SttAnalysisMetadata(
                    provider="fake-stt",
                    model="fake-model",
                    requestedLanguage="ja",
                    lowConfidenceThreshold=0.55,
                    audioDuration=1.2,
                    audioQualitySignals={
                        "rms": 0.1,
                        "peak": 0.5,
                        "silenceRatio": 0.1,
                        "sampleRate": 16000,
                        "channels": 1,
                    },
                    normalizationVersion="speaking-audio-normalization",
                    sttHintVersion="speaking-stt-hint",
                ),
            ),
            usage=SpeakingUsage(),
        )

    async def process_turn(self, *, context, audio_bytes, file_name, content_type):
        return TurnProcessResponse(
            requestId=context.request_id,
            sessionId=context.session_id,
            turnIndex=context.turn_index,
            status="READY",
            transcript=(await self.transcribe_audio(
                context=types.SimpleNamespace(
                    request_id=context.request_id,
                    session_id=context.session_id,
                    turn_index=context.turn_index,
                ),
                audio_bytes=audio_bytes,
                file_name=file_name,
                content_type=content_type,
            )).transcript,
            assistant=AssistantTurn(
                text="それは大変でしたね。",
                voice=context.voice,
                audio=AssistantAudio(
                    audioReference="tts-turn",
                    contentType="audio/wav",
                    voice=context.voice,
                    cacheKey="cache-turn",
                    status="READY",
                ),
            ),
            conversation=ConversationResult(
                intent="FOLLOW_UP",
                difficulty="A2",
                shouldEnd=False,
                assistanceLevel="NONE",
            ),
            usage=SpeakingUsage(),
        )


class FakeConversationService:
    async def generate(self, request):
        return ConversationGenerationResponse(
            requestId=request.request_id,
            sessionId=request.session_id,
            turnIndex=request.turn_index,
            assistantText="それは大変でしたね。",
            conversation=ConversationResult(
                intent="FOLLOW_UP",
                difficulty="A2",
                shouldEnd=False,
                assistanceLevel="NONE",
            ),
            usage=SpeakingUsage(),
        )


class FakeAssistanceService:
    async def generate(self, request):
        return AssistanceResponse(
            requestId=request.request_id,
            sessionId=request.session_id,
            turnIndex=request.turn_index,
            type=request.assistance_type,
            content="予定を表す表現を使ってみましょう。",
            usage=SpeakingUsage(),
            idempotentReplay=False,
        )


class FakeTtsService:
    async def synthesize(self, request):
        return TtsResponse(
            requestId=request.request_id,
            sessionId=request.session_id,
            audio=AssistantAudio(
                audioReference="tts-ref",
                contentType="audio/wav",
                voice=request.voice,
                cacheKey="cache-key",
                durationSeconds=1.0,
                status="READY",
            ),
            usage=SpeakingUsage(),
        )


class FakeEvaluationService:
    async def evaluate(self, request):
        return SpeakingEvaluationResponse(
            requestId=request.request_id,
            sessionId=request.session_id,
            status=SpeakingEvaluationStatus.INSUFFICIENT_EVIDENCE,
            overallScore=None,
            metrics=[],
            strengths=[],
            improvements=[],
            recommendedExpressions=[],
            pronunciationPractice=[],
            profileSignals=[],
            eligibility=EvaluationEligibility(
                validUserTurns=1,
                validUserSpeechSeconds=12,
                validSttTurnRatio=1.0,
                eligibleBeforeAi=False,
                missingRequirements=["VALID_USER_TURNS", "VALID_SPEECH_SECONDS"],
            ),
            evaluationVersion="speaking-evaluation",
            scoringPolicyVersion="speaking-scoring-policy",
            promptVersion="speaking-evaluation-prompt",
            usage=SpeakingUsage(),
        )


class FakeAudioStore:
    def __init__(self):
        self.path = Path(tempfile.gettempdir()) / "translacat-speaking-api-test.wav"
        self.path.write_bytes(b"RIFFfake")

    def get(self, reference):
        if reference != "tts-ref":
            return None
        return types.SimpleNamespace(path=self.path, content_type="audio/wav")


class LanguageLearningSpeakingApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake_dependencies = types.ModuleType("app.api.dependencies")
        cls.turn_service = FakeTurnService()
        cls.conversation_service = FakeConversationService()
        cls.assistance_service = FakeAssistanceService()
        cls.tts_service = FakeTtsService()
        cls.evaluation_service = FakeEvaluationService()
        cls.audio_store = FakeAudioStore()

        fake_dependencies.get_language_learning_speaking_turn_service = (
            lambda: cls.turn_service
        )
        fake_dependencies.get_language_learning_speaking_conversation_service = (
            lambda: cls.conversation_service
        )
        fake_dependencies.get_language_learning_speaking_assistance_service = (
            lambda: cls.assistance_service
        )
        fake_dependencies.get_language_learning_speaking_tts_service = (
            lambda: cls.tts_service
        )
        fake_dependencies.get_language_learning_speaking_evaluation_service = (
            lambda: cls.evaluation_service
        )
        fake_dependencies.get_language_learning_speaking_audio_store = (
            lambda: cls.audio_store
        )

        cls.original_dependencies = sys.modules.get("app.api.dependencies")
        cls.original_v1_package = sys.modules.get("app.api.v1")
        cls.original_module = sys.modules.pop(
            "app.api.v1.language_learning_speaking",
            None,
        )

        fake_v1_package = types.ModuleType("app.api.v1")
        fake_v1_package.__path__ = [
            str(Path(__file__).resolve().parents[1] / "app" / "api" / "v1")
        ]

        sys.modules["app.api.dependencies"] = fake_dependencies
        sys.modules["app.api.v1"] = fake_v1_package

        module = importlib.import_module("app.api.v1.language_learning_speaking")
        cls.module = module
        app = FastAPI()
        app.include_router(module.router, prefix="/api/v1")
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("app.api.v1.language_learning_speaking", None)
        if cls.original_module is not None:
            sys.modules["app.api.v1.language_learning_speaking"] = cls.original_module
        if cls.original_v1_package is None:
            sys.modules.pop("app.api.v1", None)
        else:
            sys.modules["app.api.v1"] = cls.original_v1_package
        if cls.original_dependencies is None:
            sys.modules.pop("app.api.dependencies", None)
        else:
            sys.modules["app.api.dependencies"] = cls.original_dependencies

    @staticmethod
    def session_context(**overrides):
        payload = {
            "requestId": "speaking-api-1",
            "idempotencyKey": "speaking-api-idem-1",
            "sessionId": "session-1",
            "turnIndex": 1,
            "originLanguage": "ko",
            "learningLanguage": "ja",
            "topic": "일상",
            "persona": "친근한 일본인 동료",
            "conversationStartMode": "AI_FIRST",
            "correctionMode": "CONVERSATION",
            "targetLevel": "A2",
        }
        payload.update(overrides)
        return payload

    def test_session_start_contract_is_camel_case(self):
        payload = self.session_context(turnIndex=0)
        response = self.client.post(
            "/api/v1/language-learning/speaking/sessions/start",
            json=payload,
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["resolvedStartMode"], "AI_FIRST")
        self.assertEqual(body["assistant"]["audio"]["audioReference"], "tts-start")

    def test_transcribe_multipart_returns_standard_metadata(self):
        response = self.client.post(
            "/api/v1/language-learning/speaking/turns/transcribe",
            data={
                "context": __import__("json").dumps(
                    {
                        "requestId": "stt-api-1",
                        "idempotencyKey": "stt-api-idem-1",
                        "sessionId": "session-1",
                        "turnIndex": 1,
                        "learningLanguage": "ja",
                    }
                )
            },
            files={"audio": ("turn.wav", b"RIFFfake", "audio/wav")},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["transcript"]["confidence"], 0.91)
        self.assertEqual(body["transcript"]["metadata"]["requestedLanguage"], "ja")
        self.assertEqual(len(body["transcript"]["segments"]), 1)

    def test_process_turn_returns_text_and_audio_reference(self):
        response = self.client.post(
            "/api/v1/language-learning/speaking/turns/process",
            data={"context": __import__("json").dumps(self.session_context())},
            files={"audio": ("turn.wav", b"RIFFfake", "audio/wav")},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "READY")
        self.assertEqual(body["assistant"]["text"], "それは大変でしたね。")
        self.assertEqual(body["assistant"]["audio"]["audioReference"], "tts-turn")

    def test_assistance_returns_generated_content(self):
        response = self.client.post(
            "/api/v1/language-learning/speaking/assistance",
            json={
                "requestId": "assist-api-1",
                "idempotencyKey": "assist-api-idem-1",
                "sessionId": "session-1",
                "turnIndex": 2,
                "assistanceType": "HINT",
                "originLanguage": "ko",
                "learningLanguage": "ja",
                "topic": "주말 계획",
                "targetLevel": "A2",
                "assistantText": "週末は何をする予定ですか？",
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["type"], "HINT")
        self.assertEqual(body["content"], "予定を表す表現を使ってみましょう。")
        self.assertFalse(body["idempotentReplay"])

    def test_tts_can_be_called_independently(self):
        response = self.client.post(
            "/api/v1/language-learning/speaking/tts",
            json={
                "requestId": "tts-api-1",
                "idempotencyKey": "tts-api-idem-1",
                "sessionId": "session-1",
                "text": "こんにちは",
                "learningLanguage": "ja",
                "voice": "Kore",
                "playbackSpeed": "SLOW",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["audio"]["audioReference"], "tts-ref")

    def test_evaluation_insufficient_evidence_contract(self):
        response = self.client.post(
            "/api/v1/language-learning/speaking/evaluate",
            json={
                "requestId": "eval-api-1",
                "idempotencyKey": "eval-api-idem-1",
                "sessionId": "session-1",
                "topic": "일상",
                "originLanguage": "ko",
                "learningLanguage": "ja",
                "userTurns": [
                    {
                        "turnId": "turn-1",
                        "turnIndex": 1,
                        "transcript": "こんにちは",
                        "sttConfidence": 0.9,
                        "durationSeconds": 12,
                    }
                ],
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "INSUFFICIENT_EVIDENCE")
        self.assertIsNone(body["overallScore"])
        self.assertEqual(body["scoringPolicyVersion"], "speaking-scoring-policy")

    def test_audio_reference_endpoint(self):
        response = self.client.get(
            "/api/v1/language-learning/speaking/audio/tts-ref"
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/wav")


if __name__ == "__main__":
    unittest.main()
