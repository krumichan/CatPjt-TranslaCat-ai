import asyncio
import io
import math
import unittest
import wave

from app.ai.ports import SpeechSynthesisResult, StructuredGenerationResult
from app.features.language_learning.speaking.assistance_service import SpeakingAssistanceService
from app.features.language_learning.speaking.audio_processor import SpeakingAudioProcessor
from app.features.language_learning.speaking.audio_store import TemporaryTtsAudioStore
from app.features.language_learning.speaking.benchmark import SpeakingEvaluationBenchmarkService
from app.features.language_learning.speaking.conversation_service import SpeakingConversationService
from app.features.language_learning.speaking.errors import SpeakingStageException
from app.features.language_learning.speaking.evaluation_service import SpeakingEvaluationService
from app.features.language_learning.speaking.policy import (
    SPEAKING_CONVERSATION_PROMPT_VERSION,
    calculate_evaluation_eligibility,
    calculate_speaking_overall,
    resolve_assistance_level,
)
from app.features.language_learning.speaking.prompts import (
    SPEAKING_ASSISTANCE_SYSTEM_PROMPT,
    SPEAKING_CONVERSATION_SYSTEM_PROMPT,
    build_conversation_prompt,
)
from app.features.language_learning.speaking.stt_service import (
    SpeakingSttService,
    SttProviderResult,
    SttProviderSegment,
)
from app.features.language_learning.speaking.tts_service import SpeakingTtsService
from app.schemas.language_learning_speaking import (
    AssistanceLevel,
    AssistanceRequest,
    AssistanceType,
    AssistanceUsage,
    BenchmarkSample,
    ConversationGenerationRequest,
    MetricEvaluationState,
    SpeakingEvaluationRequest,
    SpeakingEvaluationTurn,
    SpeakingMetricPayload,
    SpeakingMetricType,
    TtsRequest,
)


class FakeStructuredProvider:
    def __init__(self, results=None, errors=None) -> None:
        self.results = list(results or [])
        self.errors = list(errors or [])
        self.calls = []

    async def call_with_metadata(self, type_name, data, schema=None):
        self.calls.append((type_name, data, schema))
        if self.errors:
            error = self.errors.pop(0)
            if error is not None:
                raise error
        if not self.results:
            raise RuntimeError("no result")
        return StructuredGenerationResult(
            data=self.results.pop(0),
            input_tokens=12,
            output_tokens=8,
            provider="fake",
            model="fake-model",
        )


class FakeSpeechProvider:
    def __init__(self, errors=None) -> None:
        self.errors = list(errors or [])
        self.calls = 0
        self.last_kwargs = None

    async def synthesize_speech(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        if self.errors:
            error = self.errors.pop(0)
            if error is not None:
                raise error
        return SpeechSynthesisResult(
            audio_bytes=b"RIFFfake-wav",
            content_type="audio/wav",
            provider="fake-tts",
            model="fake-tts-model",
            duration_seconds=1.5,
        )


class FakeSttProvider:
    def __init__(self, result=None, errors=None) -> None:
        self.result = result or SttProviderResult(
            text="こんにちは",
            language="ja",
            language_probability=0.99,
            segments=[SttProviderSegment(0.0, 1.2, "こんにちは", -0.1)],
            provider="fake-stt",
            model="fake-stt-model",
        )
        self.errors = list(errors or [])
        self.calls = 0
        self.last_phrase_hints = None

    async def transcribe(self, wav_bytes, *, language, phrase_hints=None):
        self.calls += 1
        self.last_phrase_hints = phrase_hints
        if self.errors:
            error = self.errors.pop(0)
            if error is not None:
                raise error
        return self.result


def make_wav(seconds: float = 1.2, *, silence: bool = False) -> bytes:
    sample_rate = 16000
    count = int(seconds * sample_rate)
    samples = bytearray()
    for index in range(count):
        value = 0 if silence else int(12000 * math.sin(2 * math.pi * 440 * index / sample_rate))
        samples += int(value).to_bytes(2, byteorder="little", signed=True)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(samples)
    return buffer.getvalue()


def conversation_payload():
    return {
        "assistantText": "今日は何をしましたか？",
        "intent": "DAILY_CHAT",
        "difficulty": "A2",
        "shouldEnd": False,
        "endReason": None,
        "hint": None,
        "coachingCorrections": [],
        "sessionSummary": "日常について会話中",
    }


def conversation_request(**overrides):
    data = {
        "requestId": "conv-1",
        "idempotencyKey": "idem-1",
        "sessionId": "session-1",
        "turnIndex": 1,
        "originLanguage": "ko",
        "learningLanguage": "ja",
        "topic": "일상",
        "conversationStartMode": "AI_FIRST",
        "correctionMode": "CONVERSATION",
        "targetLevel": "A2",
        "conversationHistory": [],
        "assistanceUsage": [],
        "isInitialTurn": False,
    }
    data.update(overrides)
    return ConversationGenerationRequest.model_validate(data)



def assistance_request(assistance_type="HINT"):
    return AssistanceRequest.model_validate(
        {
            "requestId": "assist-1",
            "idempotencyKey": "assist-idem-1",
            "sessionId": "session-1",
            "turnIndex": 2,
            "assistanceType": assistance_type,
            "originLanguage": "ko",
            "learningLanguage": "ja",
            "topic": "주말 계획",
            "targetLevel": "A2",
            "assistantText": "週末は何をする予定ですか？",
            "conversationHistory": [],
            "selectedKeywords": [],
        }
    )

def evaluation_turn(index: int, *, excluded=False, audio=True, confidence=0.9):
    return {
        "turnId": f"turn-{index}",
        "turnIndex": index,
        "transcript": "今日は仕事をしました。",
        "sttConfidence": confidence,
        "durationSeconds": 12,
        "segments": [
            {
                "startMs": 0,
                "endMs": 12000,
                "text": "今日は仕事をしました。",
                "confidence": confidence,
            }
        ],
        "audioReference": f"audio-{index}" if audio else None,
        "audioAvailable": audio,
        "audioQualitySignals": (
            {
                "rms": 0.1,
                "peak": 0.5,
                "silenceRatio": 0.1,
                "sampleRate": 16000,
                "channels": 1,
            }
            if audio
            else None
        ),
        "excludedFromEvaluation": excluded,
        "assistanceUsage": [],
    }


def evaluation_request(turns=None):
    return SpeakingEvaluationRequest.model_validate(
        {
            "requestId": "eval-1",
            "idempotencyKey": "eval-idem-1",
            "sessionId": "session-1",
            "topic": "일상",
            "targetLevel": "A2",
            "originLanguage": "ko",
            "learningLanguage": "ja",
            "userTurns": turns or [evaluation_turn(i) for i in range(1, 6)],
            "assistantTurns": [],
            "evaluationPolicyVersion": "speaking-evaluation-policy-v1",
        }
    )


def metric_payload(metric_type, score=80, state="EVALUATED", turn_id="turn-1"):
    return {
        "type": metric_type,
        "state": state,
        "score": score if state == "EVALUATED" else None,
        "confidence": 0.9,
        "summary": "요약",
        "evidence": [
            {
                "turnId": turn_id,
                "startMs": 0,
                "endMs": 1000,
                "message": "근거",
            }
        ],
        "notEvaluableReason": None if state == "EVALUATED" else "근거 부족",
    }


class SpeakingKeywordPromptPolicyTest(unittest.TestCase):
    def test_system_prompt_defines_topic_and_vocabulary_roles(self):
        self.assertIn(
            "TOPIC defines the broad conversation context",
            SPEAKING_CONVERSATION_SYSTEM_PROMPT,
        )
        self.assertIn(
            "VOCABULARY defines a specific learning focus",
            SPEAKING_CONVERSATION_SYSTEM_PROMPT,
        )
        self.assertIn(
            "SYSTEM and CUSTOM have equal priority",
            SPEAKING_CONVERSATION_SYSTEM_PROMPT,
        )
        self.assertIn(
            "empty selectedKeywords list adds no keyword constraint",
            SPEAKING_CONVERSATION_SYSTEM_PROMPT,
        )
        self.assertIn(
            "TOPIC supplies broad context",
            SPEAKING_ASSISTANCE_SYSTEM_PROMPT,
        )
        self.assertEqual(
            SPEAKING_CONVERSATION_PROMPT_VERSION,
            "speaking-conversation-v2",
        )

    def test_conversation_payload_keeps_both_keyword_types_flat(self):
        prompt = build_conversation_prompt(
            conversation_request(
                selectedKeywords=[
                    {
                        "key": "shopping",
                        "text": "Shopping",
                        "source": "SYSTEM",
                        "type": "TOPIC",
                    },
                    {
                        "key": "price",
                        "text": "price",
                        "source": "CUSTOM",
                        "type": "VOCABULARY",
                    },
                ]
            )
        )

        self.assertIn('"key":"shopping"', prompt)
        self.assertIn('"type":"TOPIC"', prompt)
        self.assertIn('"key":"price"', prompt)
        self.assertIn('"type":"VOCABULARY"', prompt)

    def test_empty_keyword_selection_remains_valid(self):
        prompt = build_conversation_prompt(conversation_request())

        self.assertIn('"selectedKeywords":[]', prompt)


def evaluation_payload(confidence=0.9, pronunciation_state="EVALUATED"):
    metric_names = [
        "GRAMMAR",
        "VOCABULARY",
        "NATURALNESS",
        "MEANING",
        "EXPRESSIVENESS",
        "FLUENCY",
        "PRONUNCIATION",
        "INTERACTION",
    ]
    metrics = []
    for name in metric_names:
        state = pronunciation_state if name == "PRONUNCIATION" else "EVALUATED"
        metrics.append(metric_payload(name, state=state))
    return {
        "evaluationConfidence": confidence,
        "metrics": metrics,
        "strengths": ["대화를 유지했습니다."],
        "improvements": ["조금 더 자연스럽게 연결해 보세요."],
        "recommendedExpressions": [],
        "pronunciationPractice": [],
        "profileSignals": [
            {
                "metricType": "FLUENCY",
                "source": "SPEAKING",
                "direction": "IMPROVING",
                "confidence": 0.9,
                "evidenceTurnIds": ["turn-1"],
                "patternKey": "fluency.pause",
                "recommendedFocus": "짧은 연결 표현 연습",
            }
        ],
    }


class AudioProcessorTest(unittest.TestCase):
    def setUp(self):
        self.processor = SpeakingAudioProcessor()

    def test_valid_audio_is_normalized(self):
        result = self.processor.validate_and_normalize(
            make_wav(),
            file_name="turn.wav",
            content_type="audio/wav",
            min_seconds=1,
            max_seconds=60,
            max_bytes=10 * 1024 * 1024,
        )
        self.assertGreaterEqual(result.duration_seconds, 1)
        self.assertEqual(result.quality.sample_rate, 16000)
        self.assertTrue(result.wav_bytes.startswith(b"RIFF"))

    def test_short_audio_is_rejected(self):
        with self.assertRaises(SpeakingStageException) as context:
            self.processor.validate_and_normalize(
                make_wav(0.3),
                file_name="short.wav",
                content_type="audio/wav",
                min_seconds=1,
                max_seconds=60,
                max_bytes=10 * 1024 * 1024,
            )
        self.assertEqual(context.exception.code.value, "AUDIO_TOO_SHORT")

    def test_silence_is_rejected(self):
        with self.assertRaises(SpeakingStageException) as context:
            self.processor.validate_and_normalize(
                make_wav(silence=True),
                file_name="silent.wav",
                content_type="audio/wav",
                min_seconds=1,
                max_seconds=60,
                max_bytes=10 * 1024 * 1024,
            )
        self.assertEqual(context.exception.code.value, "SILENCE_DETECTED")

    def test_too_large_is_rejected(self):
        with self.assertRaises(SpeakingStageException) as context:
            self.processor.validate_and_normalize(
                make_wav(),
                file_name="large.wav",
                content_type="audio/wav",
                min_seconds=1,
                max_seconds=60,
                max_bytes=100,
            )
        self.assertEqual(context.exception.code.value, "AUDIO_TOO_LARGE")


class SpeakingPolicyTest(unittest.TestCase):
    def test_assisted_and_guided_levels(self):
        assisted = resolve_assistance_level([AssistanceUsage(type="HINT")])
        guided = resolve_assistance_level([AssistanceUsage(type="SAMPLE_ANSWER")])
        neutral = resolve_assistance_level([AssistanceUsage(type="REPLAY")])
        self.assertEqual(assisted, AssistanceLevel.ASSISTED)
        self.assertEqual(guided, AssistanceLevel.GUIDED)
        self.assertEqual(neutral, AssistanceLevel.NONE)

    def test_eligibility_boundary_passes(self):
        eligibility = calculate_evaluation_eligibility(
            [SpeakingEvaluationTurn.model_validate(evaluation_turn(i)) for i in range(1, 6)]
        )
        self.assertTrue(eligibility.eligible_before_ai)
        self.assertEqual(eligibility.valid_user_turns, 5)
        self.assertEqual(eligibility.valid_user_speech_seconds, 60)

    def test_eligibility_rejects_less_than_five_turns(self):
        eligibility = calculate_evaluation_eligibility(
            [SpeakingEvaluationTurn.model_validate(evaluation_turn(i)) for i in range(1, 5)]
        )
        self.assertFalse(eligibility.eligible_before_ai)
        self.assertIn("VALID_USER_TURNS", eligibility.missing_requirements)

    def test_not_evaluable_metric_is_renormalized(self):
        metrics = [
            SpeakingMetricPayload.model_validate(metric_payload(name, score=80))
            for name in [
                "GRAMMAR",
                "VOCABULARY",
                "NATURALNESS",
                "MEANING",
                "EXPRESSIVENESS",
                "FLUENCY",
                "INTERACTION",
            ]
        ]
        metrics.append(
            SpeakingMetricPayload.model_validate(
                metric_payload("PRONUNCIATION", state="NOT_EVALUABLE")
            )
        )
        self.assertEqual(calculate_speaking_overall(metrics), 80)


class SpeakingSttServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_transcript_contains_confidence_segments_and_metadata(self):
        provider = FakeSttProvider()
        service = SpeakingSttService(provider, timeout_seconds=1, automatic_retries=0)
        normalized = SpeakingAudioProcessor().validate_and_normalize(
            make_wav(),
            file_name="turn.wav",
            content_type="audio/wav",
            min_seconds=1,
            max_seconds=60,
            max_bytes=10 * 1024 * 1024,
        )
        result = await service.transcribe(
            request_id="stt-1",
            session_id="session-1",
            turn_index=1,
            learning_language="ja",
            normalized_audio=normalized,
        )
        self.assertEqual(result.transcript.text, "こんにちは")
        self.assertEqual(result.transcript.metadata.provider, "fake-stt")
        self.assertEqual(len(result.transcript.segments), 1)
        self.assertGreater(result.transcript.confidence, 0)
        self.assertFalse(result.transcript.is_low_confidence)

    async def test_low_confidence_transcript_is_marked_explicitly(self):
        provider = FakeSttProvider(
            result=SttProviderResult(
                text="こんにちは",
                language="ja",
                language_probability=0.4,
                segments=[SttProviderSegment(0.0, 1.2, "こんにちは", -1.0)],
                provider="fake-stt",
                model="fake-stt-model",
            )
        )
        service = SpeakingSttService(provider, timeout_seconds=1, automatic_retries=0)
        normalized = SpeakingAudioProcessor().validate_and_normalize(
            make_wav(),
            file_name="turn.wav",
            content_type="audio/wav",
            min_seconds=1,
            max_seconds=60,
            max_bytes=10 * 1024 * 1024,
        )

        result = await service.transcribe(
            request_id="stt-low-confidence",
            session_id="session-1",
            turn_index=1,
            learning_language="ja",
            normalized_audio=normalized,
        )

        self.assertTrue(result.transcript.is_low_confidence)
        self.assertEqual(
            result.transcript.metadata.low_confidence_threshold,
            0.55,
        )

    async def test_stt_retries_stage_twice(self):
        provider = FakeSttProvider(errors=[RuntimeError("1"), RuntimeError("2"), None])
        service = SpeakingSttService(provider, timeout_seconds=1, automatic_retries=2)
        normalized = SpeakingAudioProcessor().validate_and_normalize(
            make_wav(),
            file_name="turn.wav",
            content_type="audio/wav",
            min_seconds=1,
            max_seconds=60,
            max_bytes=10 * 1024 * 1024,
        )
        await service.transcribe(
            request_id="stt-1",
            session_id="session-1",
            turn_index=1,
            learning_language="ja",
            normalized_audio=normalized,
        )
        self.assertEqual(provider.calls, 3)

    async def test_stt_passes_phrase_hints_to_provider(self):
        provider = FakeSttProvider()
        service = SpeakingSttService(provider, timeout_seconds=1, automatic_retries=0)
        normalized = SpeakingAudioProcessor().validate_and_normalize(
            make_wav(),
            file_name="turn.wav",
            content_type="audio/wav",
            min_seconds=1,
            max_seconds=60,
            max_bytes=10 * 1024 * 1024,
        )

        await service.transcribe(
            request_id="stt-hint-1",
            session_id="session-1",
            turn_index=1,
            learning_language="ja",
            normalized_audio=normalized,
            phrase_hints=["会議", "デプロイ"],
        )

        self.assertEqual(provider.last_phrase_hints, ["会議", "デプロイ"])


class SpeakingConversationServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_conversation_contract_and_usage(self):
        provider = FakeStructuredProvider(results=[conversation_payload()])
        service = SpeakingConversationService(provider, timeout_seconds=1, automatic_retries=0)
        response = await service.generate(conversation_request())
        self.assertEqual(response.assistant_text, "今日は何をしましたか？")
        self.assertEqual(response.usage.conversation.input_tokens, 12)
        self.assertEqual(provider.calls[0][0], "LANGUAGE_LEARNING_SPEAKING_CONVERSATION")

    async def test_invalid_schema_is_retried(self):
        provider = FakeStructuredProvider(
            results=[{"assistantText": "missing fields"}, conversation_payload()]
        )
        service = SpeakingConversationService(provider, timeout_seconds=1, automatic_retries=1)
        response = await service.generate(conversation_request())
        self.assertEqual(response.conversation.intent, "DAILY_CHAT")
        self.assertEqual(len(provider.calls), 2)

    async def test_coaching_assistance_is_reflected_in_prompt(self):
        provider = FakeStructuredProvider(results=[conversation_payload()])
        service = SpeakingConversationService(provider, timeout_seconds=1, automatic_retries=0)
        await service.generate(
            conversation_request(
                correctionMode="COACHING",
                assistanceUsage=[{"type": "SAMPLE_ANSWER", "count": 1}],
            )
        )
        prompt = provider.calls[0][1]
        self.assertIn('"correctionMode":"COACHING"', prompt)
        self.assertIn('"assistanceLevel":"GUIDED"', prompt)


class SpeakingTtsServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_tts_returns_opaque_audio_reference(self):
        provider = FakeSpeechProvider()
        store = TemporaryTtsAudioStore(ttl_seconds=60)
        service = SpeakingTtsService(provider, store, timeout_seconds=1, automatic_retries=0)
        response = await service.synthesize(
            TtsRequest(
                requestId="tts-1",
                idempotencyKey="tts-idem",
                sessionId="session-1",
                text="こんにちは",
                learningLanguage="ja",
                voice="Kore",
            )
        )
        self.assertTrue(response.audio.audio_reference.startswith("tts-"))
        self.assertEqual(response.audio.content_type, "audio/wav")

    async def test_same_text_uses_tts_cache(self):
        provider = FakeSpeechProvider()
        store = TemporaryTtsAudioStore(ttl_seconds=60)
        service = SpeakingTtsService(provider, store, timeout_seconds=1, automatic_retries=0)
        request = TtsRequest(
            requestId="tts-1",
            idempotencyKey="tts-idem",
            sessionId="session-1",
            text="こんにちは",
            learningLanguage="ja",
            voice="Kore",
        )
        first = await service.synthesize(request)
        second = await service.synthesize(request)
        self.assertEqual(first.audio.audio_reference, second.audio.audio_reference)
        self.assertEqual(provider.calls, 1)


class SpeakingEvaluationServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_evaluation_returns_eight_metrics_and_overall(self):
        provider = FakeStructuredProvider(results=[evaluation_payload()])
        service = SpeakingEvaluationService(provider, timeout_seconds=1, automatic_retries=0)
        response = await service.evaluate(evaluation_request())
        self.assertEqual(response.status.value, "EVALUATED")
        self.assertEqual(response.overall_score, 80)
        self.assertEqual(len(response.metrics), 8)
        self.assertEqual(response.scoring_policy_version, "speaking-scoring-policy-v1")
        self.assertEqual(len(response.profile_signals), 1)

    async def test_precheck_insufficient_evidence_skips_provider(self):
        provider = FakeStructuredProvider(results=[evaluation_payload()])
        service = SpeakingEvaluationService(provider, timeout_seconds=1, automatic_retries=0)
        response = await service.evaluate(
            evaluation_request([evaluation_turn(i) for i in range(1, 4)])
        )
        self.assertEqual(response.status.value, "INSUFFICIENT_EVIDENCE")
        self.assertIsNone(response.overall_score)
        self.assertEqual(provider.calls, [])

    async def test_low_runtime_confidence_has_no_official_score_or_profile_signal(self):
        provider = FakeStructuredProvider(results=[evaluation_payload(confidence=0.69)])
        service = SpeakingEvaluationService(provider, timeout_seconds=1, automatic_retries=0)
        response = await service.evaluate(evaluation_request())
        self.assertEqual(response.status.value, "INSUFFICIENT_EVIDENCE")
        self.assertIsNone(response.overall_score)
        self.assertEqual(response.profile_signals, [])

    async def test_pronunciation_requires_audio_evidence(self):
        provider = FakeStructuredProvider(results=[evaluation_payload()])
        service = SpeakingEvaluationService(provider, timeout_seconds=1, automatic_retries=0)
        turns = [evaluation_turn(i, audio=False) for i in range(1, 6)]
        with self.assertRaises(SpeakingStageException):
            await service.evaluate(evaluation_request(turns))

    async def test_pronunciation_requires_usable_audio_quality(self):
        provider = FakeStructuredProvider(results=[evaluation_payload()])
        service = SpeakingEvaluationService(
            provider,
            timeout_seconds=1,
            automatic_retries=0,
        )
        turns = [evaluation_turn(i) for i in range(1, 6)]
        for turn in turns:
            turn["audioQualitySignals"]["silenceRatio"] = 0.99

        with self.assertRaises(SpeakingStageException) as context:
            await service.evaluate(evaluation_request(turns))

        self.assertEqual(context.exception.code.value, "INVALID_RESPONSE_SCHEMA")

    async def test_not_evaluable_pronunciation_without_audio_is_valid(self):
        provider = FakeStructuredProvider(
            results=[evaluation_payload(pronunciation_state="NOT_EVALUABLE")]
        )
        service = SpeakingEvaluationService(provider, timeout_seconds=1, automatic_retries=0)
        turns = [evaluation_turn(i, audio=False) for i in range(1, 6)]
        response = await service.evaluate(evaluation_request(turns))
        pronunciation = next(
            metric for metric in response.metrics if metric.type == SpeakingMetricType.PRONUNCIATION
        )
        self.assertEqual(pronunciation.state, MetricEvaluationState.NOT_EVALUABLE)
        self.assertEqual(response.overall_score, 80)


class SpeakingBenchmarkTest(unittest.TestCase):
    def test_two_human_evaluators_and_70_percent_threshold(self):
        metrics = [metric.value for metric in SpeakingMetricType]
        sample = BenchmarkSample.model_validate(
            {
                "sampleId": "sample-1",
                "aiScores": [
                    {"metricType": metric, "score": 80} for metric in metrics
                ],
                "humanEvaluators": [
                    {
                        "evaluatorId": "human-1",
                        "scores": [
                            {"metricType": metric, "score": 75} for metric in metrics
                        ],
                    },
                    {
                        "evaluatorId": "human-2",
                        "scores": [
                            {"metricType": metric, "score": 85} for metric in metrics
                        ],
                    },
                ],
            }
        )
        result = SpeakingEvaluationBenchmarkService().evaluate([sample])
        self.assertTrue(result.passed)
        self.assertEqual(result.agreement_rate, 1.0)

    def test_benchmark_fails_below_70_percent(self):
        metrics = [metric.value for metric in SpeakingMetricType]
        ai_scores = []
        human_scores = []
        for index, metric in enumerate(metrics):
            ai_scores.append({"metricType": metric, "score": 100 if index < 3 else 50})
            human_scores.append({"metricType": metric, "score": 0 if index < 3 else 50})
        sample = BenchmarkSample.model_validate(
            {
                "sampleId": "sample-1",
                "aiScores": ai_scores,
                "humanEvaluators": [
                    {"evaluatorId": "h1", "scores": human_scores},
                    {"evaluatorId": "h2", "scores": human_scores},
                ],
            }
        )
        result = SpeakingEvaluationBenchmarkService().evaluate([sample])
        self.assertFalse(result.passed)
        self.assertLess(result.agreement_rate, 0.70)


from app.features.language_learning.speaking.idempotency import InMemoryIdempotencyStore
from app.features.language_learning.speaking.turn_service import SpeakingTurnService
from app.schemas.language_learning_speaking import (
    AssistantAudio,
    ConversationGenerationResponse,
    ConversationResult,
    SessionStartRequest,
    SpeakingErrorCode,
    SpeakingStage,
    SpeakingUsage,
    StageUsage,
    SttAnalysisMetadata,
    SttResponse,
    SttSegment,
    TranscriptResult,
    TtsResponse,
)
from app.features.language_learning.speaking.audio_processor import NormalizedAudio
from app.schemas.language_learning_speaking import AudioQualitySignals


class FakeAudioProcessor:
    def validate_and_normalize(self, *args, **kwargs):
        return NormalizedAudio(
            wav_bytes=make_wav(),
            duration_seconds=1.2,
            source_format="wav",
            quality=AudioQualitySignals(
                rms=0.1,
                peak=0.5,
                silenceRatio=0.1,
                sampleRate=16000,
                channels=1,
            ),
        )


class FakeSttServiceForTurn:
    def __init__(self):
        self.calls = 0
        self.last_kwargs = None

    async def transcribe(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        return SttResponse(
            requestId=kwargs["request_id"],
            sessionId=kwargs["session_id"],
            turnIndex=kwargs["turn_index"],
            transcript=TranscriptResult(
                text="今日は仕事をしました。",
                language="ja",
                confidence=0.9,
                segments=[
                    SttSegment(
                        startMs=0,
                        endMs=1200,
                        text="今日は仕事をしました。",
                        confidence=0.9,
                    )
                ],
                metadata=SttAnalysisMetadata(
                    provider="fake-stt",
                    model="fake",
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
                    normalizationVersion="v1",
                    sttHintVersion="v1",
                ),
            ),
            usage=SpeakingUsage(stt=StageUsage(latencyMs=1, audioSeconds=1.2)),
        )


class FakeConversationServiceForTurn:
    def __init__(self, *, error=None):
        self.error = error
        self.calls = 0
        self.last_request = None

    async def generate(self, request):
        self.calls += 1
        self.last_request = request
        if self.error:
            raise self.error
        return ConversationGenerationResponse(
            requestId=request.request_id,
            sessionId=request.session_id,
            turnIndex=request.turn_index,
            assistantText="今日は何をしましたか？",
            conversation=ConversationResult(
                intent="DAILY_CHAT",
                difficulty="A2",
                shouldEnd=False,
                assistanceLevel="NONE",
            ),
            usage=SpeakingUsage(
                conversation=StageUsage(
                    latencyMs=2,
                    inputTokens=10,
                    outputTokens=5,
                )
            ),
        )


class FakeTtsServiceForTurn:
    def __init__(self, *, error=None):
        self.error = error
        self.calls = 0

    async def synthesize(self, request):
        self.calls += 1
        if self.error:
            raise self.error
        return TtsResponse(
            requestId=request.request_id,
            sessionId=request.session_id,
            audio=AssistantAudio(
                audioReference="tts-ref",
                contentType="audio/wav",
                voice=request.voice,
                cacheKey="cache",
                durationSeconds=1.0,
                status="READY",
            ),
            usage=SpeakingUsage(tts=StageUsage(latencyMs=3, ttsCharacters=len(request.text))),
        )


class SpeakingTurnServiceTest(unittest.IsolatedAsyncioTestCase):
    def build_service(self, *, conversation_error=None, tts_error=None):
        self.stt = FakeSttServiceForTurn()
        self.conversation = FakeConversationServiceForTurn(error=conversation_error)
        self.tts = FakeTtsServiceForTurn(error=tts_error)
        return SpeakingTurnService(
            audio_processor=FakeAudioProcessor(),
            stt_service=self.stt,
            conversation_service=self.conversation,
            tts_service=self.tts,
            idempotency_store=InMemoryIdempotencyStore(ttl_seconds=60),
        )

    async def test_user_first_start_does_not_generate_ai_turn(self):
        service = self.build_service()
        response = await service.start_session(
            SessionStartRequest(
                requestId="start-1",
                idempotencyKey="start-idem",
                sessionId="session-1",
                originLanguage="ko",
                learningLanguage="ja",
                topic="일상",
                conversationStartMode="USER_FIRST",
            )
        )
        self.assertEqual(response.resolved_start_mode.value, "USER_FIRST")
        self.assertIsNone(response.assistant)
        self.assertEqual(self.conversation.calls, 0)

    async def test_ai_first_start_generates_text_and_tts(self):
        service = self.build_service()
        response = await service.start_session(
            SessionStartRequest(
                requestId="start-1",
                idempotencyKey="start-idem",
                sessionId="session-1",
                originLanguage="ko",
                learningLanguage="ja",
                topic="일상",
                conversationStartMode="AI_FIRST",
            )
        )
        self.assertEqual(response.assistant.text, "今日は何をしましたか？")
        self.assertEqual(response.assistant.audio.audio_reference, "tts-ref")
        self.assertTrue(self.conversation.last_request.is_initial_turn)

    async def test_conversation_failure_preserves_stt_result(self):
        service = self.build_service(
            conversation_error=SpeakingStageException(
                code=SpeakingErrorCode.CONVERSATION_GENERATION_FAILED,
                stage=SpeakingStage.CONVERSATION,
                message="failed",
                retryable=True,
            )
        )
        response = await service.process_turn(
            context=conversation_request(),
            audio_bytes=make_wav(),
            file_name="turn.wav",
            content_type="audio/wav",
        )
        self.assertEqual(response.status, "PARTIAL_FAILURE")
        self.assertEqual(response.failed_stage, "CONVERSATION")
        self.assertEqual(response.transcript.text, "今日は仕事をしました。")
        self.assertIsNone(response.assistant)

    async def test_tts_failure_preserves_assistant_text(self):
        service = self.build_service(
            tts_error=SpeakingStageException(
                code=SpeakingErrorCode.TTS_FAILED,
                stage=SpeakingStage.TTS,
                message="tts failed",
                retryable=True,
            )
        )
        response = await service.process_turn(
            context=conversation_request(),
            audio_bytes=make_wav(),
            file_name="turn.wav",
            content_type="audio/wav",
        )
        self.assertEqual(response.status, "PARTIAL_FAILURE")
        self.assertEqual(response.failed_stage, "TTS")
        self.assertEqual(response.assistant.text, "今日は何をしましたか？")
        self.assertIsNone(response.assistant.audio)

    async def test_same_idempotency_key_replays_without_reprocessing(self):
        service = self.build_service()
        context = conversation_request()
        first = await service.process_turn(
            context=context,
            audio_bytes=make_wav(),
            file_name="turn.wav",
            content_type="audio/wav",
        )
        second = await service.process_turn(
            context=context,
            audio_bytes=make_wav(),
            file_name="turn.wav",
            content_type="audio/wav",
        )
        self.assertFalse(first.idempotent_replay)
        self.assertTrue(second.idempotent_replay)
        self.assertEqual(self.stt.calls, 1)
        self.assertEqual(self.conversation.calls, 1)
        self.assertEqual(self.tts.calls, 1)

    async def test_selected_keywords_are_forwarded_as_stt_phrase_hints(self):
        service = self.build_service()
        await service.process_turn(
            context=conversation_request(
                selectedKeywords=[
                    {
                        "key": "meeting",
                        "text": "会議",
                        "source": "SYSTEM",
                        "type": "VOCABULARY",
                    }
                ]
            ),
            audio_bytes=make_wav(),
            file_name="turn.wav",
            content_type="audio/wav",
        )

        self.assertEqual(self.stt.last_kwargs["phrase_hints"], ["会議"])


class SpeakingPhase2CoverageTest(unittest.IsolatedAsyncioTestCase):
    def test_audio_too_long_and_corrupted_are_distinguished(self):
        processor = SpeakingAudioProcessor()
        with self.assertRaises(SpeakingStageException) as too_long:
            processor.validate_and_normalize(
                make_wav(2.0),
                file_name="long.wav",
                content_type="audio/wav",
                min_seconds=1,
                max_seconds=1.5,
                max_bytes=10 * 1024 * 1024,
            )
        self.assertEqual(too_long.exception.code.value, "AUDIO_TOO_LONG")

        with self.assertRaises(SpeakingStageException) as corrupted:
            processor.validate_and_normalize(
                b"not-audio",
                file_name="broken.webm",
                content_type="audio/webm",
                min_seconds=1,
                max_seconds=60,
                max_bytes=10 * 1024 * 1024,
            )
        self.assertEqual(corrupted.exception.code.value, "UNSUPPORTED_AUDIO_FORMAT")

    def test_all_six_assistance_types_resolve_to_expected_levels(self):
        neutral = resolve_assistance_level(
            [
                AssistanceUsage(type="REPLAY"),
                AssistanceUsage(type="SLOW_PLAYBACK"),
                AssistanceUsage(type="SHOW_QUESTION"),
            ]
        )
        assisted = resolve_assistance_level(
            [AssistanceUsage(type="HINT"), AssistanceUsage(type="TRANSLATION")]
        )
        guided = resolve_assistance_level(
            [AssistanceUsage(type="SAMPLE_ANSWER")]
        )
        self.assertEqual(neutral, AssistanceLevel.NONE)
        self.assertEqual(assisted, AssistanceLevel.ASSISTED)
        self.assertEqual(guided, AssistanceLevel.GUIDED)

    def test_stt_ratio_uses_all_submitted_turns(self):
        turns = [
            SpeakingEvaluationTurn.model_validate(evaluation_turn(i))
            for i in range(1, 6)
        ]
        turns.extend(
            [
                SpeakingEvaluationTurn.model_validate(
                    evaluation_turn(6, excluded=True, confidence=0.1)
                ),
                SpeakingEvaluationTurn.model_validate(
                    evaluation_turn(7, excluded=True, confidence=0.1)
                ),
            ]
        )
        eligibility = calculate_evaluation_eligibility(turns)
        self.assertEqual(eligibility.valid_user_turns, 5)
        self.assertAlmostEqual(eligibility.valid_stt_turn_ratio, 5 / 7, places=4)
        self.assertFalse(eligibility.eligible_before_ai)
        self.assertIn("VALID_STT_TURN_RATIO", eligibility.missing_requirements)

    def test_stt_ratio_80_percent_boundary(self):
        turns = [
            SpeakingEvaluationTurn.model_validate(
                evaluation_turn(i, confidence=0.9 if i <= 4 else 0.2)
            )
            for i in range(1, 6)
        ]
        eligibility = calculate_evaluation_eligibility(turns)
        self.assertEqual(eligibility.valid_stt_turn_ratio, 0.8)
        self.assertTrue(eligibility.eligible_before_ai)

    def test_speech_seconds_are_separate_from_stt_validity(self):
        turns = [
            SpeakingEvaluationTurn.model_validate(
                evaluation_turn(index, confidence=0.9)
            )
            for index in range(1, 6)
        ]
        turns.append(
            SpeakingEvaluationTurn.model_validate(
                {
                    **evaluation_turn(6, confidence=0.0),
                    "transcript": "",
                    "segments": [],
                }
            )
        )

        eligibility = calculate_evaluation_eligibility(turns)

        self.assertEqual(eligibility.valid_user_turns, 5)
        self.assertEqual(eligibility.valid_user_speech_seconds, 72)
        self.assertAlmostEqual(eligibility.valid_stt_turn_ratio, 5 / 6, places=4)

    def test_weighted_overall_policy_v1(self):
        scores = {
            "GRAMMAR": 50,
            "VOCABULARY": 60,
            "NATURALNESS": 70,
            "MEANING": 80,
            "EXPRESSIVENESS": 90,
            "FLUENCY": 100,
            "PRONUNCIATION": 0,
            "INTERACTION": 100,
        }
        metrics = [
            SpeakingMetricPayload.model_validate(metric_payload(name, score=score))
            for name, score in scores.items()
        ]
        self.assertEqual(calculate_speaking_overall(metrics), 70)

    async def test_conversation_prompt_contains_persona_profile_keyword_and_summary(self):
        provider = FakeStructuredProvider(results=[conversation_payload()])
        service = SpeakingConversationService(
            provider,
            timeout_seconds=1,
            automatic_retries=0,
        )
        history = [
            {"role": "USER", "text": f"old-{index}"}
            for index in range(20)
        ]
        await service.generate(
            conversation_request(
                persona="친근한 일본인 동료",
                learningProfileSummary={
                    "profileVersion": "p2",
                    "strengths": ["meaning"],
                    "weaknesses": ["fluency"],
                },
                selectedKeywords=[
                    {
                        "key": "business",
                        "text": "会議",
                        "source": "SYSTEM",
                        "type": "VOCABULARY",
                    }
                ],
                focusSignals=["fluency.pause"],
                conversationHistory=history,
                sessionSummary="이전 8개 Turn의 구조화 요약",
            )
        )
        prompt = provider.calls[0][1]
        self.assertIn('"persona":"친근한 일본인 동료"', prompt)
        self.assertIn('"profileVersion":"p2"', prompt)
        self.assertIn('"text":"会議"', prompt)
        self.assertIn('"focusSignals":["fluency.pause"]', prompt)
        self.assertIn("old-19", prompt)
        self.assertNotIn("old-0", prompt)
        self.assertIn("이전 8개 Turn의 구조화 요약", prompt)

    async def test_conversation_forces_end_at_max_turns(self):
        provider = FakeStructuredProvider(results=[conversation_payload()])
        service = SpeakingConversationService(
            provider,
            timeout_seconds=1,
            automatic_retries=0,
        )
        response = await service.generate(
            conversation_request(
                turnIndex=20,
                idempotencyKey="max-turns",
            )
        )
        self.assertTrue(response.conversation.should_end)
        self.assertEqual(response.conversation.end_reason, "MAX_TURNS")

    async def test_conversation_forces_end_at_max_session_duration(self):
        provider = FakeStructuredProvider(results=[conversation_payload()])
        service = SpeakingConversationService(
            provider,
            timeout_seconds=1,
            automatic_retries=0,
        )
        response = await service.generate(
            conversation_request(
                sessionElapsedSeconds=600,
                idempotencyKey="max-duration",
            )
        )
        self.assertTrue(response.conversation.should_end)
        self.assertEqual(response.conversation.end_reason, "MAX_SESSION_DURATION")

    async def test_conversation_same_idempotency_key_calls_provider_once(self):
        provider = FakeStructuredProvider(
            results=[conversation_payload(), conversation_payload()]
        )
        service = SpeakingConversationService(
            provider,
            timeout_seconds=1,
            automatic_retries=0,
        )
        request = conversation_request()
        first = await service.generate(request)
        second = await service.generate(request)
        self.assertEqual(first.assistant_text, second.assistant_text)
        self.assertEqual(len(provider.calls), 1)

    async def test_coaching_correction_requires_improvement_link(self):
        invalid = conversation_payload()
        invalid["coachingCorrections"] = [
            {
                "original": "私は昨日会う",
                "improved": "私は昨日会いました",
                "explanation": "과거형을 사용합니다.",
                "improvementLink": None,
            }
        ]
        provider = FakeStructuredProvider(results=[invalid])
        service = SpeakingConversationService(
            provider,
            timeout_seconds=1,
            automatic_retries=0,
        )
        with self.assertRaises(SpeakingStageException) as context:
            await service.generate(conversation_request(correctionMode="COACHING"))
        self.assertEqual(context.exception.code.value, "INVALID_RESPONSE_SCHEMA")

    async def test_conversation_respects_snapshot_retry_limit(self):
        provider = FakeStructuredProvider(
            errors=[RuntimeError("first"), None],
            results=[conversation_payload()],
        )
        service = SpeakingConversationService(
            provider,
            timeout_seconds=1,
            automatic_retries=2,
        )

        with self.assertRaises(SpeakingStageException):
            await service.generate(
                conversation_request(
                    idempotencyKey="no-auto-retry",
                    sessionPolicySnapshot={"automaticRetryLimitPerStage": 0},
                )
            )

        self.assertEqual(len(provider.calls), 1)

    async def test_provider_rate_limit_is_normalized(self):
        class RateLimitError(RuntimeError):
            status_code = 429

        provider = FakeStructuredProvider(errors=[RateLimitError("quota")])
        service = SpeakingConversationService(
            provider,
            timeout_seconds=1,
            automatic_retries=0,
        )
        with self.assertRaises(SpeakingStageException) as context:
            await service.generate(conversation_request())
        self.assertEqual(context.exception.code.value, "PROVIDER_RATE_LIMITED")
        self.assertTrue(context.exception.retryable)

    async def test_provider_timeout_is_normalized(self):
        provider = FakeStructuredProvider(errors=[asyncio.TimeoutError()])
        service = SpeakingConversationService(
            provider,
            timeout_seconds=1,
            automatic_retries=0,
        )
        with self.assertRaises(SpeakingStageException) as context:
            await service.generate(
                conversation_request(idempotencyKey="timeout-test")
            )
        self.assertEqual(context.exception.code.value, "PROVIDER_TIMEOUT")
        self.assertTrue(context.exception.retryable)

    async def test_tts_passes_voice_language_and_speed(self):
        provider = FakeSpeechProvider()
        store = TemporaryTtsAudioStore(ttl_seconds=60)
        service = SpeakingTtsService(
            provider,
            store,
            timeout_seconds=1,
            automatic_retries=0,
        )
        await service.synthesize(
            TtsRequest(
                requestId="tts-speed",
                idempotencyKey="tts-speed-idem",
                sessionId="session-1",
                text="こんにちは",
                learningLanguage="ja",
                voice="Puck",
                playbackSpeed="SLOW",
            )
        )
        self.assertEqual(provider.last_kwargs["voice"], "Puck")
        self.assertEqual(provider.last_kwargs["language"], "ja")
        self.assertEqual(provider.last_kwargs["speed"], "SLOW")

    async def test_tts_respects_request_retry_limit(self):
        provider = FakeSpeechProvider(errors=[RuntimeError("tts failure"), None])
        store = TemporaryTtsAudioStore(ttl_seconds=60)
        service = SpeakingTtsService(
            provider,
            store,
            timeout_seconds=1,
            automatic_retries=2,
        )

        with self.assertRaises(SpeakingStageException):
            await service.synthesize(
                TtsRequest(
                    requestId="tts-no-retry",
                    idempotencyKey="tts-no-retry-idem",
                    sessionId="session-1",
                    text="こんにちは",
                    learningLanguage="ja",
                    automaticRetryLimit=0,
                )
            )

        self.assertEqual(provider.calls, 1)

    async def test_evaluation_same_idempotency_key_calls_provider_once(self):
        provider = FakeStructuredProvider(
            results=[evaluation_payload(), evaluation_payload()]
        )
        service = SpeakingEvaluationService(
            provider,
            timeout_seconds=1,
            automatic_retries=0,
        )
        request = evaluation_request()
        first = await service.evaluate(request)
        second = await service.evaluate(request)
        self.assertEqual(first.overall_score, second.overall_score)
        self.assertEqual(len(provider.calls), 1)

    async def test_evaluation_is_cached_once_per_session_and_policy_version(self):
        provider = FakeStructuredProvider(
            results=[evaluation_payload(), evaluation_payload()]
        )
        service = SpeakingEvaluationService(
            provider,
            timeout_seconds=1,
            automatic_retries=0,
        )
        first = await service.evaluate(evaluation_request())
        second_request = evaluation_request().model_copy(
            update={
                "request_id": "eval-2",
                "idempotency_key": "eval-idem-2",
            }
        )
        second = await service.evaluate(second_request)

        self.assertEqual(first.overall_score, second.overall_score)
        self.assertEqual(second.request_id, "eval-2")
        self.assertEqual(len(provider.calls), 1)

    async def test_evaluation_prompt_marks_guided_turns(self):
        provider = FakeStructuredProvider(results=[evaluation_payload()])
        service = SpeakingEvaluationService(
            provider,
            timeout_seconds=1,
            automatic_retries=0,
        )
        turns = [evaluation_turn(i) for i in range(1, 6)]
        turns[0]["assistanceUsage"] = [{"type": "SAMPLE_ANSWER", "count": 1}]

        await service.evaluate(evaluation_request(turns))

        prompt = provider.calls[0][1]
        self.assertIn('"assistanceLevel":"GUIDED"', prompt)
        self.assertIn('"pronunciationEvidenceAvailable":true', prompt)

    async def test_evaluation_rejects_evidence_timestamp_outside_turn(self):
        payload = evaluation_payload()
        payload["metrics"][0]["evidence"][0]["endMs"] = 13000
        provider = FakeStructuredProvider(results=[payload])
        service = SpeakingEvaluationService(
            provider,
            timeout_seconds=1,
            automatic_retries=0,
        )
        with self.assertRaises(SpeakingStageException) as context:
            await service.evaluate(evaluation_request())
        self.assertEqual(context.exception.code.value, "INVALID_RESPONSE_SCHEMA")

    async def test_topic_recommended_resolves_to_topic_default(self):
        stt = FakeSttServiceForTurn()
        conversation = FakeConversationServiceForTurn()
        tts = FakeTtsServiceForTurn()
        service = SpeakingTurnService(
            audio_processor=FakeAudioProcessor(),
            stt_service=stt,
            conversation_service=conversation,
            tts_service=tts,
        )
        response = await service.start_session(
            SessionStartRequest(
                requestId="start-rec",
                idempotencyKey="start-rec-idem",
                sessionId="session-rec",
                originLanguage="ko",
                learningLanguage="ja",
                topic="일상",
                conversationStartMode="TOPIC_RECOMMENDED",
                topicRecommendedStartMode="USER_FIRST",
            )
        )
        self.assertEqual(response.resolved_start_mode.value, "USER_FIRST")
        self.assertIsNone(response.assistant)
        self.assertEqual(conversation.calls, 0)


if __name__ == "__main__":
    unittest.main()


class SpeakingAssistanceServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_generates_hint_payload(self):
        provider = FakeStructuredProvider(
            results=[{"type": "HINT", "content": "予定を表す表現を使ってみましょう。"}]
        )
        service = SpeakingAssistanceService(
            provider,
            timeout_seconds=1,
            automatic_retries=0,
        )

        response = await service.generate(assistance_request())

        self.assertEqual(response.type, AssistanceType.HINT)
        self.assertEqual(response.content, "予定を表す表現を使ってみましょう。")
        self.assertFalse(response.idempotent_replay)
        self.assertEqual(len(provider.calls), 1)

    async def test_same_idempotency_key_reuses_assistance(self):
        provider = FakeStructuredProvider(
            results=[{"type": "TRANSLATION", "content": "주말에는 무엇을 할 예정인가요?"}]
        )
        service = SpeakingAssistanceService(
            provider,
            timeout_seconds=1,
            automatic_retries=0,
        )
        request = assistance_request("TRANSLATION")

        first = await service.generate(request)
        second = await service.generate(request)

        self.assertFalse(first.idempotent_replay)
        self.assertTrue(second.idempotent_replay)
        self.assertEqual(len(provider.calls), 1)

    async def test_rejects_non_generated_assistance(self):
        service = SpeakingAssistanceService(
            FakeStructuredProvider(),
            timeout_seconds=1,
            automatic_retries=0,
        )

        with self.assertRaises(SpeakingStageException):
            await service.generate(assistance_request("REPLAY"))
