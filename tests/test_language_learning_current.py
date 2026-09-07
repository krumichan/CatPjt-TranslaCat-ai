import asyncio
import copy
import unittest
from collections import Counter

from fastapi import HTTPException

from app.ai.ports import SpeechSynthesisResult, StructuredGenerationResult
from app.features.language_learning.level_test.benchmark import LevelTestEvaluationBenchmarkService
from app.features.language_learning.level_test.policy import LEVEL_TEST_RECIPE
from app.features.language_learning.level_test.normalizer import LevelTestGenerationNormalizer
from app.features.language_learning.level_test.service import LevelTestService
from app.features.language_learning.quality import (
    DiversityCandidate,
    DiversityValidator,
    resolve_daily_complexity_band,
    resolve_listening_complexity_band,
)
from app.features.language_learning.listening.generation_service import ListeningGenerationService
from app.features.language_learning.speaking.audio_processor import NormalizedAudio
from app.features.language_learning.writing.service import LanguageLearningWritingService
from app.schemas.language_learning import DailyWritingGenerationRequest, WritingEvaluationResponse
from app.schemas.language_learning_level_test_benchmark import LevelTestBenchmarkSample
from app.schemas.language_learning_level_test import (
    LevelTestBestOptionAdvantage,
    LevelTestChoiceSelectionPolicy,
    LevelTestChoiceSemanticVerificationPayload,
    LevelTestItemType,
    LevelTestMetricResult,
    LevelTestQuestionCandidate,
    LevelTestQuestionGenerationPayload,
    LevelTestQuestionGenerationRequest,
    LevelTestSpeakingEvaluationContext,
    LevelTestSpeakingTaskResponseStatus,
    LevelTestTextEvaluationRequest,
    LevelTestVocabContextDesignPayload,
)
from app.schemas.language_learning_listening import ListeningSetGenerationRequest
from app.schemas.language_learning_quality import (
    DiversityContext,
    DiversityHistoryEntry,
    DiversityMetadata,
    GenerationSourceType,
    ScenarioCategory,
)
from app.schemas.language_learning_speaking import (
    AudioQualitySignals,
    SpeakingUsage,
    StageUsage,
    SttAnalysisMetadata,
    SttResponse,
    SttSegment,
    TranscriptResult,
)


class QueueProvider:
    def __init__(self, *, structured=None, plain=None, audio_bytes=b"RIFFfake-wav"):
        self.structured = list(structured or [])
        self.plain = list(plain or [])
        self.audio_bytes = audio_bytes
        self.calls = []
        self.tts_calls = []

    async def call_with_metadata(self, type_name, data, schema=None):
        self.calls.append((type_name, data, schema))
        if not self.structured:
            raise RuntimeError("no structured result")
        return StructuredGenerationResult(
            data=self.structured.pop(0),
            input_tokens=10,
            output_tokens=20,
            provider="fake",
            model="fake-model",
        )

    async def call(self, type_name, data, schema=None):
        self.calls.append((type_name, data, schema))
        if not self.plain:
            raise RuntimeError("no plain result")
        return self.plain.pop(0)

    async def synthesize_speech(self, *, text, voice, language, speed):
        self.tts_calls.append((text, voice, language, speed))
        return SpeechSynthesisResult(
            audio_bytes=self.audio_bytes,
            content_type="audio/wav",
            provider="fake-tts",
            model="fake-tts-model",
            duration_seconds=1.25,
        )


class FakePresignedAudioUploader:
    def __init__(self):
        self.uploads = []

    async def put(self, upload_url, audio_bytes, content_type):
        self.uploads.append((upload_url, audio_bytes, content_type))


class FakeWritingEvaluationService:
    def __init__(self):
        self.requests = []

    async def evaluate(self, request):
        self.requests.append(request)
        return WritingEvaluationResponse.model_validate(
            {
                "requestId": request.request_id,
                "scores": {
                    "overall": 82,
                    "meaning": 85,
                    "grammar": 80,
                    "vocabulary": 78,
                    "naturalness": 84,
                    "expression": 80,
                },
                "strengths": [
                    {"originText": "의미가 명확합니다.", "learningText": "意味が明確です。"}
                ],
                "weaknesses": [],
                "corrections": [],
                "recommendedAnswers": ["例1", "例2"],
                "explanation": {"originText": "좋습니다.", "learningText": "良いです。"},
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
                "evaluationRubricVersion": "writing-evaluation-rubric",
                "scoringPolicyVersion": "writing-scoring-policy",
                "promptVersion": "writing-evaluation",
            }
        )


class NoopEvaluationService:
    async def evaluate(self, request):
        raise AssertionError("not expected")


class FakeAudioProcessor:
    def validate_and_normalize(self, *args, **kwargs):
        return NormalizedAudio(
            wav_bytes=b"wav",
            duration_seconds=3.0,
            source_format="wav",
            quality=AudioQualitySignals(
                rms=0.1,
                peak=0.4,
                silence_ratio=0.1,
                sample_rate=16000,
                channels=1,
            ),
        )


class FakeSttService:
    def __init__(self, transcript_text="自己紹介をします。東京で働いています。"):
        self.transcript_text = transcript_text

    async def transcribe(self, **kwargs):
        quality = AudioQualitySignals(
            rms=0.1,
            peak=0.4,
            silence_ratio=0.1,
            sample_rate=16000,
            channels=1,
        )
        return SttResponse(
            request_id=kwargs["request_id"],
            session_id=kwargs["session_id"],
            turn_index=kwargs["turn_index"],
            transcript=TranscriptResult(
                text=self.transcript_text,
                language="ja",
                confidence=0.92,
                segments=[
                    SttSegment(
                        start_ms=0,
                        end_ms=3000,
                        text=self.transcript_text,
                        confidence=0.92,
                    )
                ],
                metadata=SttAnalysisMetadata(
                    provider="fake-stt",
                    model="fake",
                    requested_language="ja",
                    low_confidence_threshold=0.55,
                    audio_duration=3.0,
                    audio_quality_signals=quality,
                    normalization_version="speaking-audio-normalization",
                    stt_hint_version="speaking-stt-hint",
                ),
            ),
            usage=SpeakingUsage(
                stt=StageUsage(latency_ms=5, provider="fake-stt", model="fake")
            ),
        )


def diversity_metadata(scenario="WORK", intent="REQUEST", archetype="SCENARIO_RESPONSE"):
    return {
        "scenarioCategory": scenario,
        "communicativeIntent": intent,
        "taskArchetype": archetype,
        "grammarFocusCodes": ["POLITE_REQUEST"],
        "lexicalFocusCodes": ["SCHEDULE"],
        "semanticSummary": f"{scenario}-{intent}-{archetype}",
        "requiresBackgroundKnowledge": False,
    }


def writing_current_request():
    return DailyWritingGenerationRequest.model_validate(
        {
            "requestId": "w35-1",
            "originLanguage": "ko",
            "learningLanguage": "ja",
            "sentenceCount": 2,
            "difficultyDistribution": {"review": 0, "normal": 1, "challenge": 1},
            "selectedKeywords": [],
            "generationDate": "2026-08-27",
            "languageComplexity": {"baseComplexityBand": 3},
            "contentDiversityPolicyVersion": "language-learning-diversity",
        }
    )


def writing_candidate_payload():
    return {
        "items": [
            {
                "order": 1,
                "difficulty": "NORMAL",
                "originText": "회의 시간을 오후 세 시로 바꿀 수 있는지 동료에게 정중하게 물어보세요.",
                "keywords": [],
                "focusMetrics": ["NATURALNESS"],
                "focusReason": "정중한 요청",
                "languageComplexityBand": 3,
                "diversityMetadata": diversity_metadata("WORK", "REQUEST", "SCHEDULE_REQUEST"),
            },
            {
                "order": 2,
                "difficulty": "NORMAL",
                "originText": "여행 중 호텔 체크인 시간을 확인하는 메시지를 작성하세요.",
                "keywords": [],
                "focusMetrics": ["MEANING"],
                "focusReason": "정보 확인",
                "languageComplexityBand": 3,
                "diversityMetadata": diversity_metadata("TRAVEL", "CONFIRM", "SERVICE_CONFIRM"),
            },
            {
                "order": 3,
                "difficulty": "CHALLENGE",
                "originText": "배송이 늦어진 점에 유감을 표현하고 가능한 새 도착 시간을 완곡하게 문의하세요.",
                "keywords": [],
                "focusMetrics": ["EXPRESSION", "NATURALNESS"],
                "focusReason": "완곡한 문의",
                "languageComplexityBand": 4,
                "diversityMetadata": diversity_metadata("SERVICE", "ASK_INFORMATION", "HEDGED_INQUIRY"),
            },
            {
                "order": 4,
                "difficulty": "CHALLENGE",
                "originText": "친구의 제안을 바로 거절하지 않고 사정을 설명하며 다른 날짜를 제안하세요.",
                "keywords": [],
                "focusMetrics": ["EXPRESSION"],
                "focusReason": "간접 거절과 대안",
                "languageComplexityBand": 4,
                "diversityMetadata": diversity_metadata("SOCIAL", "DECLINE", "INDIRECT_DECLINE"),
            },
        ]
    }


def listening_current_request():
    return ListeningSetGenerationRequest.model_validate(
        {
            "requestId": "l35-1",
            "idempotencyKey": "l35-idem",
            "userContext": {
                "originLanguage": "ko",
                "learningLanguage": "ja",
                "level": "MY_LEVEL",
                "profileFocus": [],
            },
            "setContext": {
                "learningDate": "2026-08-27",
                "topic": {"id": 1, "title": "일상"},
                "selectedKeywords": [],
                "itemCount": 2,
                "difficulty": "MY_LEVEL",
            },
            "constraints": {"audioSecondsMin": 8, "audioSecondsMax": 20},
            "languageComplexity": {"baseComplexityBand": 3},
            "contentDiversityPolicyVersion": "language-learning-diversity",
        }
    )


def listening_candidate_payload():
    return {
        "items": [
            {
                "itemIndex": 1,
                "sourceText": "会議は午後三時から始まりますので、少し早めに来てください。",
                "referenceMeanings": ["회의는 오후 3시에 시작하니 조금 일찍 와 주세요.", "오후 3시 회의이므로 조금 일찍 와 주세요."],
                "keyMeaningUnits": ["회의는 오후 3시에 시작", "조금 일찍 와 주세요"],
                "targetKeywords": [],
                "estimatedAudioSeconds": 10,
                "safety": {"passed": True, "categories": []},
                "languageComplexityBand": 3,
                "diversityMetadata": diversity_metadata("WORK", "GIVE_INSTRUCTION", "SCHEDULE_NOTICE"),
            },
            {
                "itemIndex": 2,
                "sourceText": "ホテルに着いたら、受付で予約した名前を伝えてください。",
                "referenceMeanings": ["호텔에 도착하면 접수처에서 예약한 이름을 말해 주세요.", "호텔 접수처에 예약자 이름을 알려 주세요."],
                "keyMeaningUnits": ["호텔에 도착", "접수처", "예약한 이름을 전달"],
                "targetKeywords": [],
                "estimatedAudioSeconds": 10,
                "safety": {"passed": True, "categories": []},
                "languageComplexityBand": 3,
                "diversityMetadata": diversity_metadata("TRAVEL", "GIVE_INSTRUCTION", "CHECKIN_INSTRUCTION"),
            },
            {
                "itemIndex": 3,
                "sourceText": "注文した商品がまだ届いていないので、配送状況を確認したいです。",
                "referenceMeanings": ["주문한 상품이 아직 도착하지 않아 배송 상황을 확인하고 싶습니다.", "주문 상품이 안 와서 배송 상태를 확인하려고 합니다."],
                "keyMeaningUnits": ["주문 상품", "아직 도착하지 않음", "배송 상황 확인"],
                "targetKeywords": [],
                "estimatedAudioSeconds": 11,
                "safety": {"passed": True, "categories": []},
                "languageComplexityBand": 3,
                "diversityMetadata": diversity_metadata("SHOPPING", "ASK_INFORMATION", "DELIVERY_INQUIRY"),
            },
            {
                "itemIndex": 4,
                "sourceText": "今日は少し疲れているので、夕食の後は家でゆっくり休むつもりです。",
                "referenceMeanings": ["오늘은 조금 피곤해서 저녁 식사 후 집에서 푹 쉴 생각입니다.", "오늘 피곤해서 저녁 뒤에는 집에서 쉬려고 합니다."],
                "keyMeaningUnits": ["오늘 조금 피곤함", "저녁 식사 후", "집에서 쉴 예정"],
                "targetKeywords": [],
                "estimatedAudioSeconds": 12,
                "safety": {"passed": True, "categories": []},
                "languageComplexityBand": 3,
                "diversityMetadata": diversity_metadata("DAILY_LIFE", "REPORT", "DAILY_PLAN"),
            },
        ]
    }


def level_question_request():
    return LevelTestQuestionGenerationRequest.model_validate(
        {
            "requestId": "level-test-q1",
            "idempotencyKey": "level-test-q1-idem",
            "sessionId": 100,
            "questionNumber": 1,
            "totalQuestions": 20,
            "domain": "VOCABULARY",
            "itemType": "VOCAB_CONTEXT_CHOICE",
            "originLanguage": "ko",
            "learningLanguage": "ja",
            "targetComplexityBand": 2,
            "diversityContext": {
                "sameFeatureRecent": [
                    {
                        "sourceType": "LEVEL_TEST",
                        "content": "明日の約束の時間を変更したいときに最も自然な表現を選んでください。",
                    }
                ]
            },
        }
    )


def level_generation_payload():
    def candidate(prompt, scenario, intent):
        return {
            "domain": "VOCABULARY",
            "itemType": "VOCAB_CONTEXT_CHOICE",
            "complexityBand": 2,
            "instruction": "文脈に最も自然な表現を選んでください。",
            "instructionLanguage": "ja",
            "answerMode": "CHOICE",
            "answerLanguage": None,
            "promptText": prompt,
            "options": [
                {"key": "A", "text": "変更したいです"},
                {"key": "B", "text": "食べたいです"},
                {"key": "C", "text": "見ました"},
                {"key": "D", "text": "寝ています"},
            ],
            "internalAnswerKey": {"correctOptionKey": "A", "correctOrder": []},
            "referencePayload": {},
            "diversityMetadata": diversity_metadata(scenario, intent, "VOCAB_CONTEXT"),
        }

    return {
        "candidates": [
            candidate("明日の約束の時間を変更したいときに最も自然な表現を選んでください。", "SCHEDULE", "REQUEST"),
            candidate("ホテルでチェックアウト時間を遅らせたいときに最も適切な表現を選んでください。", "TRAVEL", "REQUEST"),
        ]
    }


def vocab_context_design_payload():
    return {
        "designs": [
            {
                "designId": "A",
                "targetExpression": "検討する",
                "targetMeaning": "여러 안을 비교하여 검토하다",
                "semanticConstraint": "여러 방안을 비교하고 판단해야 한다는 문맥",
                "scenarioCategory": "WORK",
                "communicativeIntent": "REPORT",
            },
            {
                "designId": "B",
                "targetExpression": "延長したいです",
                "targetMeaning": "시간을 연장하고 싶다",
                "semanticConstraint": "호텔 체크아웃 시간을 더 늦게 바꾸고 싶다는 문맥",
                "scenarioCategory": "TRAVEL",
                "communicativeIntent": "REQUEST",
            },
        ]
    }


def staged_vocab_generation_payload():
    return {
        "candidates": [
            {
                "domain": "VOCABULARY",
                "itemType": "VOCAB_CONTEXT_CHOICE",
                "complexityBand": 2,
                "instruction": "문맥에 가장 자연스러운 일본어 표현을 고르세요.",
                "instructionLanguage": "ko",
                "answerMode": "CHOICE",
                "answerLanguage": None,
                "promptText": "新しいプロジェクトについて、詳細を_____必要があります。",
                "options": [
                    {"key": "A", "text": "準備する"},
                    {"key": "B", "text": "検討する"},
                    {"key": "C", "text": "変更する"},
                    {"key": "D", "text": "提出する"},
                ],
                "internalAnswerKey": {"correctOptionKey": "B", "correctOrder": []},
                "generationPlanId": "A",
                "referencePayload": {},
                "diversityMetadata": diversity_metadata(
                    "WORK", "REPORT", "Target-first project review verb"
                ),
            },
            {
                "domain": "VOCABULARY",
                "itemType": "VOCAB_CONTEXT_CHOICE",
                "complexityBand": 2,
                "instruction": "문맥에 가장 자연스러운 일본어 표현을 고르세요.",
                "instructionLanguage": "ko",
                "answerMode": "CHOICE",
                "answerLanguage": None,
                "promptText": "ホテルでチェックアウト時間をもっと遅くしたいです。時間を_____。",
                "options": [
                    {"key": "A", "text": "延長したいです"},
                    {"key": "B", "text": "食べたいです"},
                    {"key": "C", "text": "見ました"},
                    {"key": "D", "text": "寝ています"},
                ],
                "internalAnswerKey": {"correctOptionKey": "A", "correctOrder": []},
                "generationPlanId": "B",
                "referencePayload": {},
                "diversityMetadata": diversity_metadata(
                    "TRAVEL", "REQUEST", "Target-first checkout extension"
                ),
            },
        ]
    }


def repaired_vocab_candidate():
    candidate = staged_vocab_generation_payload()["candidates"][0]
    candidate["promptText"] = (
        "新しいプロジェクトについて、複数の案を比較したうえで、詳細を_____必要があります。"
    )
    candidate["options"] = [
        {"key": "A", "text": "提出する"},
        {"key": "B", "text": "検討する"},
        {"key": "C", "text": "保管する"},
        {"key": "D", "text": "発送する"},
    ]
    return {"candidate": candidate}


class CurrentQualityPolicyTest(unittest.TestCase):
    def test_character_ngram_and_structural_duplicate_guards(self):
        metadata = DiversityMetadata.model_validate(diversity_metadata())
        validator = DiversityValidator()
        first = DiversityCandidate("회의 시간을 세 시로 바꾸고 싶습니다.", metadata)
        first_decision = validator.validate(first, [])
        self.assertTrue(first_decision.accepted)
        second = DiversityCandidate("회의 시간을 오후 세 시로 변경하고 싶습니다.", metadata)
        second_decision = validator.validate(second, [DiversityCandidate(first.content, first_decision.metadata)])
        self.assertFalse(second_decision.accepted)
        self.assertIn(second_decision.reason, {"STRUCTURAL", "SIMILARITY"})

    def test_background_knowledge_is_rejected(self):
        raw = diversity_metadata()
        raw["requiresBackgroundKnowledge"] = True
        decision = DiversityValidator().validate(
            DiversityCandidate("AI의 사회적 영향을 논하세요.", DiversityMetadata.model_validate(raw)),
            [],
        )
        self.assertEqual("BACKGROUND_KNOWLEDGE", decision.reason)

    def test_challenge_complexity_is_one_band_above(self):
        self.assertEqual(4, resolve_daily_complexity_band("CHALLENGE", None))
        self.assertEqual(3, resolve_daily_complexity_band("NORMAL", None))
        self.assertEqual(2, resolve_daily_complexity_band("REVIEW", None))

    def test_writing_keeps_per_bucket_bands_while_listening_can_use_target_override(self):
        from app.schemas.language_learning_quality import LanguageComplexityContext

        context = LanguageComplexityContext(
            baseComplexityBand=3,
            targetComplexityBand=5,
        )
        self.assertEqual(2, resolve_daily_complexity_band("REVIEW", context))
        self.assertEqual(3, resolve_daily_complexity_band("NORMAL", context))
        self.assertEqual(4, resolve_daily_complexity_band("CHALLENGE", context))
        self.assertEqual(5, resolve_listening_complexity_band("MY_LEVEL", context))


class CurrentGenerationTest(unittest.TestCase):
    def test_writing_current_returns_diversity_metadata_and_exact_distribution(self):
        provider = QueueProvider(plain=[writing_candidate_payload()])
        service = LanguageLearningWritingService(provider=provider)
        response = asyncio.run(service.generate_daily(writing_current_request()))
        self.assertEqual(2, len(response.items))
        self.assertEqual("writing-generation-modes-diversity", response.prompt_version)
        self.assertEqual("language-learning-diversity", response.content_diversity_policy_version)
        self.assertEqual({"NORMAL", "CHALLENGE"}, {item.difficulty.value for item in response.items})
        self.assertTrue(all(item.diversity_metadata for item in response.items))

    def test_writing_candidate_pool_never_exceeds_current_cap(self):
        distribution = LanguageLearningWritingService._expanded_distribution(
            {"REVIEW": 18, "NORMAL": 4, "CHALLENGE": 3}
        )
        self.assertEqual(40, distribution.total)
        self.assertGreaterEqual(distribution.review, 18)
        self.assertGreaterEqual(distribution.normal, 4)
        self.assertGreaterEqual(distribution.challenge, 3)

    def test_listening_current_uses_candidate_pool_and_returns_two_distinct_items(self):
        provider = QueueProvider(structured=[listening_candidate_payload()])
        service = ListeningGenerationService(provider, automatic_retries=0)
        response = asyncio.run(service.generate(listening_current_request()))
        self.assertEqual(2, len(response.items))
        self.assertEqual("listening-generation", response.generation_version)
        self.assertEqual("language-learning-diversity", response.content_diversity_policy_version)
        self.assertNotEqual(response.items[0].content_hash, response.items[1].content_hash)
        self.assertTrue(all(item.diversity_metadata for item in response.items))

    def test_listening_current_salvages_invalid_comprehension_focus_candidate(self):
        request_data = listening_current_request().model_dump(by_alias=True, mode="json")
        request_data["setContext"]["learningMode"] = "COMPREHENSION"
        payload = listening_candidate_payload()
        focuses = ["MAIN_IDEA", "next-action", "GIST", "DETAIL"]
        for index, item in enumerate(payload["items"]):
            item.update(
                {
                    "question": "話者について最も適切な説明はどれですか？",
                    "options": [
                        {"key": "A", "text": "選択肢A"},
                        {"key": "B", "text": "選択肢B"},
                        {"key": "C", "text": "選択肢C"},
                        {"key": "D", "text": "選択肢D"},
                    ],
                    "correctOptionKey": "B",
                    "comprehensionFocus": focuses[index],
                    "summaryKeyPoints": [],
                }
            )

        provider = QueueProvider(structured=[payload])
        service = ListeningGenerationService(provider, automatic_retries=0)
        response = asyncio.run(
            service.generate(ListeningSetGenerationRequest.model_validate(request_data))
        )

        self.assertEqual(2, len(response.items))
        self.assertEqual(["NEXT_ACTION", "GIST"], [item.comprehension_focus for item in response.items])
        self.assertEqual(1, len(provider.calls))

    def test_listening_generation_schema_exposes_comprehension_focus_as_enum(self):
        from app.schemas.language_learning_listening import ListeningGenerationPayload

        schema_text = str(ListeningGenerationPayload.model_json_schema())
        for value in ["GIST", "DETAIL", "INTENT", "INFERENCE", "NEXT_ACTION"]:
            self.assertIn(value, schema_text)


class CurrentBenchmarkTest(unittest.TestCase):
    def test_level_test_human_benchmark_requires_all_segments_to_pass(self):
        samples = [
            LevelTestBenchmarkSample.model_validate(
                {
                    "sampleId": "writing-1",
                    "domain": "WRITING",
                    "itemType": "WRITING_GUIDED_SENTENCE",
                    "language": "ja",
                    "complexityBand": 2,
                    "aiScore": 82,
                    "humanScores": [
                        {"evaluatorId": "r1", "score": 80},
                        {"evaluatorId": "r2", "score": 84},
                    ],
                }
            ),
            LevelTestBenchmarkSample.model_validate(
                {
                    "sampleId": "speaking-1",
                    "domain": "SPEAKING",
                    "itemType": "SPEAKING_GUIDED_RESPONSE",
                    "language": "ja",
                    "complexityBand": 3,
                    "aiScore": 74,
                    "humanScores": [
                        {"evaluatorId": "r1", "score": 70},
                        {"evaluatorId": "r2", "score": 76},
                    ],
                }
            ),
        ]
        result = LevelTestEvaluationBenchmarkService().evaluate(samples)
        self.assertTrue(result.passed)
        self.assertEqual(1.0, result.agreement_rate)


class CurrentLevelTestTest(unittest.TestCase):
    def _service(
        self,
        provider,
        *,
        audio_uploader=None,
        writing_service=None,
        verify_vocab_context_semantics=False,
        enable_vocab_context_multistage=False,
        verify_task_sufficiency=False,
        telemetry_summary_interval=20,
        stt_service=None,
    ):
        return LevelTestService(
            provider=provider,
            writing_service=writing_service or FakeWritingEvaluationService(),
            dictation_service=NoopEvaluationService(),
            interpretation_service=NoopEvaluationService(),
            audio_processor=FakeAudioProcessor(),
            stt_service=stt_service or FakeSttService(),
            speech_provider=provider,
            audio_uploader=audio_uploader,
            verify_vocab_context_semantics=verify_vocab_context_semantics,
            enable_vocab_context_multistage=enable_vocab_context_multistage,
            verify_task_sufficiency=verify_task_sufficiency,
            telemetry_summary_interval=telemetry_summary_interval,
        )

    def test_level_test_generation_is_fixed_to_20_question_recipe_and_skips_recent_duplicate(self):
        provider = QueueProvider(structured=[level_generation_payload()])
        response = asyncio.run(self._service(provider).generate_question(level_question_request()))
        self.assertEqual(20, response.total_questions)
        self.assertEqual("VOCABULARY", response.domain.value)
        self.assertIn("ホテル", response.prompt_text)
        self.assertEqual("level-test-generation", response.generation_version)
        self.assertEqual("level-test-multiskill-prompt", response.prompt_version)
        self.assertEqual(2, response.complexity_band)

    def test_blank_prompt_text_is_not_fabricated_from_generic_instruction(self):
        payload = level_generation_payload()
        payload["candidates"][0]["promptText"] = ""
        normalized, stats = LevelTestGenerationNormalizer.normalize(
            payload,
            origin_language="ko",
            learning_language="ja",
        )
        self.assertEqual("", normalized["candidates"][0]["promptText"])
        self.assertEqual(0, stats.prompt_text_fallbacks)

    def test_vocab_paraphrase_without_recoverable_emphasis_is_rejected(self):
        payload = level_generation_payload()
        for candidate in payload["candidates"]:
            candidate["itemType"] = "VOCAB_PARAPHRASE_CHOICE"
            candidate["instruction"] = "밑줄 친 표현과 가장 의미가 가까운 것을 고르세요."
        payload["candidates"][0]["promptText"] = "この<u>表現の意味を選んでください。"
        payload["candidates"][1]["promptText"] = "この表現</u>の意味を選んでください。"
        provider = QueueProvider(structured=[copy.deepcopy(payload) for _ in range(3)])
        request = LevelTestQuestionGenerationRequest.model_validate(
            {
                "requestId": "level-test-vocab-paraphrase-bad-markup",
                "idempotencyKey": "level-test-vocab-paraphrase-bad-markup-idem",
                "sessionId": 100,
                "questionNumber": 3,
                "totalQuestions": 20,
                "domain": "VOCABULARY",
                "itemType": "VOCAB_PARAPHRASE_CHOICE",
                "originLanguage": "ko",
                "learningLanguage": "ja",
                "targetComplexityBand": 2,
            }
        )

        with self.assertRaises(HTTPException) as raised:
            asyncio.run(self._service(provider).generate_question(request))
        self.assertEqual(422, raised.exception.status_code)
        self.assertEqual("QUESTION_CONTENT_INVALID", raised.exception.detail["code"])
        self.assertTrue(raised.exception.detail["reasons"])

    def test_reading_question_without_passage_is_rejected_while_complete_candidate_is_used(self):
        payload = level_generation_payload()
        prompts = [
            "この文章の最も伝えたいことは何ですか。",
            (
                "近年、多くの企業では市場環境の変化に合わせて事業戦略を見直しています。"
                "短期的な売上だけでなく、顧客の課題を継続的に把握し、新しい価値を提供することが重要です。"
                "そのため、社内では部門を越えた情報共有と長期的な投資を進めています。\n\n"
                "この文章の最も伝えたいことは何ですか。"
            ),
        ]
        for index, candidate in enumerate(payload["candidates"]):
            candidate.update(
                {
                    "domain": "READING",
                    "itemType": "READING_GIST",
                    "complexityBand": 3,
                    "instruction": "다음 글을 읽고 가장 적절한 답을 선택하세요.",
                    "promptText": prompts[index],
                }
            )
            candidate["diversityMetadata"] = diversity_metadata(
                "WORK" if index == 0 else "LEARNING",
                "REPORT",
                f"READING_GIST_{index}",
            )

        provider = QueueProvider(structured=[payload])
        request = LevelTestQuestionGenerationRequest.model_validate(
            {
                "requestId": "level-test-reading-gist",
                "idempotencyKey": "level-test-reading-gist-idem",
                "sessionId": 100,
                "questionNumber": 7,
                "totalQuestions": 20,
                "domain": "READING",
                "itemType": "READING_GIST",
                "originLanguage": "ko",
                "learningLanguage": "ja",
                "targetComplexityBand": 3,
            }
        )

        response = asyncio.run(self._service(provider).generate_question(request))

        self.assertIn("近年、多くの企業", response.prompt_text)
        self.assertGreater(len(response.prompt_text), 60)

    def test_mixed_reading_instruction_is_repaired_to_learning_language(self):
        payload = level_generation_payload()
        reading_prompt = (
            "旅行会社から、来週の出発時間が変更されたという連絡が届きました。"
            "新しい出発時間は午前九時で、集合場所はこれまでと同じ駅前です。\n\n"
            "このメッセージの主な目的は何ですか。"
        )
        for index, candidate in enumerate(payload["candidates"]):
            candidate.update(
                {
                    "domain": "READING",
                    "itemType": "READING_GIST",
                    "complexityBand": 2,
                    "instruction": "다음 글を読み、最も適切な答えを選んでください。",
                    "promptText": reading_prompt + str(index),
                }
            )
            candidate["diversityMetadata"] = diversity_metadata(
                "TRAVEL" if index == 0 else "SCHEDULE",
                "REPORT",
                f"READING_NOTICE_{index}",
            )

        provider = QueueProvider(structured=[payload])
        request = LevelTestQuestionGenerationRequest.model_validate(
            {
                "requestId": "level-test-reading-instruction",
                "idempotencyKey": "level-test-reading-instruction-idem",
                "sessionId": 100,
                "questionNumber": 7,
                "totalQuestions": 20,
                "domain": "READING",
                "itemType": "READING_GIST",
                "originLanguage": "ko",
                "learningLanguage": "ja",
                "targetComplexityBand": 2,
            }
        )

        response = asyncio.run(self._service(provider).generate_question(request))

        self.assertEqual("次の文章を読み、最も適切な答えを選んでください。", response.instruction)

    def test_reading_underline_is_removed_instead_of_highlighting_answer_evidence(self):
        payload = level_generation_payload()
        reading_prompt = (
            "企業では新しい制度を導入する前に小規模な試験運用を行いました。"
            "その結果を確認し、問題点を修正してから全社へ展開する方針です。\n\n"
            "なぜ最初に小規模な試験運用を行ったのですか。"
        )
        for index, candidate in enumerate(payload["candidates"]):
            candidate.update(
                {
                    "domain": "READING",
                    "itemType": "READING_DETAIL",
                    "complexityBand": 4,
                    "instruction": "다음 글을 읽고 질문에 답하세요.",
                    "promptText": reading_prompt.replace(
                        "問題点を修正してから全社へ展開する方針です。",
                        "<u>問題点を修正してから全社へ展開する方針です。</u>"
                    ),
                }
            )
            candidate["diversityMetadata"] = diversity_metadata(
                "WORK" if index == 0 else "LEARNING",
                "EXPLAIN_REASON",
                f"READING_DETAIL_UNDERLINE_{index}",
            )
        provider = QueueProvider(structured=[payload])
        request = LevelTestQuestionGenerationRequest.model_validate(
            {
                "requestId": "level-test-reading-no-underline",
                "idempotencyKey": "level-test-reading-no-underline-idem",
                "sessionId": 100,
                "questionNumber": 8,
                "totalQuestions": 20,
                "domain": "READING",
                "itemType": "READING_DETAIL",
                "originLanguage": "ko",
                "learningLanguage": "ja",
                "targetComplexityBand": 4,
            }
        )

        response = asyncio.run(self._service(provider).generate_question(request))

        self.assertNotIn("<u>", response.prompt_text)
        self.assertNotIn("</u>", response.prompt_text)
        self.assertIn("問題点を修正してから全社へ展開する方針です。", response.prompt_text)

    def test_sentence_order_options_are_rotated_when_provider_returns_answer_order(self):
        payload = level_generation_payload()
        for index, candidate in enumerate(payload["candidates"]):
            candidate.update(
                {
                    "domain": "GRAMMAR",
                    "itemType": "GRAMMAR_SENTENCE_ORDER",
                    "complexityBand": 3,
                    "instruction": "与えられた要素を正しい順序に並べてください。",
                    "promptText": "次の語句を正しい順序に並べて、意味の通る文を完成させてください。",
                    "options": [
                        {"key": "A", "text": "もし明日"},
                        {"key": "B", "text": "雨が降ったら、"},
                        {"key": "C", "text": "映画を見に"},
                        {"key": "D", "text": "行きませんか？"},
                    ],
                    "internalAnswerKey": {
                        "correctOptionKey": None,
                        "correctOrder": ["A", "B", "C", "D"],
                    },
                }
            )
            candidate["diversityMetadata"] = diversity_metadata(
                "DAILY_LIFE" if index == 0 else "SOCIAL",
                "SUGGEST",
                f"SENTENCE_ORDER_{index}",
            )
        provider = QueueProvider(structured=[payload])
        request = LevelTestQuestionGenerationRequest.model_validate(
            {
                "requestId": "level-test-order-shuffle",
                "idempotencyKey": "level-test-order-shuffle-idem",
                "sessionId": 100,
                "questionNumber": 6,
                "totalQuestions": 20,
                "domain": "GRAMMAR",
                "itemType": "GRAMMAR_SENTENCE_ORDER",
                "originLanguage": "ko",
                "learningLanguage": "ja",
                "targetComplexityBand": 3,
            }
        )

        response = asyncio.run(self._service(provider).generate_question(request))

        self.assertNotEqual(
            response.internal_answer_key.correct_order,
            [option.key for option in response.options],
        )
        self.assertEqual(["A", "B", "C", "D"], response.internal_answer_key.correct_order)

    def test_sentence_order_invalid_provider_keys_are_canonicalized_before_schema_validation(self):
        payload = level_generation_payload()
        for index, candidate in enumerate(payload["candidates"]):
            candidate.update(
                {
                    "domain": "GRAMMAR",
                    "itemType": "GRAMMAR_SENTENCE_ORDER",
                    "complexityBand": 3,
                    "instruction": "与えられた要素を正しい順序に並べてください。",
                    "promptText": "次の語句を正しい順序に並べて、意味の通る文を完成させてください。",
                    "options": [
                        {"key": "1", "text": "もし明日"},
                        {"key": "2", "text": "雨が降ったら、"},
                        {"key": "3", "text": "映画を見に"},
                        {"key": "4", "text": "一緒に"},
                        {"key": "5", "text": "行き"},
                        {"key": "6", "text": "ませんか？"},
                    ],
                    "internalAnswerKey": {
                        "correctOptionKey": None,
                        "correctOrder": ["1", "2", "3", "4", "5", "6"],
                    },
                }
            )
            candidate["diversityMetadata"] = diversity_metadata(
                "DAILY_LIFE" if index == 0 else "SOCIAL",
                "SUGGEST",
                f"SENTENCE_ORDER_NUMERIC_KEYS_{index}",
            )
        provider = QueueProvider(structured=[payload])
        request = LevelTestQuestionGenerationRequest.model_validate(
            {
                "requestId": "level-test-order-key-normalization",
                "idempotencyKey": "level-test-order-key-normalization-idem",
                "sessionId": 100,
                "questionNumber": 6,
                "totalQuestions": 20,
                "domain": "GRAMMAR",
                "itemType": "GRAMMAR_SENTENCE_ORDER",
                "originLanguage": "ko",
                "learningLanguage": "ja",
                "targetComplexityBand": 3,
            }
        )

        response = asyncio.run(self._service(provider).generate_question(request))

        self.assertEqual({"A", "B", "C", "D", "E", "F"}, {option.key for option in response.options})
        self.assertEqual(["A", "B", "C", "D", "E", "F"], response.internal_answer_key.correct_order)
        self.assertNotEqual(
            response.internal_answer_key.correct_order,
            [option.key for option in response.options],
        )

    def test_choice_invalid_provider_keys_and_correct_key_are_canonicalized_together(self):
        payload = level_generation_payload()
        for candidate in payload["candidates"]:
            candidate["options"] = [
                {"key": "opt-1", "text": "変更したいです"},
                {"key": "opt-2", "text": "食べたいです"},
                {"key": "opt-3", "text": "見ました"},
                {"key": "opt-4", "text": "寝ています"},
            ]
            candidate["internalAnswerKey"] = {
                "correctOptionKey": "opt-1",
                "correctOrder": [],
            }

        provider = QueueProvider(structured=[payload])
        response = asyncio.run(
            self._service(provider).generate_question(level_question_request())
        )

        self.assertEqual(["A", "B", "C", "D"], [option.key for option in response.options])
        self.assertEqual("A", response.internal_answer_key.correct_option_key)

    def test_vocab_context_choice_rejects_ambiguous_candidate_even_when_self_audit_claims_unique(self):
        payload = level_generation_payload()
        ambiguous = payload["candidates"][0]
        ambiguous["promptText"] = "新しいプロジェクトについて、詳細を_____必要があります。"
        ambiguous["options"] = [
            {"key": "A", "text": "準備する"},
            {"key": "B", "text": "検討する"},
            {"key": "C", "text": "変更する"},
            {"key": "D", "text": "提出する"},
        ]
        ambiguous["internalAnswerKey"] = {"correctOptionKey": "B", "correctOrder": []}
        ambiguous["choiceQualityAudit"] = {
            "uniqueCorrectOption": True,
            "directlyCompatibleOptionKeys": ["B"],
        }

        provider = QueueProvider(
            structured=[
                payload,
                {"verifiable": True, "plausibleOptionKeys": ["B", "D"], "bestOptionKey": "B", "bestOptionAdvantage": "CLEAR"},
                {"verifiable": True, "plausibleOptionKeys": ["A"], "bestOptionKey": "A", "bestOptionAdvantage": "CLEAR"},
            ]
        )

        response = asyncio.run(
            self._service(
                provider,
                verify_vocab_context_semantics=True,
            ).generate_question(level_question_request())
        )

        self.assertIn("ホテル", response.prompt_text)
        self.assertEqual(3, len(provider.calls))
        self.assertEqual(
            "LANGUAGE_LEARNING_LEVEL_TEST_CHOICE_VERIFICATION",
            provider.calls[1][0],
        )
        self.assertNotIn("correctOptionKey", provider.calls[1][1])
        self.assertNotIn("choiceQualityAudit", provider.calls[1][1])

    def test_vocab_context_choice_semantic_verifier_accepts_only_expected_single_key(self):
        payload = level_generation_payload()
        provider = QueueProvider(
            structured=[
                payload,
                {"verifiable": True, "plausibleOptionKeys": ["A"], "bestOptionKey": "A", "bestOptionAdvantage": "CLEAR"},
                {"verifiable": True, "plausibleOptionKeys": ["A"], "bestOptionKey": "A", "bestOptionAdvantage": "CLEAR"},
            ]
        )

        response = asyncio.run(
            self._service(
                provider,
                verify_vocab_context_semantics=True,
            ).generate_question(level_question_request())
        )

        self.assertEqual("A", response.internal_answer_key.correct_option_key)
        self.assertEqual(3, len(provider.calls))

    def test_vocab_context_choice_rejects_nfkc_equivalent_duplicate_options_without_ai(self):
        payload = level_generation_payload()
        for candidate in payload["candidates"]:
            candidate["options"] = [
                {"key": "A", "text": "ＡＢＣ"},
                {"key": "B", "text": "ABC"},
                {"key": "C", "text": "変更する"},
                {"key": "D", "text": "提出する"},
            ]
            candidate["internalAnswerKey"] = {"correctOptionKey": "A", "correctOrder": []}
        provider = QueueProvider(structured=[copy.deepcopy(payload) for _ in range(3)])

        with self.assertRaises(HTTPException) as raised:
            asyncio.run(self._service(provider).generate_question(level_question_request()))

        self.assertEqual(422, raised.exception.status_code)
        self.assertTrue(
            any("실질적으로 동일한 표현" in reason for reason in raised.exception.detail["reasons"])
        )

    def test_vocab_context_multistage_design_generate_verify_accepts_without_repair(self):
        generated = staged_vocab_generation_payload()
        generated["candidates"][0]["promptText"] = (
            "新しいプロジェクトについて、複数の案を比較したうえで、詳細を_____必要があります。"
        )
        generated["candidates"][0]["options"] = [
            {"key": "A", "text": "提出する"},
            {"key": "B", "text": "検討する"},
            {"key": "C", "text": "保管する"},
            {"key": "D", "text": "発送する"},
        ]
        provider = QueueProvider(
            structured=[
                vocab_context_design_payload(),
                generated,
                {"verifiable": True, "plausibleOptionKeys": ["B"], "bestOptionKey": "B", "bestOptionAdvantage": "CLEAR"},
            ]
        )

        with self.assertLogs(
            "app.features.language_learning.level_test.service", level="INFO"
        ) as captured:
            response = asyncio.run(
                self._service(
                    provider,
                    verify_vocab_context_semantics=True,
                    enable_vocab_context_multistage=True,
                ).generate_question(level_question_request())
            )

        self.assertEqual("B", response.internal_answer_key.correct_option_key)
        self.assertEqual(3, len(provider.calls))
        self.assertEqual(
            "LANGUAGE_LEARNING_LEVEL_TEST_VOCAB_CONTEXT_DESIGN",
            provider.calls[0][0],
        )
        self.assertEqual("LANGUAGE_LEARNING_LEVEL_TEST_GENERATION", provider.calls[1][0])
        self.assertEqual(
            "LANGUAGE_LEARNING_LEVEL_TEST_CHOICE_VERIFICATION",
            provider.calls[2][0],
        )
        self.assertIn("vocabContextDesigns", provider.calls[1][1])
        self.assertTrue(any("designCalls=1" in line for line in captured.output))
        self.assertTrue(any("repairCalls=0" in line for line in captured.output))

    def test_vocab_context_multistage_repairs_once_after_ambiguous_verdict(self):
        provider = QueueProvider(
            structured=[
                vocab_context_design_payload(),
                staged_vocab_generation_payload(),
                {"verifiable": True, "plausibleOptionKeys": ["B", "D"], "bestOptionKey": "B", "bestOptionAdvantage": "CLEAR"},
                repaired_vocab_candidate(),
                {"verifiable": True, "plausibleOptionKeys": ["B"], "bestOptionKey": "B", "bestOptionAdvantage": "CLEAR"},
            ]
        )

        with self.assertLogs(
            "app.features.language_learning.level_test.service", level="INFO"
        ) as captured:
            response = asyncio.run(
                self._service(
                    provider,
                    verify_vocab_context_semantics=True,
                    enable_vocab_context_multistage=True,
                ).generate_question(level_question_request())
            )

        self.assertIn("複数の案を比較", response.prompt_text)
        self.assertEqual(5, len(provider.calls))
        self.assertEqual(
            "LANGUAGE_LEARNING_LEVEL_TEST_VOCAB_CONTEXT_REPAIR",
            provider.calls[3][0],
        )
        self.assertNotIn("correctOptionKey", provider.calls[2][1])
        self.assertTrue(any("repairCalls=1" in line for line in captured.output))
        self.assertTrue(any("repairAccepted=1" in line for line in captured.output))

    def test_vocab_context_multistage_rejects_generator_target_drift_before_verifier(self):
        generated = staged_vocab_generation_payload()
        generated["candidates"][0]["options"] = [
            {"key": "A", "text": "準備する"},
            {"key": "B", "text": "提出する"},
            {"key": "C", "text": "変更する"},
            {"key": "D", "text": "保管する"},
        ]
        generated["candidates"][0]["internalAnswerKey"] = {
            "correctOptionKey": "B",
            "correctOrder": [],
        }
        provider = QueueProvider(
            structured=[
                vocab_context_design_payload(),
                generated,
                {"verifiable": True, "plausibleOptionKeys": ["A"], "bestOptionKey": "A", "bestOptionAdvantage": "CLEAR"},
            ]
        )

        response = asyncio.run(
            self._service(
                provider,
                verify_vocab_context_semantics=True,
                enable_vocab_context_multistage=True,
            ).generate_question(level_question_request())
        )

        self.assertIn("チェックアウト", response.prompt_text)
        self.assertEqual(3, len(provider.calls))
        self.assertEqual(
            "LANGUAGE_LEARNING_LEVEL_TEST_CHOICE_VERIFICATION",
            provider.calls[2][0],
        )

    def test_vocab_context_multistage_falls_back_to_verified_generation_when_design_is_invalid(self):
        generated = staged_vocab_generation_payload()
        generated["candidates"][0]["generationPlanId"] = None
        generated["candidates"][0]["promptText"] = (
            "新しいプロジェクトについて、複数の案を比較したうえで、詳細を_____必要があります。"
        )
        generated["candidates"][0]["options"] = [
            {"key": "A", "text": "提出する"},
            {"key": "B", "text": "検討する"},
            {"key": "C", "text": "保管する"},
            {"key": "D", "text": "発送する"},
        ]
        provider = QueueProvider(
            structured=[
                {"designs": []},
                generated,
                {"verifiable": True, "plausibleOptionKeys": ["B"], "bestOptionKey": "B", "bestOptionAdvantage": "CLEAR"},
            ]
        )

        with self.assertLogs(
            "app.features.language_learning.level_test.service", level="INFO"
        ) as captured:
            response = asyncio.run(
                self._service(
                    provider,
                    verify_vocab_context_semantics=True,
                    enable_vocab_context_multistage=True,
                ).generate_question(level_question_request())
            )

        self.assertEqual("B", response.internal_answer_key.correct_option_key)
        self.assertEqual(3, len(provider.calls))
        self.assertTrue(any("designFallbacks=1" in line for line in captured.output))

    def test_choice_selection_policy_allows_advanced_best_answer_items(self):
        service = self._service(QueueProvider())

        self.assertEqual(
            LevelTestChoiceSelectionPolicy.UNIQUE_ANSWER,
            service._choice_selection_policy(LevelTestItemType.VOCAB_CONTEXT_CHOICE, 3),
        )
        self.assertEqual(
            LevelTestChoiceSelectionPolicy.BEST_ANSWER,
            service._choice_selection_policy(LevelTestItemType.VOCAB_CONTEXT_CHOICE, 4),
        )
        self.assertEqual(
            LevelTestChoiceSelectionPolicy.BEST_ANSWER,
            service._choice_selection_policy(LevelTestItemType.VOCAB_PARAPHRASE_CHOICE, 3),
        )
        self.assertEqual(
            LevelTestChoiceSelectionPolicy.BEST_ANSWER,
            service._choice_selection_policy(LevelTestItemType.GRAMMAR_FORM_CHOICE, 4),
        )
        self.assertEqual(
            LevelTestChoiceSelectionPolicy.BEST_ANSWER,
            service._choice_selection_policy(LevelTestItemType.READING_TEXT_INFERENCE, 3),
        )
        self.assertEqual(
            LevelTestChoiceSelectionPolicy.BEST_ANSWER,
            service._choice_selection_policy(LevelTestItemType.LISTENING_DETAIL_CHOICE, 4),
        )
        self.assertIsNone(
            service._choice_selection_policy(LevelTestItemType.GRAMMAR_SENTENCE_ORDER, 5)
        )

    def test_advanced_grammar_best_answer_accepts_multiple_plausible_when_both_verifiers_agree(self):
        payload = level_generation_payload()
        for index, candidate in enumerate(payload["candidates"]):
            candidate.update(
                {
                    "domain": "GRAMMAR",
                    "itemType": "GRAMMAR_FORM_CHOICE",
                    "complexityBand": 5,
                    "instruction": "문맥상 가장 적절한 연결 표현을 고르세요.",
                    "promptText": "彼は重い病気を患っていた。_____、最後まで職務を全うした。",
                    "options": [
                        {"key": "A", "text": "にもかかわらず"},
                        {"key": "B", "text": "とはいえ"},
                        {"key": "C", "text": "からには"},
                        {"key": "D", "text": "ところで"},
                    ],
                    "internalAnswerKey": {"correctOptionKey": "A", "correctOrder": []},
                }
            )
            candidate["diversityMetadata"] = diversity_metadata(
                "HEALTH_GENERAL",
                "DESCRIBE",
                f"ADVANCED_CONCESSION_{index}",
            )
        provider = QueueProvider(
            structured=[
                payload,
                {
                    "verifiable": True,
                    "plausibleOptionKeys": ["A", "B"],
                    "bestOptionKey": "A",
                    "bestOptionAdvantage": "CLEAR",
                },
                {
                    "verifiable": True,
                    "plausibleOptionKeys": ["A", "B"],
                    "bestOptionKey": "A",
                    "bestOptionAdvantage": "CLEAR",
                },
            ]
        )
        request = LevelTestQuestionGenerationRequest.model_validate(
            {
                "requestId": "level-test-grammar-best",
                "idempotencyKey": "level-test-grammar-best-idem",
                "sessionId": 100,
                "questionNumber": 4,
                "totalQuestions": 20,
                "domain": "GRAMMAR",
                "itemType": "GRAMMAR_FORM_CHOICE",
                "originLanguage": "ko",
                "learningLanguage": "ja",
                "targetComplexityBand": 5,
            }
        )

        response = asyncio.run(
            self._service(provider, verify_vocab_context_semantics=True).generate_question(request)
        )

        self.assertEqual("A", response.internal_answer_key.correct_option_key)
        self.assertEqual(3, len(provider.calls))
        self.assertIn("BEST_ANSWER", provider.calls[0][1])
        self.assertIn('"selectionPolicy":"BEST_ANSWER"', provider.calls[1][1])
        self.assertNotIn("correctOptionKey", provider.calls[1][1])
        self.assertIn("Adversarial second pass", provider.calls[2][1])
        self.assertEqual("BEST_ANSWER", response.internal_answer_key.selection_policy.value)
        self.assertEqual({"A": 100, "B": 30, "C": 0, "D": 0}, response.internal_answer_key.option_scores)

    def test_best_answer_partial_credit_uses_two_verifier_consensus(self):
        normalized, _ = LevelTestGenerationNormalizer.normalize(
            {"candidates": [staged_vocab_generation_payload()["candidates"][0]]},
            origin_language="ko",
            learning_language="ja",
        )
        candidate = LevelTestQuestionGenerationPayload.model_validate(normalized).candidates[0]
        candidate = candidate.model_copy(update={"complexity_band": 5})
        first = LevelTestChoiceSemanticVerificationPayload.model_validate(
            {
                "verifiable": True,
                "plausibleOptionKeys": ["B", "C", "D"],
                "bestOptionKey": "B",
                "bestOptionAdvantage": "CLEAR",
            }
        )
        second = LevelTestChoiceSemanticVerificationPayload.model_validate(
            {
                "verifiable": True,
                "plausibleOptionKeys": ["B", "D"],
                "bestOptionKey": "B",
                "bestOptionAdvantage": "CLEAR",
            }
        )

        scores = LevelTestService._derive_choice_option_scores(
            candidate, LevelTestChoiceSelectionPolicy.BEST_ANSWER, [first, second]
        )
        self.assertEqual(100, scores["B"])
        self.assertEqual(10, scores["C"])
        self.assertEqual(30, scores["D"])
        self.assertEqual(0, scores["A"])

        unique_scores = LevelTestService._derive_choice_option_scores(
            candidate, LevelTestChoiceSelectionPolicy.UNIQUE_ANSWER, [first]
        )
        self.assertEqual({"A": 0, "B": 100, "C": 0, "D": 0}, unique_scores)

    def test_best_answer_requires_clear_advantage_and_unique_answer_rejects_multiple_plausible(self):
        normalized, _ = LevelTestGenerationNormalizer.normalize(
            {"candidates": [staged_vocab_generation_payload()["candidates"][0]]},
            origin_language="ko",
            learning_language="ja",
        )
        candidate = LevelTestQuestionGenerationPayload.model_validate(normalized).candidates[0]
        service = self._service(QueueProvider())

        unique_verdict = LevelTestChoiceSemanticVerificationPayload.model_validate(
            {
                "verifiable": True,
                "plausibleOptionKeys": ["B", "D"],
                "bestOptionKey": "B",
                "bestOptionAdvantage": "CLEAR",
            }
        )
        self.assertIsNotNone(
            service._choice_semantic_rejection_reason(
                candidate, unique_verdict, LevelTestChoiceSelectionPolicy.UNIQUE_ANSWER
            )
        )

        candidate = candidate.model_copy(update={"complexity_band": 5})
        weak_best = LevelTestChoiceSemanticVerificationPayload.model_validate(
            {
                "verifiable": True,
                "plausibleOptionKeys": ["B", "D"],
                "bestOptionKey": "B",
                "bestOptionAdvantage": "WEAK",
            }
        )
        self.assertIsNotNone(
            service._choice_semantic_rejection_reason(
                candidate, weak_best, LevelTestChoiceSelectionPolicy.BEST_ANSWER
            )
        )

    def test_best_answer_rejects_functionally_near_equivalent_rival_before_partial_credit(self):
        normalized, _ = LevelTestGenerationNormalizer.normalize(
            {"candidates": [staged_vocab_generation_payload()["candidates"][0]]},
            origin_language="ko",
            learning_language="ja",
        )
        candidate = LevelTestQuestionGenerationPayload.model_validate(normalized).candidates[0]
        candidate = candidate.model_copy(update={"complexity_band": 5})
        verdict = LevelTestChoiceSemanticVerificationPayload.model_validate(
            {
                "verifiable": True,
                "plausibleOptionKeys": ["B", "D"],
                "nearEquivalentOptionKeys": ["D"],
                "bestOptionKey": "B",
                "bestOptionAdvantage": "CLEAR",
            }
        )

        reason = LevelTestService._choice_semantic_rejection_reason(
            candidate, verdict, LevelTestChoiceSelectionPolicy.BEST_ANSWER
        )
        self.assertIsNotNone(reason)
        self.assertIn("기능적으로 거의 동등", reason)
        self.assertEqual(
            {"A": 0, "B": 100, "C": 0, "D": 0},
            LevelTestService._derive_choice_option_scores(
                candidate, LevelTestChoiceSelectionPolicy.BEST_ANSWER, [verdict]
            ),
        )

    def test_reading_and_listening_choice_reject_origin_language_lane_leak(self):
        request = LevelTestQuestionGenerationRequest.model_validate(
            {
                "requestId": "lane-guard",
                "idempotencyKey": "lane-guard-idem",
                "sessionId": 100,
                "questionNumber": 11,
                "totalQuestions": 20,
                "domain": "LISTENING",
                "itemType": "LISTENING_GIST_CHOICE",
                "originLanguage": "ko",
                "learningLanguage": "ja",
                "targetComplexityBand": 2,
            }
        )
        payload = level_generation_payload()["candidates"][0]
        payload.update(
            {
                "domain": "LISTENING",
                "itemType": "LISTENING_GIST_CHOICE",
                "complexityBand": 2,
                "instruction": "음성을 듣고 중심 내용을 고르세요.",
                "instructionLanguage": "ko",
                "answerMode": "CHOICE",
                "answerLanguage": None,
                "promptText": "이 음성의 중심 내용은 무엇입니까?",
                "options": [
                    {"key": "A", "text": "회의 일정"},
                    {"key": "B", "text": "날씨"},
                    {"key": "C", "text": "여행"},
                    {"key": "D", "text": "주문"},
                ],
                "internalAnswerKey": {"correctOptionKey": "A", "correctOrder": []},
                "referencePayload": {
                    "sourceText": "会議は午後三時から始まります。",
                    "listeningQuestion": "이 음성의 중심 내용은 무엇입니까?",
                },
            }
        )
        normalized, _ = LevelTestGenerationNormalizer.normalize(
            {"candidates": [payload]}, origin_language="ko", learning_language="ja"
        )
        candidate = LevelTestQuestionGenerationPayload.model_validate(normalized).candidates[0]
        with self.assertRaisesRegex(ValueError, "learningLanguage=ja"):
            LevelTestService._validate_question_candidate(request, candidate)

    def test_vocab_design_normalizer_drops_schema_noise_without_inventing_semantics(self):
        raw = vocab_context_design_payload()
        raw["designs"][0]["designId"] = "unexpected"
        raw["designs"][0]["obsoletePresentationField"] = "ignored"
        raw["designs"].append(copy.deepcopy(raw["designs"][0]))

        normalized = LevelTestService._normalize_vocab_context_design_payload(raw)
        parsed = LevelTestVocabContextDesignPayload.model_validate(normalized)

        self.assertEqual(["A", "B"], [item.design_id for item in parsed.designs])
        self.assertNotIn("obsoletePresentationField", normalized["designs"][0])
        self.assertEqual("検討する", parsed.designs[0].target_expression)

    def test_vocab_repair_is_attempted_only_for_repairable_semantic_ambiguity(self):
        normalized, _ = LevelTestGenerationNormalizer.normalize(
            {"candidates": [staged_vocab_generation_payload()["candidates"][0]]},
            origin_language="ko",
            learning_language="ja",
        )
        candidate = LevelTestQuestionGenerationPayload.model_validate(normalized).candidates[0]
        ambiguous = LevelTestChoiceSemanticVerificationPayload.model_validate(
            {
                "verifiable": True,
                "plausibleOptionKeys": ["B", "D"],
                "bestOptionKey": "B",
                "bestOptionAdvantage": "CLEAR",
            }
        )
        unverifiable = LevelTestChoiceSemanticVerificationPayload.model_validate(
            {
                "verifiable": False,
                "plausibleOptionKeys": ["B", "D"],
                "bestOptionKey": "B",
                "bestOptionAdvantage": "WEAK",
            }
        )

        self.assertTrue(
            LevelTestService._vocab_context_repairable_semantic_failure(
                candidate, ambiguous, LevelTestChoiceSelectionPolicy.UNIQUE_ANSWER
            )
        )
        self.assertFalse(
            LevelTestService._vocab_context_repairable_semantic_failure(
                candidate, unverifiable, LevelTestChoiceSelectionPolicy.UNIQUE_ANSWER
            )
        )

    def test_advanced_listening_best_answer_is_verified_before_reference_audio_upload(self):
        payload = level_generation_payload()
        source_text = "会議は午後三時から始まる予定でしたが、都合により四時に変更されました。"
        for index, candidate in enumerate(payload["candidates"]):
            candidate.update(
                {
                    "domain": "LISTENING",
                    "itemType": "LISTENING_DETAIL_CHOICE",
                    "complexityBand": 4,
                    "instruction": "음성을 듣고 가장 적절한 답을 고르세요.",
                    "promptText": "会議の開始時刻について最も適切な説明はどれですか。",
                    "options": [
                        {"key": "A", "text": "午後四時に変更された"},
                        {"key": "B", "text": "午後三時のままである"},
                        {"key": "C", "text": "午前四時に変更された"},
                        {"key": "D", "text": "会議は中止された"},
                    ],
                    "internalAnswerKey": {"correctOptionKey": "A", "correctOrder": []},
                    "referencePayload": {"sourceText": source_text},
                }
            )
            candidate["diversityMetadata"] = diversity_metadata(
                "WORK", "REPORT", f"LISTENING_DETAIL_BEST_{index}"
            )
        verdict = {
            "verifiable": True,
            "plausibleOptionKeys": ["A"],
            "bestOptionKey": "A",
            "bestOptionAdvantage": "CLEAR",
        }
        provider = QueueProvider(structured=[payload, verdict, verdict])
        uploader = FakePresignedAudioUploader()
        request = LevelTestQuestionGenerationRequest.model_validate(
            {
                "requestId": "level-test-listening-best",
                "idempotencyKey": "level-test-listening-best-idem",
                "sessionId": 100,
                "questionNumber": 12,
                "totalQuestions": 20,
                "domain": "LISTENING",
                "itemType": "LISTENING_DETAIL_CHOICE",
                "originLanguage": "ko",
                "learningLanguage": "ja",
                "targetComplexityBand": 4,
                "referenceAudioUpload": {
                    "uploadUrl": "https://upload.invalid/audio",
                    "objectKey": "level-test/audio.wav",
                    "contentType": "audio/wav",
                    "voice": "Kore",
                    "playbackSpeed": "NORMAL",
                },
            }
        )

        response = asyncio.run(
            self._service(
                provider,
                audio_uploader=uploader,
                verify_vocab_context_semantics=True,
            ).generate_question(request)
        )

        self.assertIsNotNone(response.reference_audio)
        self.assertEqual(2, sum(call[0] == "LANGUAGE_LEARNING_LEVEL_TEST_CHOICE_VERIFICATION" for call in provider.calls))
        verifier_prompt = provider.calls[1][1]
        self.assertIn(source_text, verifier_prompt)
        self.assertIn('"selectionPolicy":"BEST_ANSWER"', verifier_prompt)
        self.assertEqual(1, len(provider.tts_calls))
        self.assertEqual(1, len(uploader.uploads))

    def test_generation_quality_summary_logs_cumulative_rates(self):
        provider = QueueProvider(structured=[level_generation_payload(), level_generation_payload()])
        service = self._service(provider, telemetry_summary_interval=2)
        first = level_question_request()
        second = first.model_copy(
            update={
                "request_id": "level-test-q1-second",
                "idempotency_key": "level-test-q1-second-idem",
            }
        )

        with self.assertLogs(
            "app.features.language_learning.level_test.service", level="INFO"
        ) as captured:
            asyncio.run(service.generate_question(first))
            asyncio.run(service.generate_question(second))

        self.assertTrue(
            any(
                "generation quality summary" in line
                and "requests=2" in line
                and "acceptanceRate=100.0" in line
                for line in captured.output
            )
        )

    def test_vocab_context_staged_schema_requires_generation_plan_id(self):
        schema = self._service(QueueProvider())._generation_schema(
            level_question_request(),
            require_vocab_plan_id=True,
        )
        candidate_schema = schema["$defs"]["LevelTestQuestionCandidate"]

        self.assertIn("generationPlanId", candidate_schema["required"])
        self.assertEqual(
            ["A", "B"],
            candidate_schema["properties"]["generationPlanId"]["enum"],
        )

    def test_level_test_generation_schema_exposes_reference_payload_fields_to_provider(self):
        schema = LevelTestQuestionGenerationPayload.model_json_schema()
        reference_schema = schema["$defs"]["LevelTestReferencePayload"]

        self.assertEqual(
            {
                "sourceText",
                "referenceMeanings",
                "keyMeaningUnits",
                "referenceText",
                "translationSourceText",
                "emphasisText",
                "readingPassage",
                "readingQuestion",
                "listeningQuestion",
                "providedFacts",
                "requiredIntents",
                "responseConstraints",
            },
            set(reference_schema["properties"]),
        )
        self.assertEqual(
            {
                "sourceText",
                "referenceMeanings",
                "keyMeaningUnits",
                "referenceText",
                "translationSourceText",
                "emphasisText",
                "readingPassage",
                "readingQuestion",
                "listeningQuestion",
                "providedFacts",
                "requiredIntents",
                "responseConstraints",
            },
            set(reference_schema["required"]),
        )

    def test_listening_provider_schema_requires_non_nullable_source_text(self):
        request = LevelTestQuestionGenerationRequest.model_validate(
            {
                "requestId": "level-test-listening-schema",
                "idempotencyKey": "level-test-listening-schema-idem",
                "sessionId": 100,
                "questionNumber": 11,
                "totalQuestions": 20,
                "domain": "LISTENING",
                "itemType": "LISTENING_GIST_CHOICE",
                "originLanguage": "ko",
                "learningLanguage": "ja",
                "targetComplexityBand": 3,
            }
        )

        schema = self._service(QueueProvider())._generation_schema(request)
        reference = schema["$defs"]["LevelTestReferencePayload"]["properties"]

        self.assertEqual({"type": "string"}, reference["sourceText"])
        self.assertEqual({"type": "string"}, reference["listeningQuestion"])

    def test_writing_translation_provider_schema_requires_source_text(self):
        request = LevelTestQuestionGenerationRequest.model_validate(
            {
                "requestId": "level-test-writing-translation-schema",
                "idempotencyKey": "level-test-writing-translation-schema-idem",
                "sessionId": 100,
                "questionNumber": 15,
                "totalQuestions": 20,
                "domain": "WRITING",
                "itemType": "WRITING_TRANSLATION",
                "originLanguage": "ko",
                "learningLanguage": "ja",
                "targetComplexityBand": 3,
            }
        )

        schema = self._service(QueueProvider())._generation_schema(request)
        reference = schema["$defs"]["LevelTestReferencePayload"]["properties"]

        self.assertEqual({"type": "string"}, reference["translationSourceText"])

    def test_guided_writing_provider_schema_requires_sufficient_guidance_lists(self):
        request = LevelTestQuestionGenerationRequest.model_validate(
            {
                "requestId": "level-test-guided-writing-schema",
                "idempotencyKey": "level-test-guided-writing-schema-idem",
                "sessionId": 100,
                "questionNumber": 17,
                "totalQuestions": 20,
                "domain": "WRITING",
                "itemType": "WRITING_SHORT_PARAGRAPH",
                "originLanguage": "ko",
                "learningLanguage": "ja",
                "targetComplexityBand": 4,
            }
        )

        schema = self._service(QueueProvider())._generation_schema(request)
        reference = schema["$defs"]["LevelTestReferencePayload"]["properties"]

        self.assertEqual(2, reference["providedFacts"]["minItems"])
        self.assertEqual(2, reference["requiredIntents"]["minItems"])
        self.assertEqual(1, reference["responseConstraints"]["minItems"])

    def test_request_specific_schema_requires_structured_reading_and_interpretation_fields(self):
        reading_request = LevelTestQuestionGenerationRequest.model_validate(
            {
                "requestId": "reading-schema",
                "idempotencyKey": "reading-schema-idem",
                "sessionId": 100,
                "questionNumber": 9,
                "totalQuestions": 20,
                "domain": "READING",
                "itemType": "READING_DISCOURSE_FUNCTION",
                "originLanguage": "ko",
                "learningLanguage": "ja",
                "targetComplexityBand": 2,
            }
        )
        reading_schema = LevelTestService._generation_schema(reading_request)
        reference = reading_schema["$defs"]["LevelTestReferencePayload"]["properties"]
        self.assertEqual({"type": "string"}, reference["readingPassage"])
        self.assertEqual({"type": "string"}, reference["readingQuestion"])
        self.assertEqual({"type": "string"}, reference["emphasisText"])

        interpretation_request = LevelTestQuestionGenerationRequest.model_validate(
            {
                "requestId": "interpretation-schema",
                "idempotencyKey": "interpretation-schema-idem",
                "sessionId": 100,
                "questionNumber": 14,
                "totalQuestions": 20,
                "domain": "LISTENING",
                "itemType": "LISTENING_INTERPRETATION",
                "originLanguage": "ko",
                "learningLanguage": "ja",
                "targetComplexityBand": 2,
            }
        )
        interpretation_schema = LevelTestService._generation_schema(interpretation_request)
        reference = interpretation_schema["$defs"]["LevelTestReferencePayload"]["properties"]
        self.assertEqual(2, reference["referenceMeanings"]["minItems"])
        self.assertEqual(3, reference["referenceMeanings"]["maxItems"])
        self.assertEqual(2, reference["keyMeaningUnits"]["minItems"])
        self.assertEqual(5, reference["keyMeaningUnits"]["maxItems"])
        candidate_properties = interpretation_schema["$defs"]["LevelTestQuestionCandidate"]["properties"]
        self.assertEqual(["TEXT"], candidate_properties["answerMode"]["enum"])
        self.assertEqual(["ko"], candidate_properties["answerLanguage"]["enum"])

    def test_reference_payload_shape_is_filled_without_fabricating_listening_source(self):
        payload = level_generation_payload()
        for candidate in payload["candidates"]:
            candidate["referencePayload"] = {}

        normalized, stats = LevelTestGenerationNormalizer.normalize(
            payload,
            origin_language="ko",
            learning_language="ja",
        )

        self.assertEqual(2, stats.reference_payload_shape_repairs)
        for candidate in normalized["candidates"]:
            self.assertEqual(
                {
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
                },
                candidate["referencePayload"],
            )

    def test_listening_nested_script_alias_is_canonicalized_to_source_text(self):
        payload = level_generation_payload()
        for candidate in payload["candidates"]:
            candidate.update(
                {
                    "domain": "LISTENING",
                    "itemType": "LISTENING_GIST_CHOICE",
                    "complexityBand": 3,
                    "instruction": "음성을 듣고 중심 내용을 고르세요.",
                    "promptText": "音声の中心内容として最も適切なものを選んでください。",
                    "referencePayload": {"script": "会議は午後三時から始まります。"},
                }
            )
        provider = QueueProvider(structured=[payload])
        request = LevelTestQuestionGenerationRequest.model_validate(
            {
                "requestId": "level-test-listening-script-alias",
                "idempotencyKey": "level-test-listening-script-alias-idem",
                "sessionId": 100,
                "questionNumber": 11,
                "totalQuestions": 20,
                "domain": "LISTENING",
                "itemType": "LISTENING_GIST_CHOICE",
                "originLanguage": "ko",
                "learningLanguage": "ja",
                "targetComplexityBand": 3,
            }
        )

        response = asyncio.run(self._service(provider).generate_question(request))

        self.assertEqual("会議は午後三時から始まります。", response.reference_payload["sourceText"])
        self.assertNotIn("script", response.reference_payload)

    def test_listening_top_level_audio_text_alias_is_moved_before_schema_validation(self):
        payload = level_generation_payload()
        for candidate in payload["candidates"]:
            candidate.update(
                {
                    "domain": "LISTENING",
                    "itemType": "LISTENING_DETAIL_CHOICE",
                    "complexityBand": 3,
                    "instruction": "음성을 듣고 질문에 답하세요.",
                    "promptText": "会議の開始時刻はいつですか。",
                    "referencePayload": {},
                    "audioText": "会議は午後三時から始まります。",
                }
            )
        provider = QueueProvider(structured=[payload])
        request = LevelTestQuestionGenerationRequest.model_validate(
            {
                "requestId": "level-test-listening-top-level-alias",
                "idempotencyKey": "level-test-listening-top-level-alias-idem",
                "sessionId": 100,
                "questionNumber": 12,
                "totalQuestions": 20,
                "domain": "LISTENING",
                "itemType": "LISTENING_DETAIL_CHOICE",
                "originLanguage": "ko",
                "learningLanguage": "ja",
                "targetComplexityBand": 3,
            }
        )

        response = asyncio.run(self._service(provider).generate_question(request))

        self.assertEqual("会議は午後三時から始まります。", response.reference_payload["sourceText"])

    def test_listening_missing_source_text_is_not_fabricated_from_prompt(self):
        payload = level_generation_payload()
        for candidate in payload["candidates"]:
            candidate.update(
                {
                    "domain": "LISTENING",
                    "itemType": "LISTENING_GIST_CHOICE",
                    "complexityBand": 3,
                    "instruction": "음성을 듣고 중심 내용을 고르세요.",
                    "promptText": "この文はsourceTextではありません。",
                    "referencePayload": {},
                }
            )
        provider = QueueProvider(structured=[copy.deepcopy(payload) for _ in range(3)])
        request = LevelTestQuestionGenerationRequest.model_validate(
            {
                "requestId": "level-test-listening-source-missing",
                "idempotencyKey": "level-test-listening-source-missing-idem",
                "sessionId": 100,
                "questionNumber": 11,
                "totalQuestions": 20,
                "domain": "LISTENING",
                "itemType": "LISTENING_GIST_CHOICE",
                "originLanguage": "ko",
                "learningLanguage": "ja",
                "targetComplexityBand": 3,
            }
        )

        with self.assertRaises(HTTPException) as raised:
            asyncio.run(self._service(provider).generate_question(request))

        self.assertEqual(422, raised.exception.status_code)
        self.assertTrue(
            any("referencePayload.sourceText" in reason for reason in raised.exception.detail["reasons"])
        )

    def test_listening_structured_question_overrides_accidental_prompt_script_copy(self):
        payload = level_generation_payload()
        source_text = "来週末のパーティーは田中さんの家で開き、料理は持ち寄りにする予定です。"
        question = "この音声メッセージの主な内容は何ですか。"
        for candidate in payload["candidates"]:
            candidate.update(
                {
                    "domain": "LISTENING",
                    "itemType": "LISTENING_GIST_CHOICE",
                    "complexityBand": 2,
                    "instruction": "음성을 듣고 주요 내용을 고르세요.",
                    "promptText": source_text,
                    "referencePayload": {
                        "sourceText": source_text,
                        "listeningQuestion": question,
                    },
                }
            )
        provider = QueueProvider(structured=[payload])
        request = LevelTestQuestionGenerationRequest.model_validate(
            {
                "requestId": "level-test-listening-structured-question",
                "idempotencyKey": "level-test-listening-structured-question-idem",
                "sessionId": 100,
                "questionNumber": 11,
                "totalQuestions": 20,
                "domain": "LISTENING",
                "itemType": "LISTENING_GIST_CHOICE",
                "originLanguage": "ko",
                "learningLanguage": "ja",
                "targetComplexityBand": 2,
            }
        )

        response = asyncio.run(self._service(provider).generate_question(request))

        self.assertEqual(question, response.prompt_text)
        self.assertEqual(question, response.reference_payload["listeningQuestion"])
        self.assertEqual(source_text, response.reference_payload["sourceText"])
        self.assertNotIn(source_text, response.prompt_text)

    def test_listening_script_copy_is_rejected_when_no_separate_question_exists(self):
        payload = level_generation_payload()
        source_text = "来週末のパーティーは田中さんの家で開き、料理は持ち寄りにする予定です。"
        for candidate in payload["candidates"]:
            candidate.update(
                {
                    "domain": "LISTENING",
                    "itemType": "LISTENING_GIST_CHOICE",
                    "complexityBand": 2,
                    "instruction": "음성을 듣고 주요 내용을 고르세요.",
                    "promptText": source_text,
                    "referencePayload": {"sourceText": source_text},
                }
            )
        provider = QueueProvider(structured=[copy.deepcopy(payload) for _ in range(3)])
        request = LevelTestQuestionGenerationRequest.model_validate(
            {
                "requestId": "level-test-listening-script-leak",
                "idempotencyKey": "level-test-listening-script-leak-idem",
                "sessionId": 100,
                "questionNumber": 11,
                "totalQuestions": 20,
                "domain": "LISTENING",
                "itemType": "LISTENING_GIST_CHOICE",
                "originLanguage": "ko",
                "learningLanguage": "ja",
                "targetComplexityBand": 2,
            }
        )

        with self.assertRaises(HTTPException) as raised:
            asyncio.run(self._service(provider).generate_question(request))

        self.assertEqual(422, raised.exception.status_code)
        self.assertTrue(
            any("sourceText가 노출" in reason for reason in raised.exception.detail["reasons"])
        )

    def test_listening_leak_guard_allows_short_keyword_reference(self):
        source_text = (
            "新製品の海外展開では、現地の流通チャネルを早期に確保することが最も重要です。"
            "法務確認や品質管理も必要ですが、販売網がなければ市場に届けられません。"
        )
        prompt_text = "この音声によると、海外展開で最も重要な課題は何ですか。"

        self.assertFalse(LevelTestService._has_listening_script_leak(prompt_text, source_text))

    def test_grammar_form_choice_rejects_duplicated_inflection_boundary(self):
        payload = level_generation_payload()
        for index, candidate in enumerate(payload["candidates"]):
            candidate.update(
                {
                    "domain": "GRAMMAR",
                    "itemType": "GRAMMAR_FORM_CHOICE",
                    "complexityBand": 3,
                    "instruction": "밑줄 친 부분에 들어갈 가장 적절한 형태를 고르세요.",
                    "promptText": "明日の会議の資料が_____たら、私に教えてください。",
                    "options": [
                        {"key": "A", "text": "できた"},
                        {"key": "B", "text": "する"},
                        {"key": "C", "text": "しない"},
                        {"key": "D", "text": "される"},
                    ],
                    "internalAnswerKey": {"correctOptionKey": "A", "correctOrder": []},
                }
            )
            candidate["diversityMetadata"] = diversity_metadata(
                "WORK" if index == 0 else "SCHEDULE",
                "REQUEST",
                f"GRAMMAR_FORM_DUP_{index}",
            )
        provider = QueueProvider(structured=[copy.deepcopy(payload) for _ in range(3)])
        request = LevelTestQuestionGenerationRequest.model_validate(
            {
                "requestId": "level-test-grammar-form-boundary",
                "idempotencyKey": "level-test-grammar-form-boundary-idem",
                "sessionId": 100,
                "questionNumber": 4,
                "totalQuestions": 20,
                "domain": "GRAMMAR",
                "itemType": "GRAMMAR_FORM_CHOICE",
                "originLanguage": "ko",
                "learningLanguage": "ja",
                "targetComplexityBand": 3,
            }
        )

        with self.assertRaises(HTTPException) as raised:
            asyncio.run(self._service(provider).generate_question(request))
        self.assertIn(raised.exception.status_code, {422, 502})

    def test_generation_metadata_and_answer_limits_are_normalized_before_schema_validation(self):
        payload = level_generation_payload()
        for index, candidate in enumerate(payload["candidates"]):
            candidate["diversityMetadata"]["taskArchetype"] = (
                "  " + ((f"long archetype {index} words ") * 10) + "  "
            )
            candidate["diversityMetadata"]["semanticSummary"] = (
                "  " + ((f"summary {index} ") * 100) + "  "
            )
            candidate["maxAudioSeconds"] = 90
            candidate["maxAnswerLength"] = 12000

        provider = QueueProvider(structured=[payload])
        response = asyncio.run(
            self._service(provider).generate_question(level_question_request())
        )

        self.assertLessEqual(len(response.diversity_metadata.task_archetype), 100)
        self.assertLessEqual(len(response.diversity_metadata.semantic_summary), 500)
        self.assertIsNone(response.max_audio_seconds)
        self.assertIsNone(response.max_answer_length)

    def test_listening_interpretation_missing_answer_language_is_repaired_from_request(self):
        payload = level_generation_payload()
        for candidate in payload["candidates"]:
            candidate.update(
                {
                    "domain": "LISTENING",
                    "itemType": "LISTENING_INTERPRETATION",
                    "complexityBand": 2,
                    "instruction": "音声を聞き、内容の意味を説明してください。",
                    "answerMode": "TEXT",
                    "answerLanguage": None,
                    "promptText": "音声を聞き、内容の意味を韓国語で説明してください。",
                    "options": [],
                    "internalAnswerKey": {"correctOptionKey": None, "correctOrder": []},
                    "referencePayload": {
                        "sourceText": "明日の会議は午後三時からです。",
                        "referenceMeanings": [
                            "내일 회의는 오후 3시부터입니다.",
                            "내일 회의 시작 시간은 오후 세 시입니다.",
                        ],
                        "keyMeaningUnits": ["내일", "회의", "오후 3시"],
                    },
                }
            )
        provider = QueueProvider(structured=[payload])
        request = LevelTestQuestionGenerationRequest.model_validate(
            {
                "requestId": "level-test-listening-interpretation",
                "idempotencyKey": "level-test-listening-interpretation-idem",
                "sessionId": 100,
                "questionNumber": 14,
                "totalQuestions": 20,
                "domain": "LISTENING",
                "itemType": "LISTENING_INTERPRETATION",
                "originLanguage": "ko",
                "learningLanguage": "ja",
                "targetComplexityBand": 2,
            }
        )

        response = asyncio.run(self._service(provider).generate_question(request))

        self.assertEqual("ko", response.answer_language)

    def test_listening_dictation_wrong_answer_language_is_repaired_to_learning_language(self):
        payload = level_generation_payload()
        for candidate in payload["candidates"]:
            candidate.update(
                {
                    "domain": "LISTENING",
                    "itemType": "LISTENING_DICTATION",
                    "complexityBand": 3,
                    "instruction": "音声を聞き、聞こえた内容を正確に書き取ってください。",
                    "answerMode": "TEXT",
                    "answerLanguage": "ko",
                    "promptText": "音声を聞いて、日本語で正確に書き取ってください。",
                    "options": [],
                    "internalAnswerKey": {"correctOptionKey": None, "correctOrder": []},
                    "referencePayload": {"sourceText": "駅まで歩いて十分ぐらいです。"},
                }
            )
        provider = QueueProvider(structured=[payload])
        request = LevelTestQuestionGenerationRequest.model_validate(
            {
                "requestId": "level-test-listening-dictation",
                "idempotencyKey": "level-test-listening-dictation-idem",
                "sessionId": 100,
                "questionNumber": 13,
                "totalQuestions": 20,
                "domain": "LISTENING",
                "itemType": "LISTENING_DICTATION",
                "originLanguage": "ko",
                "learningLanguage": "ja",
                "targetComplexityBand": 3,
            }
        )

        response = asyncio.run(self._service(provider).generate_question(request))

        self.assertEqual("ja", response.answer_language)

    def test_audio_answer_max_audio_seconds_is_clamped_to_contract(self):
        payload = level_generation_payload()
        for candidate in payload["candidates"]:
            candidate.update(
                {
                    "domain": "SPEAKING",
                    "itemType": "SPEAKING_GUIDED_RESPONSE",
                    "complexityBand": 3,
                    "instruction": "일본어로 짧게 답하세요.",
                    "answerMode": "AUDIO",
                    "answerLanguage": "ja",
                    "promptText": "最近楽しんだことを日本語で説明してください。",
                    "options": [],
                    "internalAnswerKey": {"correctOptionKey": None, "correctOrder": []},
                    "referencePayload": {
                        "providedFacts": ["最近楽しんだことを一つ述べる"],
                        "requiredIntents": ["最近の経験を説明する"],
                        "responseConstraints": ["短く説明する"],
                    },
                    "maxAudioSeconds": 90,
                }
            )
        provider = QueueProvider(structured=[payload])
        request = LevelTestQuestionGenerationRequest.model_validate(
            {
                "requestId": "level-test-q20",
                "idempotencyKey": "level-test-q20-idem",
                "sessionId": 100,
                "questionNumber": 20,
                "totalQuestions": 20,
                "domain": "SPEAKING",
                "itemType": "SPEAKING_GUIDED_RESPONSE",
                "originLanguage": "ko",
                "learningLanguage": "ja",
                "targetComplexityBand": 3,
            }
        )

        response = asyncio.run(self._service(provider).generate_question(request))

        self.assertEqual(60, response.max_audio_seconds)

    def test_listening_level_test_uploads_reference_audio_without_leaking_presigned_url_to_llm(self):
        provider = QueueProvider(
            structured=[
                {
                    "candidates": [
                        {
                            "domain": "LISTENING",
                            "itemType": "LISTENING_GIST_CHOICE",
                            "complexityBand": 3,
                            "instruction": "음성을 듣고 가장 알맞은 답을 고르세요.",
                            "instructionLanguage": "ko",
                            "answerMode": "CHOICE",
                            "answerLanguage": None,
                            "promptText": "音声の中心内容を選んでください。",
                            "options": [
                                {"key": "A", "text": "会議時間の案内"},
                                {"key": "B", "text": "天気の案内"},
                                {"key": "C", "text": "旅行の計画"},
                                {"key": "D", "text": "食事の注文"},
                            ],
                            "internalAnswerKey": {"correctOptionKey": "A", "correctOrder": []},
                            "referencePayload": {"sourceText": "会議は午後三時から始まります。"},
                            "diversityMetadata": diversity_metadata("WORK", "REPORT", "LISTENING_GIST"),
                        }
                    ]
                }
            ]
        )
        uploader = FakePresignedAudioUploader()
        request = LevelTestQuestionGenerationRequest.model_validate(
            {
                "requestId": "level-listen-1",
                "idempotencyKey": "level-listen-1-idem",
                "sessionId": 100,
                "questionNumber": 11,
                "totalQuestions": 20,
                "domain": "LISTENING",
                "itemType": "LISTENING_GIST_CHOICE",
                "originLanguage": "ko",
                "learningLanguage": "ja",
                "targetComplexityBand": 3,
                "referenceAudioUpload": {
                    "uploadUrl": "https://r2.example.invalid/presigned?X-Amz-Signature=secret",
                    "objectKey": "language-learning/level-test/reference/ja/test.wav",
                    "contentType": "audio/wav",
                    "voice": "Kore",
                    "playbackSpeed": "NORMAL",
                },
            }
        )

        response = asyncio.run(
            self._service(provider, audio_uploader=uploader).generate_question(request)
        )

        self.assertEqual(
            "language-learning/level-test/reference/ja/test.wav",
            response.reference_audio.object_key,
        )
        self.assertEqual(1250, response.reference_audio.duration_ms)
        self.assertEqual(1, len(uploader.uploads))
        self.assertEqual(
            ("会議は午後三時から始まります。", "Kore", "ja", "NORMAL"),
            provider.tts_calls[0],
        )
        llm_prompt = provider.calls[0][1]
        self.assertNotIn("X-Amz-Signature", llm_prompt)
        self.assertNotIn("referenceAudioUpload", llm_prompt)

    def test_task_sufficiency_verifier_rejects_problem_solving_and_hides_internal_answer(self):
        raw = {
            "domain": "WRITING",
            "itemType": "WRITING_SHORT_PARAGRAPH",
            "complexityBand": 3,
            "instruction": "주어진 조건을 포함해 일본어로 작성하세요.",
            "instructionLanguage": "ko",
            "answerMode": "TEXT",
            "answerLanguage": "ja",
            "promptText": "現在、同じ顧客情報を三つのファイルに入力しています。入力先を一つに統合し、自動集計を使う提案メールを書いてください。",
            "options": [],
            "internalAnswerKey": {"correctOptionKey": None, "correctOrder": []},
            "referencePayload": {
                "providedFacts": ["同じ顧客情報を三つのファイルに入力している", "入力先を一つに統合する"],
                "requiredIntents": ["EXPLAIN_PROBLEM", "PROPOSE_SOLUTION"],
                "responseConstraints": ["期待される効果も一つ述べる"],
            },
            "diversityMetadata": diversity_metadata("WORK", "SUGGEST", "GUIDED_PROPOSAL"),
        }
        normalized_task, _ = LevelTestGenerationNormalizer.normalize(
            {"candidates": [raw]}, origin_language="ko", learning_language="ja"
        )
        candidate = LevelTestQuestionGenerationPayload.model_validate(normalized_task).candidates[0]
        provider = QueueProvider(
            structured=[
                {
                    "sufficient": False,
                    "requiresExternalKnowledge": False,
                    "requiresProblemSolving": True,
                    "providedFactsSufficient": True,
                    "communicativeGoalsClear": True,
                    "instructionAndTaskRolesSeparated": True,
                    "missingInformation": [],
                }
            ]
        )
        request = LevelTestQuestionGenerationRequest.model_validate(
            {
                "requestId": "guided-check",
                "idempotencyKey": "guided-check-idem",
                "sessionId": 100,
                "questionNumber": 17,
                "totalQuestions": 20,
                "domain": "WRITING",
                "itemType": "WRITING_SHORT_PARAGRAPH",
                "originLanguage": "ko",
                "learningLanguage": "ja",
                "targetComplexityBand": 3,
            }
        )

        reason, _ = asyncio.run(
            self._service(provider, verify_task_sufficiency=True)._verify_task_sufficiency_candidate(
                request, candidate
            )
        )
        self.assertIn("상황 판단/문제 해결 요구", reason)
        verifier_prompt = provider.calls[0][1]
        self.assertNotIn("correctOptionKey", verifier_prompt)
        self.assertNotIn("internalAnswerKey", verifier_prompt)

    def test_reading_discourse_bold_markup_is_normalized_to_structured_emphasis(self):
        payload = {
            "candidates": [
                {
                    "domain": "READING",
                    "itemType": "READING_DISCOURSE_FUNCTION",
                    "complexityBand": 3,
                    "instruction": "강조된 문장의 역할을 고르세요.",
                    "instructionLanguage": "ko",
                    "answerMode": "CHOICE",
                    "answerLanguage": None,
                    "promptText": "売上が前年同期比で減少し、競合製品の台頭も確認された。**これは、新たな戦略が必要であることを示している。**今後は顧客層を再定義し、施策を見直す必要がある。上記の強調部分は文章全体でどのような役割を果たしているか。",
                    "options": [
                        {"key": "A", "text": "原因分析"},
                        {"key": "B", "text": "前文を受けた評価"},
                        {"key": "C", "text": "例示"},
                        {"key": "D", "text": "反論"},
                    ],
                    "internalAnswerKey": {"correctOptionKey": "B", "correctOrder": []},
                    "referencePayload": {},
                    "diversityMetadata": diversity_metadata("WORK", "SUMMARIZE", "DISCOURSE_FUNCTION"),
                }
            ]
        }
        normalized, _ = LevelTestGenerationNormalizer.normalize(
            payload, origin_language="ko", learning_language="ja"
        )
        candidate = LevelTestQuestionGenerationPayload.model_validate(normalized).candidates[0]
        self.assertNotIn("**", candidate.prompt_text)
        self.assertEqual(
            "これは、新たな戦略が必要であることを示している。",
            candidate.reference_payload["emphasisText"],
        )
        self.assertEqual(
            "売上が前年同期比で減少し、競合製品の台頭も確認された。これは、新たな戦略が必要であることを示している。今後は顧客層を再定義し、施策を見直す必要がある。",
            candidate.reference_payload["readingPassage"],
        )
        self.assertEqual(
            "上記の強調部分は文章全体でどのような役割を果たしているか。",
            candidate.reference_payload["readingQuestion"],
        )
        self.assertEqual(
            f"{candidate.reference_payload['readingPassage']}\n\n{candidate.reference_payload['readingQuestion']}",
            candidate.prompt_text,
        )
        LevelTestService._validate_question_candidate(
            LevelTestQuestionGenerationRequest.model_validate(
                {
                    "requestId": "reading-emphasis",
                    "idempotencyKey": "reading-emphasis-idem",
                    "sessionId": 100,
                    "questionNumber": 10,
                    "totalQuestions": 20,
                    "domain": "READING",
                    "itemType": "READING_DISCOURSE_FUNCTION",
                    "originLanguage": "ko",
                    "learningLanguage": "ja",
                    "targetComplexityBand": 3,
                }
            ),
            candidate,
        )

    def test_level_test_recipe_has_exact_current_domain_distribution(self):
        self.assertEqual(20, len(LEVEL_TEST_RECIPE))
        self.assertEqual(LevelTestItemType.WRITING_TRANSLATION, LEVEL_TEST_RECIPE[15][1])
        self.assertEqual(LevelTestItemType.WRITING_TRANSLATION, LEVEL_TEST_RECIPE[16][1])
        self.assertEqual(LevelTestItemType.WRITING_SHORT_PARAGRAPH, LEVEL_TEST_RECIPE[17][1])
        counts = Counter(domain.value for domain, _ in LEVEL_TEST_RECIPE.values())
        self.assertEqual(
            {
                "VOCABULARY": 3,
                "GRAMMAR": 3,
                "READING": 4,
                "LISTENING": 4,
                "WRITING": 3,
                "SPEAKING": 3,
            },
            dict(counts),
        )

    def test_writing_level_test_evaluation_reuses_writing_evaluator(self):
        writing_service = FakeWritingEvaluationService()
        request = LevelTestTextEvaluationRequest.model_validate(
            {
                "requestId": "eval-1",
                "idempotencyKey": "eval-idem",
                "sessionId": 10,
                "itemId": 17,
                "domain": "WRITING",
                "itemType": "WRITING_SHORT_PARAGRAPH",
                "promptText": "会議が午後三時から四時に変更されました。参加者へ変更を伝え、都合が悪い場合は知らせるよう依頼してください。",
                "answer": "今日は仕事をします。夜は家で休みます。",
                "originLanguage": "ko",
                "learningLanguage": "ja",
                "complexityBand": 2,
                "providedFacts": ["회의가 오후 3시에서 4시로 변경됨", "참석자에게 변경을 알려야 함"],
                "requiredIntents": ["일정 변경 알림", "참석이 어려우면 회신 요청"],
                "responseConstraints": ["정중한 일본어 사용"],
            }
        )
        response = asyncio.run(
            self._service(QueueProvider(), writing_service=writing_service).evaluate_text(request)
        )
        self.assertTrue(response.evaluable)
        self.assertEqual(82, response.score)
        self.assertEqual(5, len(response.metrics))
        self.assertEqual("level-test-evaluation", response.evaluation_version)
        forwarded = writing_service.requests[0]
        self.assertEqual("WRITING_SHORT_PARAGRAPH", forwarded.task_type)
        self.assertEqual(request.provided_facts, forwarded.provided_facts)
        self.assertEqual(request.required_intents, forwarded.required_intents)
        self.assertEqual(request.response_constraints, forwarded.response_constraints)

    def test_translation_level_test_evaluation_forwards_source_contract(self):
        writing_service = FakeWritingEvaluationService()
        request = LevelTestTextEvaluationRequest.model_validate(
            {
                "requestId": "eval-translation",
                "idempotencyKey": "eval-translation-idem",
                "sessionId": 10,
                "itemId": 15,
                "domain": "WRITING",
                "itemType": "WRITING_TRANSLATION",
                "promptText": "회의 시간이 변경되었으니 참석이 어려운 경우 오늘 오후 5시까지 알려주세요.",
                "answer": "会議時間が変更されましたので、参加が難しい場合は本日午後5時までにお知らせください。",
                "originLanguage": "ko",
                "learningLanguage": "ja",
                "complexityBand": 3,
                "translationSourceText": "회의 시간이 변경되었으니 참석이 어려운 경우 오늘 오후 5시까지 알려주세요.",
            }
        )
        asyncio.run(
            self._service(QueueProvider(), writing_service=writing_service).evaluate_text(request)
        )
        forwarded = writing_service.requests[0]
        self.assertEqual("WRITING_TRANSLATION", forwarded.task_type)
        self.assertEqual(request.translation_source_text, forwarded.translation_source_text)

    def test_speaking_level_test_uses_stt_and_five_metric_contract(self):
        provider = QueueProvider(
            structured=[
                {
                    "evaluationConfidence": 0.9,
                    "taskResponseStatus": "FULFILLED",
                    "metrics": [
                        {"type": "PRONUNCIATION", "state": "EVALUATED", "score": 80, "confidence": 0.9, "summary": "명료합니다.", "evidence": ["STT/acoustic"]},
                        {"type": "FLUENCY", "state": "EVALUATED", "score": 75, "confidence": 0.9, "summary": "대체로 유창합니다.", "evidence": ["3 seconds"]},
                        {"type": "GRAMMAR", "state": "EVALUATED", "score": 85, "confidence": 0.9, "summary": "문법이 안정적입니다.", "evidence": ["transcript"]},
                        {"type": "VOCABULARY", "state": "EVALUATED", "score": 80, "confidence": 0.9, "summary": "적절합니다.", "evidence": ["transcript"]},
                        {"type": "TASK_FULFILLMENT", "state": "EVALUATED", "score": 90, "confidence": 0.9, "summary": "과업을 수행했습니다.", "evidence": ["prompt/transcript"]},
                    ],
                    "strengths": ["의미가 명확합니다."],
                    "improvements": ["조금 더 자연스럽게 연결해 보세요."],
                    "recommendedAnswers": ["はじめまして。東京で会社員として働いています。よろしくお願いします。"],
                }
            ]
        )
        request = LevelTestSpeakingEvaluationContext.model_validate(
            {
                "requestId": "sp-1",
                "idempotencyKey": "sp-idem",
                "sessionId": 10,
                "itemId": 19,
                "itemType": "SPEAKING_GUIDED_RESPONSE",
                "promptText": "일본어로 간단히 자기소개하세요.",
                "originLanguage": "ko",
                "learningLanguage": "ja",
                "complexityBand": 2,
                "maxDurationSeconds": 30,
                "providedFacts": ["이름과 직업을 말한다"],
                "requiredIntents": ["자기소개"],
                "responseConstraints": ["정중한 일본어"],
            }
        )
        response = asyncio.run(
            self._service(provider).evaluate_speaking(
                request,
                audio_bytes=b"fake",
                file_name="answer.wav",
                content_type="audio/wav",
            )
        )
        self.assertTrue(response.evaluable)
        self.assertEqual(5, len(response.metrics))
        self.assertEqual("自己紹介をします。東京で働いています。", response.transcript)
        self.assertEqual("level-test-speaking-evaluation", response.evaluation_version)
        prompt = provider.calls[-1][1]
        self.assertIn('"providedFacts":["이름과 직업을 말한다"]', prompt)
        self.assertIn('"requiredIntents":["자기소개"]', prompt)
        self.assertIn('"responseConstraints":["정중한 일본어"]', prompt)

    def test_speaking_weighted_score(self):
        metrics = [
            LevelTestMetricResult(type="PRONUNCIATION", score=80, confidence=1),
            LevelTestMetricResult(type="FLUENCY", score=80, confidence=1),
            LevelTestMetricResult(type="GRAMMAR", score=80, confidence=1),
            LevelTestMetricResult(type="VOCABULARY", score=80, confidence=1),
            LevelTestMetricResult(type="TASK_FULFILLMENT", score=100, confidence=1),
        ]
        score = LevelTestService._calculate_speaking_item_score(
            item_type=level_question_request().item_type.SPEAKING_GUIDED_RESPONSE,
            metrics=metrics,
        )
        self.assertEqual(85, score)

    def test_speaking_meta_refusal_is_evaluable_but_capped_at_ten(self):
        metrics = [
            LevelTestMetricResult(type="PRONUNCIATION", score=55, confidence=1),
            LevelTestMetricResult(type="FLUENCY", score=55, confidence=1),
            LevelTestMetricResult(type="GRAMMAR", score=55, confidence=1),
            LevelTestMetricResult(type="VOCABULARY", score=55, confidence=1),
            LevelTestMetricResult(type="TASK_FULFILLMENT", score=0, confidence=1),
        ]
        raw_score = LevelTestService._calculate_speaking_item_score(
            item_type=LevelTestItemType.SPEAKING_GUIDED_RESPONSE,
            metrics=metrics,
        )
        score = LevelTestService._apply_speaking_task_response_score_cap(
            raw_score,
            LevelTestSpeakingTaskResponseStatus.META_REFUSAL,
        )

        self.assertEqual(41, raw_score)
        self.assertEqual(10, score)

    def test_speaking_meta_refusal_response_remains_evaluable_with_low_score(self):
        provider = QueueProvider(
            structured=[
                {
                    "evaluationConfidence": 0.95,
                    "taskResponseStatus": "META_REFUSAL",
                    "metrics": [
                        {"type": "PRONUNCIATION", "state": "EVALUATED", "score": 55, "confidence": 0.9, "summary": "일부 발화는 식별됩니다.", "evidence": ["acoustic/STT"]},
                        {"type": "FLUENCY", "state": "EVALUATED", "score": 55, "confidence": 0.9, "summary": "짧은 문장은 이어 말했습니다.", "evidence": ["transcript"]},
                        {"type": "GRAMMAR", "state": "EVALUATED", "score": 55, "confidence": 0.9, "summary": "부자연스러운 표현이 있습니다.", "evidence": ["お疲れして"]},
                        {"type": "VOCABULARY", "state": "EVALUATED", "score": 55, "confidence": 0.9, "summary": "제한적인 어휘를 사용했습니다.", "evidence": ["transcript"]},
                        {"type": "TASK_FULFILLMENT", "state": "EVALUATED", "score": 0, "confidence": 1.0, "summary": "과제 수행을 거부했습니다.", "evidence": ["답변할 수 없다고 말함"]},
                    ],
                    "strengths": [],
                    "improvements": ["주어진 세 가지 내용을 포함해 답하세요."],
                    "recommendedAnswers": ["写真を共有するのが趣味です。最近、知人以外には見せたくないため公開範囲を変更しました。今後は親しい友人だけに共有します。"],
                }
            ]
        )
        request = LevelTestSpeakingEvaluationContext.model_validate(
            {
                "requestId": "sp-meta-refusal",
                "idempotencyKey": "sp-meta-refusal-idem",
                "sessionId": 10,
                "itemId": 20,
                "itemType": "SPEAKING_GUIDED_RESPONSE",
                "promptText": "写真共有の趣味、公開範囲を変えた理由、今後の使い方を説明してください。",
                "originLanguage": "ko",
                "learningLanguage": "ja",
                "complexityBand": 3,
                "maxDurationSeconds": 30,
                "providedFacts": ["사진 공유가 취미", "지인 외에는 보여주고 싶지 않음", "친한 친구에게만 공유"],
                "requiredIntents": ["취미 설명", "변경 이유 설명", "향후 방침 전달"],
                "responseConstraints": ["2~4문장", "정중하고 자연스러운 표현"],
            }
        )
        transcript = "この問題は、お疲れして答えられません。"

        response = asyncio.run(
            self._service(
                provider,
                stt_service=FakeSttService(transcript),
            ).evaluate_speaking(
                request,
                audio_bytes=b"fake",
                file_name="answer.wav",
                content_type="audio/wav",
            )
        )

        self.assertTrue(response.evaluable)
        self.assertEqual(10, response.score)
        self.assertEqual(transcript, response.transcript)
        self.assertIsNone(response.reason_code)
        self.assertEqual("level-test-speaking-evaluation", response.evaluation_version)


if __name__ == "__main__":
    unittest.main()


class LevelTestRouteContractTest(unittest.TestCase):
    def test_level_test_routes_use_integrated_path(self):
        import importlib.util
        import sys
        import types
        from pathlib import Path

        dependency_name = "app.api.dependencies"
        original_dependencies = sys.modules.get(dependency_name)
        fake_dependencies = types.ModuleType(dependency_name)
        fake_dependencies.get_language_learning_level_test_service = lambda: None
        sys.modules[dependency_name] = fake_dependencies
        try:
            module_path = (
                Path(__file__).parents[1]
                / "app"
                / "api"
                / "v1"
                / "language_learning_level_test.py"
            )
            spec = importlib.util.spec_from_file_location(
                "level_test_route_contract_module",
                module_path,
            )
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            if original_dependencies is None:
                sys.modules.pop(dependency_name, None)
            else:
                sys.modules[dependency_name] = original_dependencies

        paths = {route.path for route in module.router.routes}
        self.assertIn(
            "/language-learning/level-test/questions/generate",
            paths,
        )
        self.assertIn(
            "/language-learning/level-test/evaluate/text",
            paths,
        )
        self.assertIn(
            "/language-learning/level-test/evaluate/speaking",
            paths,
        )


class LevelTestDiversityContextContractTest(unittest.TestCase):
    def test_exact_content_hashes_90d_accepts_be_camel_case_contract(self):
        context = DiversityContext.model_validate(
            {
                "currentSession": [],
                "sameFeatureRecent": [],
                "crossFeatureRecent": [],
                "exactContentHashes90d": ["hash-a", "hash-b"],
            }
        )

        self.assertEqual(["hash-a", "hash-b"], context.exact_content_hashes_90d)
        self.assertEqual(
            ["hash-a", "hash-b"],
            context.model_dump(by_alias=True)["exactContentHashes90d"],
        )
        self.assertNotIn("exactContentHashes90D", context.model_dump(by_alias=True))

class LevelTestScenarioBalanceContractTest(unittest.TestCase):
    def test_generation_schema_limits_scenario_category_to_server_preferences(self):
        request = level_question_request().model_copy(
            update={"preferred_scenario_categories": [
                ScenarioCategory.TRAVEL,
                ScenarioCategory.HOBBY,
            ]}
        )
        request = LevelTestQuestionGenerationRequest.model_validate(
            request.model_dump(mode="json", by_alias=True)
        )

        schema = LevelTestService._generation_schema(request)
        scenario_schema = schema["$defs"]["DiversityMetadata"]["properties"]["scenarioCategory"]

        self.assertEqual(scenario_schema["enum"], ["TRAVEL", "HOBBY"])

    def test_candidate_outside_server_preferred_scenarios_is_rejected(self):
        request = LevelTestQuestionGenerationRequest.model_validate(
            {
                **level_question_request().model_dump(mode="json", by_alias=True),
                "preferredScenarioCategories": ["HOBBY", "FOOD"],
            }
        )
        normalized, _ = LevelTestGenerationNormalizer.normalize(
            level_generation_payload(),
            origin_language="ko",
            learning_language="ja",
        )
        candidate = LevelTestQuestionCandidate.model_validate(
            normalized["candidates"][0]
        )

        with self.assertRaisesRegex(ValueError, "세션 균형 계획"):
            LevelTestService._validate_question_candidate(request, candidate)

    def test_prompt_contains_balancing_constraint_and_rejects_business_default(self):
        request = LevelTestQuestionGenerationRequest.model_validate(
            {
                **level_question_request().model_dump(mode="json", by_alias=True),
                "preferredScenarioCategories": ["SOCIAL", "DIGITAL_LIFE"],
            }
        )
        from app.features.language_learning.level_test.prompts import (
            LEVEL_TEST_GENERATION_SYSTEM_PROMPT,
            build_level_test_generation_prompt,
        )

        prompt = build_level_test_generation_prompt(request)

        self.assertIn('"preferredScenarioCategories":["SOCIAL","DIGITAL_LIFE"]', prompt)
        self.assertIn("Never infer that a harder/advanced item should default to business", LEVEL_TEST_GENERATION_SYSTEM_PROMPT)
