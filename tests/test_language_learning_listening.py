import asyncio
import hashlib
import io
import tempfile
import unittest
import wave

from app.ai.ports import SpeechSynthesisResult, StructuredGenerationResult
from app.features.language_learning.listening.audio_store import (
    TemporaryListeningAudioStore,
)
from app.features.language_learning.listening.benchmark import (
    ListeningEvaluationBenchmarkService,
)
from app.features.language_learning.listening.dictation_service import (
    ListeningDictationService,
)
from app.features.language_learning.listening.comprehension_service import (
    ListeningComprehensionService,
)
from app.core.config import settings
from app.features.language_learning.listening.summary_service import (
    ListeningSummaryService,
)
from app.features.language_learning.listening.errors import ListeningStageException
from app.features.language_learning.listening.explanation_service import (
    ListeningExplanationService,
)
from app.features.language_learning.listening.generation_service import (
    ListeningGenerationService,
)
from app.features.language_learning.listening.interpretation_service import (
    ListeningInterpretationService,
)
from app.features.language_learning.listening.normalization import normalize_text
from app.features.language_learning.listening.policy import (
    DICTATION_WEIGHTS,
    INTERPRETATION_WEIGHTS,
    PROFILE_SIGNAL_WEIGHTS,
    REPEAT_WEIGHTS,
    resolve_assistance_level,
)
from app.features.language_learning.listening.repeat_service import (
    ListeningRepeatService,
)
from app.features.language_learning.listening.tts_service import ListeningTtsService
from app.features.language_learning.speaking.audio_processor import NormalizedAudio
from app.features.language_learning.speaking.errors import SpeakingStageException
from app.features.language_learning.speaking.stt_service import (
    SttProviderResult,
    SttProviderSegment,
)
from app.schemas.language_learning_listening import (
    AssistanceUsage,
    DictationEvaluationRequest,
    ComprehensionEvaluationRequest,
    InterpretationEvaluationRequest,
    SummaryEvaluationRequest,
    ListeningAssistanceLevel,
    ListeningAssistanceType,
    ListeningBenchmarkSample,
    ListeningErrorCode,
    ListeningSetGenerationRequest,
    ListeningTaskStatus,
    ListeningTaskType,
    ListeningTtsRequest,
    ProfileMetric,
    RecommendationExplanationRequest,
    RepeatEvaluationContext,
)
from app.schemas.language_learning_speaking import (
    AudioQualitySignals,
    SpeakingErrorCode,
    SpeakingStage,
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
            raise RuntimeError("no fake result")
        return StructuredGenerationResult(
            data=self.results.pop(0),
            input_tokens=13,
            output_tokens=21,
            provider="fake",
            model="fake-model",
        )


class Provider503Error(RuntimeError):
    status_code = 503


class Provider429Error(RuntimeError):
    status_code = 429


class FakeSpeechProvider:
    def __init__(self, errors=None) -> None:
        self.errors = list(errors or [])
        self.calls = 0
        self.kwargs = []

    async def synthesize_speech(self, **kwargs):
        self.calls += 1
        self.kwargs.append(kwargs)
        if self.errors:
            error = self.errors.pop(0)
            if error is not None:
                raise error
        return SpeechSynthesisResult(
            audio_bytes=b"reference-audio",
            content_type="audio/wav",
            provider="fake-tts",
            model="fake-tts-model",
            duration_seconds=9.6,
        )


class FakeSttProvider:
    def __init__(self, result=None, errors=None) -> None:
        self.result = result or stt_result()
        self.errors = list(errors or [])
        self.calls = 0

    async def transcribe(self, wav_bytes, *, language, phrase_hints=None):
        self.calls += 1
        if self.errors:
            error = self.errors.pop(0)
            if error is not None:
                raise error
        return self.result


class FakeAudioProcessor:
    def __init__(
        self,
        quality=None,
        error=None,
        duration=2.0,
        wav_bytes=b"normalized-wav",
    ) -> None:
        self.quality = quality or quality_signals()
        self.error = error
        self.duration = duration
        self.wav_bytes = wav_bytes
        self.max_seconds = None

    def validate_and_normalize(
        self,
        audio_bytes,
        *,
        file_name,
        content_type,
        min_seconds,
        max_seconds,
        max_bytes,
    ):
        self.max_seconds = max_seconds
        if self.error:
            raise self.error
        return NormalizedAudio(
            wav_bytes=self.wav_bytes,
            duration_seconds=self.duration,
            source_format="wav",
            quality=self.quality,
        )


def quality_signals(rms=0.10, peak=0.50, silence_ratio=0.05):
    return AudioQualitySignals(
        rms=rms,
        peak=peak,
        silence_ratio=silence_ratio,
        sample_rate=16000,
        channels=1,
    )


def stt_result(
    text="hello world",
    *,
    language="en",
    language_probability=0.98,
    logprob=-0.05,
):
    return SttProviderResult(
        text=text,
        language=language,
        language_probability=language_probability,
        segments=[SttProviderSegment(0.0, 2.0, text, logprob)],
        provider="fake-stt",
        model="fake-stt-model",
        model_version="fake-stt",
    )


def clipped_noise_wav(seconds=1.0):
    sample_rate = 16000
    frames = bytearray()
    for index in range(int(sample_rate * seconds)):
        value = 32767 if index % 2 == 0 else -32768
        frames.extend(value.to_bytes(2, byteorder="little", signed=True))
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(frames)
    return buffer.getvalue()


def generation_request(**overrides):
    data = {
        "requestId": "generation-1",
        "idempotencyKey": "generation-idem-1",
        "userContext": {
            "originLanguage": "ko",
            "learningLanguage": "ja",
            "level": "MY_LEVEL",
            "profileFocus": ["LISTENING_RECOGNITION", "VOCABULARY"],
        },
        "setContext": {
            "learningDate": "2026-08-23",
            "topic": {"id": 10, "title": "여행"},
            "selectedKeywords": [
                {
                    "key": "travel",
                    "text": "旅行",
                    "source": "SYSTEM",
                    "type": "TOPIC",
                },
                {
                    "key": "reservation",
                    "text": "予約",
                    "source": "CUSTOM",
                    "type": "VOCABULARY",
                },
            ],
            "itemCount": 2,
            "difficulty": "MY_LEVEL",
        },
        "constraints": {
            "audioSecondsMin": 8,
            "audioSecondsMax": 20,
            "recentContentHashes": [],
            "recentSimilaritySummaries": [],
        },
        "policyVersion": "listening",
        "modelConfigVersion": "model",
        "languageComplexity": {"baseComplexityBand": 3},
        "contentDiversityPolicyVersion": "language-learning-diversity",
    }
    data.update(overrides)
    return ListeningSetGenerationRequest.model_validate(data)


def generation_payload(source_suffix=""):
    return {
        "items": [
            {
                "itemIndex": 1,
                "sourceText": f"京都で静かな寺を見学したいです。{source_suffix}",
                "referenceMeanings": [
                    "교토에서 조용한 절을 구경하고 싶습니다.",
                    "교토의 한적한 사찰을 둘러보고 싶어요.",
                ],
                "keyMeaningUnits": ["교토에서", "조용한 절", "구경하고 싶다"],
                "targetKeywords": ["京都", "寺"],
                "estimatedAudioSeconds": 9.6,
                "safety": {"passed": True, "categories": []},
                "languageComplexityBand": 3,
                "diversityMetadata": {
                    "scenarioCategory": "TRAVEL",
                    "communicativeIntent": "DESCRIBE",
                    "taskArchetype": "TRAVEL_PLAN",
                    "grammarFocusCodes": ["DESIRE"],
                    "lexicalFocusCodes": ["TRAVEL"],
                    "semanticSummary": "교토의 조용한 절을 방문하고 싶다는 여행 계획",
                    "requiresBackgroundKnowledge": False,
                },
            },
            {
                "itemIndex": 2,
                "sourceText": "旅行の前にホテルを予約しておきました。",
                "referenceMeanings": [
                    "여행 전에 호텔을 예약해 두었습니다.",
                    "여행을 떠나기 전 호텔 예약을 마쳤어요.",
                ],
                "keyMeaningUnits": ["여행 전", "호텔", "예약 완료"],
                "targetKeywords": ["旅行", "予約"],
                "estimatedAudioSeconds": 8.5,
                "safety": {"passed": True, "categories": []},
                "languageComplexityBand": 3,
                "diversityMetadata": {
                    "scenarioCategory": "TRAVEL",
                    "communicativeIntent": "REPORT",
                    "taskArchetype": "RESERVATION_STATUS",
                    "grammarFocusCodes": ["PREPARATION"],
                    "lexicalFocusCodes": ["RESERVATION"],
                    "semanticSummary": "여행 전에 호텔 예약을 완료했다는 보고",
                    "requiresBackgroundKnowledge": False,
                },
            },
        ]
    }


def evaluation_base(purpose="OFFICIAL", assistance=None, answer_revealed=False):
    return {
        "requestId": "evaluation-1",
        "idempotencyKey": "evaluation-idem-1",
        "itemId": 301,
        "attemptId": 401,
        "evaluationPurpose": purpose,
        "answerRevealed": answer_revealed,
        "assistanceUsage": assistance or [],
        "policyVersion": "listening-profile",
        "modelConfigVersion": "listening-model-config",
    }


def dictation_request(
    source="京都へ行きます。",
    answer="きょうとへ行きます",
    *,
    purpose="OFFICIAL",
    assistance=None,
    answer_revealed=False,
    accepted_variants=None,
):
    data = evaluation_base(purpose, assistance, answer_revealed)
    data.update(
        {
            "sourceText": source,
            "answer": answer,
            "learningLanguage": "ja",
            "acceptedVariants": accepted_variants or {"京都": ["きょうと"]},
        }
    )
    return DictationEvaluationRequest.model_validate(data)


def interpretation_payload(confidence=0.90):
    metric_scores = {
        "MEANING_FIDELITY": 90,
        "DETAIL_AND_NUANCE": 80,
        "ORIGIN_NATURALNESS": 70,
    }
    return {
        "evaluationConfidence": confidence,
        "metrics": [
            {
                "type": metric_type,
                "score": score,
                "confidence": confidence,
                "evidence": [
                    {
                        "metric": metric_type,
                        "severity": "INFO",
                        "feedback": "의미 근거",
                    }
                ],
            }
            for metric_type, score in metric_scores.items()
        ],
        "deliveredMeaningUnits": ["일본에 가면", "교토 방문"],
        "omittedMeaningUnits": ["천천히 보고 싶다"],
        "misunderstoodMeaningUnits": [],
        "addedInformation": [],
        "recommendedInterpretations": [
            "일본에 가면 교토를 천천히 둘러보고 싶어요.",
            "일본 여행에서는 교토를 여유롭게 보고 싶습니다.",
        ],
        "strengths": ["방문 의도를 이해했습니다."],
        "improvements": ["속도에 관한 의미도 들어 보세요."],
    }


def interpretation_request(*, purpose="OFFICIAL", answer_revealed=False):
    data = evaluation_base(purpose, answer_revealed=answer_revealed)
    data.update(
        {
            "sourceText": "日本に行ったら京都をゆっくり見たいです。",
            "referenceMeanings": [
                "일본에 가면 교토를 천천히 보고 싶어요.",
                "일본 여행에서는 교토를 여유롭게 둘러보고 싶습니다.",
            ],
            "keyMeaningUnits": ["일본에 가면", "교토 방문", "천천히 보고 싶다"],
            "answer": "일본에 가면 교토에 가고 싶어요.",
            "originLanguage": "ko",
            "learningLanguage": "ja",
        }
    )
    return InterpretationEvaluationRequest.model_validate(data)


def comprehension_request(selected="B", *, correct="B"):
    data = evaluation_base()
    data.update(
        {
            "question": "話者は次に何をしますか？",
            "options": [
                {"key": "A", "text": "電話します"},
                {"key": "B", "text": "予約します"},
                {"key": "C", "text": "帰ります"},
                {"key": "D", "text": "待ちます"},
            ],
            "selectedOptionKey": selected,
            "correctOptionKey": correct,
            "comprehensionFocus": "NEXT_ACTION",
            "originLanguage": "ko",
            "learningLanguage": "ja",
        }
    )
    return ComprehensionEvaluationRequest.model_validate(data)


def summary_request():
    data = evaluation_base()
    data.update(
        {
            "sourceText": "会議は十時から三時に変更になりました。参加できるか確認してください。",
            "summaryKeyPoints": ["会議時間の変更", "参加可否の確認"],
            "answer": "会議が三時に変わり、参加できるか確認しています。",
            "originLanguage": "ko",
            "learningLanguage": "ja",
        }
    )
    return SummaryEvaluationRequest.model_validate(data)


def summary_payload():
    scores = {
        "GIST_COVERAGE": 100,
        "KEY_POINT_COVERAGE": 80,
        "LANGUAGE_CLARITY": 60,
    }
    return {
        "evaluationConfidence": 0.9,
        "metrics": [
            {
                "type": metric_type,
                "score": score,
                "confidence": 0.9,
                "evidence": [
                    {
                        "metric": metric_type,
                        "severity": "INFO",
                        "feedback": "근거",
                    }
                ],
            }
            for metric_type, score in scores.items()
        ],
        "strengths": ["핵심을 파악했습니다."],
        "improvements": ["세부 정보를 더 포함해 보세요."],
        "recommendedSummaries": [
            "会議は三時に変更され、参加可否の確認が必要です。",
            "会議時間が変更されたため、参加できるか確認します。",
        ],
        "deliveredKeyPoints": ["会議時間の変更"],
        "omittedKeyPoints": ["参加可否の確認"],
    }


def repeat_request(*, purpose="OFFICIAL", assistance=None, answer_revealed=False):
    data = evaluation_base(purpose, assistance, answer_revealed)
    data.update(
        {
            "sourceText": "hello world",
            "sourceDurationSeconds": 2.0,
            "learningLanguage": "en",
            "phraseHints": ["hello", "world"],
        }
    )
    return RepeatEvaluationContext.model_validate(data)


def selected_task(response, task_type):
    return next(task for task in response.tasks if task.task_type == task_type)


class ListeningGenerationTest(unittest.TestCase):
    def test_generation_returns_hashes_and_uses_profile_keyword_context(self):
        provider = FakeStructuredProvider([generation_payload()])
        service = ListeningGenerationService(provider, automatic_retries=0)
        response = asyncio.run(service.generate(generation_request()))

        self.assertEqual(2, len(response.items))
        self.assertEqual("listening-generation", response.generation_version)
        self.assertEqual(64, len(response.items[0].content_hash))
        self.assertEqual(64, len(response.items[0].similarity_key))
        self.assertEqual(2, len(response.items[0].reference_meanings))
        self.assertIn("profileFocus", provider.calls[0][1])
        self.assertIn("VOCABULARY", provider.calls[0][1])

    def test_generation_accepts_comprehension_mode_contract(self):
        request_data = generation_request().model_dump(by_alias=True, mode="json")
        request_data["setContext"]["learningMode"] = "COMPREHENSION"
        payload = generation_payload()
        for item in payload["items"]:
            item.update({
                "question": "話者は次に何をしますか？",
                "options": [
                    {"key": "A", "text": "電話します"},
                    {"key": "B", "text": "予約します"},
                    {"key": "C", "text": "帰ります"},
                    {"key": "D", "text": "待ちます"},
                ],
                "correctOptionKey": "B",
                "comprehensionFocus": "NEXT_ACTION",
                "summaryKeyPoints": [],
            })
        service = ListeningGenerationService(FakeStructuredProvider([payload]), automatic_retries=0)
        response = asyncio.run(service.generate(ListeningSetGenerationRequest.model_validate(request_data)))
        self.assertEqual("B", response.items[0].correct_option_key)
        self.assertEqual(4, len(response.items[0].options))

    def test_generation_accepts_summary_mode_contract(self):
        request_data = generation_request().model_dump(by_alias=True, mode="json")
        request_data["setContext"]["learningMode"] = "SUMMARY"
        payload = generation_payload()
        for item in payload["items"]:
            item.update({"summaryKeyPoints": ["旅行の予定", "予約の状況"]})
        service = ListeningGenerationService(FakeStructuredProvider([payload]), automatic_retries=0)
        response = asyncio.run(service.generate(ListeningSetGenerationRequest.model_validate(request_data)))
        self.assertEqual(2, len(response.items[0].summary_key_points))

    def test_generation_rejects_mode_payload_mismatch(self):
        request_data = generation_request().model_dump(by_alias=True, mode="json")
        request_data["setContext"]["learningMode"] = "COMPREHENSION"
        service = ListeningGenerationService(FakeStructuredProvider([generation_payload()]), automatic_retries=0)
        with self.assertRaises(ListeningStageException) as context:
            asyncio.run(service.generate(ListeningSetGenerationRequest.model_validate(request_data)))
        self.assertEqual(ListeningErrorCode.GENERATION_FAILED, context.exception.code)

    def test_generation_is_idempotent(self):
        provider = FakeStructuredProvider([generation_payload()])
        service = ListeningGenerationService(provider, automatic_retries=0)
        request = generation_request()
        asyncio.run(service.generate(request))
        asyncio.run(
            service.generate(request.model_copy(update={"request_id": "generation-2"}))
        )
        self.assertEqual(1, len(provider.calls))

    def test_recent_duplicate_is_rejected(self):
        payload = generation_payload()
        normalized = normalize_text(payload["items"][0]["sourceText"], "ja")
        content_hash = hashlib.sha256(normalized.text.encode()).hexdigest()
        request = generation_request(
            constraints={
                "audioSecondsMin": 8,
                "audioSecondsMax": 20,
                "recentContentHashes": [content_hash],
                "recentSimilaritySummaries": [],
            }
        )
        service = ListeningGenerationService(
            FakeStructuredProvider([payload]),
            automatic_retries=0,
        )
        with self.assertRaises(ListeningStageException) as context:
            asyncio.run(service.generate(request))
        self.assertEqual(ListeningErrorCode.GENERATION_FAILED, context.exception.code)

    def test_difficulty_duration_is_enforced(self):
        payload = generation_payload()
        payload["items"][0]["estimatedAudioSeconds"] = 30
        service = ListeningGenerationService(
            FakeStructuredProvider([payload]),
            automatic_retries=0,
        )
        with self.assertRaises(ListeningStageException) as context:
            asyncio.run(service.generate(generation_request()))
        self.assertEqual(
            ListeningErrorCode.GENERATION_FAILED, context.exception.code
        )

    def test_provider_429_is_retried_within_two_retry_limit(self):
        provider = FakeStructuredProvider(
            results=[generation_payload()],
            errors=[Provider429Error("rate limited")],
        )
        service = ListeningGenerationService(provider, automatic_retries=1)
        response = asyncio.run(service.generate(generation_request()))
        self.assertEqual(2, len(response.items))
        self.assertEqual(2, len(provider.calls))


class ListeningTtsTest(unittest.TestCase):
    def _request(self, **overrides):
        source = "京都へ行きたいです。"
        normalized = normalize_text(source, "ja")
        data = {
            "requestId": "tts-1",
            "idempotencyKey": "tts-idem-1",
            "itemId": 301,
            "sourceText": source,
            "contentHash": hashlib.sha256(normalized.text.encode()).hexdigest(),
            "generationVersion": "listening-generation",
            "learningLanguage": "ja",
            "voice": {"locale": "ja-JP", "voiceKey": "standard-1", "version": "current"},
            "playbackSpeed": "SLOW",
            "policyVersion": "listening",
            "modelConfigVersion": "listening-model-config",
        }
        data.update(overrides)
        return ListeningTtsRequest.model_validate(data)

    def test_tts_returns_voice_hash_duration_checksum_and_reuses_normal_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TemporaryListeningAudioStore()
            store.base_dir = __import__("pathlib").Path(directory)
            provider = FakeSpeechProvider()
            service = ListeningTtsService(provider, store, automatic_retries=0)
            response = asyncio.run(service.synthesize(self._request()))
            cached = asyncio.run(service.synthesize(self._request(requestId="tts-2")))

        self.assertEqual("READY", response.status)
        assert response.audio is not None
        assert cached.audio is not None
        self.assertEqual(9600, response.audio.duration_ms)
        self.assertEqual(64, len(response.audio.text_hash))
        self.assertEqual(64, len(response.audio.checksum))
        self.assertEqual("NORMAL", provider.kwargs[0]["speed"])
        self.assertEqual(response.audio.audio_reference, cached.audio.audio_reference)
        self.assertEqual(1, provider.calls)

    def test_tts_failure_keeps_generated_text_and_structured_stage_error(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TemporaryListeningAudioStore()
            store.base_dir = __import__("pathlib").Path(directory)
            provider = FakeSpeechProvider(
                [
                    Provider503Error("down"),
                    Provider503Error("down"),
                    Provider503Error("down"),
                ]
            )
            service = ListeningTtsService(provider, store, automatic_retries=2)
            response = asyncio.run(service.synthesize(self._request()))

        self.assertEqual("FAILED", response.status)
        self.assertEqual(self._request().source_text, response.source_text)
        assert response.error is not None
        self.assertEqual("TTS", response.error.failed_stage.value)
        self.assertEqual(ListeningErrorCode.PROVIDER_UNAVAILABLE, response.error.code)
        self.assertTrue(response.error.retryable)
        self.assertEqual(3, provider.calls)


class ListeningDictationTest(unittest.TestCase):
    def test_ko_ja_en_locale_normalization_and_japanese_accepted_variant(self):
        service = ListeningDictationService()
        japanese = asyncio.run(service.evaluate(dictation_request()))
        task = selected_task(japanese, ListeningTaskType.DICTATION)
        self.assertEqual(100, task.score)
        self.assertTrue(
            any(item.status.value == "ACCEPTED_VARIANT" for item in task.alignment)
        )

        korean = dictation_request(
            source="안녕하세요, 반갑습니다!",
            answer="안녕하세요 반갑습니다",
            accepted_variants={},
        ).model_copy(update={"learning_language": "ko", "idempotency_key": "ko"})
        english = dictation_request(
            source="I can't go.",
            answer="i cannot go",
            accepted_variants={},
        ).model_copy(update={"learning_language": "en", "idempotency_key": "en"})
        self.assertEqual(
            100,
            selected_task(
                asyncio.run(service.evaluate(korean)), ListeningTaskType.DICTATION
            ).score,
        )
        self.assertEqual(
            100,
            selected_task(
                asyncio.run(service.evaluate(english)), ListeningTaskType.DICTATION
            ).score,
        )

    def test_omission_addition_substitution_and_order_change_scores(self):
        service = ListeningDictationService()
        cases = [
            ("one two three", "one three"),
            ("one two three", "one two three four"),
            ("one two three", "one blue three"),
            ("one two three", "three two one"),
        ]
        scores = []
        for index, (source, answer) in enumerate(cases):
            request = dictation_request(
                source=source,
                answer=answer,
                accepted_variants={},
            ).model_copy(
                update={
                    "learning_language": "en",
                    "idempotency_key": f"alignment-{index}",
                }
            )
            task = selected_task(
                asyncio.run(service.evaluate(request)),
                ListeningTaskType.DICTATION,
            )
            scores.append(task.score)
            assert task.score is not None
            self.assertLess(task.score, 100)
        self.assertEqual(4, len(scores))

    def test_improvements_show_exact_difference_and_local_context(self):
        service = ListeningDictationService()

        omission_task = selected_task(
            asyncio.run(
                service.evaluate(
                    dictation_request(
                        source="午後一時から会議を始めます。",
                        answer="午後時から会議を始めます。",
                        accepted_variants={},
                    ).model_copy(update={"idempotency_key": "detail-omission"})
                )
            ),
            ListeningTaskType.DICTATION,
        )
        self.assertTrue(any("누락: 「一」" in item for item in omission_task.improvements))
        self.assertTrue(any("원문 구간" in item for item in omission_task.improvements))
        self.assertTrue(any("내 답변 구간" in item for item in omission_task.improvements))
        self.assertEqual(
            ["대부분의 Token과 어순을 정확히 받아썼습니다."],
            omission_task.strengths,
        )
        omission_evidence = next(
            item for item in omission_task.evidence if item.reference == "一"
        )
        self.assertIn("답변에서 누락", omission_evidence.feedback)

        addition_task = selected_task(
            asyncio.run(
                service.evaluate(
                    dictation_request(
                        source="客様から問い合わせがありました。",
                        answer="お客様から問い合わせがありました。",
                        accepted_variants={},
                    ).model_copy(update={"idempotency_key": "detail-addition"})
                )
            ),
            ListeningTaskType.DICTATION,
        )
        self.assertTrue(any("추가: 「お」" in item for item in addition_task.improvements))
        self.assertTrue(any("원문 구간" in item for item in addition_task.improvements))
        self.assertTrue(any("내 답변 구간" in item for item in addition_task.improvements))
        addition_evidence = next(
            item for item in addition_task.evidence if item.recognized == "お"
        )
        self.assertIn("원문에 없는", addition_evidence.feedback)

    def test_metric_weights_and_profile_signal_allowlist_are_fixed(self):
        response = asyncio.run(
            ListeningDictationService().evaluate(dictation_request())
        )
        task = selected_task(response, ListeningTaskType.DICTATION)
        self.assertEqual(
            DICTATION_WEIGHTS, {metric.type: metric.weight for metric in task.metrics}
        )
        signal_weights = {
            signal.metric: signal.evidence_weight for signal in task.profile_signals
        }
        self.assertEqual(
            PROFILE_SIGNAL_WEIGHTS[ListeningTaskType.DICTATION], signal_weights
        )
        self.assertNotIn(
            "WRITTEN_EXPRESSION",
            {signal.metric.value for signal in task.profile_signals},
        )

    def test_practice_has_score_but_no_profile_signal(self):
        task = selected_task(
            asyncio.run(
                ListeningDictationService().evaluate(
                    dictation_request(purpose="PRACTICE")
                )
            ),
            ListeningTaskType.DICTATION,
        )
        self.assertTrue(task.evaluable)
        self.assertEqual(100, task.score)
        self.assertFalse(task.profile_eligible)
        self.assertEqual([], task.profile_signals)

    def test_show_answer_excludes_item_evaluation_and_signals(self):
        request = dictation_request(assistance=[{"type": "SHOW_ANSWER", "count": 1}])
        response = asyncio.run(ListeningDictationService().evaluate(request))
        task = selected_task(response, ListeningTaskType.DICTATION)
        self.assertEqual(ListeningTaskStatus.NOT_EVALUABLE, task.status)
        self.assertIsNone(task.score)
        self.assertEqual("ANSWER_REVEALED", task.reason_code)
        self.assertEqual(ListeningAssistanceLevel.GUIDED, task.assistance_level)
        self.assertEqual([], task.profile_signals)

    def test_replay_and_slow_are_independent_but_hints_are_assisted(self):
        replay = [
            AssistanceUsage(type=ListeningAssistanceType.REPLAY),
            AssistanceUsage(type=ListeningAssistanceType.SLOW_PLAYBACK),
        ]
        hint = [AssistanceUsage(type=ListeningAssistanceType.TOPIC_HINT)]
        self.assertEqual(
            ListeningAssistanceLevel.INDEPENDENT, resolve_assistance_level(replay)
        )
        self.assertEqual(
            ListeningAssistanceLevel.ASSISTED, resolve_assistance_level(hint)
        )


class ListeningInterpretationTest(unittest.TestCase):
    def test_semantic_metric_weights_and_recommended_interpretations(self):
        service = ListeningInterpretationService(
            FakeStructuredProvider([interpretation_payload(0.90)]),
            automatic_retries=0,
        )
        response = asyncio.run(service.evaluate(interpretation_request()))
        task = selected_task(response, ListeningTaskType.INTERPRETATION)
        self.assertEqual(85, task.score)
        self.assertEqual(
            INTERPRETATION_WEIGHTS,
            {metric.type: metric.weight for metric in task.metrics},
        )
        self.assertEqual(2, len(task.recommended_interpretations))
        self.assertEqual("천천히 보고 싶다", task.omitted_meaning_units[0])

    def test_normalized_zero_to_one_metric_scores_are_converted_to_points(self):
        payload = interpretation_payload(0.90)
        for metric in payload["metrics"]:
            metric["score"] = metric["score"] / 100

        service = ListeningInterpretationService(
            FakeStructuredProvider([payload]),
            automatic_retries=0,
        )
        response = asyncio.run(service.evaluate(interpretation_request()))
        task = selected_task(response, ListeningTaskType.INTERPRETATION)

        self.assertEqual(85, task.score)
        self.assertEqual(
            [90.0, 80.0, 70.0],
            [metric.score for metric in task.metrics],
        )

    def test_mixed_metric_score_scales_are_not_rewritten(self):
        payload = interpretation_payload(0.90)
        payload["metrics"][2]["score"] = 1

        service = ListeningInterpretationService(
            FakeStructuredProvider([payload]),
            automatic_retries=0,
        )
        response = asyncio.run(service.evaluate(interpretation_request()))
        task = selected_task(response, ListeningTaskType.INTERPRETATION)

        self.assertEqual(74, task.score)
        self.assertEqual(1.0, task.metrics[2].score)

    def test_confidence_069_is_not_evaluable_but_070_is_evaluable(self):
        low = ListeningInterpretationService(
            FakeStructuredProvider([interpretation_payload(0.69)]),
            automatic_retries=0,
        )
        boundary = ListeningInterpretationService(
            FakeStructuredProvider([interpretation_payload(0.70)]),
            automatic_retries=0,
        )
        low_task = selected_task(
            asyncio.run(low.evaluate(interpretation_request())),
            ListeningTaskType.INTERPRETATION,
        )
        boundary_task = selected_task(
            asyncio.run(boundary.evaluate(interpretation_request())),
            ListeningTaskType.INTERPRETATION,
        )
        self.assertFalse(low_task.evaluable)
        self.assertIsNone(low_task.score)
        self.assertEqual("LOW_CONFIDENCE", low_task.reason_code)
        self.assertTrue(boundary_task.evaluable)
        self.assertEqual(85, boundary_task.score)

    def test_practice_and_answer_reveal_never_emit_signals(self):
        practice = ListeningInterpretationService(
            FakeStructuredProvider([interpretation_payload()]),
            automatic_retries=0,
        )
        practice_task = selected_task(
            asyncio.run(practice.evaluate(interpretation_request(purpose="PRACTICE"))),
            ListeningTaskType.INTERPRETATION,
        )
        self.assertEqual([], practice_task.profile_signals)

        reveal_provider = FakeStructuredProvider([interpretation_payload()])
        reveal = ListeningInterpretationService(reveal_provider, automatic_retries=0)
        reveal_task = selected_task(
            asyncio.run(reveal.evaluate(interpretation_request(answer_revealed=True))),
            ListeningTaskType.INTERPRETATION,
        )
        self.assertEqual([], reveal_task.profile_signals)
        self.assertIsNone(reveal_task.score)
        self.assertEqual(0, len(reveal_provider.calls))

    def test_timeout_retry_and_official_practice_cache_separation(self):
        provider = FakeStructuredProvider(
            results=[interpretation_payload(), interpretation_payload()],
            errors=[TimeoutError("timeout")],
        )
        service = ListeningInterpretationService(provider, automatic_retries=1)
        official_request = interpretation_request()
        practice_request = interpretation_request(purpose="PRACTICE").model_copy(
            update={"request_id": "practice-2"}
        )
        official = asyncio.run(service.evaluate(official_request))
        practice = asyncio.run(service.evaluate(practice_request))
        self.assertEqual(3, len(provider.calls))
        self.assertTrue(
            selected_task(
                official,
                ListeningTaskType.INTERPRETATION,
            ).profile_signals
        )
        self.assertEqual(
            [],
            selected_task(
                practice,
                ListeningTaskType.INTERPRETATION,
            ).profile_signals,
        )


class ListeningComprehensionTest(unittest.IsolatedAsyncioTestCase):
    async def test_objective_choice_is_scored_deterministically(self):
        service = ListeningComprehensionService()
        correct = await service.evaluate(comprehension_request("B"))
        wrong = await service.evaluate(
            comprehension_request("A").model_copy(
                update={"idempotency_key": "evaluation-idem-wrong"}
            )
        )
        correct_task = next(task for task in correct.tasks if task.task_type == ListeningTaskType.COMPREHENSION)
        wrong_task = next(task for task in wrong.tasks if task.task_type == ListeningTaskType.COMPREHENSION)
        self.assertEqual(100, correct_task.score)
        self.assertEqual(0, wrong_task.score)
        self.assertEqual("ANSWER_ACCURACY", correct_task.metrics[0].type)


class ListeningSummaryTest(unittest.IsolatedAsyncioTestCase):
    def test_summary_service_uses_evaluation_timeout_by_default(self):
        provider = FakeStructuredProvider([summary_payload()])
        service = ListeningSummaryService(provider, automatic_retries=0)

        self.assertEqual(
            settings.AI_LISTENING_EVALUATION_TIMEOUT_SECONDS,
            service.timeout_seconds,
        )

    async def test_summary_prioritizes_listening_content_over_language_polish(self):
        provider = FakeStructuredProvider([summary_payload()])
        service = ListeningSummaryService(provider, timeout_seconds=1, automatic_retries=0)
        response = await service.evaluate(summary_request())
        # 100*0.55 + 80*0.30 + 60*0.15 = 88
        task = next(task for task in response.tasks if task.task_type == ListeningTaskType.SUMMARY)
        self.assertEqual(88, task.score)
        self.assertEqual(3, len(task.metrics))
        self.assertEqual("GIST_COVERAGE", task.metrics[0].type)


class ListeningRepeatTest(unittest.TestCase):
    def test_repeat_uses_alignment_acoustics_and_allowed_profile_signals(self):
        service = ListeningRepeatService(
            FakeAudioProcessor(),
            FakeSttProvider(),
            automatic_retries=0,
        )
        response = asyncio.run(
            service.evaluate(
                repeat_request(),
                audio_bytes=b"audio",
                file_name="answer.wav",
                content_type="audio/wav",
            )
        )
        task = selected_task(response, ListeningTaskType.REPEAT_AFTER_AUDIO)
        self.assertTrue(task.evaluable)
        self.assertEqual(
            REPEAT_WEIGHTS, {metric.type: metric.weight for metric in task.metrics}
        )
        signal_weights = {
            signal.metric: signal.evidence_weight for signal in task.profile_signals
        }
        self.assertEqual(
            PROFILE_SIGNAL_WEIGHTS[ListeningTaskType.REPEAT_AFTER_AUDIO],
            signal_weights,
        )
        self.assertNotIn(
            "SPOKEN_EXPRESSION",
            {signal.metric.value for signal in task.profile_signals},
        )
        self.assertNotIn(
            "INTERACTION", {signal.metric.value for signal in task.profile_signals}
        )
        self.assertFalse(task.debug_metadata["phonemeProviderScoreAvailable"])
        self.assertIn("RMS", task.debug_metadata["supportedEvidence"])

    def test_same_transcript_with_worse_acoustics_lowers_pronunciation(self):
        good_service = ListeningRepeatService(
            FakeAudioProcessor(quality_signals()),
            FakeSttProvider(),
            automatic_retries=0,
        )
        noisy_service = ListeningRepeatService(
            FakeAudioProcessor(
                quality_signals(rms=0.01, peak=0.99, silence_ratio=0.50)
            ),
            FakeSttProvider(),
            automatic_retries=0,
        )
        good = selected_task(
            asyncio.run(
                good_service.evaluate(
                    repeat_request(),
                    audio_bytes=b"good",
                    file_name="good.wav",
                    content_type="audio/wav",
                )
            ),
            ListeningTaskType.REPEAT_AFTER_AUDIO,
        )
        noisy = selected_task(
            asyncio.run(
                noisy_service.evaluate(
                    repeat_request(),
                    audio_bytes=b"noisy",
                    file_name="noisy.wav",
                    content_type="audio/wav",
                )
            ),
            ListeningTaskType.REPEAT_AFTER_AUDIO,
        )
        good_pronunciation = next(
            metric.score for metric in good.metrics if metric.type == "PRONUNCIATION"
        )
        noisy_pronunciation = next(
            metric.score for metric in noisy.metrics if metric.type == "PRONUNCIATION"
        )
        assert good_pronunciation is not None
        assert noisy_pronunciation is not None
        self.assertGreater(good_pronunciation, noisy_pronunciation)

    def test_language_mismatch_and_alignment_insufficient_are_structured(self):
        mismatch = ListeningRepeatService(
            FakeAudioProcessor(),
            FakeSttProvider(stt_result(language="ja", language_probability=0.95)),
            automatic_retries=0,
        )
        mismatch_task = selected_task(
            asyncio.run(
                mismatch.evaluate(
                    repeat_request(),
                    audio_bytes=b"audio",
                    file_name="answer.wav",
                    content_type="audio/wav",
                )
            ),
            ListeningTaskType.REPEAT_AFTER_AUDIO,
        )
        self.assertEqual("LANGUAGE_MISMATCH", mismatch_task.reason_code)
        self.assertIsNone(mismatch_task.score)

        insufficient = ListeningRepeatService(
            FakeAudioProcessor(),
            FakeSttProvider(stt_result(text="unrelated")),
            automatic_retries=0,
        )
        insufficient_task = selected_task(
            asyncio.run(
                insufficient.evaluate(
                    repeat_request(),
                    audio_bytes=b"other",
                    file_name="answer.wav",
                    content_type="audio/wav",
                )
            ),
            ListeningTaskType.REPEAT_AFTER_AUDIO,
        )
        self.assertEqual("ALIGNMENT_INSUFFICIENT", insufficient_task.reason_code)

    def test_audio_validation_error_and_duration_cap_are_structured(self):
        error = SpeakingStageException(
            code=SpeakingErrorCode.SILENCE_DETECTED,
            stage=SpeakingStage.AUDIO_VALIDATION,
            message="silence",
            retryable=False,
        )
        processor = FakeAudioProcessor(error=error)
        service = ListeningRepeatService(
            processor, FakeSttProvider(), automatic_retries=0
        )
        task = selected_task(
            asyncio.run(
                service.evaluate(
                    repeat_request(),
                    audio_bytes=b"silence",
                    file_name="answer.wav",
                    content_type="audio/wav",
                )
            ),
            ListeningTaskType.REPEAT_AFTER_AUDIO,
        )
        self.assertEqual("SILENCE", task.reason_code)
        self.assertEqual(17.0, processor.max_seconds)

    def test_clipping_and_noise_are_rejected_before_stt(self):
        processor = FakeAudioProcessor(
            quality=quality_signals(rms=0.99, peak=1.0, silence_ratio=0.0),
            wav_bytes=clipped_noise_wav(),
        )
        stt = FakeSttProvider()
        service = ListeningRepeatService(processor, stt, automatic_retries=0)
        task = selected_task(
            asyncio.run(
                service.evaluate(
                    repeat_request(),
                    audio_bytes=b"clipped-noise",
                    file_name="answer.wav",
                    content_type="audio/wav",
                )
            ),
            ListeningTaskType.REPEAT_AFTER_AUDIO,
        )
        self.assertEqual("LOW_AUDIO_QUALITY", task.reason_code)
        self.assertGreater(task.debug_metadata["clippingRatio"], 0.10)
        self.assertEqual(0, stt.calls)


class ListeningExplanationAndBenchmarkTest(unittest.TestCase):
    def test_explanation_echoes_be_decision_only(self):
        provider = FakeStructuredProvider(
            [
                {
                    "explanation": "최근 연습에서 발음 근거가 쌓였어요. 따라 말하기로 명료도를 다듬어 보세요.",
                    "ctaLabel": "따라 말하기",
                }
            ]
        )
        service = ListeningExplanationService(provider, automatic_retries=0)
        request = RecommendationExplanationRequest.model_validate(
            {
                "requestId": "explain-1",
                "idempotencyKey": "explain-idem-1",
                "type": "RECOMMENDATION",
                "targetMetric": "PRONUNCIATION",
                "recommendedActivity": "LISTENING",
                "recommendedTask": "REPEAT_AFTER_AUDIO",
                "evidenceSummary": {
                    "sources": ["LISTENING", "SPEAKING"],
                    "count": 5,
                    "recentAverage": 72,
                },
                "originLanguage": "ko",
            }
        )
        response = asyncio.run(service.explain(request))
        self.assertEqual("LISTENING", response.recommended_activity)
        self.assertEqual("REPEAT_AFTER_AUDIO", response.recommended_task)
        self.assertIn("REPEAT_AFTER_AUDIO", provider.calls[0][1])

    def test_human_gate_requires_two_raters_and_each_language_task_segment(self):
        samples = [
            ListeningBenchmarkSample.model_validate(
                {
                    "sampleId": f"sample-{index}",
                    "language": language,
                    "difficulty": "MY_LEVEL",
                    "taskType": task,
                    "aiScore": ai,
                    "humanScores": [
                        {"evaluatorId": "human-a", "score": human - 2},
                        {"evaluatorId": "human-b", "score": human + 2},
                    ],
                }
            )
            for index, (language, task, ai, human) in enumerate(
                [
                    ("ko", "DICTATION", 82, 80),
                    ("ko", "INTERPRETATION", 70, 72),
                    ("ja", "DICTATION", 77, 75),
                    ("ja", "REPEAT_AFTER_AUDIO", 88, 85),
                ]
            )
        ]
        result = ListeningEvaluationBenchmarkService().evaluate(samples)
        self.assertTrue(result.passed)
        self.assertEqual(1.0, result.agreement_rate)
        self.assertTrue(
            any(
                segment.segment.startswith("language-task:")
                for segment in result.segments
            )
        )

        with self.assertRaises(ValueError):
            ListeningBenchmarkSample.model_validate(
                {
                    "sampleId": "invalid",
                    "language": "en",
                    "difficulty": "EASY",
                    "taskType": "DICTATION",
                    "aiScore": 80,
                    "humanScores": [{"evaluatorId": "only-one", "score": 80}],
                }
            )


class ListeningPolicyInvariantTest(unittest.TestCase):
    def test_all_metric_weight_sets_sum_to_one(self):
        self.assertAlmostEqual(1.0, sum(DICTATION_WEIGHTS.values()))
        self.assertAlmostEqual(1.0, sum(INTERPRETATION_WEIGHTS.values()))
        self.assertAlmostEqual(1.0, sum(REPEAT_WEIGHTS.values()))

    def test_profile_allowlist_has_only_documented_pairs(self):
        self.assertEqual(
            {
                ProfileMetric.LISTENING_RECOGNITION,
                ProfileMetric.VOCABULARY,
                ProfileMetric.ORTHOGRAPHY,
            },
            set(PROFILE_SIGNAL_WEIGHTS[ListeningTaskType.DICTATION]),
        )
        self.assertEqual(
            {ProfileMetric.MEANING, ProfileMetric.ORIGIN_NATURALNESS},
            set(PROFILE_SIGNAL_WEIGHTS[ListeningTaskType.INTERPRETATION]),
        )


if __name__ == "__main__":
    unittest.main()
