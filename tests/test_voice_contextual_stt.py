from __future__ import annotations

import asyncio
import unittest

from app.ai.ports import VoiceTranslationGenerationResult
from app.core.config import settings
from app.features.voice_translation.speech_detector import PassThroughSpeechEvidenceGuard
from app.features.voice_translation.stream import VoiceChannelStreamContext
from app.features.voice_translation.stt import VoiceSttResult
from app.features.voice_translation.translation import VoiceTranslationService
from app.schemas.voice_translation import VoiceStreamOpen

REPEATED_UNSUPPORTED = "ac yn yw'r gweithio yn yw'r gweithio yn yw'r gweithio yn yw'r gweithio"
CLEAR_ENGLISH = "Four strong ideas can make our words more useful to listeners."


def stt_result(text: str, language: str, confidence: float) -> VoiceSttResult:
    return VoiceSttResult(
        text=text,
        language=language,
        language_confidence=confidence,
        no_speech_probability=0.01,
        provider="fake",
        model="fake-stt",
    )


class RecordingStt:
    ready = True
    model_version = "fake-stt"

    def __init__(self, original: VoiceSttResult, candidate: VoiceSttResult | Exception) -> None:
        self.original = original
        self.candidate = candidate
        self.languages: list[str | None] = []

    async def transcribe_pcm(
        self,
        pcm_bytes: bytes,
        *,
        language: str | None,
        is_final: bool,
        initial_prompt: str | None = None,
    ) -> VoiceSttResult:
        del pcm_bytes, is_final, initial_prompt
        self.languages.append(language)
        if language is None:
            return self.original
        if isinstance(self.candidate, Exception):
            raise self.candidate
        return self.candidate


class UnusedTranslationProvider:
    async def translate_voice_utterance(
        self,
        *,
        source_text: str,
        source_language: str,
        target_language: str,
        previous_source_text: str | None = None,
    ) -> VoiceTranslationGenerationResult:
        del source_text, source_language, target_language, previous_source_text
        raise AssertionError("Contextual STT tests must not call translation")


def context(
    stt: RecordingStt,
    *,
    locked: str | None = "en",
    manual: str | None = None,
) -> VoiceChannelStreamContext:
    payload = {
        "type": "STREAM_OPEN",
        "requestId": "contextual-stt-test",
        "sessionId": "contextual-stt-session",
        "channel": "REMOTE",
        "mode": "MEDIA",
        "sourceLanguageMode": "MANUAL" if manual else "AUTO",
        "manualSourceLanguage": manual,
        "lastLockedLanguage": locked,
        "targetLanguage": "ko",
        "audioFormat": {
            "encoding": "PCM_S16LE",
            "sampleRate": 16000,
            "channels": 1,
            "frameDurationMs": 100,
        },
    }
    return VoiceChannelStreamContext(
        VoiceStreamOpen.model_validate(payload),
        stt_provider=stt,
        translation_service=VoiceTranslationService(UnusedTranslationProvider()),
        speech_evidence_guard=PassThroughSpeechEvidenceGuard(),
    )


class VoiceContextualSttTest(unittest.IsolatedAsyncioTestCase):
    async def test_low_confidence_repetition_uses_existing_language_only_when_retry_is_clear(self):
        # 준비
        original = stt_result(REPEATED_UNSUPPORTED, "cy", 0.58)
        stt = RecordingStt(original, stt_result(CLEAR_ENGLISH, "en", 1.0))

        # 실행
        result = await context(stt)._transcribe_final_with_context(b"speech")

        # 검증: 강제 추론의 언어 확신도를 실제 감지 결과처럼 공개하지 않는다.
        self.assertEqual(result.text, CLEAR_ENGLISH)
        self.assertEqual(result.language, "en")
        self.assertIsNone(result.language_confidence)
        self.assertEqual(stt.languages, [None, "en"])

    async def test_natural_or_confident_unsupported_language_is_not_forced_to_english(self):
        # 준비 / 실행: 반복 환각과 구분되는 비지원 언어 입력을 보존한다.
        cases = [
            stt_result("Bonjour, je voudrais parler de notre réunion ce matin.", "fr", 0.58),
            stt_result(REPEATED_UNSUPPORTED, "cy", 0.92),
        ]
        for original in cases:
            stt = RecordingStt(original, stt_result(CLEAR_ENGLISH, "en", 1.0))
            result = await context(stt)._transcribe_final_with_context(b"speech")

            # 검증
            self.assertIs(result, original)
            self.assertEqual(stt.languages, [None])

    async def test_manual_language_and_unlocked_channel_do_not_use_contextual_retry(self):
        # 준비 / 실행: 사용자의 수동 언어 필터와 잠금 전 자동 감지를 보존한다.
        original = stt_result(REPEATED_UNSUPPORTED, "cy", 0.58)
        for locked, manual in [("en", "en"), (None, None)]:
            stt = RecordingStt(original, stt_result(CLEAR_ENGLISH, "en", 1.0))
            result = await context(
                stt, locked=locked, manual=manual
            )._transcribe_final_with_context(b"speech")

            # 검증
            self.assertIs(result, original)
            self.assertEqual(stt.languages, [None])

    async def test_weak_or_failed_retry_keeps_original_language_rejection(self):
        # 준비 / 실행: 강제 언어 출력이 반복되거나 공급자가 실패하면 원래 판정을 유지한다.
        original = stt_result(REPEATED_UNSUPPORTED, "cy", 0.58)
        for candidate in [stt_result(REPEATED_UNSUPPORTED, "en", 1.0), RuntimeError("unavailable")]:
            stt = RecordingStt(original, candidate)
            result = await context(stt)._transcribe_final_with_context(b"speech")

            # 검증
            self.assertIs(result, original)
            self.assertEqual(stt.languages, [None, "en"])

    async def test_retry_does_not_extend_the_final_stt_deadline(self):
        # 준비
        original_timeout = settings.AI_VOICE_STT_FINAL_TIMEOUT_SECONDS
        settings.AI_VOICE_STT_FINAL_TIMEOUT_SECONDS = 0.1
        original = stt_result(REPEATED_UNSUPPORTED, "cy", 0.58)
        stt = RecordingStt(original, stt_result(CLEAR_ENGLISH, "en", 1.0))

        # 실행 / 검증: 잔여 시간이 없으면 두 번째 모델 호출을 시작하지 않는다.
        try:
            result = await asyncio.wait_for(
                context(stt)._transcribe_final_with_context(b"speech"),
                timeout=settings.AI_VOICE_STT_FINAL_TIMEOUT_SECONDS,
            )
            self.assertIs(result, original)
            self.assertEqual(stt.languages, [None])
        finally:
            settings.AI_VOICE_STT_FINAL_TIMEOUT_SECONDS = original_timeout
