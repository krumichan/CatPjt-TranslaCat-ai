from __future__ import annotations

import asyncio
import struct
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.ai.ports import VoiceTranslationGenerationResult
from app.api.internal import voice as voice_api
from app.core.config import settings
from app.features.voice_translation.stream import VoiceStreamApplicationService
from app.features.voice_translation.stt import VoiceSttResult
from app.features.voice_translation.translation import VoiceTranslationService


def pcm_frame(value: int, duration_ms: int = 100) -> bytes:
    return struct.pack("<h", value) * (16 * duration_ms)


def stream_open_payload():
    return {
        "type": "STREAM_OPEN",
        "requestId": "trace-api-1",
        "sessionId": "session-api-1",
        "channel": "SELF",
        "mode": "MIC",
        "sourceLanguageMode": "AUTO",
        "manualSourceLanguage": None,
        "targetLanguage": "ja",
        "audioFormat": {
            "encoding": "PCM_S16LE",
            "sampleRate": 16000,
            "channels": 1,
            "frameDurationMs": 100,
        },
        "policy": {
            "endpointingSilenceMs": 300,
            "minUtteranceDurationMs": 250,
            "maxUtteranceDurationMs": 10000,
            "languageLockConfidence": 0.80,
            "languageSwitchConfidence": 0.85,
            "languageSwitchConsecutiveCount": 3,
        },
    }


class ApiSttProvider:
    ready = True
    model_version = "api-fake-stt"

    async def transcribe_pcm(
        self,
        pcm_bytes: bytes,
        *,
        language: str | None,
        is_final: bool,
        initial_prompt: str | None = None,
    ) -> VoiceSttResult:
        del pcm_bytes, language, initial_prompt
        await asyncio.sleep(0)
        return VoiceSttResult(
            text="안녕하세요" if is_final else "안녕",
            language="ko",
            language_confidence=0.97,
            no_speech_probability=0.01,
            provider="fake",
            model="api-fake-stt",
        )


class ApiTranslationProvider:
    ready = True

    def __init__(self) -> None:
        self.calls = 0

    async def translate_voice_utterance(
        self,
        *,
        source_text: str,
        source_language: str,
        target_language: str,
    ) -> VoiceTranslationGenerationResult:
        del source_text, source_language, target_language
        self.calls += 1
        return VoiceTranslationGenerationResult(
            translated_text="こんにちは",
            source_reading_tokens=[],
            provider="fake",
            model="api-fake-translation",
        )


class VoiceTranslationApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_api_key = settings.SERVER_API_KEY
        settings.SERVER_API_KEY = "internal-secret"
        cls.translation_provider = ApiTranslationProvider()
        cls.translation_service = VoiceTranslationService(
            cls.translation_provider,
            max_retries=0,
        )
        cls.stream_service = VoiceStreamApplicationService(
            stt_provider=ApiSttProvider(),
            translation_service=cls.translation_service,
        )

        app = FastAPI()
        app.include_router(voice_api.router, prefix="/internal/v1")
        app.dependency_overrides[voice_api.get_voice_stream_service] = (
            lambda: cls.stream_service
        )
        app.dependency_overrides[voice_api.get_voice_translation_service] = (
            lambda: cls.translation_service
        )
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        settings.SERVER_API_KEY = cls.original_api_key

    def test_internal_websocket_contract_and_event_order(self):
        with self.client.websocket_connect(
            "/internal/v1/voice/streams",
            headers={"X-API-KEY": "internal-secret"},
        ) as websocket:
            websocket.send_json(stream_open_payload())
            ready = websocket.receive_json()
            self.assertEqual(ready["type"], "STREAM_READY")
            self.assertEqual(ready["requestId"], "trace-api-1")

            for _ in range(4):
                websocket.send_bytes(pcm_frame(4000))
            for _ in range(3):
                websocket.send_bytes(pcm_frame(0))

            event_types = []
            completed = None
            for _ in range(8):
                event = websocket.receive_json()
                event_types.append(event["type"])
                if event["type"] == "VOICE_PIPELINE_COMPLETED":
                    completed = event
                    break

            self.assertIsNotNone(completed)
            self.assertLess(
                event_types.index("SPEECH_STARTED"),
                event_types.index("TRANSCRIPT_FINAL"),
            )
            self.assertLess(
                event_types.index("TRANSCRIPT_FINAL"),
                event_types.index("VOICE_PIPELINE_COMPLETED"),
            )
            assert completed is not None
            self.assertEqual(completed["translatedText"], "こんにちは")
            self.assertNotIn("provider", completed)

            websocket.send_json({"type": "STREAM_CLOSE", "reason": "TEST_DONE"})
            closed = websocket.receive_json()
            self.assertEqual(closed["type"], "STREAM_CLOSED")

        self.assertEqual(self.stream_service.active_stream_count, 0)

    def test_websocket_rejects_missing_service_credential(self):
        with self.assertRaises(WebSocketDisconnect) as caught:
            with self.client.websocket_connect("/internal/v1/voice/streams"):
                pass
        self.assertEqual(caught.exception.code, 4401)

    def test_stream_open_maps_unsupported_format_to_typed_error(self):
        payload = stream_open_payload()
        payload["audioFormat"]["encoding"] = "OPUS"
        with self.client.websocket_connect(
            "/internal/v1/voice/streams",
            headers={"X-API-KEY": "internal-secret"},
        ) as websocket:
            websocket.send_json(payload)
            failure = websocket.receive_json()
            self.assertEqual(failure["type"], "VOICE_PIPELINE_FAILED")
            self.assertEqual(
                failure["error"]["code"],
                "VOICE_AI_UNSUPPORTED_AUDIO_FORMAT",
            )

    def test_second_stream_open_is_rejected(self):
        with self.client.websocket_connect(
            "/internal/v1/voice/streams",
            headers={"X-API-KEY": "internal-secret"},
        ) as websocket:
            websocket.send_json(stream_open_payload())
            self.assertEqual(websocket.receive_json()["type"], "STREAM_READY")
            websocket.send_json(stream_open_payload())
            failure = websocket.receive_json()
            self.assertEqual(failure["type"], "VOICE_PIPELINE_FAILED")
            self.assertEqual(
                failure["error"]["code"],
                "VOICE_AI_INVALID_STREAM_OPEN",
            )
            self.assertEqual(websocket.receive_json()["type"], "STREAM_CLOSED")

    def test_translation_retry_uses_camel_case_and_is_idempotent(self):
        before = self.translation_provider.calls
        payload = {
            "requestId": "retry-api-1",
            "sessionId": "session-api-1",
            "segmentId": 101,
            "sourceText": "안녕하세요",
            "sourceLanguage": "ko",
            "targetLanguage": "ja",
        }
        first = self.client.post(
            "/internal/v1/voice/translation/retry",
            headers={"X-API-KEY": "internal-secret"},
            json=payload,
        )
        second = self.client.post(
            "/internal/v1/voice/translation/retry",
            headers={"X-API-KEY": "internal-secret"},
            json=payload,
        )
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(first.json(), second.json())
        self.assertEqual(first.json()["translatedText"], "こんにちは")
        self.assertEqual(self.translation_provider.calls, before + 1)

    def test_readiness_is_service_authenticated(self):
        unauthorized = self.client.get("/internal/v1/voice/readiness")
        self.assertEqual(unauthorized.status_code, 401)

        ready = self.client.get(
            "/internal/v1/voice/readiness",
            headers={"X-API-KEY": "internal-secret"},
        )
        self.assertEqual(ready.status_code, 200)
        self.assertTrue(ready.json()["ready"])
        self.assertEqual(ready.json()["activeStreams"], 0)


if __name__ == "__main__":
    unittest.main()
