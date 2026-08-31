from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

from app.ai.providers.gemini.client import GeminiService, GeminiTtsQuotaCooldownError


class _FakeModels:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    async def generate_content(self, **kwargs):
        del kwargs
        self.calls += 1
        return self.responses.pop(0)


def _empty_response():
    return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[]))])


def _audio_response():
    inline = SimpleNamespace(data=b"\x01\x00" * 2400)
    part = SimpleNamespace(inline_data=inline)
    return SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))]
    )


class GeminiTtsRetryTest(IsolatedAsyncioTestCase):
    def setUp(self):
        GeminiService._tts_quota_cooldown_until = 0.0

    async def test_empty_audio_response_is_retried_then_succeeds(self):
        models = _FakeModels([_empty_response(), _audio_response()])
        service = GeminiService()
        service._client = SimpleNamespace(aio=SimpleNamespace(models=models))

        with patch(
            "app.ai.providers.gemini.client.asyncio.sleep",
            new=AsyncMock(),
        ) as sleep:
            result = await service.synthesize_speech(
                text="こんにちは",
                voice="Kore",
                language="ja",
                speed="NORMAL",
            )

        self.assertEqual(2, models.calls)
        self.assertTrue(result.audio_bytes.startswith(b"RIFF"))
        sleep.assert_awaited_once()

    async def test_empty_audio_response_fails_after_bounded_retries(self):
        models = _FakeModels([_empty_response(), _empty_response(), _empty_response()])
        service = GeminiService()
        service._client = SimpleNamespace(aio=SimpleNamespace(models=models))

        with patch(
            "app.ai.providers.gemini.client.asyncio.sleep",
            new=AsyncMock(),
        ) as sleep:
            with self.assertRaisesRegex(ValueError, "no audio part"):
                await service.synthesize_speech(
                    text="こんにちは",
                    voice="Kore",
                    language="ja",
                    speed="NORMAL",
                )

        self.assertEqual(3, models.calls)
        self.assertEqual(2, sleep.await_count)


class _QuotaError(RuntimeError):
    def __init__(self):
        super().__init__(
            "429 RESOURCE_EXHAUSTED. {'error': {'details': "
            "[{'@type': 'type.googleapis.com/google.rpc.RetryInfo', "
            "'retryDelay': '41520s'}]}}"
        )


class GeminiTtsQuotaCooldownTest(IsolatedAsyncioTestCase):
    def setUp(self):
        GeminiService._tts_quota_cooldown_until = 0.0

    async def test_retry_info_activates_fail_fast_quota_cooldown(self):
        class _RaisingModels:
            def __init__(self):
                self.calls = 0

            async def generate_content(self, **kwargs):
                del kwargs
                self.calls += 1
                raise _QuotaError()

        models = _RaisingModels()
        service = GeminiService()
        service._client = SimpleNamespace(aio=SimpleNamespace(models=models))

        with self.assertRaises(_QuotaError):
            await service.synthesize_speech(
                text="こんにちは", voice="Kore", language="ja", speed="NORMAL"
            )
        self.assertEqual(1, models.calls)

        with self.assertRaises(GeminiTtsQuotaCooldownError) as raised:
            await service.synthesize_speech(
                text="次の文", voice="Kore", language="ja", speed="NORMAL"
            )
        self.assertGreater(raised.exception.retry_after_seconds, 41_000)
        self.assertEqual(1, models.calls)
