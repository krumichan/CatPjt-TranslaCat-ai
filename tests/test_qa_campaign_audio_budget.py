from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest

from app.ai.providers.gemini.client import GeminiService
from scripts.qa_campaign_audio_budget import (
    BudgetedSpeechProvider,
    BudgetedSttProvider,
    inspect_local_stt_cache,
    make_budgeted_stt_provider,
    prepare_local_stt,
)
from scripts.qa_campaign_budget import CampaignBudgetExceeded, CampaignLedger
from tests.test_language_learning_speaking import make_wav


@pytest.fixture(autouse=True)
def isolate_shared_tts_quota_cooldown(monkeypatch: pytest.MonkeyPatch) -> None:
    # Quota tests elsewhere deliberately activate this class-level runtime state.
    # Offline fake-transport cases start independently and restore the prior value.
    # Production cooldown behavior is not bypassed or changed.
    monkeypatch.setattr(GeminiService, "_tts_quota_cooldown_until", 0.0)


def _response(*, usage=True, audio=True):
    return SimpleNamespace(
        usage_metadata=SimpleNamespace(prompt_token_count=100, candidates_token_count=1000,
                                       total_token_count=1100, thoughts_token_count=0) if usage else None,
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=[
            SimpleNamespace(inline_data=SimpleNamespace(data=b"\x00" * 48000))
        ]))] if audio else [],
    )


def _gemini_with_fake_transport(response):
    generated = AsyncMock(return_value=response)
    models = SimpleNamespace(generate_content=generated,
                             _api_client=SimpleNamespace(_http_options=SimpleNamespace(retry_options=None)))
    service = GeminiService()
    service._client = cast(Any, SimpleNamespace(aio=SimpleNamespace(models=models)))
    return service, models, generated


@pytest.mark.asyncio
async def test_gemini_real_service_receipts_settle_audio_cost_not_text_output(tmp_path: Path) -> None:
    upstream, models, generated = _gemini_with_fake_transport(_response())
    ledger = CampaignLedger(tmp_path / "budget.json", "audio-test")
    provider = BudgetedSpeechProvider(upstream, ledger, phase="speaking")
    response = await provider.synthesize_speech(text="こんにちは", voice="Kore", language="ja", speed="NORMAL")
    assert response.audio_bytes.startswith(b"RIFF")
    call = ledger.snapshot()["calls"][0]
    assert call["reserved"]["input_tokens"] == 8192 * 6
    assert call["reserved"]["output_tokens"] == 0
    assert call["reserved"]["tts_characters"] == 5 * 6
    assert call["accounted"]["input_tokens"] == 200
    assert call["accounted"]["output_tokens"] == 0
    assert call["accounted"]["tts_characters"] == 10
    assert call["accounted"]["usd"] == pytest.approx(0.0201)
    assert models.generate_content is generated
    receipt = json.loads((tmp_path / "audio-budget-receipts/tts-0001.json").read_text())
    assert receipt["receipts"][0]["audioOutputTokens"] == 1000
    assert "こんにちは" not in json.dumps(receipt, ensure_ascii=False)


@pytest.mark.asyncio
async def test_service_empty_audio_retries_are_preserved_and_all_receipts_count(tmp_path: Path) -> None:
    upstream, _, generated = _gemini_with_fake_transport(_response())
    generated.side_effect = [_response(audio=False), _response(audio=False), _response()]
    ledger = CampaignLedger(tmp_path / "budget.json", "audio-test")
    with patch("app.ai.providers.gemini.client.asyncio.sleep", new=AsyncMock()):
        await BudgetedSpeechProvider(upstream, ledger, phase="speaking").synthesize_speech(
            text="hello", voice="Kore", language="en", speed="NORMAL")
    assert generated.call_count == 3
    call = ledger.snapshot()["calls"][0]
    assert call["accounted"]["input_tokens"] == 600
    assert call["accounted"]["usd"] == pytest.approx(0.0603)


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [TimeoutError, asyncio.CancelledError])
async def test_failed_tts_keeps_full_reservation_and_restores_interception(tmp_path: Path, error) -> None:
    upstream, models, generated = _gemini_with_fake_transport(_response())
    generated.side_effect = error()
    ledger = CampaignLedger(tmp_path / "budget.json", "audio-test")
    with pytest.raises(error):
        await BudgetedSpeechProvider(upstream, ledger, phase="speaking").synthesize_speech(
            text="hello", voice="Kore", language="en", speed="NORMAL")
    call = ledger.snapshot()["calls"][0]
    assert call["accounted"] is None
    assert call["failureType"] == error.__name__
    assert models.generate_content is generated


@pytest.mark.asyncio
async def test_missing_receipt_keeps_full_reservation(tmp_path: Path) -> None:
    upstream, _, _ = _gemini_with_fake_transport(_response(usage=False))
    ledger = CampaignLedger(tmp_path / "budget.json", "audio-test")
    await BudgetedSpeechProvider(upstream, ledger, phase="speaking").synthesize_speech(
        text="hello", voice="Kore", language="en", speed="NORMAL")
    assert ledger.snapshot()["calls"][0]["accounted"] is None


@pytest.mark.asyncio
async def test_tts_timeout_records_attempt_without_second_underlying_start(tmp_path: Path) -> None:
    upstream, _, generated = _gemini_with_fake_transport(_response())

    async def blocked(**kwargs):
        await asyncio.Event().wait()

    generated.side_effect = blocked
    ledger = CampaignLedger(tmp_path / "budget.json", "audio-test")
    with pytest.raises(TimeoutError):
        await BudgetedSpeechProvider(upstream, ledger, phase="speaking", call_seconds=0.01).synthesize_speech(
            text="hello", voice="Kore", language="en", speed="NORMAL")
    assert generated.call_count == 1
    assert ledger.snapshot()["calls"][0]["failureType"] == "TimeoutError"


@pytest.mark.asyncio
async def test_stt_uses_real_wav_duration_and_local_zero_dollar_accounting(tmp_path: Path) -> None:
    upstream = SimpleNamespace(transcribe=AsyncMock(return_value="transcript"))
    ledger = CampaignLedger(tmp_path / "budget.json", "audio-test")
    response = await BudgetedSttProvider(upstream, ledger, phase="speaking").transcribe(make_wav(1.2), language="ja")
    assert response == "transcript"
    call = ledger.snapshot()["calls"][0]
    assert call["accounted"]["stt_seconds"] == 2.4
    assert call["accounted"]["usd"] == 0
    assert call["accounted"]["output_tokens"] == 0


@pytest.mark.asyncio
async def test_exhausted_ledger_prevents_audio_provider_start(tmp_path: Path) -> None:
    upstream, _, generated = _gemini_with_fake_transport(_response())
    ledger = CampaignLedger(tmp_path / "budget.json", "audio-test")
    with patch.object(ledger, "reserve", side_effect=CampaignBudgetExceeded("TEST_CAP")):
        with pytest.raises(CampaignBudgetExceeded):
            await BudgetedSpeechProvider(upstream, ledger, phase="speaking").synthesize_speech(
                text="hello", voice="Kore", language="en", speed="NORMAL")
    assert generated.call_count == 0


@pytest.mark.asyncio
async def test_changed_sdk_retry_policy_is_denied_without_provider_call(tmp_path: Path) -> None:
    upstream, models, generated = _gemini_with_fake_transport(_response())
    models._api_client._http_options.retry_options = SimpleNamespace(attempts=5)
    ledger = CampaignLedger(tmp_path / "budget.json", "audio-test")
    with pytest.raises(CampaignBudgetExceeded, match="UNPRICED_GEMINI_RETRY_POLICY"):
        await BudgetedSpeechProvider(upstream, ledger, phase="speaking").synthesize_speech(
            text="hello", voice="Kore", language="en", speed="NORMAL")
    assert generated.call_count == 0
    assert ledger.snapshot()["calls"] == []


def test_stt_cache_inspection_is_local_files_only_and_reports_missing_files(tmp_path: Path) -> None:
    with patch("faster_whisper.utils.download_model", return_value=str(tmp_path)) as lookup:
        report = inspect_local_stt_cache()
    assert lookup.call_args.kwargs["local_files_only"] is True
    assert report["cacheAvailable"] is False
    assert report["downloadRequired"] is True


def test_stt_cache_uses_shared_runtime_instead_of_standalone_speaking_model(tmp_path: Path) -> None:
    from app.core.config import settings

    with patch.object(settings, "AI_VOICE_STT_MODEL_NAME", "base"), patch.object(
        settings, "AI_VOICE_STT_MODEL_REVISION", "shared-revision"
    ), patch.object(settings, "AI_SPEAKING_STT_MODEL_NAME", "tiny"), patch(
        "faster_whisper.utils.download_model", return_value=str(tmp_path)
    ) as lookup:
        report = inspect_local_stt_cache()
    assert lookup.call_args.args == ("base",)
    assert lookup.call_args.kwargs["revision"] == "shared-revision"
    assert lookup.call_args.kwargs["local_files_only"] is True
    assert report["configuredModel"] == "base"
    assert report["modelRevision"] == "shared-revision"


@pytest.mark.asyncio
async def test_shared_runtime_keeps_listening_detection_and_speaking_forced_language(tmp_path: Path) -> None:
    from app.features.speech_to_text import WhisperRuntimeResult

    runtime = SimpleNamespace(
        ready=True, model_name="base", transcribe=AsyncMock(return_value=WhisperRuntimeResult(
            text="こんにちは", language="ja", language_probability=0.99,
            duration_seconds=1.2, segments=[], provider="faster-whisper",
            model="base", model_version="base@faster-whisper-1.2.1")),
    )
    ledger = CampaignLedger(tmp_path / "budget.json", "audio-test")
    listening = make_budgeted_stt_provider(runtime, ledger, phase="listening", mode="listening")
    speaking = make_budgeted_stt_provider(runtime, ledger, phase="speaking", mode="speaking")
    await listening.transcribe(make_wav(1.2), language="ja")
    await speaking.transcribe(make_wav(1.2), language="ja")
    assert runtime.transcribe.call_args_list[0].kwargs["options"]["language"] is None
    assert runtime.transcribe.call_args_list[1].kwargs["options"]["language"] == "ja"
    calls = ledger.snapshot()["calls"]
    assert [item["model"] for item in calls] == ["base", "base"]
    assert [item["accounted"]["stt_seconds"] for item in calls] == [1.2, 2.4]


@pytest.mark.asyncio
async def test_missing_stt_cache_does_not_prepare_runtime_or_download(tmp_path: Path) -> None:
    ledger = CampaignLedger(tmp_path / "budget.json", "audio-test")
    report = {"cacheAvailable": False, "downloadRequired": True, "configuredModel": "base"}
    with patch("scripts.qa_campaign_audio_budget.inspect_local_stt_cache", return_value=report), patch(
        "app.features.speech_to_text.FasterWhisperRuntime.warm_up", new=AsyncMock()
    ) as warm:
        provider, runtime, saved = await prepare_local_stt(ledger, phase="speaking", report_path=tmp_path / "stt.json")
    assert provider is None and runtime is None
    assert saved == report
    warm.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["tts", "stt"])
@pytest.mark.parametrize("after_reservation", [False, True])
async def test_audio_expired_deadline_does_not_enter_upstream(tmp_path: Path, kind, after_reservation) -> None:
    ledger = CampaignLedger(tmp_path / "budget.json", "audio-test")
    if kind == "tts":
        upstream, _, generated = _gemini_with_fake_transport(_response())
        provider = BudgetedSpeechProvider(upstream, ledger, phase="test")

        async def invoke():
            return await provider.synthesize_speech(text="hello", voice="Kore", language="en", speed="NORMAL")
        # TTS checks once before and once after its serial capture lock.
        times = [1, 1, 0] if after_reservation else [0]
    else:
        generated = AsyncMock(return_value="transcript")
        provider = BudgetedSttProvider(SimpleNamespace(transcribe=generated), ledger, phase="test")

        async def invoke():
            return await provider.transcribe(make_wav(1.2), language="ja")
        times = [1, 0] if after_reservation else [0]
    with patch.object(ledger, "remaining_seconds", side_effect=times):
        with pytest.raises(CampaignBudgetExceeded, match="CAMPAIGN_DEADLINE"):
            await invoke()
    generated.assert_not_called()
    state = ledger.snapshot()
    assert len(state["calls"]) == int(after_reservation)
    assert state["stoppedReason"] == "CAMPAIGN_DEADLINE"
    if after_reservation:
        assert state["calls"][0]["status"] == "FAILED"
        assert state["calls"][0]["usageStatus"] == "UNKNOWN_RESERVED"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["tts", "stt"])
async def test_audio_swallowed_timeout_cannot_succeed_or_start_another_stage(tmp_path: Path, kind) -> None:
    ledger = CampaignLedger(tmp_path / "budget.json", "audio-test")

    async def ignores_cancel(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return _response() if kind == "tts" else "transcript"

    if kind == "tts":
        upstream, models, generated = _gemini_with_fake_transport(_response())
        generated.side_effect = ignores_cancel
        provider = BudgetedSpeechProvider(upstream, ledger, phase="test", call_seconds=.005)
        with pytest.raises(TimeoutError):
            await provider.synthesize_speech(text="hello", voice="Kore", language="en", speed="NORMAL")
        assert models.generate_content is generated
        receipt = json.loads((tmp_path / "audio-budget-receipts/tts-0001.json").read_text())
        assert receipt["status"] == "FAILED"
        assert receipt["partial"] is True
        assert receipt["receipts"][0]["status"] == "COMPLETED"
    else:
        generated = AsyncMock(side_effect=ignores_cancel)
        provider = BudgetedSttProvider(SimpleNamespace(transcribe=generated), ledger, phase="test", call_seconds=.005)
        with pytest.raises(TimeoutError):
            await provider.transcribe(make_wav(1.2), language="ja")
    assert generated.call_count == 1
    state = ledger.snapshot()
    assert state["calls"][0]["status"] == "FAILED"
    assert state["calls"][0]["failureType"] == "TimeoutError"
    assert state["calls"][0]["usageStatus"] == "UNKNOWN_RESERVED"
    assert state["stoppedReason"] is None
    assert ledger.run_stop_reason == "QA_CALL_TIMEOUT"
    next_stage = AsyncMock()
    with pytest.raises(CampaignBudgetExceeded, match="QA_CALL_TIMEOUT"):
        await BudgetedSttProvider(SimpleNamespace(transcribe=next_stage), ledger, phase="same-run").transcribe(
            make_wav(1.2), language="ja")
    next_stage.assert_not_called()
    assert len(ledger.snapshot()["calls"]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["tts", "stt"])
async def test_audio_swallowed_cancellation_still_flushes_cancelled_attempt(tmp_path: Path, kind) -> None:
    ledger = CampaignLedger(tmp_path / "budget.json", "audio-test")
    started = asyncio.Event()

    async def ignores_cancel(*args, **kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return _response() if kind == "tts" else "transcript"

    if kind == "tts":
        upstream, _, generated = _gemini_with_fake_transport(_response())
        generated.side_effect = ignores_cancel
        speech = BudgetedSpeechProvider(upstream, ledger, phase="test")
        task = asyncio.create_task(speech.synthesize_speech(text="hello", voice="Kore", language="en", speed="NORMAL"))
    else:
        generated = AsyncMock(side_effect=ignores_cancel)
        stt = BudgetedSttProvider(SimpleNamespace(transcribe=generated), ledger, phase="test")
        task = asyncio.create_task(stt.transcribe(make_wav(1.2), language="ja"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    state = ledger.snapshot()
    assert generated.call_count == 1
    assert state["calls"][0]["status"] == "CANCELLED"
    assert state["calls"][0]["failureType"] == "CancelledError"
    assert state["calls"][0]["usageStatus"] == "UNKNOWN_RESERVED"
    assert state["stoppedReason"] is None
    assert ledger.run_stop_reason == "QA_EXECUTION_INTERRUPTED"
