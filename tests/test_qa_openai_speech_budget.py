from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from scripts.qa_campaign_audio_budget import BudgetedOpenAISpeechProvider
from scripts.qa_campaign_budget import CampaignBudgetExceeded, CampaignLedger


def _fake_speech() -> SimpleNamespace:
    return SimpleNamespace(
        provider_name="openai", ready=True,
        synthesize_speech=AsyncMock(return_value=SimpleNamespace(
            audio_bytes=b"fake-wav-only-for-budget-boundary",
            duration_seconds=2.0,
        )),
        warm_up=AsyncMock(), shutdown=AsyncMock(),
    )


@pytest.mark.asyncio
async def test_openai_speech_keeps_unknown_cost_reservation_after_success(tmp_path: Path) -> None:
    upstream = _fake_speech()
    ledger = CampaignLedger(tmp_path / "ledger.json", "new", prior_ledger_sha256="prior-sha")
    await BudgetedOpenAISpeechProvider(upstream, ledger, phase="VOICE_SELECT").synthesize_speech(
        text="こんにちは", voice="marin", language="ja", speed="NORMAL")
    entry = ledger.snapshot()["calls"][0]
    assert entry["task"] == "OPENAI_TTS"
    assert entry["status"] == "COMPLETED"
    assert entry["usageStatus"] == "UNKNOWN_RESERVED"
    assert entry["accounted"] is None and entry["reserved"]["usd"] >= 0.5
    assert ledger.snapshot()["priorLedgerSha256"] == "prior-sha"
    assert upstream.synthesize_speech.await_count == 1


@pytest.mark.asyncio
async def test_openai_speech_budget_denial_prevents_upstream_start(tmp_path: Path) -> None:
    upstream = _fake_speech()
    ledger = CampaignLedger(tmp_path / "ledger.json", "new", limits={
        "starts": 2000, "input_tokens": 8_000_000, "output_tokens": 2_000_000,
        "tts_characters": 200_000, "stt_seconds": 7200, "usd": 0.2,
        "wall_seconds": 604_800, "active_seconds": 43_200,
    })
    with pytest.raises(CampaignBudgetExceeded, match="RESERVATION_DENIED_USD"):
        await BudgetedOpenAISpeechProvider(upstream, ledger, phase="test").synthesize_speech(
            text="こんにちは", voice="marin", language="ja", speed="NORMAL")
    upstream.synthesize_speech.assert_not_awaited()
    assert ledger.snapshot()["stoppedReason"] == "RESERVATION_DENIED_USD"


@pytest.mark.asyncio
async def test_openai_speech_timeout_keeps_unknown_reservation_and_stops_followup(tmp_path: Path) -> None:
    upstream = _fake_speech()
    async def blocked(**_: object) -> None:
        await asyncio.Event().wait()
    upstream.synthesize_speech.side_effect = blocked
    ledger = CampaignLedger(tmp_path / "ledger.json", "new")
    provider = BudgetedOpenAISpeechProvider(upstream, ledger, phase="test", call_seconds=0.01)
    with pytest.raises(TimeoutError):
        await provider.synthesize_speech(text="こんにちは", voice="marin", language="ja", speed="NORMAL")
    assert upstream.synthesize_speech.await_count == 1
    assert ledger.snapshot()["calls"][0]["usageStatus"] == "UNKNOWN_RESERVED"
    with pytest.raises(CampaignBudgetExceeded, match="QA_CALL_TIMEOUT"):
        await provider.synthesize_speech(text="こんにちは", voice="marin", language="ja", speed="NORMAL")
    assert upstream.synthesize_speech.await_count == 1


def test_gemini_speech_upstream_is_rejected_at_wrapper_construction(tmp_path: Path) -> None:
    ledger = CampaignLedger(tmp_path / "ledger.json", "new")
    with pytest.raises(CampaignBudgetExceeded, match="GEMINI_GENERATION_DISABLED"):
        BudgetedOpenAISpeechProvider(SimpleNamespace(provider_name="gemini"), ledger, phase="test")
