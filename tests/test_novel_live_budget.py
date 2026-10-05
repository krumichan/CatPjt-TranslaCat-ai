import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.ai.model_policy import AiModelTier
from app.core.config import settings
from scripts.novel_live_verification import (
    LIMITS,
    BudgetedExplicitProvider,
    fixed_corpus,
    provider_slot,
)
from scripts.qa_campaign_budget import CampaignBudgetExceeded, CampaignLedger


def _command():
    return {
        "tier": AiModelTier.LUNA,
        "max_output_tokens": 8192,
        "reasoning_effort": "none",
        "instructions": "synthetic",
        "messages": [{"role": "user", "content": "synthetic"}],
    }


@pytest.mark.asyncio
async def test_budget_failure_keeps_full_reservation_across_restart(tmp_path, monkeypatch):
    # 준비
    monkeypatch.setattr(settings, "AI_TEXT_PROVIDER", "openai")
    monkeypatch.setattr(settings, "OPENAI_MODEL_LUNA", "gpt-5.6-luna")
    path = tmp_path / "ledger.json"
    ledger = CampaignLedger(path, "novel-test", limits=LIMITS)
    upstream = SimpleNamespace(execute_explicit=AsyncMock(side_effect=TimeoutError))
    provider = BudgetedExplicitProvider(upstream, ledger, slot_root=tmp_path)

    # 실행
    with pytest.raises(TimeoutError):
        await provider.execute_explicit(**_command())
    resumed = CampaignLedger(path, "novel-test", limits=LIMITS)

    # 검증
    result = resumed.snapshot()
    assert result["totalsIncludingReserved"]["starts"] == 1
    assert result["totalsIncludingReserved"]["usd"] > 0.03
    assert result["calls"][0]["accounted"] is None
    assert result["calls"][0]["status"] == "FAILED"


@pytest.mark.asyncio
async def test_budget_rejects_over_limit_before_paid_call(tmp_path, monkeypatch):
    # 준비
    monkeypatch.setattr(settings, "AI_TEXT_PROVIDER", "openai")
    monkeypatch.setattr(settings, "OPENAI_MODEL_LUNA", "gpt-5.6-luna")
    ledger = CampaignLedger(tmp_path / "ledger.json", "novel-test", limits={**LIMITS, "usd": 0.001})
    upstream = SimpleNamespace(execute_explicit=AsyncMock())

    # 실행 / 검증
    with pytest.raises(CampaignBudgetExceeded, match="RESERVATION_DENIED_USD"):
        await BudgetedExplicitProvider(upstream, ledger, slot_root=tmp_path).execute_explicit(
            **_command()
        )
    upstream.execute_explicit.assert_not_awaited()
    assert len(ledger.snapshot()["calls"]) == 0


@pytest.mark.asyncio
async def test_only_openai_provider_and_priced_models_are_allowed(tmp_path, monkeypatch):
    # 준비
    ledger = CampaignLedger(tmp_path / "ledger.json", "novel-test", limits=LIMITS)
    upstream = SimpleNamespace(execute_explicit=AsyncMock())
    provider = BudgetedExplicitProvider(upstream, ledger, slot_root=tmp_path)
    monkeypatch.setattr(settings, "AI_TEXT_PROVIDER", "gemini")

    # 실행 / 검증
    with pytest.raises(CampaignBudgetExceeded, match="OPENAI_ONLY_REQUIRED"):
        await provider.execute_explicit(**_command())
    monkeypatch.setattr(settings, "AI_TEXT_PROVIDER", "openai")
    monkeypatch.setattr(settings, "OPENAI_MODEL_LUNA", "unpriced-model")
    with pytest.raises(CampaignBudgetExceeded, match="UNPRICED_MODEL"):
        await provider.execute_explicit(**_command())
    upstream.execute_explicit.assert_not_awaited()


@pytest.mark.asyncio
async def test_global_file_slots_limit_parallel_calls(tmp_path):
    # 준비
    active = maximum = 0

    async def work():
        nonlocal active, maximum
        async with provider_slot(tmp_path):
            active += 1
            maximum = max(active, maximum)
            await asyncio.sleep(0.03)
            active -= 1

    # 실행
    await asyncio.gather(*(work() for _ in range(6)))

    # 검증
    assert maximum == 2


def test_fixed_corpus_contains_long_cases_and_stable_unique_ids():
    # 실행
    corpus = fixed_corpus()

    # 검증
    assert [case["targetCharacters"] for case in corpus["cases"]] == [500, 2000, 5000, 10000]
    for case in corpus["cases"]:
        assert case["characters"] >= case["targetCharacters"]
        assert len({item["id"] for item in case["items"]}) == len(case["items"])
    assert LIMITS["starts"] == 200
    assert LIMITS["usd"] == 10
