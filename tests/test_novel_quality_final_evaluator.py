from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from scripts import novel_quality_verification as verification
from scripts.qa_campaign_budget import CampaignBudgetExceeded


def fake_runner(monkeypatch, tmp_path):
    from app.ai.providers.openai import client

    create = AsyncMock(side_effect=TimeoutError("synthetic network stub"))
    sdk = Mock()
    sdk.with_options.return_value = SimpleNamespace(
        responses=SimpleNamespace(with_raw_response=SimpleNamespace(create=create))
    )
    monkeypatch.setattr(client, "OpenAIService", lambda: SimpleNamespace(client=sdk))
    ledger = verification.QualityLedger(tmp_path / "ledger.json")
    return verification.LiveResponses(phase="final", concurrency=1, ledger=ledger), create, ledger


@pytest.mark.asyncio
async def test_final_independent_evaluator_keeps_final_budget_phase_and_evaluator_price(
    monkeypatch, tmp_path
):
    # 준비: 실제 네트워크 대신 예외를 내는 stub으로 가격 예약까지만 검증한다.
    runner, create, ledger = fake_runner(monkeypatch, tmp_path)
    # 실행 / 검증: 최종 평가를 탐색으로 위장하지 않고도 평가 모델 가격이 적용된다.
    with pytest.raises(TimeoutError):
        await runner.create(
            task="blind-independent-quality-judge",
            model="gpt-6-astra",
            effort="low",
            instructions="synthetic",
            messages=[{"role": "user", "content": "synthetic"}],
            max_output_tokens=128,
        )
    create.assert_awaited_once()
    call = ledger.snapshot()["calls"][0]
    assert call["metadata"]["phase"] == "final"
    assert call["status"] == "FAILED"
    assert call["accounted"] is None


@pytest.mark.asyncio
async def test_final_phase_does_not_open_evaluator_only_model_to_translation(monkeypatch, tmp_path):
    runner, create, ledger = fake_runner(monkeypatch, tmp_path)
    with pytest.raises(CampaignBudgetExceeded, match="UNPRICED_MODEL"):
        await runner.create(
            task="translation",
            model="gpt-6-astra",
            effort="low",
            instructions="synthetic",
            messages=[{"role": "user", "content": "synthetic"}],
            max_output_tokens=128,
        )
    create.assert_not_awaited()
    assert ledger.snapshot()["calls"] == []
