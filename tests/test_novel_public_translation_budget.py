from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.config import settings
from scripts.novel_quality_server import BudgetedLegacyExecutor
from scripts.novel_quality_verification import QualityLedger
from scripts.qa_campaign_budget import CampaignBudgetExceeded


@pytest.mark.asyncio
async def test_public_catalog_metadata_uses_existing_ledger_and_model(tmp_path, monkeypatch):
    # 준비: provider 진입 시점에 이미 비용 예약이 있는지 확인한다.
    monkeypatch.setattr(settings, "OPENAI_MODEL_LUNA", "gpt-6-luna")
    ledger = QualityLedger(tmp_path / "ledger.json")

    async def generate(**kwargs):
        assert ledger.snapshot()["calls"][0]["status"] == "STARTED"
        assert kwargs["type_name"] == "NOVEL"
        return SimpleNamespace(
            provider="openai",
            model="gpt-6-luna",
            data=["첫 회"],
            input_tokens=100,
            output_tokens=20,
        )

    upstream = SimpleNamespace(call_with_metadata=AsyncMock(side_effect=generate))

    # 실행
    result = await BudgetedLegacyExecutor(upstream, ledger, "final").call(
        type_name="NOVEL", data='["第一話"]', schema={}
    )

    # 검증
    assert result == ["첫 회"]
    call = ledger.snapshot()["calls"][0]
    assert call["task"] == "NOVEL_LEGACY"
    assert call["status"] == "COMPLETED"
    assert call["metadata"]["transport"] == "PUBLIC_TRANSLATE_BATCH"
    upstream.call_with_metadata.assert_awaited_once()


@pytest.mark.asyncio
async def test_public_failure_keeps_reservation_and_other_tasks_never_call_provider(
    tmp_path, monkeypatch
):
    # 준비
    monkeypatch.setattr(settings, "OPENAI_MODEL_LUNA", "gpt-6-luna")
    ledger = QualityLedger(tmp_path / "ledger.json")
    upstream = SimpleNamespace(call_with_metadata=AsyncMock(side_effect=TimeoutError))
    wrapper = BudgetedLegacyExecutor(upstream, ledger, "final")

    # 실행 / 검증: 미확정 과금은 예약을 유지하며 무관한 작업은 유료 진입 전에 막는다.
    with pytest.raises(TimeoutError):
        await wrapper.call(type_name="RANK", data="作品", schema=None)
    with pytest.raises(CampaignBudgetExceeded, match="QA_TRANSLATION_TASK_FORBIDDEN"):
        await wrapper.call(type_name="VOICE", data="音声", schema=None)
    call = ledger.snapshot()["calls"][0]
    assert call["status"] == "FAILED"
    assert call["accounted"] is None
    assert ledger.snapshot()["totalsIncludingReserved"]["usd"] > 0
    upstream.call_with_metadata.assert_awaited_once()
