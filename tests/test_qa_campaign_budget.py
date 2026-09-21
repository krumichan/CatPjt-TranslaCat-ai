import asyncio
import json
from types import SimpleNamespace

import pytest

from scripts.qa_campaign_budget import (
    LIMITS, BudgetedTextProvider, CampaignBudgetExceeded, CampaignLedger, Reservation,
    _text_cost_reservation,
)


def test_budget_checkpoint_retries_transient_windows_sharing_violation(tmp_path, monkeypatch):
    from scripts import qa_campaign_budget as budget

    path = tmp_path / "budget-ledger.json"
    budget.atomic_json(path, {"calls": [1]})
    original = budget.os.replace
    starts = 0

    def sharing_once(source, destination):
        nonlocal starts
        starts += 1
        if starts == 1:
            raise PermissionError(5, "temporary sharing violation")
        return original(source, destination)

    monkeypatch.setattr(budget.os, "replace", sharing_once)
    budget.atomic_json(path, {"calls": [1, 2]})
    assert starts == 2
    assert json.loads(path.read_text(encoding="utf-8"))["calls"] == [1, 2]


def test_sol_reservation_uses_its_own_conservative_price_not_small_model_price():
    assert _text_cost_reservation("gpt-5.6-sol", 1_000_000, 1_000_000) == 30
    assert _text_cost_reservation("gpt-5.6-luna", 1_000_000, 1_000_000) == 5
    with pytest.raises(CampaignBudgetExceeded, match="UNPRICED_MODEL"):
        _text_cost_reservation("unknown", 1, 1)


def test_reservation_survives_restart_and_unknown_usage(tmp_path):
    path = tmp_path / "budget-ledger.json"
    first = CampaignLedger(path, "fixed")
    n = first.reserve("text", "model", Reservation(10, 20, usd=19.9))
    first.finish(n, status="FAILED", elapsed=1, failure="CancelledError")
    second = CampaignLedger(path, "fixed")
    assert second.snapshot()["totalsIncludingReserved"]["usd"] == 19.9
    with pytest.raises(CampaignBudgetExceeded):
        second.reserve("text", "model", Reservation(usd=.2))
    assert len(second.snapshot()["calls"]) == 1


@pytest.mark.parametrize("resource", list(Reservation.__dataclass_fields__))
def test_reservation_denial_stops_all_tasks_and_survives_restart(tmp_path, resource):
    path = tmp_path / "ledger.json"
    ledger = CampaignLedger(path, "fixed")
    concurrent_ledger = CampaignLedger(path, "fixed")
    attempt = ledger.reserve("unknown-audio", "fake", Reservation(**{resource: LIMITS[resource]}))
    ledger.finish(attempt, status="FAILED", elapsed=0, failure="SyntheticUnknownUsage")
    before = ledger.snapshot()
    reason = "RESERVATION_DENIED_" + resource.upper()
    with pytest.raises(CampaignBudgetExceeded, match=reason):
        ledger.reserve("too-expensive", "fake", Reservation(**{resource: 1}))
    after = ledger.snapshot()
    assert after["stoppedReason"] == reason
    assert after["calls"] == before["calls"]
    assert after["totalsIncludingReserved"] == before["totalsIncludingReserved"]
    for instance in (ledger, concurrent_ledger, CampaignLedger(path, "fixed")):
        with pytest.raises(CampaignBudgetExceeded, match=reason):
            instance.reserve("otherwise-free-followup", "fake", Reservation())
    assert len(ledger.snapshot()["calls"]) == 1


def test_start_limit_denial_is_durable_without_counting_rejected_start(tmp_path, monkeypatch):
    from scripts import qa_campaign_budget as budget

    # Exercise the exact boundary cheaply; the production constant remains 600.
    monkeypatch.setattr(budget, "LIMITS", {**LIMITS, "starts": 1})
    path = tmp_path / "ledger.json"
    ledger = CampaignLedger(path, "fixed")
    attempt = ledger.reserve("first", "fake", Reservation())
    ledger.finish(attempt, status="COMPLETED", elapsed=0, accounted=Reservation())
    with pytest.raises(CampaignBudgetExceeded, match="APPLICATION_START_LIMIT"):
        ledger.reserve("denied", "fake", Reservation())
    assert ledger.snapshot()["stoppedReason"] == "APPLICATION_START_LIMIT"
    with pytest.raises(CampaignBudgetExceeded, match="APPLICATION_START_LIMIT"):
        CampaignLedger(path, "fixed").reserve("after-restart", "fake", Reservation())
    assert len(ledger.snapshot()["calls"]) == 1


@pytest.mark.parametrize("settled_usd", [.1, 20.1])
def test_late_usage_settlement_neither_reopens_nor_overwrites_denial(tmp_path, settled_usd):
    ledger = CampaignLedger(tmp_path / "ledger.json", "fixed")
    active = ledger.reserve("in-flight", "fake", Reservation(usd=19.9))
    with pytest.raises(CampaignBudgetExceeded, match="RESERVATION_DENIED_USD"):
        ledger.reserve("too-expensive", "fake", Reservation(usd=.2))
    ledger.finish(active, status="COMPLETED", elapsed=1, accounted=Reservation(usd=settled_usd))
    state = ledger.snapshot()
    assert state["stoppedReason"] == "RESERVATION_DENIED_USD"
    assert state["totalsIncludingReserved"]["usd"] == settled_usd
    with pytest.raises(CampaignBudgetExceeded, match="RESERVATION_DENIED_USD"):
        ledger.reserve("cheap-followup", "fake", Reservation())
    assert len(ledger.snapshot()["calls"]) == 1


@pytest.mark.asyncio
async def test_tts_reservation_denial_blocks_otherwise_affordable_text_provider(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock

    from app.core.config import settings
    from scripts.qa_campaign_audio_budget import BudgetedSpeechProvider, GEMINI_TTS_MODEL

    monkeypatch.setattr(settings, "AI_TEXT_PROVIDER", "openai")
    monkeypatch.setattr(settings, "GEMINI_MODEL_TTS", GEMINI_TTS_MODEL)
    ledger = CampaignLedger(tmp_path / "ledger.json", "fixed")
    attempt = ledger.reserve("previous-unknown", "fake", Reservation(usd=19.9))
    ledger.finish(attempt, status="FAILED", elapsed=0, failure="SyntheticUnknownUsage")
    generated = AsyncMock()
    models = SimpleNamespace(generate_content=generated,
                             _api_client=SimpleNamespace(_http_options=SimpleNamespace(retry_options=None)))
    speech = SimpleNamespace(client=SimpleNamespace(aio=SimpleNamespace(models=models)),
                             synthesize_speech=AsyncMock())
    with pytest.raises(CampaignBudgetExceeded, match="RESERVATION_DENIED_USD"):
        await BudgetedSpeechProvider(speech, ledger, phase="offline").synthesize_speech(
            text="synthetic", voice="Kore", language="en", speed="NORMAL")
    text = SimpleNamespace(call_with_metadata=AsyncMock())
    with pytest.raises(CampaignBudgetExceeded, match="RESERVATION_DENIED_USD"):
        await BudgetedTextProvider(text, ledger, phase="offline").call(
            "LANGUAGE_LEARNING_PRACTICE_GENERATION", "synthetic")
    speech.synthesize_speech.assert_not_called()
    generated.assert_not_called()
    text.call_with_metadata.assert_not_called()
    assert len(ledger.snapshot()["calls"]) == 1
    assert not (tmp_path / "audio-budget-receipts").exists()


def test_success_settles_and_overage_stops(tmp_path):
    ledger = CampaignLedger(tmp_path / "ledger.json", "fixed")
    n = ledger.reserve("text", "model", Reservation(100, 200, usd=.5))
    ledger.finish(n, status="COMPLETED", elapsed=1, accounted=Reservation(10, 20, usd=.1))
    n = ledger.reserve("text", "model", Reservation(10, 20, usd=.1))
    ledger.finish(n, status="COMPLETED", elapsed=1, accounted=Reservation(11, 20, usd=.1))
    with pytest.raises(CampaignBudgetExceeded, match="USAGE_EXCEEDED"):
        ledger.reserve("text", "model", Reservation())


@pytest.mark.asyncio
async def test_wrapper_timeout_flushes_without_second_start(tmp_path, monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "AI_TEXT_PROVIDER", "openai")
    calls = []

    async def call_with_metadata(**kwargs):
        calls.append(kwargs)
        await asyncio.Event().wait()

    ledger = CampaignLedger(tmp_path / "ledger.json", "fixed")
    provider = BudgetedTextProvider(SimpleNamespace(call_with_metadata=call_with_metadata), ledger, phase="test", call_seconds=.005)
    with pytest.raises(TimeoutError):
        await provider.call("LANGUAGE_LEARNING_PRACTICE_GENERATION", "safe synthetic")
    state = ledger.snapshot()
    assert len(calls) == 1
    assert state["calls"][0]["status"] == "FAILED"
    assert state["calls"][0]["failureType"] == "TimeoutError"
    assert state["calls"][0]["usageStatus"] == "UNKNOWN_RESERVED"
    assert state["stoppedReason"] is None
    assert ledger.run_stop_reason == "QA_CALL_TIMEOUT"
    with pytest.raises(CampaignBudgetExceeded, match="QA_CALL_TIMEOUT"):
        await provider.call("LANGUAGE_LEARNING_PRACTICE_GENERATION", "safe synthetic")
    assert len(calls) == 1


def test_campaign_identity_cannot_change(tmp_path):
    path = tmp_path / "ledger.json"
    CampaignLedger(path, "fixed")
    with pytest.raises(ValueError, match="identity"):
        CampaignLedger(path, "other")


@pytest.mark.asyncio
async def test_expired_before_reservation_never_starts_provider(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock

    upstream = SimpleNamespace(call_with_metadata=AsyncMock())
    ledger = CampaignLedger(tmp_path / "ledger.json", "fixed")
    monkeypatch.setattr(ledger, "remaining_seconds", lambda: 0)
    provider = BudgetedTextProvider(upstream, ledger, phase="test")
    with pytest.raises(CampaignBudgetExceeded, match="CAMPAIGN_DEADLINE"):
        await provider.call("LANGUAGE_LEARNING_PRACTICE_GENERATION", "synthetic")
    upstream.call_with_metadata.assert_not_called()
    assert ledger.snapshot()["calls"] == []
    assert ledger.snapshot()["stoppedReason"] == "CAMPAIGN_DEADLINE"


@pytest.mark.asyncio
@pytest.mark.parametrize("remaining", [[1, 0], [1, 1, 0]])
async def test_expiry_during_reservation_or_context_entry_keeps_unknown_charge_without_start(tmp_path, monkeypatch, remaining):
    from unittest.mock import AsyncMock, Mock

    upstream = SimpleNamespace(call_with_metadata=AsyncMock())
    ledger = CampaignLedger(tmp_path / "ledger.json", "fixed")
    monkeypatch.setattr(ledger, "remaining_seconds", Mock(side_effect=remaining))
    provider = BudgetedTextProvider(upstream, ledger, phase="test")
    with pytest.raises(CampaignBudgetExceeded, match="CAMPAIGN_DEADLINE"):
        await provider.call("LANGUAGE_LEARNING_PRACTICE_GENERATION", "synthetic")
    upstream.call_with_metadata.assert_not_called()
    calls = ledger.snapshot()["calls"]
    assert len(calls) == 1
    assert calls[0]["status"] == "FAILED"
    assert calls[0].get("accounted") is None
    assert calls[0]["usageStatus"] == "UNKNOWN_RESERVED"


@pytest.mark.asyncio
async def test_swallowed_timeout_is_failure_and_halts_only_current_run(tmp_path):
    calls = []

    async def ignores_cancel(**kwargs):
        calls.append(kwargs)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return SimpleNamespace(data={}, input_tokens=1, output_tokens=1)

    path = tmp_path / "ledger.json"
    ledger = CampaignLedger(path, "fixed")
    upstream = SimpleNamespace(call_with_metadata=ignores_cancel)
    provider = BudgetedTextProvider(upstream, ledger, phase="test", call_seconds=.005)
    with pytest.raises(TimeoutError):
        await provider.call("LANGUAGE_LEARNING_PRACTICE_GENERATION", "synthetic")
    state = ledger.snapshot()
    assert state["calls"][0]["status"] == "FAILED"
    assert state["calls"][0]["failureType"] == "TimeoutError"
    assert state["calls"][0]["usageStatus"] == "UNKNOWN_RESERVED"
    assert state["stoppedReason"] is None
    with pytest.raises(CampaignBudgetExceeded, match="QA_CALL_TIMEOUT"):
        await BudgetedTextProvider(upstream, ledger, phase="same-run").call(
            "LANGUAGE_LEARNING_PRACTICE_GENERATION", "synthetic")
    assert len(calls) == 1
    next_authorized_run = CampaignLedger(path, "fixed")
    assert next_authorized_run.run_stop_reason is None
    assert next_authorized_run.snapshot()["totalsIncludingReserved"] == state["totalsIncludingReserved"]


@pytest.mark.asyncio
async def test_swallowed_external_cancellation_stays_cancelled_and_blocks_followup(tmp_path):
    started = asyncio.Event()
    calls = []

    async def ignores_cancel(**kwargs):
        calls.append(kwargs)
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return SimpleNamespace(data={}, input_tokens=1, output_tokens=1)

    ledger = CampaignLedger(tmp_path / "ledger.json", "fixed")
    provider = BudgetedTextProvider(SimpleNamespace(call_with_metadata=ignores_cancel), ledger, phase="test")
    task = asyncio.create_task(provider.call("LANGUAGE_LEARNING_PRACTICE_GENERATION", "synthetic"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    state = ledger.snapshot()
    assert state["calls"][0]["status"] == "CANCELLED"
    assert state["calls"][0]["usageStatus"] == "UNKNOWN_RESERVED"
    assert state["stoppedReason"] is None
    with pytest.raises(CampaignBudgetExceeded, match="QA_EXECUTION_INTERRUPTED"):
        await provider.call("LANGUAGE_LEARNING_PRACTICE_GENERATION", "synthetic")
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_intrinsic_provider_timeout_does_not_change_existing_retry_policy(tmp_path):
    from unittest.mock import AsyncMock

    call = AsyncMock(side_effect=[TimeoutError(), SimpleNamespace(data={}, input_tokens=1, output_tokens=1)])
    ledger = CampaignLedger(tmp_path / "ledger.json", "fixed")
    provider = BudgetedTextProvider(SimpleNamespace(call_with_metadata=call), ledger, phase="test")
    with pytest.raises(TimeoutError):
        await provider.call("LANGUAGE_LEARNING_PRACTICE_GENERATION", "synthetic")
    assert ledger.run_stop_reason is None
    assert await provider.call("LANGUAGE_LEARNING_PRACTICE_GENERATION", "synthetic") == {}
    assert call.call_count == 2


@pytest.mark.asyncio
async def test_request_scoped_cancellation_preserves_real_service_timeout_retry(tmp_path, monkeypatch):
    from app.core.config import settings
    from app.features.language_learning.speaking.assistance_service import SpeakingAssistanceService
    from app.schemas.language_learning_speaking import AssistanceRequest
    from scripts.run_contextual_choice_stabilization_qa import QaCaps, RecordingProvider

    monkeypatch.setattr(settings, "AI_TEXT_PROVIDER", "openai")
    starts = []

    async def call_with_metadata(**kwargs):
        starts.append(kwargs)
        if len(starts) == 1:
            await asyncio.Event().wait()
        return SimpleNamespace(data={"type": "HINT", "content": "Synthetic hint"},
                               input_tokens=1, output_tokens=1, provider="openai", model="gpt-5-mini")

    ledger = CampaignLedger(tmp_path / "ledger.json", "fixed", stop_on_caller_cancel=False)
    budgeted = BudgetedTextProvider(SimpleNamespace(call_with_metadata=call_with_metadata), ledger, phase="test")
    recorder = RecordingProvider(budgeted, QaCaps(12, 120000, 40000, 90, 600),
                                 stop_on_caller_cancel=False)
    # Use the production wait_for + retry implementation, accelerating only its
    # timeout. Both QA deadlines remain 90s and must not claim they expired.
    service = SpeakingAssistanceService(recorder, timeout_seconds=.1, automatic_retries=1)
    request = AssistanceRequest.model_validate({
        "requestId": "offline", "idempotencyKey": "offline-idem", "sessionId": "offline-session",
        "turnIndex": 2, "assistanceType": "HINT", "originLanguage": "ko", "learningLanguage": "ja",
        "topic": "Synthetic test", "targetLevel": "A2", "assistantText": "Synthetic question",
        "conversationHistory": [], "selectedKeywords": [],
    })
    response = await service.generate(request)
    assert response.content == "Synthetic hint"
    assert len(starts) == 2
    assert ledger.run_stop_reason is None
    assert recorder.budget_exhausted_reason is None
    calls = ledger.snapshot()["calls"]
    assert [call["status"] for call in calls] == ["CANCELLED", "COMPLETED"]
    assert calls[0]["failureType"] == "CancelledError"
    assert calls[0]["usageStatus"] == "UNKNOWN_RESERVED"
    assert calls[0]["accounted"] is None
    assert [call["status"] for call in recorder.calls] == ["CANCELLED", "SUCCEEDED"]
    assert recorder.calls[0]["failure"]["source"] == "EXTERNAL_OR_UPSTREAM_UNDETERMINED"


@pytest.mark.asyncio
async def test_request_scoped_opt_in_still_halts_on_qa_own_timeout(tmp_path):
    starts = []

    async def call_with_metadata(**kwargs):
        starts.append(kwargs)
        await asyncio.Event().wait()

    ledger = CampaignLedger(tmp_path / "ledger.json", "fixed", stop_on_caller_cancel=False)
    provider = BudgetedTextProvider(SimpleNamespace(call_with_metadata=call_with_metadata), ledger,
                                    phase="test", call_seconds=.005)
    with pytest.raises(TimeoutError):
        await provider.call("LANGUAGE_LEARNING_PRACTICE_GENERATION", "synthetic")
    assert ledger.run_stop_reason == "QA_CALL_TIMEOUT"
    with pytest.raises(CampaignBudgetExceeded, match="QA_CALL_TIMEOUT"):
        await provider.call("LANGUAGE_LEARNING_PRACTICE_GENERATION", "synthetic")
    assert len(starts) == 1


@pytest.mark.asyncio
async def test_request_scoped_opt_in_rejects_still_cancelled_task_only(tmp_path):
    from unittest.mock import AsyncMock

    ledger = CampaignLedger(tmp_path / "ledger.json", "fixed", stop_on_caller_cancel=False)
    call = AsyncMock(return_value=SimpleNamespace(data={}, input_tokens=1, output_tokens=1))
    provider = BudgetedTextProvider(SimpleNamespace(call_with_metadata=call), ledger, phase="test")

    async def cancelled_caller():
        task = asyncio.current_task()
        assert task is not None
        task.cancel()
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            pass
        with pytest.raises(asyncio.CancelledError):
            await provider.call("LANGUAGE_LEARNING_PRACTICE_GENERATION", "synthetic")

    await asyncio.create_task(cancelled_caller())
    call.assert_not_called()
    assert ledger.snapshot()["calls"] == []
    assert ledger.run_stop_reason is None
    # An independent request can continue; no interruption of its task occurred.
    assert await provider.call("LANGUAGE_LEARNING_PRACTICE_GENERATION", "synthetic") == {}
    assert call.call_count == 1


@pytest.mark.asyncio
async def test_request_scoped_opt_in_keeps_keyboard_interrupt_sticky(tmp_path):
    from scripts.qa_campaign_budget import campaign_call_boundary

    ledger = CampaignLedger(tmp_path / "ledger.json", "fixed", stop_on_caller_cancel=False)
    with pytest.raises(KeyboardInterrupt):
        async with campaign_call_boundary(ledger, 90):
            raise KeyboardInterrupt
    assert ledger.run_stop_reason == "QA_EXECUTION_INTERRUPTED"
