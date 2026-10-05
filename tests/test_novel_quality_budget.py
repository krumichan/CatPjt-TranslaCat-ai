import asyncio
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock

import pytest

from scripts.novel_quality_verification import QualityLedger, preserve_response, text_reservation
from scripts.qa_campaign_budget import CampaignBudgetExceeded, Reservation


def test_final_reserve_survives_restart_and_exploration_denial(tmp_path):
    # 준비
    path = tmp_path / "ledger.json"
    ledger = QualityLedger(path)
    ledger.reserve("explore", "gpt-6-luna", Reservation(usd=30), metadata={"phase": "explore"})

    # 실행 / 검증: 탐색 한도 거절은 최종 검증을 중단시키지 않는다.
    resumed = QualityLedger(path)
    with pytest.raises(CampaignBudgetExceeded, match="FINAL_RESERVE_PROTECTED"):
        resumed.reserve(
            "explore", "gpt-6-luna", Reservation(usd=0.01), metadata={"phase": "explore"}
        )
    resumed.reserve("final", "gpt-6-luna", Reservation(usd=20), metadata={"phase": "final"})
    assert resumed.snapshot()["totalsIncludingReserved"]["usd"] == 50
    assert len(resumed.snapshot()["calls"]) == 2


def test_cross_instance_atomic_phase_admission(tmp_path):
    # 준비
    path = tmp_path / "ledger.json"
    QualityLedger(path)

    def reserve():
        try:
            return QualityLedger(path).reserve(
                "explore", "gpt-6-luna", Reservation(usd=10), metadata={"phase": "explore"}
            )
        except CampaignBudgetExceeded:
            return None

    # 실행
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: reserve(), range(8)))

    # 검증
    assert len([value for value in results if value]) == 3
    assert QualityLedger(path).snapshot()["totalsIncludingReserved"]["usd"] == 30


def test_unknown_usage_remains_reserved_and_unpriced_models_fail(tmp_path):
    # 준비
    ledger = QualityLedger(tmp_path / "ledger.json")
    amount = text_reservation("gpt-6-luna", 10000, 2000)
    attempt = ledger.reserve("smoke", "gpt-6-luna", amount, metadata={"phase": "smoke"})

    # 실행
    ledger.finish(attempt, status="FAILED", elapsed=0.2, failure="TimeoutError")

    # 검증
    assert QualityLedger(ledger.path).snapshot()["totalsIncludingReserved"]["usd"] == amount.usd
    with pytest.raises(CampaignBudgetExceeded, match="UNPRICED_MODEL"):
        text_reservation("unknown", 100, 100)


def test_stronger_evaluator_price_is_not_a_translation_model_allowlist():
    # 실행 / 검증: 표준 요금에 캐시 쓰기·안전 여유를 더하며 평가 호출만 별도 허용한다.
    with pytest.raises(CampaignBudgetExceeded, match="UNPRICED_MODEL"):
        text_reservation("gpt-6-astra", 1000, 1000)
    amount = text_reservation("gpt-6-astra", 1000, 1000, evaluator_only=True)
    assert amount.usd == pytest.approx((1000 * 10 * 1.25 + 1000 * 50) * 1.25 / 1_000_000)


@pytest.mark.asyncio
async def test_profile_wrapper_failure_is_one_persisted_attempt(tmp_path):
    from app.schemas.model_profile_execution import ModelProfileExecutionRequest
    from scripts.novel_quality_server import BudgetedProfileExecutor

    # 준비
    ledger = QualityLedger(tmp_path / "ledger.json")
    upstream = AsyncMock()
    upstream.execute.side_effect = TimeoutError
    command = ModelProfileExecutionRequest.model_validate(
        {
            "traceId": "child-1",
            "instructions": "Synthetic",
            "messages": [{"role": "user", "content": "value"}],
            "profileId": "luna-6-none-v1",
            "maxOutputTokens": 2048,
            "remainingMilliseconds": 30000,
            "maxProviderCalls": 1,
        }
    )

    # 실행
    with pytest.raises(TimeoutError):
        await BudgetedProfileExecutor(upstream, ledger).execute(
            command=command, parent_trace_id="parent"
        )

    # 검증
    upstream.execute.assert_awaited_once()
    call = QualityLedger(ledger.path).snapshot()["calls"][0]
    assert call["status"] == "FAILED"
    assert call["metadata"]["traceId"] == "parent"
    assert call["metadata"]["childTraceId"] == "child-1"
    assert call["accounted"] is None


@pytest.mark.asyncio
async def test_speech_full_global_slots_reject_without_queue_or_paid_admission(
    tmp_path, monkeypatch
):
    from filelock import FileLock

    from app.core.config import settings
    from scripts.novel_quality_server import BudgetedSpeech

    # 준비: 외부 작업이 모든 슬롯을 점유한다. 잠금 해제 후의 지연 시작도 허용하지 않는다.
    ledger = QualityLedger(tmp_path / "ledger.json")
    monkeypatch.setattr(settings, "OPENAI_SPEECH_MODEL", "gpt-4o-mini-tts-2025-12-15")
    upstream = AsyncMock()
    locks = [FileLock(str(tmp_path / f"provider-slot-{number}.lock")) for number in range(10)]
    for lock in locks:
        lock.acquire()
    try:
        # 실행
        with pytest.raises(Exception) as caught:
            await asyncio.wait_for(
                BudgetedSpeech(upstream, ledger).synthesize_speech(
                    text="合成音声テスト", voice="marin", language="ja", speed="NORMAL"
                ),
                timeout=0.2,
            )
    finally:
        for lock in locks:
            lock.release()

    # 검증
    assert getattr(caught.value, "status_code", None) == 429
    upstream.synthesize_speech.assert_not_awaited()
    assert ledger.snapshot()["totalsIncludingReserved"]["starts"] == 0


def test_raw_qa_response_preservation_excludes_instructions_and_auth(tmp_path):
    import json
    from unittest.mock import Mock

    # 준비
    ledger = QualityLedger(tmp_path / "ledger.json")
    response = Mock()
    response.model_dump.return_value = {
        "id": "synthetic-response",
        "output": [{"type": "message", "content": [{"text": "malformed {"}]}],
        "instructions": "not retained",
        "headers": {"Authorization": "not retained"},
    }

    # 실행 / 검증: 잘못된 JSON 출력도 decoder 실행 전에 그대로 보존할 수 있다.
    path = preserve_response(ledger, 1, response)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["output"] == response.model_dump.return_value["output"]
    assert "instructions" not in saved and "headers" not in saved


def test_profile_recorder_preserves_raw_before_decoder_failure(tmp_path):
    from unittest.mock import Mock

    from scripts.novel_quality_server import ACTIVE_PROFILE_ATTEMPT, response_recorder

    # 준비
    ledger = QualityLedger(tmp_path / "ledger.json")
    attempt = ledger.reserve(
        "test", "gpt-6-luna", Reservation(usd=0.1), metadata={"phase": "smoke"}
    )
    response = Mock()
    response.model_dump.return_value = {
        "id": "synthetic",
        "output": [{"content": [{"text": "broken {"}]}],
    }
    decoder = Mock(side_effect=ValueError("invalid structured response"))
    token = ACTIVE_PROFILE_ATTEMPT.set(attempt)
    try:
        # 실행
        with pytest.raises(ValueError):
            response_recorder(ledger, decoder)(response, structured=True)
    finally:
        ACTIVE_PROFILE_ATTEMPT.reset(token)

    # 검증
    decoder.assert_called_once()
    assert (tmp_path / "provider-responses/attempt-000001.json").exists()
    assert ledger.snapshot()["calls"][0]["evidence"]["rawResponseFile"]
    assert ACTIVE_PROFILE_ATTEMPT.get() is None


def test_legacy_recorder_adds_evidence_without_changing_decoder_result(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from scripts.novel_quality_server import ACTIVE_LEGACY_ATTEMPT, legacy_response_recorder

    # 준비: 기존 응답 DTO와 decoder는 그대로이고 QA 관측만 추가한다.
    ledger = QualityLedger(tmp_path / "ledger.json")
    attempt = ledger.reserve(
        "test", "gpt-5.6-luna", Reservation(usd=0.1), metadata={"phase": "explore"}
    )
    response = SimpleNamespace(
        id="response-id", _request_id="request-id", model="gpt-5.6-luna",
        usage=SimpleNamespace(input_tokens=100, output_tokens=20,
            input_tokens_details=SimpleNamespace(cached_tokens=30),
            output_tokens_details=SimpleNamespace(reasoning_tokens=2)),
        model_dump=lambda **kwargs: {"id": "response-id", "output": [{"text": "actual"}]},
    )
    expected = {"s0": "실제 반환"}
    decoder = Mock(return_value=expected)
    token = ACTIVE_LEGACY_ATTEMPT.set(attempt)
    try:
        # 실행
        actual = legacy_response_recorder(ledger, decoder)(response, structured=True)
    finally:
        ACTIVE_LEGACY_ATTEMPT.reset(token)

    # 검증: 미관측 값을 실제 provider 시간으로 대체하지 않는다.
    assert actual is expected
    decoder.assert_called_once_with(response, structured=True)
    evidence = ledger.snapshot()["calls"][0]["evidence"]
    assert evidence["providerRequestId"] == "request-id"
    assert evidence["cachedInputTokens"] == 30 and evidence["reasoningTokens"] == 2
    assert evidence["providerMs"] is None and evidence["queueMs"] is None
    assert evidence["providerFirstTokenMs"] is None
    assert (tmp_path / "provider-responses/attempt-000001.json").exists()
