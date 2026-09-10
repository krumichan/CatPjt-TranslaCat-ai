"""Responses consumed by the BE release-contract tests. No network/model calls."""
import asyncio
import json
from pathlib import Path

import pytest

from app.features.language_learning.speaking.evaluation_service import SpeakingEvaluationService
from app.schemas.language_learning_speaking import SpeakingEvaluationRequest
from tests.test_language_learning_speaking import (
    FakeStructuredProvider,
    evaluation_payload,
    evaluation_request,
    evaluation_turn,
)


def unscorable_payload():
    payload = evaluation_payload()
    for metric in payload["metrics"]:
        metric.update(state="NOT_EVALUABLE", score=None, evidence=[], notEvaluableReason="INSUFFICIENT_AUDIO_EVIDENCE")
    return payload


def contract_cases():
    return [
        ("evaluated", evaluation_request(), evaluation_payload()),
        ("precheck-insufficient", evaluation_request([evaluation_turn(i, confidence=0.2) for i in range(1, 6)]), None),
        ("model-insufficient", evaluation_request(), evaluation_payload(confidence=0.69)),
        ("all-metrics-not-evaluable", evaluation_request(), unscorable_payload()),
    ]


async def make_case(request, payload):
    provider = FakeStructuredProvider(results=[] if payload is None else [payload])
    result = await SpeakingEvaluationService(provider, timeout_seconds=1, automatic_retries=0).evaluate(request)
    return result, provider


@pytest.mark.parametrize("name,evaluation_input,payload", contract_cases(), ids=lambda value: value if isinstance(value, str) else None)
def test_actual_ai_response_has_a_be_compatible_terminal_shape(name, evaluation_input, payload):
    request = evaluation_input
    result, provider = asyncio.run(make_case(request, payload))
    assert result.request_id == request.request_id
    assert result.session_id == request.session_id
    if name == "evaluated":
        assert result.status.value == "EVALUATED"
        assert result.overall_score == 80
        assert len(result.metrics) == 8
    else:
        assert result.status.value == "INSUFFICIENT_EVIDENCE"
        assert result.overall_score is None
        assert result.profile_signals == []
    if name == "precheck-insufficient":
        assert provider.calls == []
        assert result.metrics == []
        assert not result.eligibility.eligible_before_ai
        assert result.eligibility.missing_requirements
    elif name == "all-metrics-not-evaluable":
        assert result.evaluation_confidence == 0.9
        assert all(metric.score is None for metric in result.metrics)


@pytest.mark.parametrize("confidence,expected", [(0.5499, "INSUFFICIENT_EVIDENCE"), (0.55, "EVALUATED")])
def test_confidence_boundary_remains_strict(confidence, expected):
    request = evaluation_request([evaluation_turn(i, confidence=confidence) for i in range(1, 6)])
    result, _ = asyncio.run(make_case(request, evaluation_payload()))
    assert result.status.value == expected


def test_retry_same_frozen_evidence_keeps_one_provider_result_and_rebinds_request_id():
    async def run():
        provider = FakeStructuredProvider(results=[evaluation_payload()])
        service = SpeakingEvaluationService(provider, timeout_seconds=1, automatic_retries=0)
        request = evaluation_request()
        first = await service.evaluate(request)
        retried = await service.evaluate(SpeakingEvaluationRequest.model_validate({
            **request.model_dump(by_alias=True), "requestId": "accepted-manual-retry", "manualRetryAttempt": 1,
        }))
        assert len(provider.calls) == 1
        assert first.overall_score == retried.overall_score
        assert retried.request_id == "accepted-manual-retry"
    asyncio.run(run())


@pytest.mark.parametrize("name,evaluation_input,payload", contract_cases())
def test_checked_in_contract_fixtures_still_match_actual_ai(name, evaluation_input, payload):
    request = evaluation_input
    fixture = json.loads((Path(__file__).parent / "fixtures" / "speaking-release" / f"{name}.json").read_text(encoding="utf-8"))
    result, _ = asyncio.run(make_case(request, payload))
    actual = result.model_dump(by_alias=True, mode="json")
    # Provider latency is runtime telemetry, not part of the deterministic score/status contract.
    for body in [actual, fixture["response"]]:
        evaluation_usage = body.get("usage", {}).get("evaluation")
        if evaluation_usage:
            evaluation_usage["latencyMs"] = 0
    assert fixture["request"] == request.model_dump(by_alias=True, mode="json")
    assert fixture["response"] == actual
