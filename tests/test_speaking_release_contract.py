"""Responses consumed by the BE release-contract tests. No network/model calls."""
import asyncio
import json
from pathlib import Path

import pytest

from app.features.language_learning.speaking.evaluation_service import SpeakingEvaluationService
from app.features.language_learning.speaking.errors import SpeakingStageException
from app.schemas.language_learning_speaking import SpeakingEvaluationRequest, SpeakingEvaluationResponse
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
    payload["profileSignals"] = []
    return payload


def contract_cases():
    read_aloud = evaluation_request([evaluation_turn(i) for i in (1, 2)]).model_dump(by_alias=True, mode="json")
    read_aloud.update(practiceMode="READ_ALOUD", evaluationScope="READ_ALOUD_PROBLEM",
                      assistantTurns=[{"turnId": "reference-script", "turnIndex": 0,
                                       "text": "今日は仕事をしました。", "scriptText": "今日は仕事をしました。"}])
    read_aloud_payload = evaluation_payload()
    for metric in read_aloud_payload["metrics"]:
        if metric["type"] != "MEANING":
            metric.update(state="NOT_EVALUABLE", score=None, evidence=[], notEvaluableReason="COPIED_SCRIPT_OR_UNSUPPORTED_ACOUSTIC_EVIDENCE")
    read_aloud_payload["profileSignals"] = []
    return [
        ("evaluated", evaluation_request(), evaluation_payload()),
        ("precheck-insufficient", evaluation_request([evaluation_turn(i, confidence=0.2) for i in range(1, 6)]), None),
        ("model-insufficient", evaluation_request(), evaluation_payload(confidence=0.69)),
        ("read-aloud-script-observation", SpeakingEvaluationRequest.model_validate(read_aloud), read_aloud_payload),
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
    if name in {"evaluated", "read-aloud-script-observation"}:
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
    assert result.evidence_policy_version == "speaking-transcript-evidence-v2"


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


@pytest.mark.parametrize("name", ["evaluated", "precheck-insufficient", "model-insufficient", "all-metrics-not-evaluable"])
def test_immutable_legacy_contract_fixtures_remain_readable_without_relabelling(name):
    # V1 snapshots remain historical evidence, not expected V2 production output.
    fixture = json.loads((Path(__file__).parent / "fixtures" / "speaking-release" / f"{name}.json").read_text(encoding="utf-8"))
    SpeakingEvaluationRequest.model_validate(fixture["request"])
    result = SpeakingEvaluationResponse.model_validate(fixture["response"])
    assert result.status.value == fixture["response"]["status"]
    assert result.overall_score == fixture["response"]["overallScore"]
    assert result.scoring_policy_version == fixture["response"]["scoringPolicyVersion"]
    assert result.evidence_policy_version is None


def test_v2_rejects_all_not_evaluable_shortcut_for_usable_text_evidence():
    with pytest.raises(SpeakingStageException) as error:
        asyncio.run(make_case(evaluation_request(), unscorable_payload()))
    assert error.value.code.value == "INVALID_RESPONSE_SCHEMA"


@pytest.mark.parametrize("name,evaluation_input,payload", contract_cases())
def test_v2_checked_in_cross_contract_preserves_behavior_across_v3_prompt(name, evaluation_input, payload):
    fixture = json.loads((Path(__file__).parent / "fixtures" / "speaking-release-v2" / f"{name}.json").read_text(encoding="utf-8"))
    result, _ = asyncio.run(make_case(evaluation_input, payload))
    actual = result.model_dump(by_alias=True, mode="json")
    if actual["usage"].get("evaluation"):
        actual["usage"]["evaluation"]["latencyMs"] = 0
    assert fixture["request"] == evaluation_input.model_dump(by_alias=True, mode="json")
    # The immutable V2 fixture records the previous prompt instrument. V3
    # changes task binding/confidence instructions, not the BE response shape.
    assert fixture["response"]["promptVersion"] == "speaking-evaluation-prompt-v2"
    assert actual["promptVersion"] == "speaking-evaluation-prompt-v3"
    fixture["response"]["promptVersion"] = actual["promptVersion"]
    if fixture["response"]["usage"].get("evaluation"):
        assert fixture["response"]["usage"]["evaluation"]["promptVersion"] == "speaking-evaluation-prompt-v2"
        assert actual["usage"]["evaluation"]["promptVersion"] == "speaking-evaluation-prompt-v3"
        fixture["response"]["usage"]["evaluation"]["promptVersion"] = actual["usage"]["evaluation"]["promptVersion"]
    assert fixture["response"] == actual
