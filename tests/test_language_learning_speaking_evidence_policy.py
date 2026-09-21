"""Policy tests prove application boundaries, not real STT/acoustic accuracy."""
from __future__ import annotations

from copy import deepcopy
import json

import pytest

from app.ai.providers.openai.schema import build_openai_text_config
from app.features.language_learning.speaking.evaluation_service import SpeakingEvaluationService, _evaluation_response_schema
from app.features.language_learning.speaking.errors import SpeakingStageException
from app.features.language_learning.speaking.prompts import build_evaluation_prompt
from app.schemas.language_learning_speaking import SpeakingEvaluationRequest
from tests.test_language_learning_speaking import FakeStructuredProvider, evaluation_payload, evaluation_request, metric_payload


@pytest.mark.asyncio
@pytest.mark.parametrize("metric", ["PRONUNCIATION", "FLUENCY"])
async def test_available_waveform_metadata_cannot_authorize_unconsumed_acoustic_claim(metric: str) -> None:
    payload = evaluation_payload()
    payload["metrics"] = [metric_payload(metric) if item["type"] == metric else item for item in payload["metrics"]]
    provider = FakeStructuredProvider(results=[payload])
    with pytest.raises(SpeakingStageException) as error:
        await SpeakingEvaluationService(provider, automatic_retries=0).evaluate(evaluation_request())
    assert error.value.code.value == "INVALID_RESPONSE_SCHEMA"
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_clean_conversation_keeps_six_text_axes_and_available_weight_score() -> None:
    provider = FakeStructuredProvider(results=[evaluation_payload()])
    response = await SpeakingEvaluationService(provider, automatic_retries=0).evaluate(evaluation_request())
    assert response.status.value == "EVALUATED" and response.overall_score == 80
    assert len(response.evaluated_axes) == 6
    assert response.evaluation_coverage == .65
    assert response.evidence_policy_version == "speaking-transcript-evidence-v2"
    assert response.evidence_source == "TRANSCRIPT_OBSERVATION"
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_missing_only_server_owned_acoustic_reasons_do_not_discard_six_supported_axes() -> None:
    payload = evaluation_payload()
    for metric in payload["metrics"]:
        if metric["type"] in {"PRONUNCIATION", "FLUENCY"}:
            metric.pop("notEvaluableReason", None)
    original = deepcopy(payload)
    provider = FakeStructuredProvider(results=[payload])
    response = await SpeakingEvaluationService(provider, automatic_retries=0).evaluate(evaluation_request())
    assert response.status.value == "EVALUATED"
    assert len(response.evaluated_axes) == 6
    assert response.evaluation_coverage == .65
    assert all(metric.not_evaluable_reason for metric in response.metrics
               if metric.type.value in {"PRONUNCIATION", "FLUENCY"})
    assert payload == original and len(provider.calls) == 1


@pytest.mark.asyncio
async def test_server_owned_reason_never_repairs_acoustic_score_or_evidence() -> None:
    payload = evaluation_payload()
    acoustic = next(metric for metric in payload["metrics"] if metric["type"] == "FLUENCY")
    acoustic.pop("notEvaluableReason", None)
    acoustic["score"] = 92
    provider = FakeStructuredProvider(results=[payload])
    with pytest.raises(SpeakingStageException) as error:
        await SpeakingEvaluationService(provider, automatic_retries=0).evaluate(evaluation_request())
    assert error.value.code.value == "INVALID_RESPONSE_SCHEMA"
    assert len(provider.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["profileSignals", "pronunciationPractice"])
async def test_unsupported_acoustic_signal_or_practice_is_rejected(field: str) -> None:
    payload = evaluation_payload()
    if field == "profileSignals":
        payload[field][0]["metricType"] = "PRONUNCIATION"
    else:
        payload[field] = [{"target": "synthetic", "practicePhrase": "synthetic", "reason": "synthetic",
                           "evidenceTurnIds": ["turn-1"]}]
    with pytest.raises(SpeakingStageException):
        await SpeakingEvaluationService(FakeStructuredProvider(results=[payload]), automatic_retries=0).evaluate(evaluation_request())


def test_provider_input_labels_observation_uncertainty_without_correcting_transcript() -> None:
    request = evaluation_request()
    request.user_turns[0].segments[0].confidence = .4
    original = request.model_dump_json()
    prompt = json.loads(build_evaluation_prompt(request).rsplit("\n\n", 1)[1])
    assert prompt["evaluationCapabilities"]["acousticEvidenceConsumed"] is False
    assert prompt["evaluationCapabilities"]["uncertainTranscriptTurnIds"] == ["turn-1"]
    assert prompt["userTurns"][0]["transcriptObservation"]["usableForTextEvaluation"] is False
    assert prompt["userTurns"][0]["transcript"] == request.user_turns[0].transcript
    assert request.model_dump_json() == original


def test_free_task_bindings_exclude_unanswered_final_assistant_reply() -> None:
    data = evaluation_request().model_dump(mode="json", by_alias=True)
    data["assistantTurns"] = [
        {"turnId": f"assistant-{index}", "turnIndex": index,
         "text": f"Synthetic prompt {index}"}
        for index in range(6)
    ]
    request = SpeakingEvaluationRequest.model_validate(data)
    original = request.model_dump_json()
    prompt = json.loads(build_evaluation_prompt(request).rsplit("\n\n", 1)[1])
    assert [turn["turnId"] for turn in prompt["assistantTurns"]] == [
        f"assistant-{index}" for index in range(5)
    ]
    assert prompt["evaluationTaskBindings"] == [
        {"userTurnId": f"turn-{index}",
         "referenceAssistantTurnId": f"assistant-{index - 1}"}
        for index in range(1, 6)
    ]
    assert prompt["evaluationCapabilities"]["evaluationConfidenceScope"] == (
        "MODEL_ASSESSABLE_METRICS_AND_USABLE_TEXT_EVIDENCE"
    )
    assert request.model_dump_json() == original


@pytest.mark.asyncio
async def test_free_confidence_gate_still_blocks_low_confidence_supported_metrics() -> None:
    payload = evaluation_payload(confidence=0.62)
    provider = FakeStructuredProvider(results=[payload])
    response = await SpeakingEvaluationService(provider, automatic_retries=0).evaluate(
        evaluation_request()
    )
    assert response.status.value == "INSUFFICIENT_EVIDENCE"
    assert response.overall_score is None
    assert len(response.evaluated_axes) == 6
    assert response.evaluation_confidence == 0.62
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_uncertain_segment_cannot_be_used_for_grammar_penalty_despite_high_turn_confidence() -> None:
    request = evaluation_request()
    request.user_turns[0].segments[0].confidence = .4
    payload = evaluation_payload()
    payload["metrics"][0]["score"] = 20
    with pytest.raises(SpeakingStageException):
        await SpeakingEvaluationService(FakeStructuredProvider(results=[payload]), automatic_retries=0).evaluate(request)


@pytest.mark.asyncio
async def test_read_aloud_is_script_observation_not_spontaneous_language_ability() -> None:
    data = evaluation_request().model_dump(mode="json", by_alias=True)
    data.update(practiceMode="READ_ALOUD", evaluationScope="READ_ALOUD_PROBLEM",
                assistantTurns=[{"turnId": "reference", "turnIndex": 0, "text": "Synthetic script", "scriptText": "Synthetic script"}])
    request = SpeakingEvaluationRequest.model_validate(data)
    payload = evaluation_payload()
    for metric in payload["metrics"]:
        if metric["type"] != "MEANING":
            metric.update(state="NOT_EVALUABLE", score=None, evidence=[], notEvaluableReason="Unsupported")
    payload["profileSignals"] = []
    response = await SpeakingEvaluationService(FakeStructuredProvider(results=[payload]), automatic_retries=0).evaluate(request)
    assert [axis.value for axis in response.evaluated_axes] == ["MEANING"]
    assert response.evaluation_coverage == .1 and response.overall_score == 80
    assert response.profile_signals == []
    unsupported = deepcopy(payload)
    unsupported["metrics"][0] = metric_payload("GRAMMAR")
    with pytest.raises(SpeakingStageException):
        await SpeakingEvaluationService(FakeStructuredProvider(results=[unsupported]), automatic_retries=0).evaluate(request)


@pytest.mark.asyncio
async def test_usable_evidence_cannot_be_hidden_by_all_not_evaluable() -> None:
    payload = evaluation_payload()
    for metric in payload["metrics"]:
        metric.update(state="NOT_EVALUABLE", score=None, evidence=[], notEvaluableReason="Unsupported")
    payload["profileSignals"] = []
    with pytest.raises(SpeakingStageException):
        await SpeakingEvaluationService(FakeStructuredProvider(results=[payload]), automatic_retries=0).evaluate(evaluation_request())


def test_provider_normalization_preserves_unsupported_metric_contract() -> None:
    wire = build_openai_text_config(type_name=SpeakingEvaluationService.TYPE_NAME,
                                    schema=_evaluation_response_schema(evaluation_request()), verbosity="low")["format"]["schema"]
    supported = {
        branch["properties"]["type"]["enum"][0]
        for branch in wire["$defs"]["SpeakingMetricPayload"]["anyOf"]
    }
    assert supported == {"GRAMMAR", "VOCABULARY", "NATURALNESS", "MEANING", "EXPRESSIVENESS", "INTERACTION"}
    assert wire["properties"]["metrics"]["maxItems"] == 6
    assert wire["properties"]["pronunciationPractice"]["maxItems"] == 0


@pytest.mark.asyncio
async def test_model_only_assesses_supported_axes_and_server_adds_capability_results() -> None:
    payload = evaluation_payload()
    payload["metrics"] = [metric for metric in payload["metrics"]
                          if metric["type"] not in {"PRONUNCIATION", "FLUENCY"}]
    provider = FakeStructuredProvider(results=[payload])
    response = await SpeakingEvaluationService(provider, automatic_retries=0).evaluate(evaluation_request())
    assert len(provider.calls) == 1
    assert len(response.metrics) == 8
    assert [axis.value for axis in response.evaluated_axes] == [
        "GRAMMAR", "VOCABULARY", "NATURALNESS", "MEANING", "EXPRESSIVENESS", "INTERACTION"
    ]
    assert response.overall_score == 80
    for metric in response.metrics:
        if metric.type.value in {"PRONUNCIATION", "FLUENCY"}:
            assert metric.state.value == "NOT_EVALUABLE"
            assert metric.score is None and not metric.evidence


@pytest.mark.asyncio
async def test_read_aloud_provider_only_returns_meaning_without_losing_score() -> None:
    data = evaluation_request().model_dump(mode="json", by_alias=True)
    data.update(practiceMode="READ_ALOUD", evaluationScope="READ_ALOUD_PROBLEM",
                assistantTurns=[{"turnId": "reference", "turnIndex": 0,
                                 "text": "Synthetic script", "scriptText": "Synthetic script"}])
    request = SpeakingEvaluationRequest.model_validate(data)
    payload = evaluation_payload()
    payload["metrics"] = [metric for metric in payload["metrics"] if metric["type"] == "MEANING"]
    payload["profileSignals"] = []
    provider = FakeStructuredProvider(results=[payload])
    response = await SpeakingEvaluationService(provider, automatic_retries=0).evaluate(request)
    assert len(provider.calls) == 1
    assert [axis.value for axis in response.evaluated_axes] == ["MEANING"]
    assert response.overall_score == 80 and response.evaluation_coverage == .1
    assert len(response.metrics) == 8
    wire = build_openai_text_config(type_name=SpeakingEvaluationService.TYPE_NAME,
                                    schema=_evaluation_response_schema(request), verbosity="low")["format"]["schema"]
    assert wire["properties"]["metrics"]["maxItems"] == 1
    assert wire["$defs"]["SpeakingMetricPayload"]["properties"]["type"]["enum"] == ["MEANING"]
    prompt = json.loads(build_evaluation_prompt(request).rsplit("\n\n", 1)[1])
    assert prompt["evaluationCapabilities"]["modelAssessableMetrics"] == ["MEANING"]
