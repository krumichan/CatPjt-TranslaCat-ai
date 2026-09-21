"""Synthetic reproductions of measured-audio/evidence-authority boundaries."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

from app.ai.providers.openai.schema import build_openai_text_config
from app.features.language_learning.speaking.errors import SpeakingStageException
from app.features.language_learning.speaking.evaluation_service import (
    SpeakingEvaluationService, _evaluation_response_schema,
)
from app.features.language_learning.speaking.prompts import build_evaluation_prompt
from app.schemas.language_learning_speaking import SpeakingEvaluationPayload, SpeakingEvaluationRequest
from tests.test_language_learning_speaking import (
    FakeStructuredProvider, evaluation_payload, evaluation_request, evaluation_turn,
)


def _request() -> SpeakingEvaluationRequest:
    turns = [evaluation_turn(index) for index in (1, 2)]
    for index, turn in enumerate(turns, 1):
        turn.update(turnId=str(index), durationSeconds=3.1709583333333335)
        turn["segments"][0].update(startMs=0, endMs=4000)
    data = evaluation_request(turns).model_dump(mode="json", by_alias=True)
    data.update(practiceMode="READ_ALOUD", evaluationScope="READ_ALOUD_PROBLEM",
                assistantTurns=[{"turnId": "read-aloud-problem-1", "turnIndex": 0,
                                 "text": "Synthetic reference", "scriptText": "Synthetic reference"}])
    return SpeakingEvaluationRequest.model_validate(data)


def _payload(*, evidence_turn_id: str = "1", end_ms: int | None = None) -> dict:
    payload = evaluation_payload()
    for metric in payload["metrics"]:
        if metric["type"] != "MEANING":
            metric.update(state="NOT_EVALUABLE", score=None, evidence=[], notEvaluableReason="Unsupported in READ_ALOUD")
        else:
            metric["evidence"][0].update(turnId=evidence_turn_id, startMs=None, endMs=end_ms)
    payload["profileSignals"] = []
    return payload


def test_prompt_withholds_invalid_stt_times_without_changing_request_or_transcript() -> None:
    request = _request()
    before = request.model_dump_json()
    original_hash = hashlib.sha256(before.encode()).hexdigest()
    prompt = json.loads(build_evaluation_prompt(request).rsplit("\n\n", 1)[1])
    for original, transmitted in zip(request.user_turns, prompt["userTurns"], strict=True):
        segment = transmitted["segments"][0]
        assert "startMs" not in segment and "endMs" not in segment
        assert segment["timestampUsableForEvidence"] is False
        assert segment["text"] == original.segments[0].text
        assert segment["confidence"] == original.segments[0].confidence
        assert transmitted["durationSeconds"] == original.duration_seconds
    contract = prompt["evidenceContract"]
    assert contract["allowedUserTurnIds"] == ["1", "2"]
    assert contract["timestampBoundsByUserTurn"] == {
        "1": {"minMs": 0, "maxMs": 3420}, "2": {"minMs": 0, "maxMs": 3420},
    }
    assert contract["assistantTurnsAreReferenceContextOnly"] is True
    assert contract["unsupportedTimestampsMustBeNull"] is True
    assert hashlib.sha256(request.model_dump_json().encode()).hexdigest() == original_hash
    assert request.model_dump_json() == before
    assert request.user_turns[0].segments[0].end_ms == 4000


def test_prompt_preserves_valid_stt_times_and_excludes_ineligible_evidence_ids() -> None:
    request = _request()
    request.user_turns[0].segments[0].end_ms = 3000
    request.user_turns[1].excluded_from_evaluation = True
    prompt = json.loads(build_evaluation_prompt(request).rsplit("\n\n", 1)[1])
    assert prompt["userTurns"][0]["segments"][0] == request.user_turns[0].segments[0].model_dump(by_alias=True)
    assert prompt["evidenceContract"]["allowedUserTurnIds"] == ["1"]


def test_wire_schema_preserves_exact_turn_bound_branches_and_nullable_timing() -> None:
    request = _request()
    request.user_turns[1].duration_seconds = 10
    public_before = SpeakingEvaluationPayload.model_json_schema()
    original_request = request.model_dump_json()
    schema = _evaluation_response_schema(request)
    config = build_openai_text_config(type_name=SpeakingEvaluationService.TYPE_NAME,
                                      schema=schema, verbosity="low")
    wire = config["format"]["schema"]
    branches = wire["$defs"]["MetricEvidence"]["anyOf"]
    assert len(branches) == 2
    for branch, turn_id, maximum in zip(branches, ("1", "2"), (3420, 10250), strict=True):
        assert branch["properties"]["turnId"]["enum"] == [turn_id]
        for field in ("startMs", "endMs"):
            variants = branch["properties"][field]["anyOf"]
            assert {"type": "null"} in variants
            assert next(value for value in variants if value.get("type") == "integer") == {
                "minimum": 0, "maximum": maximum, "type": "integer",
            }
            assert field not in branch["required"]
    for name in ("SpeakingProfileSignal", "RecommendedExpression", "PronunciationPractice"):
        assert wire["$defs"][name]["properties"]["evidenceTurnIds"]["items"]["enum"] == ["1", "2"]
    assert config["format"]["strict"] is False
    assert SpeakingEvaluationPayload.model_json_schema() == public_before
    assert request.model_dump_json() == original_request


@pytest.mark.asyncio
@pytest.mark.parametrize("turn_id,end_ms", [("1", 4000), ("read-aloud-problem-1", None)])
async def test_final_output_still_rejects_invalid_time_or_assistant_evidence_without_repair(
    turn_id: str, end_ms: int | None,
) -> None:
    payload = _payload(evidence_turn_id=turn_id, end_ms=end_ms)
    original = deepcopy(payload)
    provider = FakeStructuredProvider(results=[payload, payload, payload])
    service = SpeakingEvaluationService(provider, timeout_seconds=1, automatic_retries=2)
    with pytest.raises(SpeakingStageException) as caught:
        await service.evaluate(_request())
    assert caught.value.code.value == "INVALID_RESPONSE_SCHEMA"
    assert len(provider.calls) == 3
    assert payload == original


@pytest.mark.asyncio
async def test_null_timestamps_remain_valid_with_one_normal_evaluation_call() -> None:
    provider = FakeStructuredProvider(results=[_payload()])
    request = _request()
    original = request.model_dump_json()
    response = await SpeakingEvaluationService(provider, timeout_seconds=1).evaluate(request)
    assert response.status.value == "EVALUATED"
    assert len(provider.calls) == 1
    assert all(e.start_ms is None and e.end_ms is None for m in response.metrics for e in m.evidence)
    assert request.model_dump_json() == original


def test_per_turn_validation_does_not_borrow_longer_turn_bound() -> None:
    request = _request()
    request.user_turns[1].duration_seconds = 10
    payload = SpeakingEvaluationPayload.model_validate(_payload(evidence_turn_id="1", end_ms=4000))
    with pytest.raises(ValueError, match="timestamp"):
        SpeakingEvaluationService._validate_evidence(request, payload)
    for metric in payload.metrics:
        for evidence in metric.evidence:
            evidence.turn_id = "2"
    SpeakingEvaluationService._validate_evidence(request, payload)


@pytest.mark.parametrize("field", ["profileSignals", "recommendedExpressions", "pronunciationPractice"])
def test_all_supporting_evidence_lists_still_reject_assistant_id(field: str) -> None:
    raw = _payload()
    if field == "recommendedExpressions":
        raw[field] = [{"recommended": "Synthetic", "explanation": "Synthetic"}]
    elif field == "pronunciationPractice":
        raw[field] = [{"target": "Synthetic", "practicePhrase": "Synthetic", "reason": "Synthetic"}]
    else:
        raw[field] = [{"metricType": "MEANING", "direction": "WEAKNESS", "confidence": .9,
                       "patternKey": "synthetic", "recommendedFocus": "synthetic"}]
    raw[field][0]["evidenceTurnIds"] = ["read-aloud-problem-1"]
    payload = SpeakingEvaluationPayload.model_validate(raw)
    with pytest.raises(ValueError, match="evidenceTurnIds"):
        SpeakingEvaluationService._validate_evidence(_request(), payload)
