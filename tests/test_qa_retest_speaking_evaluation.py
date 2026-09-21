from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.features.language_learning.speaking.errors import SpeakingStageException
from app.features.language_learning.speaking.prompts import build_evaluation_prompt
from app.schemas.language_learning_speaking import SpeakingPracticeMode
from scripts.qa_retest_speaking_evaluation import (
    captured_read_aloud_request,
    run_captured_speaking_evaluation_retest,
)
from tests.test_language_learning_speaking import (
    FakeStructuredProvider, evaluation_payload, evaluation_request,
)


def _source() -> dict:
    request = evaluation_request().model_copy(update={
        "practice_mode": SpeakingPracticeMode.READ_ALOUD,
        "evaluation_scope": "READ_ALOUD_PROBLEM",
    })
    return {"attempts": [{"sequence": 9, "request": {
        "type": "LANGUAGE_LEARNING_SPEAKING_EVALUATION",
        "data": build_evaluation_prompt(request),
    }}]}


@pytest.mark.asyncio
async def test_captured_retest_keeps_identical_prompt_and_never_regenerates_audio(tmp_path: Path) -> None:
    source = _source()
    path = tmp_path / "source.json"
    original = json.dumps(source).encode()
    path.write_bytes(original)
    payload = evaluation_payload()
    for metric in payload["metrics"]:
        if metric["type"] != "MEANING":
            metric.update(state="NOT_EVALUABLE", score=None, evidence=[], notEvaluableReason="Unsupported")
    payload["profileSignals"] = []
    provider = FakeStructuredProvider(results=[payload])
    result = await run_captured_speaking_evaluation_retest(provider, path, tmp_path / "retest")
    assert result["status"] == "COMPLETED"
    assert result["existingProviderAttemptCap"] == 3
    assert len(provider.calls) == 1
    assert provider.calls[0][0] == "LANGUAGE_LEARNING_SPEAKING_EVALUATION"
    assert provider.calls[0][1] == source["attempts"][0]["request"]["data"]
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="already exists"):
        await run_captured_speaking_evaluation_retest(provider, path, tmp_path / "retest")
    assert len(provider.calls) == 1


def test_retest_refuses_derived_evidence_drift_without_provider_call() -> None:
    source = _source()
    prompt = source["attempts"][0]["request"]["data"]
    header, body = prompt.rsplit("\n\n", 1)
    payload = json.loads(body)
    payload["pronunciationEvidenceAvailable"] = not payload["pronunciationEvidenceAvailable"]
    source["attempts"][0]["request"]["data"] = header + "\n\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    with pytest.raises(ValueError, match="original prompt exactly"):
        captured_read_aloud_request(source)


def test_retest_never_reconstructs_removed_stt_timestamps() -> None:
    source = _source()
    prompt = source["attempts"][0]["request"]["data"]
    header, body = prompt.rsplit("\n\n", 1)
    payload = json.loads(body)
    payload["userTurns"][0]["segments"] = [{
        "text": "Retained transcript", "confidence": 0.9,
        "timestampUsableForEvidence": False,
    }]
    source["attempts"][0]["request"]["data"] = header + "\n\n" + json.dumps(payload)
    with pytest.raises(ValueError, match="original request snapshot"):
        captured_read_aloud_request(source)


@pytest.mark.asyncio
async def test_retest_preserves_three_attempt_failure_and_partial_payloads(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text(json.dumps(_source()), encoding="utf-8")
    invalid = evaluation_payload()
    invalid["profileSignals"][0]["direction"] = "POSITIVE"
    provider = FakeStructuredProvider(results=[invalid, invalid, invalid])
    with pytest.raises(SpeakingStageException):
        await run_captured_speaking_evaluation_retest(provider, source, tmp_path / "retest")
    result = json.loads((tmp_path / "retest/speaking-evaluation-retest.json").read_text(encoding="utf-8"))
    assert result["status"] == "FAILED"
    assert result["partial"] is True
    assert result["error"]["code"] == "INVALID_RESPONSE_SCHEMA"
    assert len(provider.calls) == len(result["attempts"]) == 3
    assert all(attempt["response"]["data"]["profileSignals"][0]["direction"] == "POSITIVE"
               for attempt in result["attempts"])
