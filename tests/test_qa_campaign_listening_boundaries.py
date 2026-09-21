"""Synthetic orchestration checks; no provider or Japanese-quality assertion."""
from __future__ import annotations

import asyncio
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
from typing import Any
import wave

import pytest

from scripts.qa_campaign_listening_boundaries import (
    BoundaryHttpResponse, prepare_manifest, run_listening_boundaries,
)


def source_request() -> dict[str, Any]:
    return {"requestId": "captured-request", "idempotencyKey": "captured-idempotency",
            "userContext": {"originLanguage": "ko", "learningLanguage": "ja", "level": "MY_LEVEL", "profileFocus": ["LISTENING"]},
            "setContext": {"learningDate": "2026-09-19", "learningMode": "DICTATION", "topic": {"id": "source-topic", "title": "source title"},
                           "selectedKeywords": [], "itemCount": 1, "difficulty": "MY_LEVEL"},
            "languageComplexity": {"baseLevelScore": 61, "baseComplexityBand": 3, "targetComplexityBand": 3},
            "referenceVoice": {"locale": "ja", "voiceKey": "Kore", "version": "current", "accent": "STANDARD"}}


def json_response(value: dict[str, Any], status: int = 200) -> BoundaryHttpResponse:
    return BoundaryHttpResponse(status, json.dumps(value).encode())


class SyntheticTransport:
    def __init__(self, *, duration_failure: bool = False, invalid_actual_duration: bool = False):
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []
        self.audio = b""
        self.duration_failure = duration_failure
        self.invalid_actual_duration = invalid_actual_duration

    async def __call__(self, method: str, path: str, *, json_body: dict[str, Any] | None,
                       timeout_seconds: float) -> BoundaryHttpResponse:
        self.calls.append((method, path, deepcopy(json_body)))
        assert timeout_seconds > 0 and path.startswith("/api/v1/language-learning/listening/")
        if path.endswith("/sets/generate"):
            assert json_body is not None
            text = "synthetic " + json_body["requestId"]
            return json_response({"requestId": json_body["requestId"], "generationVersion": "unchanged-generation-version",
                                  "policyVersion": json_body["policyVersion"], "modelConfigVersion": json_body["modelConfigVersion"],
                                  "usage": {}, "items": [{
                                      "itemIndex": 1, "sourceText": text, "normalizedSourceText": text,
                                      "referenceMeanings": ["reference1", "reference2"], "keyMeaningUnits": ["unit"],
                                      "targetKeywords": [], "estimatedAudioSeconds": 99, "contentHash": hashlib.sha256(text.encode()).hexdigest(),
                                      "similarityKey": "similarity-" + json_body["requestId"], "safety": {"passed": True},
                                      "durationDemand": {"minSeconds": json_body["constraints"]["audioSecondsMin"],
                                                         "maxSeconds": json_body["constraints"]["audioSecondsMax"],
                                                         "policyVersion": "listening-audio-duration-v1", "playbackSpeed": "NORMAL"},
                                  }]})
        if path.endswith("/tts"):
            assert json_body is not None
            result = {key: json_body[key] for key in ("requestId", "itemId", "sourceText", "contentHash", "generationVersion")}
            result["usage"] = {}
            if self.duration_failure:
                return json_response({**result, "status": "FAILED", "error": {
                    "code": "AUDIO_TOO_SHORT", "failedStage": "TTS", "retryable": False,
                    "message": "synthetic duration failure", "details": {"measuredSeconds": 3.0}}})
            seconds = 3 if self.invalid_actual_duration else json_body["durationDemand"]["minSeconds"]
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as output:
                output.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
                output.writeframes(b"\0\0" * round(seconds * 8000))
            self.audio = buffer.getvalue()
            checksum = hashlib.sha256(self.audio).hexdigest()
            return json_response({**result, "status": "READY", "audio": {
                "audioReference": "listening-tts-" + "a" * 32, "durationMs": round(seconds * 1000),
                "format": "WAV", "sampleRate": 8000, "channels": 1, "voice": json_body["voice"],
                "textHash": json_body["contentHash"], "checksum": checksum, "cacheKey": "synthetic-cache",
                "ttsVersion": "unchanged-tts-version", "durationValidated": True,
                "durationPolicyVersion": "listening-audio-duration-v1"}})
        assert method == "GET" and "/audio/listening-tts-" in path
        return BoundaryHttpResponse(200, self.audio, "audio/wav")


def test_manifest_is_offline_fixed_two_by_five_preserves_source_profile_and_date():
    source = source_request()
    original = deepcopy(source)
    manifest = prepare_manifest(source, campaign_id="test", source_provenance={"fixture": "synthetic-test"})
    assert source == original
    assert len(manifest["cases"]) == 10
    for index, case in enumerate(manifest["cases"]):
        payload = case["generationRequest"]
        assert case["difficulty"] == ("EASY" if index < 5 else "CHALLENGE")
        assert payload["setContext"]["learningDate"] == source["setContext"]["learningDate"]
        assert payload["setContext"]["topic"] == source["setContext"]["topic"]
        assert payload["userContext"]["profileFocus"] == source["userContext"]["profileFocus"]
        assert payload["languageComplexity"]["baseLevelScore"] == 61
        assert payload["languageComplexity"]["targetComplexityBand"] == (2 if index < 5 else 4)
        assert payload["setContext"]["itemCount"] == 1 and payload["durationCorrection"] is None
    assert manifest["contentCorrectionAttempts"] == 0
    assert manifest["maximumHttpStarts"] == 30


@pytest.mark.parametrize("missing", ["referenceVoice", "languageComplexity"])
def test_source_cannot_silently_invent_profile_or_voice(missing):
    source = source_request()
    source.pop(missing)
    with pytest.raises(ValueError, match="PROFILE_REQUEST_REQUIRED"):
        prepare_manifest(source, campaign_id="test", source_provenance={"fixture": "test"})


@pytest.mark.asyncio
async def test_normal_chain_ten_items_thirty_http_starts_no_sdk_no_db_and_no_replay(tmp_path):
    transport = SyntheticTransport()
    report = await run_listening_boundaries(transport, source_request(), tmp_path, campaign_id="test",
                                             source_provenance={"fixture": "synthetic-test"})
    assert report["status"] == "COMPLETED" and report["completedItems"] == 10
    assert report["httpStarts"] == 30 and not report["bePublicationEvidence"]
    assert all(row["audio"]["withinEffectiveBounds"] for row in report["items"])
    generations = [payload for _, path, payload in transport.calls if path.endswith("/sets/generate")]
    assert [len(payload["diversityContext"]["currentSession"]) for payload in generations if payload] == [0, 1, 2, 3, 4] * 2
    tts = [payload for _, path, payload in transport.calls if path.endswith("/tts")]
    assert all(payload and payload["automaticRetryLimit"] == 2 and payload["manualRetryAttempt"] == 0 for payload in tts)
    assert len(list(tmp_path.glob("*.wav"))) == 10
    with pytest.raises(ValueError, match="NO_REPLAY"):
        await run_listening_boundaries(transport, source_request(), tmp_path, campaign_id="test",
                                        source_provenance={"fixture": "synthetic-test"})
    assert len(transport.calls) == 30


@pytest.mark.asyncio
async def test_duration_failures_keep_fixed_denominator_no_correction_retry_or_audio_download(tmp_path):
    transport = SyntheticTransport(duration_failure=True)
    report = await run_listening_boundaries(transport, source_request(), tmp_path, campaign_id="test",
                                             source_provenance={"fixture": "synthetic-test"})
    assert report["status"] == "COMPLETED_WITH_FAILURES" and report["completedItems"] == 0
    assert len(report["items"]) == 10 and report["httpStarts"] == 20
    assert all(row["error"]["code"] == "AUDIO_TOO_SHORT" for row in report["items"])
    assert not list(tmp_path.glob("*.wav"))
    assert all(method == "POST" for method, _, _ in transport.calls)


@pytest.mark.asyncio
async def test_ready_flag_cannot_override_actual_audio_length(tmp_path):
    transport = SyntheticTransport(invalid_actual_duration=True)
    with pytest.raises(ValueError, match="OUTSIDE_EFFECTIVE_BOUNDS"):
        await run_listening_boundaries(transport, source_request(), tmp_path, campaign_id="test",
                                        source_provenance={"fixture": "synthetic-test"})
    report = json.loads((tmp_path / "result.json").read_text())
    assert report["status"] == "FAILED" and report["partial"] is True and report["httpStarts"] == 3
    assert Path(report["items"][0]["audioArtifact"]).exists()
    assert len(transport.calls) == 3


@pytest.mark.parametrize("failure", ["cancel", "provider-exception", "timeout"])
@pytest.mark.asyncio
async def test_transport_failure_is_reserved_then_flushed_with_no_followup(tmp_path, failure):
    completed = SyntheticTransport()
    calls = []

    async def transport(method, path, *, json_body, timeout_seconds):
        calls.append(path)
        if len(calls) == 1:
            return await completed(method, path, json_body=json_body, timeout_seconds=timeout_seconds)
        reserved = json.loads((tmp_path / "result.json").read_text())
        assert reserved["httpAttempts"][-1]["status"] == "STARTED"
        if failure == "cancel":
            raise asyncio.CancelledError()
        if failure == "provider-exception":
            raise OSError("sensitive exception detail must not be serialized")
        await asyncio.sleep(5)
        pytest.fail("Timeout wrapper must cancel this call")

    error_type = asyncio.CancelledError if failure == "cancel" else OSError if failure == "provider-exception" else TimeoutError
    with pytest.raises(error_type):
        await run_listening_boundaries(transport, source_request(), tmp_path, campaign_id="test",
                                        source_provenance={"fixture": "synthetic-test"}, http_timeout_seconds=1)
    text = (tmp_path / "result.json").read_text()
    report = json.loads(text)
    assert report["partial"] is True and report["httpStarts"] == 2 and len(calls) == 2
    assert report["httpAttempts"][0]["response"]["items"][0]["sourceText"]
    assert report["httpAttempts"][1]["request"]["sourceText"]
    assert report["httpAttempts"][1]["errorType"] == error_type.__name__
    assert "latencyMs" in report["httpAttempts"][1] and "sensitive exception detail" not in text
