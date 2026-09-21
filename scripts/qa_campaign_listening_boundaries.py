"""Private, one-shot direct-AI Listening boundary QA; not BE/DB completion evidence.

The root caller supplies authenticated HTTP transport to the already budgeted QA
AI server. This module never creates providers, credentials, or network clients.
No endpoint is called by importing this module or preparing its manifest.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import re
import time
from typing import Any, Protocol
import wave

from app.features.language_learning.quality.policy import resolve_daily_complexity_band
from app.schemas.language_learning_listening import (
    ListeningItem, ListeningSetGenerationRequest, ListeningSetGenerationResponse,
    ListeningTtsRequest, ListeningTtsResponse,
)
from scripts.qa_campaign_be_api import utc_now
from scripts.qa_campaign_integration import _write_json


_BASE = "/api/v1/language-learning/listening"
_BOUNDS = {"EASY": (5.0, 12.0), "CHALLENGE": (15.0, 30.0)}


@dataclass(frozen=True)
class BoundaryHttpResponse:
    status: int
    body: bytes
    content_type: str = "application/json"


class BoundaryTransport(Protocol):
    async def __call__(self, method: str, path: str, *, json_body: dict[str, Any] | None,
                       timeout_seconds: float) -> BoundaryHttpResponse: ...


def prepare_manifest(source_request: dict[str, Any], *, campaign_id: str,
                     source_provenance: dict[str, Any]) -> dict[str, Any]:
    """Require captured/external source identity; do not manufacture learner data."""
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,48}", campaign_id) or not source_provenance:
        raise ValueError("BOUNDARY_QA_CAMPAIGN_AND_SOURCE_PROVENANCE_REQUIRED")
    source = ListeningSetGenerationRequest.model_validate(source_request)
    voice = source.reference_voice
    if (source.user_context.learning_language != "ja" or voice is None
            or voice.locale != "ja" or voice.voice_key != "Kore"
            or source.language_complexity is None or source.duration_correction is not None
            or source.manual_retry_attempt != 0):
        raise ValueError("BOUNDARY_QA_INITIAL_JA_KORE_PROFILE_REQUEST_REQUIRED")
    base = source.model_dump(mode="json", by_alias=True)
    cases = []
    for difficulty, (minimum, maximum) in _BOUNDS.items():
        for order in range(1, 6):
            identifier = f"{campaign_id}-{difficulty.lower()}-{order}"
            payload = deepcopy(base)
            payload.update(requestId=identifier, idempotencyKey=identifier)
            payload["userContext"]["level"] = difficulty
            payload["setContext"].update(itemCount=1, learningMode="DICTATION", difficulty=difficulty)
            payload["constraints"].update(audioSecondsMin=minimum, audioSecondsMax=maximum)
            payload["languageComplexity"]["targetComplexityBand"] = resolve_daily_complexity_band(
                difficulty, source.language_complexity)
            validated = ListeningSetGenerationRequest.model_validate(payload)
            cases.append({"caseId": identifier, "difficulty": difficulty, "logicalOrder": order,
                          "generationRequest": validated.model_dump(mode="json", by_alias=True)})
    return {
        "version": "listening-direct-ai-boundaries-v1", "campaignId": campaign_id,
        "sourceProvenance": deepcopy(source_provenance),
        "normalizedSourceRequestSha256": hashlib.sha256(json.dumps(base, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
        "sourceRequest": base, "cases": cases, "expectedItems": 10,
        "scope": "Direct AI API only; no BE set/session/DB publication or authenticated browser completion claim",
        "requestChanges": ["QA request/idempotency identity", "DICTATION single item", "difficulty and derived target band",
                           "difficulty duration boundary", "previous accepted boundary items added to diversity history"],
        "maximumHttpStarts": 30, "maximumGenerationPostStarts": 10, "maximumTtsPostStarts": 10,
        "contentCorrectionAttempts": 0, "transportRetries": 0,
        "providerBudgetAuthority": "Existing root-owned QA server global ledger; HTTP starts are not provider starts",
        "providerRetries": "Unchanged production stage retries; SDK internal retries/failover remain separately unobservable",
        "timeoutBoundary": "HTTP cancellation stops this runner, not proof that server/native provider work stopped; root must inspect global ledger",
        "rejectedAudioBoundary": "TTS duration failure may expose measured metadata only; production does not publish rejected audio bytes",
        "humanContentQualityVerified": False,
    }


def _with_prefix(payload: dict[str, Any], prefix: list[ListeningItem]) -> dict[str, Any]:
    request = deepcopy(payload)
    for item in prefix:
        metadata = item.diversity_metadata
        request["diversityContext"]["currentSession"].append({
            "sourceType": "LISTENING", "content": item.source_text, "contentHash": item.content_hash,
            "scenarioCategory": metadata.scenario_category.value if metadata else None,
            "communicativeIntent": metadata.communicative_intent.value if metadata else None,
            "taskArchetype": metadata.task_archetype if metadata else None,
            "grammarFocusCodes": metadata.grammar_focus_codes if metadata else [],
            "semanticSummary": metadata.semantic_summary if metadata else None, "ageDays": 0,
        })
        request["constraints"]["recentContentHashes"].append(item.content_hash)
        request["constraints"]["recentSimilaritySummaries"].append(item.similarity_key)
        request["diversityContext"]["exactContentHashes90d"].append(item.content_hash)
    # Preserve the actual source history. If its capacity cannot fit the prefix,
    # fail before a call instead of silently deleting personalization/history.
    return ListeningSetGenerationRequest.model_validate(request).model_dump(mode="json", by_alias=True)


def _verify_waveform(raw: bytes, content_type: str, response: ListeningTtsResponse,
                     item: ListeningItem) -> dict[str, Any]:
    audio, demand = response.audio, item.duration_demand
    if audio is None or demand is None or not audio.duration_validated or audio.duration_policy_version != demand.policy_version:
        raise ValueError("BOUNDARY_AUDIO_DURATION_AUTHORITY_MISSING")
    if content_type.split(";", 1)[0] not in {"audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave"}:
        raise ValueError("BOUNDARY_AUDIO_CONTENT_TYPE_INVALID")
    with wave.open(io.BytesIO(raw), "rb") as source:
        frames, rate, channels = source.getnframes(), source.getframerate(), source.getnchannels()
        if (source.getcomptype() != "NONE" or frames <= 0 or rate < 8000
                or channels not in (1, 2) or source.getsampwidth() not in (1, 2, 3, 4)
                or len(source.readframes(frames)) != frames * channels * source.getsampwidth()):
            raise ValueError("BOUNDARY_AUDIO_FRAMES_INVALID")
    seconds = frames / rate
    checksum = hashlib.sha256(raw).hexdigest()
    if (checksum != audio.checksum or channels != audio.channels or rate != audio.sample_rate
            or abs(seconds * 1000 - audio.duration_ms) > 1):
        raise ValueError("BOUNDARY_AUDIO_METADATA_MISMATCH")
    if not demand.min_seconds <= seconds <= demand.max_seconds:
        raise ValueError("BOUNDARY_AUDIO_OUTSIDE_EFFECTIVE_BOUNDS")
    return {"durationSeconds": seconds, "minSeconds": demand.min_seconds, "maxSeconds": demand.max_seconds,
            "frames": frames, "sampleRate": rate, "channels": channels, "sha256": checksum,
            "withinEffectiveBounds": True, "fullWaveformDecoded": True}


async def run_listening_boundaries(transport: BoundaryTransport, source_request: dict[str, Any],
                                   output_dir: Path, *, campaign_id: str,
                                   source_provenance: dict[str, Any], http_timeout_seconds: float = 300) -> dict[str, Any]:
    """One fixed 2×5 run. Root must authorize/serialize all paid activity externally."""
    if not 1 <= http_timeout_seconds <= 600:
        raise ValueError("BOUNDARY_HTTP_TIMEOUT_MUST_BE_1_TO_600")
    manifest = prepare_manifest(source_request, campaign_id=campaign_id, source_provenance=source_provenance)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path, report_path = output_dir / "manifest.json", output_dir / "result.json"
    if manifest_path.exists() or report_path.exists():
        raise ValueError("BOUNDARY_RUN_ALREADY_STARTED_NO_REPLAY")
    _write_json(manifest_path, manifest)
    report: dict[str, Any] = {"status": "STARTED", "partial": True, "startedAt": utc_now(),
                              "items": [], "httpAttempts": [], "manifest": str(manifest_path),
                              "bePublicationEvidence": False, "humanQualityEvidence": False}

    async def call(method: str, path: str, payload: dict[str, Any] | None, *, case_id: str, stage: str) -> BoundaryHttpResponse:
        if len(report["httpAttempts"]) >= manifest["maximumHttpStarts"]:
            raise ValueError("BOUNDARY_HTTP_START_LIMIT")
        row: dict[str, Any] = {"ordinal": len(report["httpAttempts"]) + 1, "caseId": case_id, "stage": stage,
                               "method": method, "path": path, "request": payload, "status": "STARTED",
                               "startedAt": utc_now(), "timeoutSeconds": http_timeout_seconds}
        report["httpAttempts"].append(row)
        _write_json(report_path, report)  # Reserve before any possible network/provider start.
        started = time.monotonic()
        try:
            response = await asyncio.wait_for(transport(method, path, json_body=payload,
                                                        timeout_seconds=http_timeout_seconds), http_timeout_seconds)
            row.update(httpStatus=response.status, responseBytes=len(response.body),
                       responseSha256=hashlib.sha256(response.body).hexdigest(),
                       status="COMPLETED" if response.status == 200 else "HTTP_FAILED")
            if stage != "DOWNLOAD":
                row["response"] = json.loads(response.body)
            return response
        except BaseException as error:
            row.update(status="INTERRUPTED" if isinstance(error, (KeyboardInterrupt, asyncio.CancelledError)) else "FAILED",
                       errorType=type(error).__name__)
            raise
        finally:
            row.update(latencyMs=round((time.monotonic() - started) * 1000), finishedAt=utc_now())
            _write_json(report_path, report)

    prefixes: dict[str, list[ListeningItem]] = {difficulty: [] for difficulty in _BOUNDS}
    try:
        for case in manifest["cases"]:
            case_id, difficulty = case["caseId"], case["difficulty"]
            row = {key: case[key] for key in ("caseId", "difficulty", "logicalOrder")}
            row["status"] = "STARTED"
            report["items"].append(row)
            payload = _with_prefix(case["generationRequest"], prefixes[difficulty])
            generated_http = await call("POST", _BASE + "/sets/generate", payload, case_id=case_id, stage="GENERATION")
            if generated_http.status != 200:
                row.update(status="GENERATION_FAILED", httpStatus=generated_http.status)
                if generated_http.status >= 500 or generated_http.status == 429:
                    report["status"] = "STOPPED_INFRASTRUCTURE_FAILURE"
                    return report
                continue
            generated = ListeningSetGenerationResponse.model_validate_json(generated_http.body)
            if (generated.request_id != case_id or len(generated.items) != 1 or generated.items[0].item_index != 1):
                raise ValueError("BOUNDARY_GENERATION_IDENTITY_OR_ITEM_COUNT_INVALID")
            item = generated.items[0]
            demand = item.duration_demand
            minimum, maximum = _BOUNDS[difficulty]
            if (demand is None or (demand.min_seconds, demand.max_seconds) != (minimum, maximum)
                    or demand.policy_version != "listening-audio-duration-v1" or item.quality_correction_count != 0):
                raise ValueError("BOUNDARY_GENERATION_DURATION_CONTRACT_INVALID")
            row["generatedItem"] = item.model_dump(mode="json", by_alias=True)
            tts = ListeningTtsRequest.model_validate({
                "requestId": case_id + "-tts", "idempotencyKey": case_id + "-tts", "itemId": case_id,
                "sourceText": item.source_text, "contentHash": item.content_hash, "generationVersion": generated.generation_version,
                "learningLanguage": payload["userContext"]["learningLanguage"], "voice": payload["referenceVoice"],
                "playbackSpeed": "NORMAL", "policyVersion": generated.policy_version,
                "modelConfigVersion": generated.model_config_version, "durationDemand": demand.model_dump(mode="json", by_alias=True),
            })
            tts_http = await call("POST", _BASE + "/tts", tts.model_dump(mode="json", by_alias=True), case_id=case_id, stage="TTS")
            if tts_http.status != 200:
                row.update(status="TTS_FAILED", httpStatus=tts_http.status)
                if tts_http.status >= 500 or tts_http.status == 429:
                    report["status"] = "STOPPED_INFRASTRUCTURE_FAILURE"
                    return report
                continue
            response = ListeningTtsResponse.model_validate_json(tts_http.body)
            if (response.request_id != tts.request_id or response.item_id != case_id or response.content_hash != item.content_hash
                    or response.source_text != item.source_text or response.generation_version != generated.generation_version):
                raise ValueError("BOUNDARY_TTS_IDENTITY_MISMATCH")
            if response.status == "FAILED":
                row.update(status="TTS_FAILED", error=response.error.model_dump(mode="json", by_alias=True) if response.error else None)
                continue  # No content correction or re-synthesis in this direct-AI probe.
            audio = response.audio
            if audio is None or audio.voice != tts.voice or not re.fullmatch(r"listening-tts-[a-f0-9]{32}", audio.audio_reference):
                raise ValueError("BOUNDARY_AUDIO_REFERENCE_OR_VOICE_INVALID")
            downloaded = await call("GET", _BASE + "/audio/" + audio.audio_reference, None, case_id=case_id, stage="DOWNLOAD")
            if downloaded.status != 200:
                row.update(status="DOWNLOAD_FAILED", httpStatus=downloaded.status)
                continue
            audio_path = output_dir / f"{case_id}.wav"
            audio_path.write_bytes(downloaded.body)
            row["audioArtifact"] = str(audio_path)
            row["audio"] = _verify_waveform(downloaded.body, downloaded.content_type, response, item)
            row["status"] = "COMPLETED"
            prefixes[difficulty].append(item)
            _write_json(report_path, report)
        completed = sum(item["status"] == "COMPLETED" for item in report["items"])
        report.update(status="COMPLETED" if completed == 10 else "COMPLETED_WITH_FAILURES", partial=False)
        return report
    except BaseException as error:
        report.update(status="INTERRUPTED" if isinstance(error, (KeyboardInterrupt, asyncio.CancelledError)) else "FAILED",
                      errorType=type(error).__name__)
        raise
    finally:
        report.update(finishedAt=utc_now(), httpStarts=len(report["httpAttempts"]),
                      completedItems=sum(item["status"] == "COMPLETED" for item in report["items"]))
        _write_json(report_path, report)
