"""One-shot copied 4.45s Listening correction through the owned QA AI server.

This is a direct AI diagnostic, not evidence of BE storage/browser publication.
The original short WAV is not reconstructed; only its saved measured duration
and source text enter the existing correction contract. Never retries HTTP.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import httpx

from app.features.language_learning.listening.duration import (
    decode_reference_audio, duration_demand, minimum_short_correction_characters,
)
from app.features.language_learning.listening.normalization import normalize_text
from app.schemas.language_learning_listening import (
    ListeningSetGenerationRequest, ListeningSetGenerationResponse,
    ListeningTtsRequest, ListeningTtsResponse,
)
from scripts.qa_campaign_budget import atomic_json


def prepare(snapshot: dict[str, Any]) -> tuple[ListeningSetGenerationRequest, dict[str, Any]]:
    if (snapshot.get("dailySetId") != 5 or snapshot.get("mode") != "DICTATION"
            or snapshot.get("difficulty") != "EASY" or snapshot.get("status") != "PARTIAL"):
        raise ValueError("Not the identified isolated short-audio source set")
    rows = [row for row in snapshot["items"] if row["itemIndex"] == 5
            and row["replacementSequence"] == 0 and row["failureReason"] == "AUDIO_TOO_SHORT"]
    if len(rows) != 1 or rows[0]["generationMetadata"]["qualityCorrectionCount"] != 1:
        raise ValueError("Original item or reserved correction not present")
    original = rows[0]["sourceText"]
    prefix = []
    for row in snapshot["items"]:
        if row["itemIndex"] >= 5 or row["status"] != "READY":
            continue
        metadata = row["generationMetadata"].get("diversityMetadata") or {}
        prefix.append({
            "sourceType": "LISTENING", "content": row["sourceText"],
            "contentHash": row["generationMetadata"]["contentHash"],
            "scenarioCategory": metadata.get("scenarioCategory"),
            "communicativeIntent": metadata.get("communicativeIntent"),
            "taskArchetype": metadata.get("taskArchetype"),
            "grammarFocusCodes": metadata.get("grammarFocusCodes") or [],
            "semanticSummary": metadata.get("semanticSummary"), "ageDays": 0,
        })
    if len(prefix) != 4:
        raise ValueError("The accepted four-item prefix is not intact")
    request = ListeningSetGenerationRequest.model_validate({
        "requestId": "closure-short-correction-set5-copy-v2",
        "idempotencyKey": "closure-short-correction-set5-copy-v2",
        "userContext": {"originLanguage": "ko", "learningLanguage": "ja", "level": "EASY",
                        "profileFocus": ["LISTENING_RECOGNITION"]},
        "setContext": {"learningDate": "2026-09-20", "learningMode": "DICTATION",
                       "topic": {"id": "owned-qa-copy", "title": "환경과 생활 습관"},
                       "selectedKeywords": [{"key": "environment", "text": "環境",
                                             "source": "SYSTEM", "type": "TOPIC"}],
                       "itemCount": 1, "difficulty": "EASY"},
        "constraints": {"audioSecondsMin": 5.0, "audioSecondsMax": 12.0},
        "diversityContext": {"currentSession": prefix,
                              "exactContentHashes90d": [row["contentHash"] for row in prefix]},
        "contentDiversityPolicyVersion": "language-learning-diversity",
        "durationCorrection": {"previousSourceText": original,
                               "previousMeasuredSeconds": 4.45,
                               "qualityCorrectionCount": 1},
        "referenceVoice": {"locale": "ja", "voiceKey": "marin",
                           "version": "openai-speech-v1", "accent": "STANDARD"},
        "languageComplexity": {"baseComplexityBand": 4, "targetComplexityBand": 3},
        "policyVersion": "listening", "modelConfigVersion": "listening-model-config",
    })
    source = {"sourceSetId": 5, "sourceItemId": rows[0]["itemId"],
              "sourceText": original, "copiedMeasuredSeconds": 4.45,
              "originalAudioBytesAvailable": False,
              "sourceGenerationMetadata": rows[0]["generationMetadata"],
              "minimumCorrectionCharacters": minimum_short_correction_characters(request),
              "acceptedPrefixCount": len(prefix)}
    return request, source


async def run(snapshot_path: Path, output: Path, *, environment: Path, port: int) -> dict[str, Any]:
    if output.exists():
        raise ValueError("Existing diagnostic output cannot be replayed")
    if port != 18084:
        raise ValueError("Only owned isolated QA AI port is allowed")
    manifest = json.loads((environment / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("owner") != "translacat-isolated-qa"
            or manifest.get("campaign") != "openai-speech-campaign-20260920"
            or manifest["ports"]["ai"] != port):
        raise ValueError("Owned QA AI environment changed")
    private = json.loads((environment / "secrets.private.json").read_text(encoding="utf-8"))
    api_key = private["QA_AI_API_KEY"]
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    request, source = prepare(snapshot)
    result: dict[str, Any] = {
        "status": "STARTED", "partial": True, "provenance": source,
        "sourceSnapshotSha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
        "generationRequest": request.model_dump(mode="json", by_alias=True),
        "httpAttempts": [], "providerBudgetAuthority": "existing global campaign QA AI wrapper",
        "beStorageVerified": False, "browserPlaybackVerified": False,
    }
    atomic_json(output, result)
    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=300,
                                 headers={"X-API-KEY": api_key}, trust_env=False) as client:
        async def post(stage: str, path: str, payload: dict[str, Any]) -> httpx.Response:
            row = {"stage": stage, "status": "STARTED", "path": path, "startedAt": time.time()}
            result["httpAttempts"].append(row)
            atomic_json(output, result)
            start = time.monotonic()
            try:
                response = await client.post(path, json=payload)
                row.update(status="COMPLETED" if response.status_code == 200 else "HTTP_FAILED",
                           httpStatus=response.status_code,
                           responseSha256=hashlib.sha256(response.content).hexdigest())
                if response.status_code != 200:
                    result["safeError"] = {"stage": stage, "httpStatus": response.status_code}
                return response
            except BaseException as error:
                row.update(status="INTERRUPTED" if isinstance(error, (KeyboardInterrupt, asyncio.CancelledError))
                           else "FAILED", errorType=type(error).__name__)
                raise
            finally:
                row["latencyMs"] = round((time.monotonic() - start) * 1000)
                atomic_json(output, result)

        try:
            gen = await post("GENERATION", "/api/v1/language-learning/listening/sets/generate",
                             request.model_dump(mode="json", by_alias=True))
            gen.raise_for_status()
            generated = ListeningSetGenerationResponse.model_validate_json(gen.content)
            if len(generated.items) != 1 or generated.items[0].quality_correction_count != 1:
                raise ValueError("CORRECTION_ITEM_CONTRACT_INVALID")
            item = generated.items[0]
            result["generatedItem"] = item.model_dump(mode="json", by_alias=True)
            result["minimumCharactersMet"] = len(item.source_text.strip()) >= source["minimumCorrectionCharacters"]
            atomic_json(output, result)
            text_hash = hashlib.sha256(normalize_text(item.source_text, "ja").text.encode()).hexdigest()
            tts_req = ListeningTtsRequest.model_validate({
                "requestId": "closure-short-correction-tts-v1",
                "idempotencyKey": "closure-short-correction-tts-v1", "itemId": "closure-copy-item5",
                "sourceText": item.source_text, "contentHash": text_hash,
                "generationVersion": generated.generation_version, "learningLanguage": "ja",
                "voice": request.reference_voice.model_dump(mode="json", by_alias=True),
                "durationDemand": duration_demand(request).model_dump(mode="json", by_alias=True),
            })
            tts = await post("TTS", "/api/v1/language-learning/listening/tts",
                             tts_req.model_dump(mode="json", by_alias=True))
            tts.raise_for_status()
            response = ListeningTtsResponse.model_validate_json(tts.content)
            result["ttsResponse"] = response.model_dump(mode="json", by_alias=True)
            atomic_json(output, result)
            if response.status != "READY" or response.audio is None:
                result["status"] = "TTS_REJECTED"
                return result
            audio = await client.get("/api/v1/language-learning/listening/audio/"
                                     + response.audio.audio_reference)
            audio.raise_for_status()
            decoded = decode_reference_audio(audio.content, audio.headers["content-type"])
            result["wav"] = {"sha256": hashlib.sha256(audio.content).hexdigest(),
                             "frames": decoded.frame_count, "sampleRate": decoded.sample_rate,
                             "seconds": decoded.duration_seconds,
                             "withinDemand": duration_demand(request).min_seconds <= decoded.duration_seconds
                             <= duration_demand(request).max_seconds}
            result["status"] = "DIRECT_AI_CORRECTION_READY" if result["wav"]["withinDemand"] else "WAV_OUT_OF_RANGE"
            return result
        except BaseException as error:
            result["status"] = "INTERRUPTED" if isinstance(error, (KeyboardInterrupt, asyncio.CancelledError)) else "FAILED"
            result["errorType"] = type(error).__name__
            return result
        finally:
            result["partial"] = result["status"] != "DIRECT_AI_CORRECTION_READY"
            atomic_json(output, result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18084)
    args = parser.parse_args()
    result = asyncio.run(run(args.snapshot, args.output, environment=args.environment, port=args.port))
    print(json.dumps({"status": result["status"], "httpStages": len(result["httpAttempts"]),
                      "measuredSeconds": result.get("wav", {}).get("seconds")}))


if __name__ == "__main__":
    main()
