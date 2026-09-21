"""One production evaluation invocation from immutable captured evidence.

No provider construction, generation, TTS, STT or audio alteration. The caller
must inject its existing campaign-budgeted provider. This is not a CLI approval
to execute a paid comparison.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from app.ai.ports import StructuredTextGenerationProvider
from app.features.language_learning.speaking.evaluation_service import SpeakingEvaluationService
from app.features.language_learning.speaking.prompts import build_evaluation_prompt
from app.schemas.language_learning_speaking import SpeakingEvaluationRequest
from scripts.qa_campaign_speaking import _Capture, _TextCapture


def captured_read_aloud_request(source: dict[str, Any]) -> tuple[SpeakingEvaluationRequest, str, int]:
    """Restore request fields; recompute only the existing prompt-derived fields."""
    for attempt in source.get("attempts", []):
        captured = attempt.get("request", {})
        if captured.get("type") != SpeakingEvaluationService.TYPE_NAME:
            continue
        prompt = captured["data"]
        body = json.loads(prompt.rsplit("\n\n", 1)[1])
        if body.get("practiceMode") != "READ_ALOUD" or body.get("evaluationScope") != "READ_ALOUD_PROBLEM":
            continue
        # These are computed by build_evaluation_prompt, not public request fields.
        body.pop("pronunciationEvidenceAvailable", None)
        body.pop("evidenceContract", None)
        body.pop("evaluationCapabilities", None)
        body.pop("evaluationTaskBindings", None)
        for turn in body["userTurns"]:
            turn.pop("assistanceLevel", None)
            turn.pop("transcriptObservation", None)
            for segment in turn.get("segments", []):
                if segment.get("timestampUsableForEvidence") is False:
                    raise ValueError("Sanitized prompt requires the original request snapshot")
        request = SpeakingEvaluationRequest.model_validate(body)
        if build_evaluation_prompt(request) != prompt:
            raise ValueError("Captured request no longer reproduces the original prompt exactly")
        return request, prompt, attempt["sequence"]
    raise ValueError("No captured READ_ALOUD_PROBLEM evaluation request")


async def run_captured_speaking_evaluation_retest(
    text_provider: StructuredTextGenerationProvider,
    source_artifact: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Call evaluate exactly once; retain its existing automatic retry limit."""
    source_bytes = Path(source_artifact).read_bytes()
    source = json.loads(source_bytes)
    request, prompt, sequence = captured_read_aloud_request(source)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    artifact = directory / "speaking-evaluation-retest.json"
    if artifact.exists():
        raise ValueError("Retest artifact already exists; a repeat run is not authorized by this helper")
    result: dict[str, Any] = {
        "status": "RUNNING", "partial": True, "attempts": [],
        "sourceSha256": hashlib.sha256(source_bytes).hexdigest(),
        "originalEvaluationSequence": sequence,
        "originalPromptSha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "request": request.model_dump(mode="json", by_alias=True),
        "boundary": "ONE_EVALUATE_INVOCATION_WITH_EXISTING_STAGE_RETRIES_NO_GENERATION_TTS_STT",
        "qualityBoundary": "SYNTHETIC_AUDIO_TRANSCRIPT_METADATA_NOT_HUMAN_PRONUNCIATION_GOLD",
    }
    capture = _Capture(directory, result)
    capture.path = artifact
    service = SpeakingEvaluationService(_TextCapture(text_provider, capture))
    result["existingProviderAttemptCap"] = 1 + service.automatic_retries
    capture.flush()
    try:
        response = await service.evaluate(request)
        capture.raise_if_halted()
        result.update(status="COMPLETED", partial=False,
                      response=response.model_dump(mode="json", by_alias=True))
    except BaseException as exc:
        result.update(status="INTERRUPTED" if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt)) else "FAILED",
                      partial=True, error={"type": type(exc).__name__,
                                           "code": getattr(getattr(exc, "code", None), "value", None)})
        raise
    finally:
        capture.flush()
    return result
