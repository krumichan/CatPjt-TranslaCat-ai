"""One-shot, budgeted FREE session 9 evaluation-only replay.

The source is the exact durable QA request. No TTS, recording, STT, BE write,
or browser action occurs. Raw evidence and model output stay in an owned private
artifact; this is not a fresh Speaking E2E or a human-pronunciation benchmark.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time

from app.ai.model_policy import get_model_name_for_task, get_task_model_policy
from app.ai.prompt_registry import get_prompt_rule
from app.ai.provider_factory import create_text_generation_provider
from app.core.config import settings
from app.features.language_learning.speaking.evaluation_service import (
    SpeakingEvaluationService,
    _evaluation_response_schema,
)
from app.features.language_learning.speaking.prompts import build_evaluation_prompt
from app.schemas.language_learning_speaking import SpeakingEvaluationRequest
from scripts.qa_campaign_budget import (
    BudgetedTextProvider,
    CampaignLedger,
    _text_cost_reservation,
    atomic_json,
)
from scripts.qa_campaign_integration import _owned_manifest


SOURCE_SHA256 = "1a76bffd8237f5ab5c0d5fd63f2fc24acf0895baed4e3e8712f245a30d908319"
PHASE = "FREE_FOLLOWUP_FIXED_EVALUATION"


def _fixed_request(source: Path) -> SpeakingEvaluationRequest:
    captured = json.loads(source.read_text(encoding="utf-8"))
    if (captured.get("requestSha256") != SOURCE_SHA256
            or captured.get("sessionId") != 9 or captured.get("jobStatus") != "SUCCEEDED"):
        raise ValueError("Fixed owned session 9 source mismatch")
    request = SpeakingEvaluationRequest.model_validate(captured["request"])
    if request.session_id != "9" or request.practice_mode.value != "FREE":
        raise ValueError("Fixed request identity mismatch")
    return request


def _admission(ledger: CampaignLedger, request: SpeakingEvaluationRequest) -> dict:
    snapshot = ledger.snapshot()
    phase_calls = [call for call in snapshot["calls"] if call.get("metadata", {}).get("phase") == PHASE]
    if phase_calls:
        raise ValueError("One fixed FREE replay already started; no same-revision repetition")
    if snapshot.get("stoppedReason") or ledger.remaining_seconds() <= 0:
        raise ValueError("Campaign halted or expired")
    if settings.AI_TEXT_PROVIDER != "openai":
        raise ValueError("Expected OpenAI QA text provider")
    task = SpeakingEvaluationService.TYPE_NAME
    model = get_model_name_for_task(task)
    if model != "gpt-5-mini":
        raise ValueError("Production Mini evaluation model changed; stop before cost")
    prompt = build_evaluation_prompt(request)
    schema = _evaluation_response_schema(request)
    conservative_input = len((prompt + (get_prompt_rule(task) or "")
                              + json.dumps(schema, ensure_ascii=True)).encode("utf-8")) + 4096
    conservative_output = get_task_model_policy(task).max_output_tokens
    conservative_usd = _text_cost_reservation(model, conservative_input, conservative_output)
    if (conservative_input > 100_000 or conservative_output > 30_000
            or conservative_usd > 10 or ledger.remaining_seconds() < 90):
        raise ValueError("FREE local reservation cannot fit before provider start")
    return {
        "task": task, "model": model,
        "inputTokenReservation": conservative_input,
        "outputTokenReservation": conservative_output,
        "usdReservation": conservative_usd,
        "promptSha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "assistantReferenceIds": [turn["referenceAssistantTurnId"] for turn in
                                  json.loads(prompt.rsplit("\n\n", 1)[1])["evaluationTaskBindings"]],
    }


async def _run(campaign: Path, source: Path, output: Path, *, live: bool) -> dict:
    _owned_manifest(campaign / "integration-environment")
    if output.exists():
        raise ValueError("Refusing to replace an existing replay result")
    request = _fixed_request(source)
    ledger = CampaignLedger(campaign / "budget-ledger.json", campaign.name)
    reservation = _admission(ledger, request)
    report = {
        "status": "DRY_RUN" if not live else "STARTED", "partial": live,
        "sourceSha256": SOURCE_SHA256,
        "sourceArtifactSha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source": str(source), "mode": "EVALUATION_ONLY", "providerCalls": 0,
        "reservation": reservation,
        "boundary": "Stored request replay; no TTS/STT/new session; existing Mini and 0.7 gate",
    }
    if not live:
        atomic_json(output, report)
        return report
    atomic_json(output, report)
    provider = BudgetedTextProvider(create_text_generation_provider(), ledger, phase=PHASE)
    service = SpeakingEvaluationService(provider, automatic_retries=0)
    started = time.monotonic()
    try:
        response = await service.evaluate(request)
        report.update(status="COMPLETED", partial=False, providerCalls=1,
                      response=response.model_dump(mode="json", by_alias=True))
    except BaseException as exc:
        report.update(status="INTERRUPTED" if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt))
                      else "FAILED", partial=True, errorType=type(exc).__name__,
                      errorCode=getattr(getattr(exc, "code", None), "value", None))
        raise
    finally:
        report["elapsedSeconds"] = round(time.monotonic() - started, 3)
        report["ledgerPhaseAttempts"] = len([
            call for call in ledger.snapshot()["calls"]
            if call.get("metadata", {}).get("phase") == PHASE
        ])
        atomic_json(output, report)
        await provider.shutdown()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    report = asyncio.run(_run(args.campaign.resolve(), args.source.resolve(),
                              args.output.resolve(), live=args.live))
    print(json.dumps({key: report.get(key) for key in
                      ("status", "providerCalls", "elapsedSeconds", "ledgerPhaseAttempts")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
