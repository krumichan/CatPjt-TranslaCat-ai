"""One-shot B5 STRUCTURE follow-up on the owned, unchanged campaign ledger.

The three inputs are copied from the prior failed comparison/holdout reports.
Each is generated once with the selected Reading Sol configuration. Private
diagnostic output is not a gold educational-quality label or a BE result.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.qa_campaign_budget import atomic_json
from scripts.qa_campaign_integration import _owned_manifest


SOURCE_NAMES = {
    "research": "b5-structure-research-B.json",
    "technology": "b5-structure-technology-B.json",
    "tourism": "holdout-b5-tourism-B.json",
}
PHASE = "B5_STRUCTURE_FOLLOWUP"
CAPS = {"starts": 80, "input_tokens": 200_000, "output_tokens": 60_000,
        "usd": 15.0, "active_seconds": 2700.0}


class B5LocalCapReached(BaseException):
    """Stop QA immediately; production's ordinary Exception retry must not eat it."""


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(output: Path) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[1]
    return {
        "version": "b5-structure-followup-v1",
        "campaignLedger": str(output.parent / "budget-ledger.json"),
        "cases": {name: {"source": filename, "sha256": _hash(output / filename)}
                  for name, filename in SOURCE_NAMES.items()},
        "sourceSha256": {name: _hash(repo / relative) for name, relative in {
            "recipe": "app/features/language_learning/reading_vocabulary/reading_difficulty_recipe.py",
            "prompts": "app/features/language_learning/reading_vocabulary/prompts.py",
            "service": "app/features/language_learning/reading_vocabulary/service.py",
        }.items()},
        "trackCaps": CAPS,
        "provider": "selected Reading Sol/high; existing Mini/Nano, attempts and 3+2",
        "boundary": "One new revision per old failure, no original artifact/DB overwrite",
    }


async def _run(output: Path, name: str, manifest: dict[str, Any]) -> dict[str, Any]:
    # Must precede importing settings/model policy/provider factory.
    os.environ["AI_READING_GENERATION_MODEL"] = "SOL"
    from app.ai.model_policy import get_model_name_for_task, get_task_model_policy
    from app.ai.prompt_registry import get_prompt_rule
    from app.ai.provider_factory import create_text_generation_provider
    from app.core.config import settings
    from app.features.language_learning.reading_vocabulary.service import ReadingVocabularyGenerationService
    from app.schemas.language_learning_practice import PracticeGenerationRequest
    from scripts.qa_campaign_budget import (
        BudgetedTextProvider, CampaignLedger,
        _text_cost_reservation,
    )
    from scripts.run_contextual_choice_stabilization_qa import QaCaps, RecordingProvider

    if settings.AI_READING_GENERATION_MODEL != "SOL":
        raise ValueError("Selected Reading Sol setting was not applied")
    source_path = output / SOURCE_NAMES[name]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if len(source["requests"]) != 2 or source.get("status") == "GENERATED_3_PLUS_2":
        raise ValueError("Source is not an incomplete, fixed 3+2 input")
    first_data, second_data = source["requests"]
    if ([first_data["questionCount"], second_data["questionCount"]] != [3, 2]
            or first_data["mode"] != "STRUCTURE"
            or second_data["mode"] != "STRUCTURE"
            or first_data["complexityBand"] != 5
            or second_data["complexityBand"] != 5):
        raise ValueError("Fixed B5 input contract changed")

    ledger = CampaignLedger(output.parent / "budget-ledger.json", output.parent.name)

    class TrackCappedProvider(BudgetedTextProvider):
        async def call_with_metadata(self, type_name: str, data: str, schema: dict | None = None) -> Any:
            model = get_model_name_for_task(type_name)
            policy = get_task_model_policy(type_name)
            text = data + (get_prompt_rule(type_name) or "") + json.dumps(schema, ensure_ascii=True)
            reserve_in = len(text.encode("utf-8")) + 4096
            reserve_out = policy.max_output_tokens
            reserve_usd = _text_cost_reservation(model, reserve_in, reserve_out)
            entries = [row for row in ledger.snapshot()["calls"]
                       if row.get("metadata", {}).get("phase") == PHASE]
            used = {key: 0.0 for key in CAPS}
            used["starts"] = len(entries)
            for row in entries:
                charged = row.get("accounted") or row["reserved"]
                for key in ("input_tokens", "output_tokens", "usd"):
                    used[key] += float(charged[key])
                used["active_seconds"] += float(row.get("latencyMs", 0)) / 1000
            proposed = {"starts": 1, "input_tokens": reserve_in,
                        "output_tokens": reserve_out, "usd": reserve_usd,
                        "active_seconds": self.call_seconds}
            for key, value in proposed.items():
                if used[key] + value > CAPS[key]:
                    raise B5LocalCapReached("B5_LOCAL_CAP_" + key.upper())
            return await super().call_with_metadata(type_name, data, schema)

    provider = TrackCappedProvider(create_text_generation_provider(), ledger, phase=PHASE)
    recorder = RecordingProvider(provider, QaCaps(35, 200_000, 60_000, 90, 900),
                                 diagnostic_capture=True)
    result_path = output / f"b5-followup-{name}.private.json"
    report: dict[str, Any] = {
        "case": name, "sourceSha256": manifest["cases"][name]["sha256"],
        "sourceCodeSha256": manifest["sourceSha256"],
        "status": "STARTED", "partial": True,
        "modelSetting": settings.AI_READING_GENERATION_MODEL,
        "passageModel": get_model_name_for_task("LANGUAGE_LEARNING_READING_PASSAGE_GENERATION"),
        "questionModel": get_model_name_for_task("LANGUAGE_LEARNING_READING_QUESTION_GENERATION_SOL"),
        "requests": [first_data, second_data], "providerAttempts": [], "responses": [],
        "beDbIntegrated": False, "boundary": "DIRECT_AI_3_PLUS_2_NOT_CONTENT_GOLD",
    }

    def checkpoint() -> None:
        report["providerAttempts"] = recorder.calls
        atomic_json(result_path, report)

    recorder.checkpoint = checkpoint
    phase_starts_before = sum(
        row.get("metadata", {}).get("phase") == PHASE
        for row in ledger.snapshot()["calls"]
    )
    started = time.monotonic()
    checkpoint()
    try:
        await provider.warm_up()
        service = ReadingVocabularyGenerationService(provider=recorder)
        first = PracticeGenerationRequest.model_validate({
            **first_data, "requestId": f"b5-followup-{name}-p1",
        })
        first_result = await service.generate(first)
        if len(first_result.questions) != 3 or first_result.reading_bundle is None:
            raise ValueError("P1_BUNDLE_INCOMPLETE")
        report["responses"].append(first_result.model_dump(mode="json", by_alias=True))
        checkpoint()
        prefix = [question.model_copy(update={"order": index})
                  for index, question in enumerate(first_result.questions, 1)]
        second = PracticeGenerationRequest.model_validate({
            **second_data, "requestId": f"b5-followup-{name}-p2",
            "previousQuestions": [question.model_dump(mode="json", by_alias=True)
                                  for question in prefix],
            "readingBundles": {"p1": first_result.reading_bundle.model_dump(
                mode="json", by_alias=True,
            )},
        })
        second_result = await service.generate(second)
        if len(second_result.questions) != 2 or second_result.reading_bundle is None:
            raise ValueError("P2_BUNDLE_INCOMPLETE")
        report["responses"].append(second_result.model_dump(mode="json", by_alias=True))
        report.update(status="GENERATED_3_PLUS_2", partial=False)
    except BaseException as exc:
        report["status"] = "INTERRUPTED" if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)) else "FAILED"
        report["failureType"] = type(exc).__name__
        report["failureMessage"] = str(exc)[:500]
        if isinstance(exc, B5LocalCapReached):
            report["stopReason"] = str(exc)
    finally:
        report["latencySeconds"] = round(time.monotonic() - started, 3)
        report["recordedProviderAttempts"] = len(recorder.calls)
        report["providerStarts"] = sum(
            row.get("metadata", {}).get("phase") == PHASE
            for row in ledger.snapshot()["calls"]
        ) - phase_starts_before
        report["inputTokens"] = sum(row["inputTokens"] for row in recorder.calls)
        report["outputTokens"] = sum(row["outputTokens"] for row in recorder.calls)
        checkpoint()
        await provider.shutdown()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--case", choices=tuple(SOURCE_NAMES))
    args = parser.parse_args()
    output = args.output.resolve()
    _owned_manifest(output.parent / "integration-environment")
    if output.name != "closure-20260920-v1":
        raise ValueError("Only the owned closure directory is admitted")
    path = output / "b5-followup-manifest.json"
    expected = _manifest(output)
    if args.prepare:
        if args.case or path.exists():
            raise ValueError("Prepare is one-shot and cannot call a provider")
        atomic_json(path, expected)
        print(json.dumps({"status": "PREPARED", "cases": list(SOURCE_NAMES), "providerStarts": 0}))
        return
    if not args.case or not path.exists() or json.loads(path.read_text(encoding="utf-8")) != expected:
        raise ValueError("Fixed manifest/source mismatch")
    if (output / f"b5-followup-{args.case}.private.json").exists():
        raise ValueError("This revision/case already started; no repeat")
    result = asyncio.run(_run(output, args.case, expected))
    print(json.dumps({key: result.get(key) for key in (
        "case", "status", "failureType", "latencySeconds", "providerStarts",
        "inputTokens", "outputTokens", "passageModel", "questionModel",
    )}))


if __name__ == "__main__":
    main()
