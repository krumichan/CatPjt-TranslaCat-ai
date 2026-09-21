"""One-shot, private Reading A/B corpus on the existing campaign ledger.

Each subprocess gets one fixed case/arm. It generates a complete p1 (3) and
p2 (2) in that order, reusing the accepted p1 prefix and private passage plan.
It never writes BE/production data; a successful model verdict is not a gold
content-quality label. Inspect the saved questions independently before choice.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import date
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


CASES = (
    {"id": "b1-comprehension-home", "band": 1, "mode": "COMPREHENSION",
     "keywords": ["家族", "生活"], "weak": ["DETAIL"], "mistakes": []},
    {"id": "b1-comprehension-shopping", "band": 1, "mode": "COMPREHENSION",
     "keywords": ["買い物", "時間"], "weak": ["GIST"], "mistakes": ["細部を取り違える"]},
    {"id": "b3-inference-transport", "band": 3, "mode": "CONTEXT_INFERENCE",
     "keywords": ["交通", "仕事"], "weak": ["CONTEXT_INFERENCE"], "mistakes": []},
    {"id": "b3-inference-environment", "band": 3, "mode": "CONTEXT_INFERENCE",
     "keywords": ["環境", "地域"], "weak": ["INFERENCE"], "mistakes": ["暗示と事実を混同"]},
    {"id": "b5-structure-research", "band": 5, "mode": "STRUCTURE",
     "keywords": ["研究", "社会"], "weak": ["STRUCTURE"], "mistakes": []},
    {"id": "b5-structure-technology", "band": 5, "mode": "STRUCTURE",
     "keywords": ["技術", "教育"], "weak": ["GIST"], "mistakes": ["筆者の立場を読み違える"]},
)


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _requests(case: dict[str, Any], learning_date: str) -> tuple[dict[str, Any], dict[str, Any]]:
    shared = {
        "domain": "READING", "mode": case["mode"], "originLanguage": "ko",
        "learningLanguage": "ja", "complexityBand": case["band"],
        "selectedKeywords": case["keywords"], "weakSignals": case["weak"],
        "recentMistakes": case["mistakes"], "generationDate": learning_date,
    }
    return (
        {**shared, "questionCount": 3, "easierCount": 1, "currentCount": 2, "challengeCount": 0},
        {**shared, "questionCount": 2, "easierCount": 0, "currentCount": 1, "challengeCount": 1},
    )


def _manifest(output: Path, *, learning_date: str | None = None) -> dict[str, Any]:
    source = Path(__file__).resolve()
    return {
        "version": "reading-generation-comparison-v1",
        "campaignLedger": str(output.parent / "budget-ledger.json"),
        "sourceSha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "learningDate": learning_date or date.today().isoformat(),
        "arms": {"A": "LUNA", "B": "SOL"},
        "cases": list(CASES),
        "runOrder": [{"caseId": case["id"], "arm": arm}
                     for i, case in enumerate(CASES)
                     for arm in (("A", "B") if i % 2 == 0 else ("B", "A"))],
        "qualityBoundary": "Independent blinded review; verifier PASS/READY is not a gold label",
        "providerBudget": "One existing campaign ledger, no reset; one-shot per case/arm",
        "maxApplicationStartsPerSet": 35,
    }


async def _run(case: dict[str, Any], arm: str, output: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    # Environment must be set before importing app.core.config or the provider.
    os.environ["AI_READING_GENERATION_MODEL"] = manifest["arms"][arm]
    from app.ai.provider_factory import create_text_generation_provider
    from app.ai.model_policy import get_model_name_for_task
    from app.core.config import settings
    from app.features.language_learning.reading_vocabulary.service import ReadingVocabularyGenerationService
    from app.schemas.language_learning_practice import PracticeGenerationRequest
    from scripts.qa_campaign_budget import BudgetedTextProvider, CampaignLedger
    from scripts.run_contextual_choice_stabilization_qa import QaCaps, RecordingProvider

    if settings.AI_READING_GENERATION_MODEL != manifest["arms"][arm]:
        raise ValueError("Reading model setting not applied in this process")
    ledger = CampaignLedger(output.parent / "budget-ledger.json", output.parent.name)
    recorder = RecordingProvider(
        BudgetedTextProvider(create_text_generation_provider(), ledger, phase="READING_AB_" + arm),
        QaCaps(35, 250_000, 75_000, 90, 1800),
        diagnostic_capture=True,
    )
    path = output / f"{case['id']}-{arm}.json"
    a, b = _requests(case, manifest["learningDate"])
    report: dict[str, Any] = {
        "caseId": case["id"], "arm": arm, "status": "STARTED", "partial": True,
        "caseSha256": _sha(case), "sameInputSha256": _sha([a, b]),
        "modelSetting": settings.AI_READING_GENERATION_MODEL,
        "passageModel": get_model_name_for_task("LANGUAGE_LEARNING_READING_PASSAGE_GENERATION"),
        "questionModel": get_model_name_for_task(
            "LANGUAGE_LEARNING_READING_QUESTION_GENERATION_SOL" if arm == "B"
            else "LANGUAGE_LEARNING_READING_VOCABULARY_GENERATION"),
        "requests": [a, b], "providerAttempts": [], "responses": [],
        "startedMonotonic": time.monotonic(),
        "beDbIntegrated": False,
    }
    def checkpoint() -> None:
        report["providerAttempts"] = recorder.calls
        atomic_json(path, report)
    recorder.checkpoint = checkpoint
    checkpoint()
    service = ReadingVocabularyGenerationService(provider=recorder)
    try:
        await recorder.upstream.warm_up()
        first = PracticeGenerationRequest.model_validate({**a, "requestId": f"comparison-{case['id']}-{arm}-p1"})
        result1 = await service.generate(first)
        if len(result1.questions) != 3 or result1.reading_bundle is None:
            raise ValueError("P1_BUNDLE_INCOMPLETE")
        report["responses"].append(result1.model_dump(mode="json", by_alias=True))
        checkpoint()
        prefix = [q.model_copy(update={"order": i}) for i, q in enumerate(result1.questions, 1)]
        second = PracticeGenerationRequest.model_validate({
            **b, "requestId": f"comparison-{case['id']}-{arm}-p2",
            "previousQuestions": [q.model_dump(mode="json", by_alias=True) for q in prefix],
            "readingBundles": {"p1": result1.reading_bundle.model_dump(mode="json", by_alias=True)},
        })
        result2 = await service.generate(second)
        if len(result2.questions) != 2 or result2.reading_bundle is None:
            raise ValueError("P2_BUNDLE_INCOMPLETE")
        report["responses"].append(result2.model_dump(mode="json", by_alias=True))
        report["status"] = "GENERATED_3_PLUS_2"
        report["partial"] = False
    except BaseException as exc:
        report["status"] = "INTERRUPTED" if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)) else "FAILED"
        report["failureType"] = type(exc).__name__
        report["failureMessage"] = str(exc)[:500]
    finally:
        report["latencySeconds"] = round(time.monotonic() - report["startedMonotonic"], 3)
        report["providerStarts"] = len(recorder.calls)
        report["inputTokens"] = sum(row["inputTokens"] for row in recorder.calls)
        report["outputTokens"] = sum(row["outputTokens"] for row in recorder.calls)
        report["modelCounts"] = {model: sum(row.get("model") == model for row in recorder.calls)
                                 for model in {str(row.get("model")) for row in recorder.calls}}
        checkpoint()
        await recorder.upstream.shutdown()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--case-id", choices=[case["id"] for case in CASES])
    parser.add_argument("--arm", choices=("A", "B"))
    args = parser.parse_args()
    output = args.output.resolve()
    _owned_manifest(output.parent / "integration-environment")
    if output.name != "closure-20260920-v1":
        raise ValueError("Only the owned closure directory is admitted")
    path = output / "reading-comparison-manifest.json"
    if args.prepare:
        if args.arm or args.case_id or path.exists():
            raise ValueError("Prepare is one-shot and cannot start providers")
        atomic_json(path, _manifest(output))
        print(json.dumps({"status": "PREPARED", "caseCount": len(CASES), "providerStarts": 0}))
        return
    if not path.exists() or not args.case_id or not args.arm:
        raise ValueError("Fixed manifest, case and arm required")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest != _manifest(output, learning_date=manifest["learningDate"]):
        raise ValueError("Manifest source/date/corpus changed after precommit")
    case = next(case for case in CASES if case["id"] == args.case_id)
    if (output / f"{case['id']}-{args.arm}.json").exists():
        raise ValueError("One-shot case/arm already started")
    result = asyncio.run(_run(case, args.arm, output, manifest))
    print(json.dumps({key: result.get(key) for key in (
        "caseId", "arm", "status", "failureType", "latencySeconds", "providerStarts",
        "inputTokens", "outputTokens", "modelCounts",
    )}))


if __name__ == "__main__":
    main()
