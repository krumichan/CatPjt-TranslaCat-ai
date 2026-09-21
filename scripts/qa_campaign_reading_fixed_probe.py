"""One-call, budgeted Reading Mini probe of a private fixed QA candidate.

This is an offline verifier comparison, not production acceptance or a gold label.
The raw candidate/verdict stays in the owned campaign directory, never stdout.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import date
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ai.provider_factory import create_text_generation_provider
from app.features.language_learning.reading_vocabulary.prompts import (
    build_practice_verification_prompt,
)
from app.features.language_learning.reading_vocabulary.service import (
    ReadingVocabularyGenerationService,
    _READING_VERIFICATION_SCHEMA,
    _ReadingPracticeVerificationPayload,
)
from app.schemas.language_learning_practice import PracticeGenerationRequest
from scripts.qa_campaign_budget import BudgetedTextProvider, CampaignLedger, atomic_json
from scripts.qa_campaign_integration import _owned_manifest


async def _run(output: Path, case: dict[str, Any]) -> dict[str, Any]:
    request = PracticeGenerationRequest.model_validate({
        "requestId": "qa-reading-fixed-probe-one-call",
        "domain": "READING", "mode": case["mode"],
        "originLanguage": "ko", "learningLanguage": case["learningLanguage"],
        "questionCount": 1, "complexityBand": 3,
        "easierCount": 0, "currentCount": 1, "challengeCount": 0,
        "generationDate": date.today().isoformat(),
    })
    question = {key: case[key] for key in (
        "passageId", "passageText", "prompt", "options", "skillTag",
    )}
    question.update({"order": 1, "questionDemand": None})
    prompt = build_practice_verification_prompt(request, [question])
    provider = BudgetedTextProvider(
        create_text_generation_provider(),
        CampaignLedger(output / "budget-ledger.json", output.name, stop_on_caller_cancel=False),
        phase="QA_READING_FIXED_ONE_CALL",
    )
    result: dict[str, Any] = {
        "caseId": case["caseId"], "status": "STARTED", "partial": True,
        "caseSha256": hashlib.sha256(json.dumps(case, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
        "promptSha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "providerStartsAllowed": 1,
    }
    try:
        await provider.warm_up()
        raw = await provider.call(
            ReadingVocabularyGenerationService.VERIFICATION_TYPE_NAME,
            prompt, _READING_VERIFICATION_SCHEMA,
        )
        result["verdictRaw"] = raw
        verdicts = _ReadingPracticeVerificationPayload.model_validate(raw).verdicts
        if len(verdicts) != 1 or verdicts[0].order != 1:
            raise ValueError("Fixed one-question verdict coverage mismatch")
        verdict = verdicts[0]
        result.update({
            "status": "COMPLETED", "partial": False,
            "readingOperation": verdict.readingOperation,
            "modeFit": verdict.modeFit,
            "supported": verdict.supported,
            "stemPresuppositionsSupported": verdict.stemPresuppositionsSupported,
        })
    except BaseException as exc:
        result.update({"status": "INTERRUPTED" if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)) else "FAILED",
                       "errorType": type(exc).__name__})
    finally:
        atomic_json(output / "reading-fixed-probe-result.json", result)
        await provider.shutdown()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    _owned_manifest(output / "integration-environment")
    case_path = args.case.resolve()
    if case_path.parent != output or case_path.name != "fixed-case-reading-set2-q4.json":
        raise ValueError("Only the fixed owned campaign case is allowed")
    result_path = output / "reading-fixed-probe-result.json"
    if result_path.exists():
        raise ValueError("Probe is one-shot; preserve prior verdict")
    case = json.loads(case_path.read_text(encoding="utf-8"))
    if case.get("caseId") != "reading-set2-q4-explicit-conclusion":
        raise ValueError("Fixed candidate identity mismatch")
    result = asyncio.run(_run(output, case))
    print(json.dumps({key: result.get(key) for key in (
        "status", "errorType", "readingOperation", "modeFit", "supported",
        "stemPresuppositionsSupported", "caseSha256", "promptSha256",
    )}))


if __name__ == "__main__":
    main()
