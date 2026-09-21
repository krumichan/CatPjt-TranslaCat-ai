"""Five unseen Reading sets after selecting Reading-only Sol/high.

The fixed corpus, 3+2 contract, old verifier/budgets, and every failed set are
preserved. This is direct AI content evidence, not BE persistence or gold labels.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import date
import hashlib
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.qa_campaign_budget import atomic_json
from scripts.qa_campaign_integration import _owned_manifest
from scripts.qa_reading_model_comparison import _run


CASES = (
    {"id": "holdout-b1-library", "band": 1, "mode": "COMPREHENSION",
     "keywords": ["図書館", "学校"], "weak": ["DETAIL"], "mistakes": ["場所を取り違える"]},
    {"id": "holdout-b2-travel", "band": 2, "mode": "COMPREHENSION",
     "keywords": ["旅行", "食事"], "weak": ["GIST"], "mistakes": []},
    {"id": "holdout-b3-community", "band": 3, "mode": "CONTEXT_INFERENCE",
     "keywords": ["防災", "地域"], "weak": ["CONTEXT_INFERENCE"], "mistakes": ["暗示と事実を混同"]},
    {"id": "holdout-b4-workplace", "band": 4, "mode": "STRUCTURE",
     "keywords": ["働き方", "制度"], "weak": ["STRUCTURE"], "mistakes": []},
    {"id": "holdout-b5-tourism", "band": 5, "mode": "STRUCTURE",
     "keywords": ["観光", "公共政策"], "weak": ["STRUCTURE"], "mistakes": ["筆者の立場を読み違える"]},
)


def manifest(output: Path, *, learning_date: str | None = None) -> dict:
    from app.core.config import Settings
    if Settings.model_fields["AI_READING_GENERATION_MODEL"].default != "SOL":
        raise ValueError("Selected production Reading model must default to SOL")
    return {"version": "reading-selected-holdout-v1", "model": "SOL",
            "sourceSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "comparisonSha256": hashlib.sha256(
                (output / "reading-comparison-summary.json").read_bytes()).hexdigest(),
            "learningDate": learning_date or date.today().isoformat(), "cases": list(CASES),
            "maxApplicationStartsPerSet": 35,
            "boundary": "Single run per fixed unseen case, 3+2, all failure evidence retained; no BE DB integration"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--case-id", choices=[case["id"] for case in CASES])
    args = parser.parse_args()
    output = args.output.resolve()
    _owned_manifest(output.parent / "integration-environment")
    if output.name != "closure-20260920-v1":
        raise ValueError("Only the existing owned closure directory is admitted")
    path = output / "reading-selected-holdout-manifest.json"
    if args.prepare:
        if args.case_id or path.exists():
            raise ValueError("Holdout manifest is one-shot")
        atomic_json(path, manifest(output))
        print(json.dumps({"status": "PREPARED", "cases": len(CASES), "providerStarts": 0}))
        return
    if not path.exists() or not args.case_id:
        raise ValueError("Fixed manifest and one case required")
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved != manifest(output, learning_date=saved["learningDate"]):
        raise ValueError("Selected holdout input/source changed after precommit")
    case = next(value for value in CASES if value["id"] == args.case_id)
    if (output / f"{case['id']}-B.json").exists():
        raise ValueError("Holdout already started; preserve result")
    result = asyncio.run(_run(case, "B", output, {"arms": {"B": "SOL"},
                                          "learningDate": saved["learningDate"]}))
    print(json.dumps({key: result.get(key) for key in (
        "caseId", "status", "failureType", "latencySeconds", "providerStarts",
        "inputTokens", "outputTokens", "modelCounts",
    )}))


if __name__ == "__main__":
    main()
