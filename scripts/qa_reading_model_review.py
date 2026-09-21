"""Offline Reading A/B outcome accounting and answer/band-blinded review packet.

No provider call. Keeps learner-visible text in the owned private QA directory.
Reviewers must not treat generator/verifier PASS or QA oracle score as gold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.qa_campaign_budget import atomic_json
from scripts.qa_campaign_integration import _owned_manifest


def _estimate(attempts: list[dict[str, Any]]) -> float:
    """QA reservation-rate estimate, not an invoice or billing receipt."""
    amount = 0.0
    for attempt in attempts:
        model = attempt.get("model")
        input_tokens = int(attempt.get("inputTokens") or 0)
        output_tokens = int(attempt.get("outputTokens") or 0)
        if model == "gpt-5.6-sol":
            amount += (5 * input_tokens + 25 * output_tokens) / 1_000_000
        elif model in {"gpt-5.6-luna", "gpt-5-mini-2025-08-07", "gpt-5-nano-2025-08-07"}:
            amount += (input_tokens + 4 * output_tokens) / 1_000_000
        elif attempt["status"] == "STARTED" or attempt["status"] == "FAILED":
            # Unknown charged usage stays in the real shared ledger; do not
            # fabricate zero as the actual failed-call cost.
            continue
        else:
            raise ValueError("Unpriced successful model in review source")
    return round(amount, 5)


def review(output: Path) -> dict[str, Any]:
    _owned_manifest(output.parent / "integration-environment")
    manifest = json.loads((output / "reading-comparison-manifest.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    blinded: list[dict[str, Any]] = []
    key: list[dict[str, Any]] = []
    for target in manifest["runOrder"]:
        case_id, arm = target["caseId"], target["arm"]
        artifact = output / f"{case_id}-{arm}.json"
        if not artifact.exists():
            rows.append({"caseId": case_id, "arm": arm, "status": "NOT_RUN"})
            continue
        source = json.loads(artifact.read_text(encoding="utf-8"))
        if source["caseId"] != case_id or source["arm"] != arm:
            raise ValueError("A/B report identity mismatch")
        attempts = source["providerAttempts"]
        row = {"caseId": case_id, "arm": arm, "status": source["status"],
               "failureType": source.get("failureType"), "responseBundles": len(source["responses"]),
               "questions": sum(len(response["questions"]) for response in source["responses"]),
               "latencySeconds": source.get("latencySeconds"), "applicationStarts": len(attempts),
               "inputTokens": sum(int(a.get("inputTokens") or 0) for a in attempts),
               "outputTokens": sum(int(a.get("outputTokens") or 0) for a in attempts),
               "qaReservationRateUsdObservedTokenEstimate": _estimate(attempts),
               "unknownOrFailedUsage": any(a["status"] != "SUCCEEDED" for a in attempts),
               "actualProviderModels": source.get("modelCounts")}
        rows.append(row)
        label = "X" if (int(hashlib.sha256(case_id.encode()).hexdigest(), 16) % 2 == 0) == (arm == "A") else "Y"
        key.append({"caseId": case_id, "blindLabel": label, "arm": arm,
                    "requestedBand": next(case["band"] for case in manifest["cases"] if case["id"] == case_id),
                    "answerKeys": [[q["correctAnswer"] for q in response["questions"]]
                                  for response in source["responses"]]})
        for response in source["responses"]:
            questions = response["questions"]
            if not questions:
                continue
            blinded.append({"caseId": case_id, "blindLabel": label,
                            "mode": response["mode"], "passageId": questions[0]["passageId"],
                            "passageText": questions[0]["passageText"],
                            "questions": [{"localOrder": q["order"], "skillTag": q["skillTag"],
                                           "stem": q["prompt"], "options": q["options"]}
                                          for q in questions]})
    result = {"status": "REVIEW_PACKET_PREPARED", "rows": rows,
              "actualGoldLabelAvailable": False,
              "qualityJudgment": "PENDING_INDEPENDENT_REVIEW",
              "costBoundary": "Known usage QA reservation-rate estimate; failed/unknown usage retained in ledger",
              "blindedPacket": "reading-review-blinded.json",
              "separateAnswerAndArmKey": "reading-review-key-private.json"}
    atomic_json(output / "reading-comparison-summary.json", result)
    atomic_json(output / "reading-review-blinded.json", blinded)
    atomic_json(output / "reading-review-key-private.json", key)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = review(args.output.resolve())
    from collections import Counter
    print(json.dumps({"rows": len(result["rows"]),
                      "byStatus": dict(Counter(row["status"] for row in result["rows"])),
                      "providerStarts": sum(row.get("applicationStarts", 0) for row in result["rows"])}))


if __name__ == "__main__":
    main()
