"""Initialize one private, persistent OpenAI-only learning QA campaign.

This never contacts a provider or database. Existing campaign files are not
overwritten; all live entry points adopt the ledger's immutable limits.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.qa_campaign_budget import CampaignLedger, atomic_json


OPENAI_CAMPAIGN_LIMITS = {
    "starts": 2000, "input_tokens": 8_000_000,
    "output_tokens": 2_000_000, "tts_characters": 200_000,
    "stt_seconds": 7200, "usd": 90.0, "wall_seconds": 604_800,
    "active_seconds": 43_200,
}


def initialize(output: Path, prior: Path) -> dict:
    if not prior.is_file():
        raise ValueError("Prior campaign ledger is required")
    output = output.resolve()
    if output.exists():
        raise ValueError("Campaign output must not exist")
    prior_hash = hashlib.sha256(prior.read_bytes()).hexdigest()
    roots = {
        "AI": Path(__file__).resolve().parents[1],
        "BE": Path(__file__).resolve().parents[2] / "CatPjt-TranslaCat-be",
        "FE": Path(__file__).resolve().parents[2] / "CatPjt-TranslaCat-fe",
    }
    heads = {
        name: subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                      text=True).strip()
        for name, root in roots.items()
    }
    cases = []
    for voice in ("marin", "cedar"):
        for sample in range(1, 4):
            cases.append({"id": f"VOICE-{voice}-{sample}", "phase": "P1",
                          "attempts": 1, "voice": voice, "expected": "complete WAV, measured duration"})
    for mode in ("READ_ALOUD", "GUIDED", "FREE"):
        for pass_number in (1, 2):
            cases.append({"id": f"SPK-{mode}-{pass_number}", "phase": "P2" if pass_number == 1 else "P5",
                          "attempts": 1, "expected": "browser record/upload/STT/evaluate/DB/re-entry"})
    for mode in ("DICTATION", "COMPREHENSION", "SUMMARY"):
        for difficulty in ("EASY", "MY_LEVEL", "CHALLENGE"):
            cases.append({"id": f"LST-{mode}-{difficulty}", "phase": "P3", "items": 5,
                          "attempts": 1, "expected": "normal WAV duration/BE rows/evaluation"})
    for mode in ("COMPREHENSION", "STRUCTURE", "CONTEXT_INFERENCE"):
        for band in (1, 3, 5):
            cases.append({"id": f"RDG-{mode}-B{band}", "phase": "P4", "items": 5,
                          "attempts": 1, "expected": "passage/question quality + BE rows"})
    output.mkdir(parents=True, exist_ok=False)
    campaign_id = output.name
    manifest = {"campaignId": campaign_id, "createdAt": datetime.now(UTC).isoformat(),
                "authorizationUsdExclusive": 100, "normalAdmissionUsdInclusive": 90,
                "providerPolicy": "OpenAI only; no new Gemini generation or fallback",
                "priorCampaignLedger": str(prior.resolve()), "priorLedgerSha256": prior_hash,
                "sourceHeadAtStart": heads, "cases": cases}
    atomic_json(output / "CASE_MATRIX.json", manifest)
    CampaignLedger(output / "budget-ledger.json", campaign_id,
                   limits=OPENAI_CAMPAIGN_LIMITS, prior_ledger_sha256=prior_hash)
    atomic_json(output / "preflight.json", {
        "sourceHeads": heads, "priorLedgerSha256": prior_hash,
        "firstLiveAt": None, "externalProviderStarts": 0,
        "dbIsolation": "NOT_VERIFIED", "browser": "NOT_RUN",
    })
    return {"campaignId": campaign_id, "caseCount": len(cases),
            "priorLedgerSha256": prior_hash}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior-ledger", type=Path, required=True)
    args = parser.parse_args()
    print(initialize(args.output, args.prior_ledger))


if __name__ == "__main__":
    main()
