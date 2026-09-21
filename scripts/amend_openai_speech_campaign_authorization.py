"""Append one user-approved dollar-limit extension to the existing QA ledger.

This command performs no provider or database work. It never changes an
existing call or a non-dollar guard, and refuses an unfamiliar campaign.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path

from filelock import FileLock

from scripts.qa_campaign_budget import CampaignLedger, atomic_json


CAMPAIGN_ID = "openai-speech-campaign-20260920"
NEW_NORMAL_ADMISSION_USD = 190.0
NEW_EXCLUSIVE_AUTHORIZATION_USD = 200.0


def amend(ledger_path: Path, revision_dir: Path) -> dict:
    ledger_path = ledger_path.resolve()
    revision_dir = revision_dir.resolve()
    if ledger_path.parent.name != CAMPAIGN_ID or ledger_path.name != "budget-ledger.json":
        raise ValueError("Unexpected campaign ledger")
    if revision_dir.exists():
        raise ValueError("Authorization revision already exists")
    with FileLock(str(ledger_path) + ".lock"):
        raw = ledger_path.read_bytes()
        state = json.loads(raw)
        if state["campaignId"] != CAMPAIGN_ID or state["limits"]["usd"] != 90.0:
            raise ValueError("Unexpected campaign identity or pre-approval dollar limit")
        if state["stoppedReason"] is not None:
            raise ValueError("A halted ledger requires a separate audited resume")
        if any(call["status"] == "STARTED" for call in state["calls"]):
            raise ValueError("Cannot amend while a provider call is in flight")
        total = CampaignLedger.totals(state)
        if total["usd"] >= NEW_NORMAL_ADMISSION_USD:
            raise ValueError("Current exposure already exceeds new admission")
        revision_dir.mkdir(parents=True, exist_ok=False)
        snapshot = revision_dir / "budget-ledger-before-authorization.json"
        snapshot.write_bytes(raw)
        event = {
            "at": datetime.now(UTC).isoformat(),
            "source": "user instruction adopting EXECUTION_SPEC (2).md on 2026-09-20",
            "campaignId": CAMPAIGN_ID,
            "oldNormalAdmissionUsd": 90.0,
            "newNormalAdmissionUsd": NEW_NORMAL_ADMISSION_USD,
            "newAuthorizationUsdExclusive": NEW_EXCLUSIVE_AUTHORIZATION_USD,
            "oldLedgerSha256": hashlib.sha256(raw).hexdigest(),
            "snapshot": str(snapshot),
            "exposureAtAmendmentUsd": total["usd"],
            "nonDollarLimitsUnchanged": True,
        }
        state["limits"] = {**state["limits"], "usd": NEW_NORMAL_ADMISSION_USD}
        state.setdefault("authorizationEvents", []).append(event)
        atomic_json(ledger_path, state)
        atomic_json(revision_dir / "authorization-event.json", event)
        return event


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--revision-dir", type=Path, required=True)
    args = parser.parse_args()
    event = amend(args.ledger, args.revision_dir)
    print(json.dumps({"campaignId": event["campaignId"],
                      "newNormalAdmissionUsd": event["newNormalAdmissionUsd"],
                      "exposureAtAmendmentUsd": event["exposureAtAmendmentUsd"]}))


if __name__ == "__main__":
    main()
