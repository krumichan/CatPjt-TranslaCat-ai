"""Read-only publication/plan observation for one owned QA Google Reading set.

Never starts generation, changes rows, or stores the passage/question text.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import qa_campaign_integration as qa


def observe(environment: Path, set_id: int) -> dict[str, Any]:
    manifest = qa._owned_manifest(environment)
    if manifest["campaign"] != "openai-speech-campaign-20260920":
        raise ValueError("Wrong QA campaign")
    private = json.loads((environment / "secrets.private.json").read_text(encoding="utf-8"))
    def query(sql: str) -> str:
        return qa._qa_mysql_query(manifest, private, sql)

    # One SQL statement gives a consistent InnoDB statement snapshot. Separate
    # request/row SELECTs can falsely report a question before its bundle when
    # the worker commits between those reads.
    row = query(
        "SELECT s.id,s.domain,s.mode,s.generation_status,s.question_count,"
        "DATE_FORMAT(s.created_at,'%Y-%m-%dT%H:%i:%s.%f'),"
        "HEX(s.generation_request_json),"
        "(SELECT COALESCE(JSON_ARRAYAGG(JSON_OBJECT('order',q.order_no,"
        "'passageId',q.passage_id,'createdAt',DATE_FORMAT(q.created_at,'%Y-%m-%dT%H:%i:%s.%f'),"
        "'passageSha256',SHA2(q.passage_text,256))),JSON_ARRAY()) "
        "FROM language_learning_practice_question q WHERE q.practice_set_id=s.id) "
        "FROM language_learning_practice_set s JOIN user u ON u.id=s.user_id "
        f"WHERE s.id={set_id} AND u.id=3 AND u.social_type='GOOGLE';"
    ).split("\t")
    if len(row) != 8 or row[1] != "READING" or row[4] != "5":
        raise ValueError("Exact owned five-question Reading set required")
    request = json.loads(bytes.fromhex(row[6]).decode("utf-8"))
    bundles = request.get("readingBundles") or {}
    if not isinstance(bundles, dict):
        raise ValueError("Persisted readingBundles must be a passage-ID map")
    questions = sorted(json.loads(row[7]), key=lambda item: item["order"])
    bundle_summary = [
        {"passageId": passage_id, "questionOrders": [q.get("order") for q in bundle.get("questions", [])],
         "planOrders": [p.get("globalOrder") for p in bundle.get("questionPlans", [])],
         "passageSha256": bundle.get("passageSha256"), "promptVersion": bundle.get("promptVersion")}
        for passage_id, bundle in sorted(bundles.items())
    ]
    return {
        "observedAt": datetime.now(timezone.utc).isoformat(), "setId": set_id,
        "mode": row[2], "generationStatus": row[3], "setCreatedAt": row[5],
        "requestSha256": hashlib.sha256(bytes.fromhex(row[6])).hexdigest(),
        "storedBundles": bundle_summary, "questionRows": questions,
        "consistentStatementSnapshot": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--set-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=900)
    args = parser.parse_args()
    if args.set_id <= 0 or not 0 < args.max_seconds <= 900:
        parser.error("Positive set ID and bounded duration required")
    environment = args.environment.resolve()
    output = args.output.resolve()
    if output.parent != environment.parent or output.exists():
        parser.error("New output must be in the owned campaign directory")
    result: dict[str, Any] = {"status": "OBSERVING", "setId": args.set_id,
                              "snapshots": [], "providerCallsStartedByObserver": 0}
    deadline = time.monotonic() + args.max_seconds
    try:
        while time.monotonic() < deadline:
            snapshot = observe(environment, args.set_id)
            previous = result["snapshots"][-1] if result["snapshots"] else None
            if previous is None or any(snapshot[key] != previous[key] for key in
                                       ("generationStatus", "storedBundles", "questionRows")):
                result["snapshots"].append(snapshot)
                qa._write_json(output, result)
            if snapshot["generationStatus"] in {"READY", "FAILED", "PARTIAL"}:
                result["status"] = snapshot["generationStatus"]
                break
            time.sleep(2)
        else:
            result["status"] = "OBSERVATION_TIMEOUT"
    except BaseException as exc:
        result["status"] = "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED_TO_OBSERVE"
        result["errorType"] = type(exc).__name__
        raise
    finally:
        result["finishedAt"] = datetime.now(timezone.utc).isoformat()
        qa._write_json(output, result)
    print(json.dumps({"status": result["status"], "setId": args.set_id,
                      "snapshotCount": len(result["snapshots"])}))


if __name__ == "__main__":
    main()
