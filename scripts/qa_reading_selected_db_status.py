"""Read-only, exact-owner Reading QA set status without learner text or secrets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.qa_campaign_budget import atomic_json
from scripts.qa_campaign_integration import _owned_manifest, _qa_mysql_query


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--set-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.set_id not in {8, 9, 10, 11, 12}:
        raise ValueError("This QA snapshot is only authorized for owned Reading sets 8-12")
    environment = args.campaign.resolve() / "integration-environment"
    manifest = _owned_manifest(environment)
    private = json.loads((environment / "secrets.private.json").read_text(encoding="utf-8"))
    rows = _qa_mysql_query(manifest, private,
        "SELECT ps.id,ps.mode,ps.complexity_band,ps.status,ps.official_score,"
        "ps.generation_status,ps.generation_failure_message,"
        "ps.generation_retry_count,JSON_KEYS(ps.generation_request_json,'$.readingBundles'),"
        "q.order_no,q.passage_id,"
        "(SELECT COUNT(*) FROM language_learning_practice_attempt a WHERE a.question_id=q.id) "
        "FROM language_learning_practice_set ps JOIN user u ON u.id=ps.user_id "
        "LEFT JOIN language_learning_practice_question q ON q.practice_set_id=ps.id "
        "WHERE ps.id=" + str(args.set_id) + " AND u.id=3 AND u.social_type='GOOGLE' "
        "AND u.public_id='TC-GA5T-4LRB' ORDER BY q.order_no;")
    parts = [line.split("\t") for line in rows.splitlines() if line]
    if not parts or any(len(part) != 12 or part[0] != str(args.set_id) for part in parts):
        raise ValueError("Exact owner and row shape not established")
    head = parts[0]
    result = {"setId": args.set_id, "mode": head[1], "complexityBand": int(head[2]),
              "learningStatus": head[3],
              "officialScore": None if head[4] == "NULL" else float(head[4]),
              "generationStatus": head[5],
              "failureCode": None if head[6] == "NULL" else head[6],
              "retryCount": int(head[7]),
              "storedBundleKeys": None if head[8] == "NULL" else json.loads(head[8]),
              "rows": [{"order": int(line[9]), "passageId": line[10],
                        "attemptCount": int(line[11])}
                       for line in parts if line[9] != "NULL"],
              "readOnly": True}
    atomic_json(args.output, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
