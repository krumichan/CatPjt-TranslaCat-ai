"""Read-only proof of the one new FREE session in the owned isolated QA schema."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.qa_campaign_budget import atomic_json
from scripts.qa_campaign_integration import DOCKER, _verify_local_docker
from scripts.qa_free_owned_schema import (
    COACHING_RETEST_SCHEMA,
    COACHING_SCHEMA,
    NEW_SCHEMA,
    RETEST_SCHEMA,
)


def verify(campaign: Path, session_id: int, *, revision_retest: bool = False,
           coaching_v1: bool = False, coaching_v1_retest: bool = False,
           expected_public_id: str | None = None) -> dict[str, object]:
    if campaign.name != "openai-speech-campaign-20260920" or type(session_id) is not int or session_id <= 0:
        raise ValueError("Exact campaign and positive session ID required")
    original = campaign / "integration-environment"
    if sum((revision_retest, coaching_v1, coaching_v1_retest)) > 1:
        raise ValueError("Choose exactly one owned QA schema")
    schema = (COACHING_RETEST_SCHEMA if coaching_v1_retest else COACHING_SCHEMA if coaching_v1
              else RETEST_SCHEMA if revision_retest else NEW_SCHEMA)
    directory = ("free-coaching-v1-retest-owned-qa" if coaching_v1_retest
                 else "free-coaching-v1-owned-qa" if coaching_v1
                 else "free-v3-retest-owned-qa" if revision_retest else "free-v3-owned-qa")
    fresh = campaign / "closure-20260920-v1" / directory
    owner = json.loads((original / "manifest.json").read_text(encoding="utf-8"))
    manifest = json.loads((fresh / "manifest.json").read_text(encoding="utf-8"))
    if (owner["owner"] != "translacat-isolated-qa" or owner["campaign"] != campaign.name
            or manifest["database"] != schema or manifest["originalDatabase"] != owner["names"]["database"]
            or manifest["status"] != "BE_HEALTHY_NORMAL_LOGIN_REQUIRED"):
        raise ValueError("Owned QA schema/profile changed")
    _verify_local_docker()
    container = owner["names"]["mysqlContainer"]
    actual = subprocess.run([str(DOCKER), "inspect", container, "--format", "{{.Id}}"],
                            capture_output=True, text=True, timeout=20, check=True).stdout.strip()
    if actual != manifest["mysqlContainerId"]:
        raise ValueError("QA MySQL owner changed")
    private = json.loads((fresh / "secrets.private.json").read_text(encoding="utf-8"))
    environment = {**os.environ, "MYSQL_PWD": private["QA_DB_PASSWORD"]}

    def query(sql: str) -> list[list[str]]:
        result = subprocess.run([str(DOCKER), "exec", "--env", "MYSQL_PWD", container,
                                 "mysql", "--batch", "--skip-column-names", "-uqa_app", schema, "-e", sql],
                                env=environment, capture_output=True, text=True, timeout=30, check=False)
        if result.returncode:
            raise RuntimeError("Owned QA read-only SELECT failed; SQL output suppressed")
        return [line.split("\t") for line in result.stdout.strip().splitlines()]

    sessions = query("SELECT id,practice_mode,status,evaluation_status,completed_turns,total_duration_seconds,"
                     "result_kind,result_policy_version "
                     f"FROM language_learning_speaking_session WHERE id={session_id}")
    turns = query("SELECT COUNT(*),MIN(turn_index),MAX(turn_index),"
                  "SUM(status='READY'),SUM(excluded_from_evaluation) "
                  f"FROM language_learning_speaking_turn WHERE session_id={session_id}")
    evaluations = query("SELECT COUNT(*),MAX(overall_score),MAX(evaluation_confidence),MAX(status) "
                        f"FROM language_learning_speaking_evaluation WHERE session_id={session_id}")
    coaching = query("SELECT COUNT(*),MAX(result_policy_version),MAX(schema_version),MAX(content_status),"
                     "MAX(source_snapshot_hash) FROM language_learning_speaking_coaching_result "
                     f"WHERE session_id={session_id}")
    metric_count = query("SELECT COUNT(*) FROM language_learning_speaking_evaluation_metric m "
                         "JOIN language_learning_speaking_evaluation e ON e.id=m.evaluation_id "
                         f"WHERE e.session_id={session_id}")
    profile_count = query("SELECT COUNT(*) FROM language_learning_profile_evidence pe "
                          "JOIN language_learning_speaking_session s ON s.user_id=pe.user_id "
                          f"WHERE s.id={session_id} AND pe.source='SPEAKING'")
    activity = query("SELECT COUNT(*),MAX(status) FROM language_learning_activity "
                     f"WHERE source='SPEAKING' AND reference_id='{session_id}'")
    owner_public_ids = query("SELECT u.public_id FROM user u JOIN language_learning_speaking_session s "
                             f"ON s.user_id=u.id WHERE s.id={session_id}")
    owner_matches = (len(owner_public_ids) == 1 and len(owner_public_ids[0]) == 1
                     and owner_public_ids[0][0] == expected_public_id)
    if expected_public_id is not None and not owner_matches:
        raise ValueError("Browser Google identity does not own the QA session")
    result: dict[str, object] = {
        "owner": manifest["owner"], "schema": schema, "session": sessions,
        "turnSummary": turns, "evaluationSummary": evaluations,
        "coachingSummary": coaching, "metricRowsForSession": metric_count,
        "speakingProfileEvidenceForOwner": profile_count, "activitySummary": activity,
        "sessionOwnerMatchesAuthenticatedPublicId": owner_matches,
        "readOnly": True, "originalSchemaUntouched": owner["names"]["database"],
    }
    output = fresh / f"speaking-free-session{session_id}-db-verify-v3.json"
    if output.exists():
        raise ValueError("Existing DB evidence preserved")
    atomic_json(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--session-id", type=int, required=True)
    parser.add_argument("--revision-retest", action="store_true")
    parser.add_argument("--coaching-v1", action="store_true")
    parser.add_argument("--coaching-v1-retest", action="store_true")
    parser.add_argument("--expected-public-id", type=str, required=True)
    args = parser.parse_args()
    evidence = verify(args.campaign.resolve(), args.session_id, revision_retest=args.revision_retest,
                      coaching_v1=args.coaching_v1, coaching_v1_retest=args.coaching_v1_retest,
                      expected_public_id=args.expected_public_id)
    print(json.dumps(evidence, ensure_ascii=False))
