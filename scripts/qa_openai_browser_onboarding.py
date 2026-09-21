"""One-shot level prerequisite fixture for the campaign-owned Google QA user.

This does not claim that the 20-question Level Test ran. It changes no auth
policy and refuses databases or accounts outside this isolated campaign.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import qa_campaign_integration as bootstrap


def prepare(directory: Path, user_id: int) -> dict[str, object]:
    directory = directory.resolve()
    manifest = bootstrap._owned_manifest(directory)
    if manifest["campaign"] != "openai-speech-campaign-20260920":
        raise ValueError("Only this owned QA campaign is eligible")
    if type(user_id) is not int or user_id <= 0:
        raise ValueError("Explicit positive Google QA user ID required")
    artifact = directory / f"browser-onboarding-user-{user_id}.json"
    if artifact.exists():
        raise ValueError("One-shot fixture already applied")
    private = json.loads((directory / "secrets.private.json").read_text(encoding="utf-8"))

    def query(sql: str) -> str:
        return bootstrap._qa_mysql_query(manifest, private, sql)

    before = query(
        "SELECT u.id,u.social_type,p.id,p.state,COALESCE(p.base_level_score,-1),"
        "s.origin_language,s.learning_language,"
        "(SELECT COUNT(*) FROM language_learning_practice_set x WHERE x.user_id=u.id),"
        "(SELECT COUNT(*) FROM language_learning_listening_daily_set x WHERE x.user_id=u.id) "
        "FROM user u JOIN language_learning_profile p ON p.user_id=u.id "
        "JOIN language_learning_user_setting s ON s.user_id=u.id "
        f"WHERE u.id={user_id};"
    ).split("\t")
    if before[:2] != [str(user_id), "GOOGLE"] or before[3:] != [
        "LEVEL_TEST_REQUIRED", "-1", "ko", "ja", "0", "0"
    ]:
        raise ValueError("Exact new Google QA account/prerequisites do not match")
    # The profile belongs to the just-authenticated user, not to the synthetic
    # LOCAL account whose earlier practice sets must remain untouched.
    rows = query(
        "START TRANSACTION; "
        "UPDATE language_learning_profile p JOIN user u ON u.id=p.user_id "
        "SET p.state='CALIBRATING',p.base_level_score=30,p.updated_at=NOW(6) "
        f"WHERE u.id={user_id} AND u.social_type='GOOGLE' AND p.id={before[2]} "
        "AND p.state='LEVEL_TEST_REQUIRED' AND p.base_level_score IS NULL; "
        "SELECT ROW_COUNT(); COMMIT;"
    ).splitlines()
    if rows != ["1"]:
        raise RuntimeError("Exact QA profile compare-and-set failed")
    after = query(
        "SELECT p.state,p.base_level_score FROM language_learning_profile p "
        f"WHERE p.user_id={user_id};"
    ).split("\t")
    if after != ["CALIBRATING", "30"]:
        raise RuntimeError("QA profile postcondition failed")
    result: dict[str, object] = {
        "campaign": manifest["campaign"],
        "database": manifest["names"]["database"],
        "userId": user_id,
        "authType": "GOOGLE",
        "previousProfile": before[3:5],
        "fixtureProfile": after,
        "levelTestPerformed": False,
        "scope": "isolated QA profile prerequisite only; normal Google auth unchanged",
    }
    artifact.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--user-id", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.environment, args.user_id), ensure_ascii=False))
