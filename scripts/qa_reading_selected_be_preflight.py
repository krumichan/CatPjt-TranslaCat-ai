"""Read-only exact-owner preflight for selected Reading normal-login browser QA."""
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
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    environment = campaign / "integration-environment"
    manifest = _owned_manifest(environment)
    private = json.loads((environment / "secrets.private.json").read_text(encoding="utf-8"))
    raw = _qa_mysql_query(manifest, private,
        "SELECT u.id,p.state,p.base_level_score,s.timezone,s.origin_language,s.learning_language "
        "FROM user u JOIN language_learning_profile p ON p.user_id=u.id "
        "JOIN language_learning_user_setting s ON s.user_id=u.id "
        "WHERE u.id=3 AND u.social_type='GOOGLE' AND u.public_id='TC-GA5T-4LRB';")
    fields = raw.split("\t")
    if len(fields) != 6 or fields[0] != "3":
        raise ValueError("Normal Google QA user profile mismatch")
    rows = _qa_mysql_query(manifest, private,
        "SELECT ps.id,ps.mode,ps.generation_status,ps.learning_date,"
        "(SELECT COUNT(*) FROM language_learning_practice_question q WHERE q.practice_set_id=ps.id) "
        "FROM language_learning_practice_set ps JOIN user u ON u.id=ps.user_id "
        "WHERE u.id=3 AND u.social_type='GOOGLE' AND u.public_id='TC-GA5T-4LRB' "
        "AND ps.domain='READING' ORDER BY ps.id;")
    report = {"userId": 3, "normalGoogleAuthRequired": True,
              "profileState": fields[1], "baseLevelScore": float(fields[2]),
              "timezone": fields[3], "originLanguage": fields[4], "learningLanguage": fields[5],
              "existingReadingSets": [dict(zip(("id", "mode", "generationStatus", "learningDate", "questions"),
                                                 line.split("\t"), strict=True))
                                      for line in rows.splitlines() if line],
              "readOnly": True}
    output = campaign / "closure-20260920-v1" / "reading-selected-be-preflight.json"
    atomic_json(output, report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
