"""One owned synthetic B5 comparison account; preserves every earlier Reading set.

This QA-only path uses the normal registration/login endpoints. The existing
registration password-storage defect is repaired for this exact synthetic row
using the first owned QA account's BCrypt hash of the same private password.
No token or credential is written to the public diagnostic artifact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import qa_campaign_integration as bootstrap
from scripts.qa_campaign_be_api import Client
from scripts.qa_campaign_reading import NEW_RUN_CASES, reading
from scripts.qa_campaign_reading_fixture import _sql

CASE = next(case for case in NEW_RUN_CASES if case.case_id == "structure-environment-b5")
LABEL = "after-b5-bounded-scope"
AFTER_AUDIT_FIX = "after-b5-bounded-scope-after-be-audit-fix"


def secondary_client(directory: Path, *, create: bool) -> Client:
    output = directory / "be-api-runs" / (LABEL if create else AFTER_AUDIT_FIX)
    client = Client(directory, output_directory=output)
    manifest = client.manifest
    private = client.private
    email = f"qa-{manifest['campaign']}-b5@example.invalid"
    base_id = client.user_id
    marker_path = directory / "be-api-runs" / LABEL / "owned-secondary-account.json"

    def query(sql: str) -> str:
        return bootstrap._qa_mysql_query(manifest, private, sql)

    if create:
        if marker_path.exists() or query("SELECT COUNT(*) FROM user WHERE email=" + _sql(email) + ";") != "0":
            raise ValueError("Secondary QA identity already exists; never register twice")
        status, response = bootstrap._qa_request("/api/v1/auth/register", {
            "email": email, "password": private["QA_APP_PASSWORD"], "username": "B5 Isolated QA",
        })
        user_id = (response.get("body") or {}).get("id")
        marker = {"campaign": manifest["campaign"], "email": email, "userId": user_id,
                  "registerHttpStatus": status, "status": "REGISTERED_FIXTURE_PENDING"}
        bootstrap._write_json(marker_path, marker)
        if status != 200 or type(user_id) is not int or user_id <= 0:
            raise ValueError("Secondary normal registration failed; preserve response boundary")
        base = ("b.id=" + str(base_id) + " AND b.email=" + _sql(private["QA_APP_EMAIL"])
                + " AND b.social_type='LOCAL' AND b.password LIKE '$2%' ")
        owned = ("u.id=" + str(user_id) + " AND u.email=" + _sql(email)
                 + " AND u.social_type='LOCAL'")
        raw_password = _sql(private["QA_APP_PASSWORD"])
        if query("SELECT COUNT(*) FROM user u WHERE " + owned
                 + " AND u.password=" + raw_password + ";") != "1":
            raise ValueError("Registered synthetic raw-password row does not match")
        updated = query(
            "UPDATE user u JOIN user b ON " + base + " SET u.password=b.password WHERE "
            + owned + " AND u.password=" + raw_password + "; SELECT ROW_COUNT();"
        )
        if updated != "1":
            raise ValueError("Exact synthetic BCrypt fixture update failed")
        marker["status"] = "AUTH_FIXTURE_READY"
        bootstrap._write_json(marker_path, marker)
        statements = (
            "START TRANSACTION; SET @qa_ok=1; "
            "INSERT INTO language_learning_profile "
            "(created_at,updated_at,created_by,updated_by,confidence,evaluation_count,"
            "profile_version,state,base_level_score,user_id) "
            "SELECT NOW(6),NOW(6),'QA_FIXTURE','QA_FIXTURE',p.confidence,0,"
            "p.profile_version,'CALIBRATING',90,u.id FROM user u "
            "JOIN language_learning_profile p ON p.user_id=" + str(base_id)
            + " WHERE " + owned + " AND @qa_ok=1; SET @qa_ok=(@qa_ok AND ROW_COUNT()=1); "
            "INSERT INTO language_learning_user_setting "
            "(created_at,updated_at,created_by,updated_by,user_id,daily_listening_goal_count,"
            "daily_sentence_count,daily_speaking_goal_minutes,default_listening_task_types,"
            "learning_language,origin_language,speaking_playback_speed,speaking_voice_id,timezone) "
            "SELECT NOW(6),NOW(6),'QA_FIXTURE','QA_FIXTURE',u.id,s.daily_listening_goal_count,"
            "s.daily_sentence_count,s.daily_speaking_goal_minutes,s.default_listening_task_types,"
            "s.learning_language,s.origin_language,s.speaking_playback_speed,s.speaking_voice_id,"
            "s.timezone FROM user u JOIN language_learning_user_setting s ON s.user_id=" + str(base_id)
            + " WHERE " + owned + " AND @qa_ok=1; SET @qa_ok=(@qa_ok AND ROW_COUNT()=1); "
            "INSERT INTO language_learning_custom_keyword "
            "(created_at,updated_at,created_by,updated_by,user_id,text,normalized_text,"
            "keyword_type,canonical_key,active,available_from,pending_parent_changed) "
            "SELECT NOW(6),NOW(6),'QA_FIXTURE','QA_FIXTURE',u.id,k.text,k.normalized_text,"
            "k.keyword_type,k.canonical_key,1,k.available_from,0 FROM user u "
            "JOIN language_learning_custom_keyword k ON k.user_id=" + str(base_id)
            + " AND k.active=1 AND k.created_by='QA_FIXTURE' WHERE " + owned
            + " AND @qa_ok=1; SET @qa_ok=(@qa_ok AND ROW_COUNT()=1); "
            "SET @qa_finish=IF(@qa_ok=1,'COMMIT','ROLLBACK'); "
            "PREPARE qa_finish FROM @qa_finish; EXECUTE qa_finish; "
            "DEALLOCATE PREPARE qa_finish; SELECT @qa_ok;"
        )
        if query(statements) != "1":
            raise ValueError("Secondary QA profile/settings/keyword CAS failed")
        marker["status"] = "READY"
        bootstrap._write_json(marker_path, marker)
    else:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        user_id = marker.get("userId")
        if (marker.get("campaign") != manifest["campaign"] or marker.get("email") != email
                or marker.get("status") != "READY" or type(user_id) is not int or user_id <= 0):
            raise ValueError("Exact owned secondary QA marker required")
        failed_output = directory / "be-api-runs" / LABEL
        failed = json.loads((failed_output / f"reading-{CASE.case_id}-workflow.json").read_text(encoding="utf-8"))
        first_start = json.loads((failed_output / "0001-reading-start.json").read_text(encoding="utf-8"))
        if (failed.get("status") != "FAILED" or failed.get("practiceSetId") is not None
                or first_start.get("httpStatus") != 500
                or first_start.get("action") != "reading-start"
                or query("SELECT COUNT(*) FROM language_learning_practice_set WHERE user_id="
                         + str(user_id) + ";") != "0"):
            raise ValueError("Only the proved pre-start BE audit 500 may be retried")
    count = query(
        "SELECT COUNT(*) FROM user u JOIN language_learning_profile p ON p.user_id=u.id "
        "JOIN language_learning_user_setting s ON s.user_id=u.id "
        "JOIN language_learning_custom_keyword k ON k.user_id=u.id WHERE u.id=" + str(user_id)
        + " AND u.email=" + _sql(email) + " AND u.social_type='LOCAL' "
        "AND p.base_level_score=90 AND s.origin_language='ko' AND s.learning_language='ja' "
        "AND k.active=1 AND k.created_by='QA_FIXTURE';"
    )
    if count != "1":
        raise ValueError("Secondary QA user binding/profile prerequisites changed")
    login_status, response = bootstrap._qa_request("/api/v1/auth/login", {
        "email": email, "password": private["QA_APP_PASSWORD"],
    })
    token = (response.get("body") or {}).get("accessToken")
    if login_status != 200 or not isinstance(token, str) or not token:
        raise ValueError("Secondary normal login failed")
    client.user_id = user_id
    client.private["QA_APP_EMAIL"] = email
    client.private["QA_APP_ACCESS_TOKEN"] = token
    return client


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    client = secondary_client(args.directory.resolve(), create=args.action == "prepare")
    if args.action == "prepare":
        if bootstrap._qa_mysql_query(client.manifest, client.private,
                "SELECT COUNT(*) FROM language_learning_practice_set WHERE user_id="
                + str(client.user_id) + ";") != "0":
            raise ValueError("New QA identity already has practice sets")
        result = {"status": "PREPARED", "userId": client.user_id,
                  "caseId": CASE.case_id, "providerCalls": 0}
    else:
        result = reading(client, CASE)
    print(json.dumps({key: result.get(key) for key in
                      ("status", "userId", "caseId", "practiceSetId")}, ensure_ascii=True))


if __name__ == "__main__":
    main()
