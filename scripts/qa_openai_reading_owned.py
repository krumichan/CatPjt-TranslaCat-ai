"""Three fixed Reading blocker cases on the new, owned LOCAL QA account.

The synthetic prerequisite is not a Level Test result. All mutations are CAS
scoped to this campaign's exact registered user and newly owned QA schema.
Public Reading generation still selects the actual band and persists the set.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from zoneinfo import ZoneInfo

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import qa_campaign_integration as bootstrap
from scripts.qa_campaign_be_api import Client
from scripts.qa_campaign_reading import NEW_RUN_CASES, reading
from scripts.qa_campaign_reading_fixture import _sql

BLOCKERS = (NEW_RUN_CASES[3], NEW_RUN_CASES[2], NEW_RUN_CASES[4])
BASE_SCORE = {1: 30, 3: 60, 5: 90}


def prepare(client: Client, case_id: str) -> dict:
    if client.browser_session is not None or client.manifest["campaign"] != "openai-speech-campaign-20260920":
        raise ValueError("Only this isolated LOCAL QA account is eligible")
    case = next((value for value in BLOCKERS if value.case_id == case_id), None)
    if case is None:
        raise ValueError("Only three declared blocker cases are eligible")
    evidence = client.output / "reading-openai-preconditions.json"
    records = json.loads(evidence.read_text(encoding="utf-8")) if evidence.exists() else []
    followup_b1 = client.output.name == "after-b1-plan-fix"
    followup_b3 = client.output.name == "after-b3-plan-fix"
    followup_b5 = client.output.name == "after-b5-scope-fix"
    followup = followup_b1 or followup_b3 or followup_b5
    order = ((BLOCKERS[0],) if followup_b1 else (BLOCKERS[1],) if followup_b3
             else (BLOCKERS[2],) if followup_b5 else BLOCKERS)
    if len(records) >= len(order) or case != order[len(records)]:
        raise ValueError("Fixed case order or unique preparation violated")
    def query(sql: str) -> str:
        return bootstrap._qa_mysql_query(client.manifest, client.private, sql)
    identity = (f"u.id={client.user_id} AND u.social_type='LOCAL' AND u.email="
                + _sql(client.private["QA_APP_EMAIL"]))
    row = query(
        "SELECT u.id,p.id,p.state,p.base_level_score,s.timezone,s.origin_language,"
        "s.learning_language,s.pending_effective_date,"
        "(SELECT COUNT(*) FROM language_learning_custom_keyword k WHERE k.user_id=u.id) "
        "FROM user u JOIN language_learning_user_setting s ON s.user_id=u.id "
        "JOIN language_learning_profile p ON p.user_id=u.id WHERE " + identity + ";"
    ).split("\t")
    if (len(row) != 9 or row[0] != str(client.user_id) or row[2] not in {"CALIBRATING", "ACTIVE"}
            or row[5:7] != ["ko", "ja"] or row[7] != "NULL" or row[8] not in {"0", "1"}):
        raise ValueError("Owned synthetic account/profile/settings prerequisite mismatch")
    if followup_b3 or followup_b5:
        prior_path = (client.directory / "be-api-runs"
                      / ("after-auth-renewal" if followup_b3 else "after-language-setup")
                      / f"reading-{case.case_id}-workflow.json")
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        if (prior.get("status") != "GENERATION_TERMINAL_NO_RETRY"
                or type(prior.get("practiceSetId")) is not int or prior.get("answers")):
            raise ValueError("Preserved failed Reading set is required before a fresh comparison")
        prior_rows = query(
            "SELECT s.generation_status,COUNT(q.id) FROM language_learning_practice_set s "
            "LEFT JOIN language_learning_practice_question q ON q.practice_set_id=s.id "
            "JOIN user u ON u.id=s.user_id WHERE " + identity
            + " AND s.id=" + str(prior["practiceSetId"]) + " GROUP BY s.id;"
        ).split("\t")
        if prior_rows != ["PARTIAL", "1" if followup_b3 else "2"]:
            raise ValueError("Failed Reading accepted prefix changed; preserve it")
    target_zone = ("Etc/GMT+12" if followup_b1 or followup_b5 else
                   "Asia/Seoul" if followup_b3 else row[4])
    target_date = datetime.now(timezone.utc).astimezone(ZoneInfo(target_zone)).date().isoformat()
    existing_set = query(
        "SELECT COUNT(*) FROM language_learning_practice_set s JOIN user u ON u.id=s.user_id "
        "WHERE " + identity + " AND s.domain='READING' AND s.mode=" + _sql(case.mode)
        + " AND s.learning_date=" + _sql(target_date) + ";"
    )
    if existing_set != "0":
        raise ValueError("Existing Reading set on this date/mode; never overwrite or regenerate")
    score = BASE_SCORE[case.band]
    old_score = float(row[3])
    profile_change = (
        "UPDATE language_learning_profile p JOIN user u ON u.id=p.user_id "
        "SET p.base_level_score=" + str(score) + ",p.updated_at=NOW(6) WHERE "
        + identity + " AND p.id=" + row[1] + " AND p.state=" + _sql(row[2])
        + " AND p.base_level_score=" + str(old_score) + "; "
        "SET @qa_ok=(@qa_ok AND ROW_COUNT()=1); "
    ) if old_score != score else ""
    if row[8] == "0":
        keyword_change = (
            "INSERT INTO language_learning_custom_keyword "
            "(user_id,text,normalized_text,keyword_type,canonical_key,active,available_from,"
            "pending_parent_changed,created_at,updated_at,created_by,updated_by) "
            "SELECT u.id," + ",".join((_sql(case.keyword), _sql(case.keyword), "'TOPIC'", _sql(case.keyword)))
            + ",1," + _sql(target_date) + ",0,NOW(6),NOW(6),'QA_FIXTURE','QA_FIXTURE' FROM user u WHERE "
            + identity + " AND @qa_ok=1;"
        )
    else:
        owned_keyword = query(
            "SELECT HEX(k.text) FROM language_learning_custom_keyword k "
            "JOIN user u ON u.id=k.user_id WHERE " + identity
            + " AND k.created_by='QA_FIXTURE' AND k.active=1;"
        ).splitlines()
        if (len(owned_keyword) != 1 or not owned_keyword[0]
                or re.fullmatch(r"[0-9A-F]+", owned_keyword[0]) is None):
            raise ValueError("Exactly one owned active QA keyword is required")
        old_keyword_sql = "CONVERT(0x" + owned_keyword[0] + " USING utf8mb4)"
        keyword_change = (
            "UPDATE language_learning_custom_keyword k JOIN user u ON u.id=k.user_id "
            "SET k.text=" + _sql(case.keyword) + ",k.normalized_text=" + _sql(case.keyword)
            + ",k.canonical_key=" + _sql(case.keyword) + ",k.available_from=" + _sql(target_date)
            + ",k.updated_at=NOW(6) WHERE "
            + identity + " AND @qa_ok=1 AND k.created_by='QA_FIXTURE' AND k.active=1 "
            "AND k.text=" + old_keyword_sql + ";"
        )
    statement = (
        "START TRANSACTION; SET @qa_ok=1; "
        + ("UPDATE language_learning_user_setting s JOIN user u ON u.id=s.user_id "
           "SET s.timezone=" + _sql(target_zone) + ",s.updated_at=NOW(6) WHERE " + identity
           + " AND s.timezone=" + _sql(row[4]) + " AND s.pending_effective_date IS NULL; "
           "SET @qa_ok=(@qa_ok AND ROW_COUNT()=1); " if followup else "")
        + profile_change + keyword_change
        + " SET @qa_ok=(@qa_ok AND ROW_COUNT()=1); "
        "SET @qa_finish=IF(@qa_ok=1,'COMMIT','ROLLBACK'); "
        "PREPARE qa_finish FROM @qa_finish; EXECUTE qa_finish; DEALLOCATE PREPARE qa_finish; SELECT @qa_ok;"
    )
    if query(statement).strip() != "1":
        raise ValueError("Owned profile/keyword CAS failed; no Reading start permitted")
    result = {"caseId": case.case_id, "mode": case.mode, "expectedBand": case.band,
              "keyword": case.keyword, "ownedUserId": client.user_id,
              "learningDate": target_date, "timezone": target_zone,
              "syntheticProfilePrerequisite": True, "status": "PREPARED"}
    records.append(result)
    bootstrap._write_json(evidence, records)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--case-id", required=True, choices=[case.case_id for case in BLOCKERS])
    args = parser.parse_args()
    if args.run_label not in {"after-language-setup", "after-b1-plan-fix",
                              "after-auth-renewal", "after-b3-plan-fix", "after-b5-scope-fix"}:
        raise ValueError("This new campaign's owned synthetic run only")
    client = Client(args.directory, output_directory=args.directory / "be-api-runs" / args.run_label)
    if args.action == "prepare":
        if args.run_label == "after-auth-renewal":
            raise ValueError("Auth renewal must reuse the already prepared B3 case")
        result = prepare(client, args.case_id)
    else:
        original_output = args.directory / "be-api-runs" / "after-language-setup"
        evidence = (original_output if args.run_label == "after-auth-renewal" else client.output) / "reading-openai-preconditions.json"
        records = json.loads(evidence.read_text(encoding="utf-8"))
        if not records or records[-1]["caseId"] != args.case_id:
            raise ValueError("Prepared case must match the next unstarted Reading case")
        case = next(value for value in BLOCKERS if value.case_id == args.case_id)
        if args.run_label == "after-auth-renewal":
            if args.case_id != "context-inference-travel-b3":
                raise ValueError("Only the 401-before-start B3 case may use renewed auth")
            previous = json.loads((original_output / f"reading-{case.case_id}-workflow.json").read_text(encoding="utf-8"))
            starts = sorted(original_output.glob("*-reading-start.json"))
            failed_start = json.loads(starts[-1].read_text(encoding="utf-8")) if starts else {}
            if (previous.get("status") != "FAILED" or previous.get("errorType") != "ValueError"
                    or previous.get("practiceSetId") is not None or previous.get("observations")
                    or previous.get("answers") or failed_start.get("httpStatus") != 401
                    or failed_start.get("action") != "reading-start"):
                raise ValueError("Only a proved pre-start 401 may be retried")
            existing = bootstrap._qa_mysql_query(client.manifest, client.private,
                "SELECT COUNT(*) FROM language_learning_practice_set WHERE user_id=" + str(client.user_id)
                + " AND domain='READING' AND mode=" + _sql(case.mode)
                + " AND learning_date=" + _sql(records[-1]["learningDate"]) + ";")
            if existing != "0":
                raise ValueError("Never restart an already persisted Reading set")
        result = reading(client, case, expected_learning_date=records[-1]["learningDate"])
    print(json.dumps({key: result.get(key) for key in ("caseId", "status", "expectedBand", "practiceSetId", "keyword")}, ensure_ascii=True))


if __name__ == "__main__":
    main()
