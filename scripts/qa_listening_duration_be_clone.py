"""Narrow QA-only pre-correction clone of owned Listening set 5.

Copies the four READY prefix rows and original 4.45-second item from the
owned LOCAL QA user2 into a new normal-login Google QA-user3 set on
2026-09-21, then enqueues one real BE correction command.
The original six rows and all prior outbox events are never updated/deleted.
This is a derived fixture, not a captured production wire request.
"""
from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.qa_campaign_budget import atomic_json
from scripts.qa_campaign_integration import _owned_manifest, _qa_mysql_query


SOURCE_IDS = (22, 23, 24, 25, 26, 27)
TARGET_DATE = date(2026, 9, 21)
PUBLIC_ID = "TC-GA5T-4LRB"


def _sql_text(value: str) -> str:
    return "CONVERT(0x" + value.encode("utf-8").hex() + " USING utf8mb4)"


def inspect(environment: Path, snapshot: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = _owned_manifest(environment)
    if (manifest.get("campaign") != "openai-speech-campaign-20260920"
            or snapshot != environment / "be-api-runs" / "continuation-listening-user2-dictation-easy" /
            "listening-snapshot-5-1789908022437726200.json"):
        raise ValueError("Only preserved owned source snapshot/set admitted")
    private = json.loads((environment / "secrets.private.json").read_text(encoding="utf-8"))
    saved = json.loads(snapshot.read_text(encoding="utf-8"))
    rows = saved["items"]
    if (saved["dailySetId"] != 5 or saved["mode"] != "DICTATION"
            or saved["difficulty"] != "EASY" or saved["status"] != "PARTIAL"
            or [row["itemId"] for row in rows] != list(SOURCE_IDS)
            or [row["status"] for row in rows] != ["READY"] * 4 + ["REPLACED", "NOT_EVALUABLE"]
            or rows[4]["generationMetadata"]["qualityCorrectionCount"] != 1
            or rows[4]["failureReason"] != "AUDIO_TOO_SHORT"):
        raise ValueError("Preserved source set is not the exact 4.45s case")
    identity = _qa_mysql_query(manifest, private,
        "SELECT COUNT(*) FROM user source JOIN user browser ON browser.id=3 "
        "WHERE source.id=2 AND source.social_type='LOCAL' "
        "AND browser.social_type='GOOGLE' AND browser.public_id=" + _sql_text(PUBLIC_ID) + ";")
    if identity != "1":
        raise ValueError("Exact source LOCAL and normal browser Google QA identities changed")
    raw = _qa_mysql_query(manifest, private,
        "SELECT i.id,i.item_index,i.replacement_sequence,i.status,SHA2(i.source_text,256),"
        "COALESCE(JSON_EXTRACT(i.generation_metadata,'$.qualityCorrectionCount'),-1) "
        "FROM language_learning_listening_item i JOIN language_learning_listening_daily_set s "
        "ON s.id=i.daily_set_id WHERE s.id=5 AND s.user_id=2 ORDER BY i.id;")
    found = [line.split("\t") for line in raw.splitlines()]
    if len(found) != 6 or [int(item[0]) for item in found] != list(SOURCE_IDS):
        raise ValueError("Current original QA DB rows differ from copied snapshot")
    for row, stored in zip(rows, found, strict=True):
        if (int(stored[1]) != row["itemIndex"] or int(stored[2]) != row["replacementSequence"]
                or stored[3] != row["status"] or stored[4].lower() != hashlib.sha256(row["sourceText"].encode()).hexdigest()):
            raise ValueError("Original item identity/text changed; clone denied")
    if found[4][5] != "1":
        raise ValueError("Source correction budget not reserved")
    destination = _qa_mysql_query(manifest, private,
        "SELECT COUNT(*) FROM language_learning_listening_daily_set WHERE user_id=3 "
        "AND learning_date='2026-09-21' AND learning_language='ja' AND learning_mode='DICTATION';")
    report = {"status": "INSPECTED_NO_WRITES", "originalSetId": 5,
              "originalOwner": "owned LOCAL QA user2", "derivedOwner": "normal Google QA user3",
              "originalRowIds": list(SOURCE_IDS), "originalStatus": saved["status"],
              "sourceSha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
              "sourceTextSha256": found[4][4].lower(), "targetDate": str(TARGET_DATE),
              "dateIsActualToday": date.today() == TARGET_DATE,
              "destinationCollisionCount": int(destination),
              "fixtureBoundary": "Five copied rows; source item26 reverts to its pre-correction NOT_EVALUABLE state in clone only",
              "plannedWrites": {"newSets": 1, "newItems": 5, "newOutboxCommands": 1,
                                "originalUpdates": 0, "originalDeletes": 0},
              "providerStarts": 0}
    return manifest, private, report


def prepare(environment: Path, snapshot: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise ValueError("One-shot clone artifact already exists")
    manifest, private, report = inspect(environment, snapshot)
    if date.today() != TARGET_DATE or report["destinationCollisionCount"] != 0:
        raise ValueError("Only the actual new QA learning day without a conflicting set is admitted")
    atomic_json(output, {**report, "status": "PREPARING"})
    # All statements share one MySQL connection. Any SQL failure closes it and
    # rolls back; the explicit CAS flag chooses COMMIT only after exact counts.
    sql = """
START TRANSACTION;
SET @qa_ok=(SELECT COUNT(*)=1 FROM language_learning_listening_daily_set s
  JOIN user u ON u.id=s.user_id WHERE s.id=5 AND u.id=2 AND u.social_type='LOCAL'
  AND (SELECT COUNT(*) FROM user browser WHERE browser.id=3
       AND browser.social_type='GOOGLE' AND browser.public_id='TC-GA5T-4LRB')=1
  AND s.status='PARTIAL'
  AND s.learning_mode='DICTATION' AND s.difficulty='EASY'
  AND NOT EXISTS(SELECT 1 FROM language_learning_listening_daily_set d
      WHERE d.user_id=3 AND d.learning_date='2026-09-21'
      AND d.learning_language='ja' AND d.learning_mode='DICTATION'));
INSERT INTO language_learning_listening_daily_set
 (user_id,learning_date,origin_language,learning_language,learning_mode,difficulty,
  topic_snapshot,keyword_snapshot,profile_snapshot,policy_version,generation_version,
  target_item_count,physical_item_count,completed_item_count,status,failure_reason,
  completed_at,version,created_at,updated_at,created_by,updated_by)
SELECT 3,'2026-09-21',origin_language,learning_language,learning_mode,difficulty,
 topic_snapshot,keyword_snapshot,profile_snapshot,policy_version,generation_version,
 5,5,4,'GENERATING',NULL,NULL,0,NOW(6),NOW(6),'QA_CLOSURE','QA_CLOSURE'
FROM language_learning_listening_daily_set WHERE id=5 AND @qa_ok=1;
SET @qa_ok=(@qa_ok AND ROW_COUNT()=1);
SET @new_set=LAST_INSERT_ID();
INSERT INTO language_learning_listening_item
 (daily_set_id,item_index,replacement_sequence,source_text,normalized_source_text,
  reference_meanings,key_meaning_units,target_keywords,estimated_audio_seconds,
  audio_object_key,audio_duration_ms,audio_content_type,audio_checksum,audio_retention_until,
  audio_deleted_at,voice_snapshot,content_hash,similarity_key,generation_metadata,
  replacement_for_item_id,status,failure_reason,automatic_tts_retry_count,
  manual_tts_retry_count,version,created_at,updated_at,created_by,updated_by)
SELECT @new_set,item_index,replacement_sequence,source_text,normalized_source_text,
 reference_meanings,key_meaning_units,target_keywords,estimated_audio_seconds,
 IF(id=26,NULL,audio_object_key),IF(id=26,NULL,audio_duration_ms),
 IF(id=26,NULL,audio_content_type),IF(id=26,NULL,audio_checksum),
 IF(id=26,NULL,audio_retention_until),IF(id=26,NULL,audio_deleted_at),voice_snapshot,
 content_hash,similarity_key,generation_metadata,NULL,
 IF(id=26,'NOT_EVALUABLE','READY'),IF(id=26,'AUDIO_TOO_SHORT',NULL),0,0,0,
 NOW(6),NOW(6),'QA_CLOSURE','QA_CLOSURE'
FROM language_learning_listening_item WHERE daily_set_id=5 AND id IN (22,23,24,25,26)
 AND @qa_ok=1 ORDER BY item_index;
SET @qa_ok=(@qa_ok AND ROW_COUNT()=5);
SET @new_source=(SELECT id FROM language_learning_listening_item
 WHERE daily_set_id=@new_set AND item_index=5 AND replacement_sequence=0);
INSERT INTO language_learning_listening_outbox
 (event_type,aggregate_id,payload_json,idempotency_key,status,attempt_count,
  available_at,last_error,processed_at,version,created_at,updated_at,created_by,updated_by)
SELECT 'GENERATE_SET',@new_set,CAST(JSON_OBJECT(
 'replacementForItemId',@new_source,'logicalItemIndex',5,'replacementSequence',1,
 'manualRetryAttempt',0,'durationCorrection',JSON_OBJECT(
 'previousSourceText',i.source_text,'previousMeasuredSeconds',4.45,
 'qualityCorrectionCount',1)) AS CHAR),
 CONCAT('qa-closure-listening-',@new_set,'-duration-correction-5'),
 'PENDING',0,NOW(6),NULL,NULL,0,NOW(6),NOW(6),'QA_CLOSURE','QA_CLOSURE'
FROM language_learning_listening_item i WHERE i.id=@new_source AND @qa_ok=1;
SET @qa_ok=(@qa_ok AND ROW_COUNT()=1);
SET @new_event=LAST_INSERT_ID();
SET @qa_finish=IF(@qa_ok=1,'COMMIT','ROLLBACK');
PREPARE qa_finish FROM @qa_finish; EXECUTE qa_finish; DEALLOCATE PREPARE qa_finish;
SELECT @qa_ok,@new_set,@new_source,@new_event;
"""
    try:
        values = _qa_mysql_query(manifest, private, sql).split("\t")
        if len(values) != 4 or values[0] != "1" or any(not x.isdigit() for x in values[1:]):
            raise ValueError("QA clone transaction rolled back or returned invalid identity")
        report.update(status="PREPARED_BE_WORKER_OWNED", newSetId=int(values[1]),
                      newSourceItemId=int(values[2]), newOutboxEventId=int(values[3]))
    except BaseException as exc:
        report.update(status="FAILED_OR_COMMIT_UNKNOWN", errorType=type(exc).__name__)
        raise
    finally:
        atomic_json(output, report)
    return report


def verify(environment: Path, snapshot: Path, output: Path) -> dict[str, Any]:
    manifest, private, original = inspect(environment, snapshot)
    recorded = json.loads(output.read_text(encoding="utf-8"))
    if (recorded.get("status") != "PREPARED_BE_WORKER_OWNED"
            or recorded.get("sourceSha256") != original["sourceSha256"]
            or recorded.get("originalRowIds") != list(SOURCE_IDS)):
        raise ValueError("Clone manifest/source authority mismatch")
    set_id = recorded["newSetId"]
    event_id = recorded["newOutboxEventId"]
    if type(set_id) is not int or type(event_id) is not int or set_id <= 5 or event_id <= 0:
        raise ValueError("New DB identities invalid")
    raw = _qa_mysql_query(manifest, private,
        "SELECT s.status,s.physical_item_count,s.completed_item_count,"
        "(SELECT COUNT(*) FROM language_learning_listening_item i WHERE i.daily_set_id=s.id),"
        "(SELECT COUNT(*) FROM language_learning_listening_item i WHERE i.daily_set_id=s.id AND i.status='READY'),"
        "(SELECT COUNT(*) FROM language_learning_listening_item i WHERE i.daily_set_id=s.id AND i.item_index=5 AND i.replacement_sequence=1 AND i.status='READY'),"
        "e.status,e.attempt_count,COALESCE(e.last_error,'') FROM language_learning_listening_daily_set s JOIN user u ON u.id=s.user_id "
        "JOIN language_learning_listening_outbox e ON e.aggregate_id=s.id "
        "WHERE s.id=" + str(set_id) + " AND u.id=3 AND u.social_type='GOOGLE' AND u.public_id="
        + _sql_text(PUBLIC_ID) + " AND e.id=" + str(event_id) + ";")
    parts = raw.split("\t")
    if len(parts) != 9:
        raise ValueError("New set owner/worker record missing")
    status, physical, completed, total, ready, corrected, event_status, attempts, last_error = parts
    charset_artifact = output.with_name("listening-be-clone-charset-repair.json")
    retry_event_status = None
    if charset_artifact.exists():
        repair = json.loads(charset_artifact.read_text(encoding="utf-8"))
        retry_id = repair.get("newOutboxEventId")
        if (repair.get("status") == "CORRECTED_COMMAND_ENQUEUED"
                and repair.get("derivedSetId") == set_id and type(retry_id) is int):
            retry_event_status = _qa_mysql_query(manifest, private,
                "SELECT e.status FROM language_learning_listening_outbox e "
                "JOIN language_learning_listening_daily_set s ON s.id=e.aggregate_id "
                "JOIN user u ON u.id=s.user_id WHERE e.id=" + str(retry_id)
                + " AND s.id=" + str(set_id) + " AND u.id=3 AND u.social_type='GOOGLE' "
                "AND u.public_id=" + _sql_text(PUBLIC_ID) + ";")
    result = {"newSetId": set_id, "originalSetId": 5,
              "originalSnapshotSha256Unchanged": original["sourceSha256"] == recorded["sourceSha256"],
              "newSetStatus": status, "physicalItemCount": int(physical),
              "completedItemCount": int(completed), "totalRows": int(total),
              "readyRows": int(ready), "correctedItem5Ready": int(corrected),
              "correctionOutboxStatus": event_status,
              "correctedFixtureOutboxStatus": retry_event_status,
              "correctionOutboxAttempts": int(attempts), "failureDiagnosticPrivate": last_error,
              "browserPlaybackVerified": False,
              "result": ("BE_READY_FIVE_ITEMS" if status == "READY" and int(ready) == 5
                         and int(corrected) == 1 and (retry_event_status or event_status) == "SUCCEEDED"
                         else "BE_NOT_COMPLETE")}
    if int(corrected) == 1:
        correction = _qa_mysql_query(manifest, private,
            "SELECT i.id,i.audio_duration_ms,i.audio_content_type,"
            "SHA2(i.source_text,256),CHAR_LENGTH(i.source_text),"
            "IF(i.audio_object_key IS NULL,0,1),i.voice_snapshot "
            "FROM language_learning_listening_item i "
            "JOIN language_learning_listening_daily_set s ON s.id=i.daily_set_id "
            "JOIN user u ON u.id=s.user_id WHERE s.id=" + str(set_id)
            + " AND u.id=3 AND u.social_type='GOOGLE' AND u.public_id=" + _sql_text(PUBLIC_ID)
            + " AND i.item_index=5 AND i.replacement_sequence=1 AND i.status='READY';")
        details = correction.split("\t", 6)
        if len(details) == 7:
            result["correctedItem"] = {"id": int(details[0]), "measuredAudioMs": int(details[1]),
                                       "mime": details[2], "sourceSha256": details[3],
                                       "sourceCharacters": int(details[4]),
                                       "audioObjectExists": details[5] == "1",
                                       "voiceSnapshot": json.loads(details[6])}
    sessions = _qa_mysql_query(manifest, private,
        "SELECT se.id,a.id,i.id,i.item_index,a.status "
        "FROM language_learning_listening_session se "
        "JOIN language_learning_listening_item_attempt a ON a.session_id=se.id "
        "JOIN language_learning_listening_item i ON i.id=a.item_id "
        "JOIN user u ON u.id=se.user_id WHERE se.daily_set_id=" + str(set_id)
        + " AND u.id=3 AND u.social_type='GOOGLE' AND u.public_id=" + _sql_text(PUBLIC_ID)
        + " ORDER BY a.id;")
    result["browserSessionAttempts"] = [dict(zip(("sessionId", "attemptId", "itemId", "itemIndex", "status"),
                                                   line.split("\t"), strict=True))
                                        for line in sessions.splitlines() if line]
    if result["result"] == "BE_NOT_COMPLETE":
        diagnostic = _qa_mysql_query(manifest, private,
            "SELECT i.status,IF(i.audio_object_key IS NULL,1,0),"
            "COALESCE(JSON_EXTRACT(i.generation_metadata,'$.qualityCorrectionCount'),-1),"
            "IF(JSON_EXTRACT(i.generation_metadata,'$.durationDemand') IS NULL,0,1),"
            "COALESCE(JSON_EXTRACT(e.payload_json,'$.durationCorrection.qualityCorrectionCount'),-1),"
            "IF(i.source_text=JSON_UNQUOTE(JSON_EXTRACT(e.payload_json,"
            "'$.durationCorrection.previousSourceText')),1,0),"
            "IF(i.daily_set_id=e.aggregate_id,1,0),"
            "CHAR_LENGTH(i.source_text),"
            "CHAR_LENGTH(JSON_UNQUOTE(JSON_EXTRACT(e.payload_json,'$.durationCorrection.previousSourceText'))),"
            "SHA2(i.source_text,256),"
            "SHA2(JSON_UNQUOTE(JSON_EXTRACT(e.payload_json,'$.durationCorrection.previousSourceText')),256),"
            "HEX(i.source_text),"
            "HEX(JSON_UNQUOTE(JSON_EXTRACT(e.payload_json,'$.durationCorrection.previousSourceText'))) "
            "FROM language_learning_listening_item i "
            "JOIN language_learning_listening_daily_set s ON s.id=i.daily_set_id "
            "JOIN user u ON u.id=s.user_id JOIN language_learning_listening_outbox e "
            "ON e.aggregate_id=s.id WHERE i.id=" + str(recorded["newSourceItemId"])
            + " AND s.id=" + str(set_id) + " AND e.id=" + str(event_id)
            + " AND u.id=3 AND u.social_type='GOOGLE' AND u.public_id=" + _sql_text(PUBLIC_ID) + ";")
        values = diagnostic.split("\t")
        if len(values) == 13:
            result["authorityDiagnostic"] = dict(zip(("sourceStatus", "audioKeyNull", "metadataCorrectionCount",
                "metadataDemandPresent", "commandCorrectionCount", "sourceTextMatchesCommand",
                "sourceSetMatchesEvent", "sourceCharacters", "commandCharacters", "sourceSha256", "commandSha256"),
                values[:11], strict=True))
            source_bytes, command_bytes = bytes.fromhex(values[11]), bytes.fromhex(values[12])
            source_text = source_bytes.decode("utf-8", errors="replace")
            command_text = command_bytes.decode("utf-8", errors="replace")
            result["authorityDiagnostic"]["mismatchCodePoints"] = [
                {"index": index, "source": f"U+{ord(a):04X}", "command": f"U+{ord(b):04X}"}
                for index, (a, b) in enumerate(zip(source_text, command_text)) if a != b
            ]
    atomic_json(output.with_name("listening-be-clone-verify.json"), result)
    return result


def repair_fixture_charset(environment: Path, snapshot: Path, output: Path) -> dict[str, Any]:
    """Resume the exact owned clone once after its pre-provider latin1 JSON corruption.

    Preserve the failed event and original rows. This fixes only QA fixture
    encoding, not Listening production acceptance, recovery or retry budgets.
    """
    artifact = output.with_name("listening-be-clone-charset-repair.json")
    if artifact.exists():
        raise ValueError("One-shot QA charset repair already has preserved evidence")
    previous = verify(environment, snapshot, output)
    diagnostic = previous.get("authorityDiagnostic", {})
    if (previous["result"] != "BE_NOT_COMPLETE" or previous["newSetStatus"] != "PARTIAL"
            or previous["correctionOutboxStatus"] != "FAILED"
            or previous["failureDiagnosticPrivate"] != "LISTENING_DURATION_CORRECTION_AUTHORITY_INVALID"
            or diagnostic.get("sourceTextMatchesCommand") != "0"
            or diagnostic.get("sourceStatus") != "NOT_EVALUABLE"
            or diagnostic.get("audioKeyNull") != "1"
            or diagnostic.get("sourceSha256") != previous_source_hash(environment, snapshot)):
        raise ValueError("Only the exact pre-provider QA character-conversion failure is repairable")
    manifest = _owned_manifest(environment)
    private = json.loads((environment / "secrets.private.json").read_text(encoding="utf-8"))
    recorded = json.loads(output.read_text(encoding="utf-8"))
    set_id, source_id, failed_id = (recorded[key] for key in
                                    ("newSetId", "newSourceItemId", "newOutboxEventId"))
    report = {"status": "PREPARING", "derivedSetId": set_id, "sourceItemId": source_id,
              "preservedFailedEventId": failed_id, "originalSetId": 5,
              "reason": "QA mysql client default charset serialized Japanese in JSON_OBJECT as question marks",
              "productionCodeModified": False, "newProviderCallsBeforeRepair": 0}
    atomic_json(artifact, report)
    sql = f"""
SET NAMES utf8mb4;
START TRANSACTION;
SET @qa_ok=(SELECT COUNT(*)=1 FROM language_learning_listening_daily_set s
 JOIN user u ON u.id=s.user_id JOIN language_learning_listening_item i ON i.daily_set_id=s.id
 JOIN language_learning_listening_outbox old ON old.aggregate_id=s.id
 WHERE s.id={set_id} AND u.id=3 AND u.social_type='GOOGLE' AND u.public_id='TC-GA5T-4LRB'
 AND s.status='PARTIAL' AND s.completed_item_count=4 AND s.physical_item_count=5
 AND i.id={source_id} AND i.item_index=5 AND i.replacement_sequence=0
 AND i.status='NOT_EVALUABLE' AND i.audio_object_key IS NULL
 AND SHA2(i.source_text,256)='{diagnostic['sourceSha256']}'
 AND old.id={failed_id} AND old.status='FAILED'
 AND old.last_error='LISTENING_DURATION_CORRECTION_AUTHORITY_INVALID'
 AND NOT EXISTS(SELECT 1 FROM language_learning_listening_outbox later
     WHERE later.idempotency_key='qa-closure-listening-charset-corrected-{set_id}'));
UPDATE language_learning_listening_daily_set SET status='GENERATING',failure_reason=NULL,
 version=version+1,updated_at=NOW(6),updated_by='QA_CLOSURE' WHERE id={set_id} AND @qa_ok=1;
SET @qa_ok=(@qa_ok AND ROW_COUNT()=1);
INSERT INTO language_learning_listening_outbox
 (event_type,aggregate_id,payload_json,idempotency_key,status,attempt_count,
  available_at,last_error,processed_at,version,created_at,updated_at,created_by,updated_by)
SELECT 'GENERATE_SET',{set_id},JSON_OBJECT(
 'replacementForItemId',{source_id},'logicalItemIndex',5,'replacementSequence',1,
 'manualRetryAttempt',0,'durationCorrection',JSON_OBJECT(
 'previousSourceText',i.source_text,'previousMeasuredSeconds',4.45,
 'qualityCorrectionCount',1)),
 'qa-closure-listening-charset-corrected-{set_id}',
 'PENDING',0,NOW(6),NULL,NULL,0,NOW(6),NOW(6),'QA_CLOSURE','QA_CLOSURE'
FROM language_learning_listening_item i WHERE i.id={source_id} AND @qa_ok=1;
SET @qa_ok=(@qa_ok AND ROW_COUNT()=1);
SET @new_event=LAST_INSERT_ID();
SET @qa_ok=(@qa_ok AND (SELECT COUNT(*)=1 FROM language_learning_listening_outbox e
 JOIN language_learning_listening_item i ON i.id={source_id}
 WHERE e.id=@new_event AND e.aggregate_id={set_id}
 AND BINARY i.source_text=BINARY JSON_UNQUOTE(JSON_EXTRACT(
 e.payload_json,'$.durationCorrection.previousSourceText'))));
SET @qa_finish=IF(@qa_ok=1,'COMMIT','ROLLBACK');
PREPARE qa_finish FROM @qa_finish; EXECUTE qa_finish; DEALLOCATE PREPARE qa_finish;
SELECT @qa_ok,@new_event;
"""
    try:
        values = _qa_mysql_query(manifest, private, sql).split("\t")
        if len(values) != 2 or values[0] != "1" or not values[1].isdigit():
            raise ValueError("QA charset repair transaction rolled back")
        report.update(status="CORRECTED_COMMAND_ENQUEUED", newOutboxEventId=int(values[1]))
    except BaseException as exc:
        report.update(status="FAILED_OR_COMMIT_UNKNOWN", errorType=type(exc).__name__)
        raise
    finally:
        atomic_json(artifact, report)
    return report


def previous_source_hash(environment: Path, snapshot: Path) -> str:
    _, _, report = inspect(environment, snapshot)
    return report["sourceTextSha256"]


def normalize_fixture_learning_count(environment: Path, snapshot: Path, output: Path) -> dict[str, Any]:
    """Do not inherit a different QA user's completed learning from source set 5."""
    artifact = output.with_name("listening-be-clone-learning-count-fixture.json")
    if artifact.exists():
        raise ValueError("One-shot owned learning-count correction already exists")
    current = verify(environment, snapshot, output)
    if (current["result"] != "BE_READY_FIVE_ITEMS" or current["completedItemCount"] != 4
            or current["readyRows"] != 5):
        raise ValueError("Only the exact newly ready cloned QA set with inherited count is eligible")
    set_id = current["newSetId"]
    manifest = _owned_manifest(environment)
    private = json.loads((environment / "secrets.private.json").read_text(encoding="utf-8"))
    report = {"status": "PREPARING", "derivedSetId": set_id,
              "inheritedCompletedLearningCount": 4, "targetCompletedLearningCount": 0,
              "readyItemCountUnchanged": 5, "originalSetIdUntouched": 5}
    atomic_json(artifact, report)
    sql = f"""
START TRANSACTION;
UPDATE language_learning_listening_daily_set s JOIN user u ON u.id=s.user_id
 SET s.completed_item_count=0,s.version=s.version+1,s.updated_at=NOW(6),s.updated_by='QA_CLOSURE'
 WHERE s.id={set_id} AND u.id=3 AND u.social_type='GOOGLE' AND u.public_id='TC-GA5T-4LRB'
 AND s.created_by='QA_CLOSURE' AND s.status='READY' AND s.completed_item_count=4
 AND (SELECT COUNT(*) FROM language_learning_listening_item i
      WHERE i.daily_set_id=s.id AND i.status='READY')=5
 AND NOT EXISTS(SELECT 1 FROM language_learning_listening_session se
      WHERE se.daily_set_id=s.id);
SET @qa_ok=(ROW_COUNT()=1);
SET @qa_finish=IF(@qa_ok=1,'COMMIT','ROLLBACK');
PREPARE qa_finish FROM @qa_finish; EXECUTE qa_finish; DEALLOCATE PREPARE qa_finish;
SELECT @qa_ok;
"""
    try:
        if _qa_mysql_query(manifest, private, sql) != "1":
            raise ValueError("Owned QA set already used or count correction rolled back")
        report["status"] = "LEARNING_COUNT_NORMALIZED_FOR_NEW_BROWSER_USER"
    except BaseException as exc:
        report.update(status="FAILED_OR_COMMIT_UNKNOWN", errorType=type(exc).__name__)
        raise
    finally:
        atomic_json(artifact, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--repair-fixture-charset", action="store_true")
    parser.add_argument("--normalize-learning-count", action="store_true")
    args = parser.parse_args()
    environment, snapshot, output = (args.environment.resolve(), args.snapshot.resolve(), args.output.resolve())
    if output.parent != environment.parent / "closure-20260920-v1":
        raise ValueError("Clone manifest must stay in the owned closure directory")
    if sum((args.prepare, args.verify, args.repair_fixture_charset, args.normalize_learning_count)) > 1:
        raise ValueError("Choose one action")
    if args.normalize_learning_count:
        result = normalize_fixture_learning_count(environment, snapshot, output)
    elif args.repair_fixture_charset:
        result = repair_fixture_charset(environment, snapshot, output)
    elif args.verify:
        result = verify(environment, snapshot, output)
    elif args.prepare:
        result = prepare(environment, snapshot, output)
    else:
        _, _, result = inspect(environment, snapshot)
    print(json.dumps({key: result.get(key) for key in (
        "status", "sourceSha256", "dateIsActualToday", "destinationCollisionCount",
        "newSetId", "newSourceItemId", "newOutboxEventId", "providerStarts",
        "result", "newSetStatus", "readyRows", "correctedItem5Ready",
    )}))


if __name__ == "__main__":
    main()
