"""Owned Google QA-user Reading prerequisites; no client/auth creation or live entry.

Only the caller executes these explicit fixture writes. The normal BE API still
selects difficulty and creates sets. No PracticeSet/question/attempt/history row
is written here. Profile initialization is SYNTHETIC, not a performed Level Test.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Protocol
from zoneinfo import ZoneInfo

from scripts import qa_campaign_integration as bootstrap
from scripts.qa_campaign_reading import NEW_RUN_CASES, ReadingCase


CASE_ORDER = (3, 1, 2, 0, 4)
FOLLOWUP_CASE_ORDER = (3, 2, 4)
FOLLOWUP_ZONE = "Asia/Seoul"
EARLY_ZONE = "Etc/GMT+12"
LATE_ZONE = "Pacific/Kiritimati"


class FixtureClient(Protocol):
    user_id: int
    browser_public_id: str
    manifest: dict[str, Any]
    private: dict[str, Any]

    def call(self, action: str, *, payload: dict[str, Any] | None = None) -> dict[str, Any]: ...


def dry_run_plan() -> dict[str, Any]:
    return {"dryRun": True, "databaseCalls": 0, "providerCalls": 0, "userId": 2,
            "profileBasis": "Explicit synthetic completed-level-test prerequisite, never real assessment",
            "cases": [{"caseId": NEW_RUN_CASES[index].case_id, "mode": NEW_RUN_CASES[index].mode,
                       "keyword": NEW_RUN_CASES[index].keyword, "targetBand": NEW_RUN_CASES[index].band,
                       "timezone": EARLY_ZONE if position < 3 else LATE_ZONE}
                      for position, index in enumerate(CASE_ORDER)],
            "immutableTables": ["language_learning_practice_set", "language_learning_practice_question",
                                "language_learning_practice_attempt", "language_learning_activity"],
            "onlyMutableFixtures": ["owned user's active timezone", "owned user's profile base score",
                                    "one manifest-owned new custom TOPIC keyword"],
            "writesRequireExplicitCallerExecution": True}


def fixture_date(position: int, now: datetime) -> str:
    if now.tzinfo is None or position not in range(5):
        raise ValueError("Aware clock and declared case position required")
    offset = -12 if position < 3 else 14
    return now.astimezone(timezone(timedelta(hours=offset))).date().isoformat()


def score_for_band(target_band: int, completed_scores: list[float | None]) -> tuple[float, int]:
    if target_band not in range(1, 6) or len(completed_scores) > 5:
        raise ValueError("Source-defined target band/history range required")
    scores = [float(value) for value in completed_scores if value is not None]
    if any(not 0 <= score <= 100 for score in scores):
        raise ValueError("Persisted completion score invalid")
    mean = sum(scores) / len(scores) if scores else None
    adjustment = 1 if mean is not None and mean >= 85 else -1 if mean is not None and mean < 55 else 0
    for base, score in enumerate((30.0, 50.0, 60.0, 75.0, 90.0), start=1):
        if max(1, min(5, base + adjustment)) == target_band:
            return score, adjustment
    raise ValueError("Requested boundary band impossible under preserved history; do not rewrite past scores")


def _sql(value: str | int | float | None) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, str):
        return "CONVERT(0x" + value.encode("utf-8").hex() + " USING utf8mb4)"
    if isinstance(value, bool):
        raise ValueError("Boolean is not an SQL numeric identity")
    return str(value)


class ReadingFixture:
    def __init__(self, client: FixtureClient, manifest_path: Path,
                 query: Callable[[str], str] | None = None, *,
                 case_order: tuple[int, ...] = CASE_ORDER,
                 fixed_zone: str | None = None):
        if type(client.user_id) is not int or client.user_id != 2 or not client.browser_public_id:
            raise ValueError("Normal browser-owned Google QA user2 only")
        if (not case_order or len(set(case_order)) != len(case_order)
                or any(type(index) is not int or index not in range(len(NEW_RUN_CASES)) for index in case_order)):
            raise ValueError("Unique source-defined Reading case order required")
        if fixed_zone is not None:
            ZoneInfo(fixed_zone)
        self.client = client
        self.path = manifest_path
        self.query = query or (lambda sql: bootstrap._qa_mysql_query(client.manifest, client.private, sql))
        self.identity = "u.id=2 AND u.social_type='GOOGLE' AND u.public_id=" + _sql(client.browser_public_id)
        self.case_order = case_order
        self.fixed_zone = fixed_zone

    def _case_time(self, position: int, now: datetime) -> tuple[str, str]:
        if now.tzinfo is None:
            raise ValueError("Aware clock required")
        if self.fixed_zone is None:
            return (EARLY_ZONE if position < 3 else LATE_ZONE, fixture_date(position, now))
        return (self.fixed_zone, now.astimezone(ZoneInfo(self.fixed_zone)).date().isoformat())

    def _snapshot(self) -> dict[str, Any]:
        raw = self.query(
            "SELECT JSON_OBJECT('userId',u.id,'googleIdentityMatches',1,"
            "'timezone',s.timezone,'origin',s.origin_language,'learning',s.learning_language,"
            "'pendingEffectiveDate',s.pending_effective_date,'profileId',p.id,'profileState',p.state,"
            "'baseScore',p.base_level_score,'calibrationStartedDate',p.calibration_started_date,"
            "'customCount',(SELECT COUNT(*) FROM language_learning_custom_keyword k WHERE k.user_id=u.id),"
            "'systemSelectionCount',(SELECT COUNT(*) FROM language_learning_user_system_keyword k WHERE k.user_id=u.id)) "
            "FROM user u JOIN language_learning_user_setting s ON s.user_id=u.id "
            "LEFT JOIN language_learning_profile p ON p.user_id=u.id WHERE " + self.identity + ";"
        )
        if not raw or len(raw.splitlines()) != 1:
            raise ValueError("Exact Google identity/settings fixture missing")
        return json.loads(raw)

    def _transaction(self, statements: str) -> None:
        # MySQL prepared COMMIT/ROLLBACK keeps every CAS mutation in one session.
        # A statement error closes the noninteractive connection and rolls back.
        result = self.query(
            "START TRANSACTION; SET @qa_ok=1; " + statements
            + " SET @qa_finish=IF(@qa_ok=1,'COMMIT','ROLLBACK'); PREPARE qa_finish FROM @qa_finish; "
            "EXECUTE qa_finish; DEALLOCATE PREPARE qa_finish; SELECT @qa_ok;"
        )
        if result.strip() != "1":
            raise ValueError("Reading fixture CAS failed; transaction rolled back")

    def initialize(self) -> dict[str, Any]:
        if self.path.exists():
            raise ValueError("Fixture initialization already recorded; no reseed")
        before = self._snapshot()
        if before["origin"] is None and before["learning"] is None:
            result = self.client.call("configure", payload={"originLanguage": "ko", "learningLanguage": "ja"})
            if result.get("httpStatus") != 200:
                raise ValueError("Normal settings configuration failed")
            before = self._snapshot()
        if before["origin"] != "ko" or before["learning"] != "ja" or before["pendingEffectiveDate"] is not None:
            raise ValueError("Existing configured language or scheduled settings must not be overwritten")
        if before["customCount"] != 0 or before["systemSelectionCount"] != 0:
            raise ValueError("Existing keywords/selections forbid fixture seeding")
        if before["profileState"] not in {None, "LEVEL_TEST_REQUIRED"} or before["baseScore"] is not None:
            raise ValueError("Only uninitialized profile can receive synthetic level-test prerequisite")
        report = {"status": "INITIALIZING", "userId": 2,
                  "googlePublicIdSha256": hashlib.sha256(self.client.browser_public_id.encode()).hexdigest(),
                  "syntheticLevelTestFixture": True, "realLevelTestExecuted": False,
                  "caseOrder": list(self.case_order), "fixedZone": self.fixed_zone,
                  "before": before, "cases": [], "createdAt": datetime.now(timezone.utc).isoformat()}
        bootstrap._write_json(self.path, report)
        profile_statement = (
            "INSERT INTO language_learning_profile (user_id,profile_version,state,base_level_score,evaluation_count,confidence,trend,additional_signals_json,created_at,updated_at,created_by,updated_by) "
            "SELECT u.id,'PROFILE','CALIBRATING',30,0,0,'stable','{}',NOW(6),NOW(6),'QA_FIXTURE','QA_FIXTURE' FROM user u WHERE "
            + self.identity + " AND NOT EXISTS (SELECT 1 FROM language_learning_profile p WHERE p.user_id=u.id); "
            if before["profileId"] is None else
            "UPDATE language_learning_profile p JOIN user u ON u.id=p.user_id SET p.state='CALIBRATING',p.base_level_score=30,p.updated_at=NOW(6) WHERE "
            + self.identity + " AND p.id=" + str(before["profileId"]) + " AND p.state='LEVEL_TEST_REQUIRED' AND p.base_level_score IS NULL; "
        )
        keyword = NEW_RUN_CASES[self.case_order[0]].keyword
        available_from = ("DATE(UTC_TIMESTAMP()-INTERVAL 12 HOUR)" if self.fixed_zone is None else
                          _sql(datetime.now(timezone.utc).astimezone(ZoneInfo(self.fixed_zone)).date().isoformat()))
        self._transaction(
            profile_statement + "SET @qa_ok=(@qa_ok AND ROW_COUNT()=1); "
            "INSERT INTO language_learning_custom_keyword (user_id,text,normalized_text,keyword_type,canonical_key,active,available_from,pending_parent_changed,created_at,updated_at,created_by,updated_by) "
            "SELECT u.id," + ",".join((_sql(keyword), _sql(keyword), "'TOPIC'", _sql(keyword)))
            + ",1," + available_from + ",0,NOW(6),NOW(6),'QA_FIXTURE','QA_FIXTURE' FROM user u WHERE "
            + self.identity + " AND @qa_ok=1 AND NOT EXISTS (SELECT 1 FROM language_learning_custom_keyword k WHERE k.user_id=u.id); "
            "SET @qa_ok=(@qa_ok AND ROW_COUNT()=1);"
        )
        owned = self.query("SELECT id FROM language_learning_custom_keyword WHERE user_id=2 AND created_by='QA_FIXTURE' AND text=" + _sql(keyword) + ";")
        if not owned.isdigit():
            raise ValueError("Unique newly created keyword ownership was not proven")
        report.update(status="INITIALIZED", keywordId=int(owned), currentKeyword=keyword, after=self._snapshot())
        bootstrap._write_json(self.path, report)
        return report

    def prepare_case(self, case_id: str, *, now: datetime | None = None) -> dict[str, Any]:
        report = json.loads(self.path.read_text(encoding="utf-8"))
        position = len(report["cases"])
        if (report["status"] != "INITIALIZED" or position >= len(self.case_order)
                or report.get("caseOrder") != list(self.case_order)
                or report.get("fixedZone") != self.fixed_zone):
            raise ValueError("Fixture not ready or all declared cases already prepared")
        case = NEW_RUN_CASES[self.case_order[position]]
        if case.case_id != case_id:
            raise ValueError("Fixed case order required; do not use fixture changes to rerun a failed case")
        if report["googlePublicIdSha256"] != hashlib.sha256(self.client.browser_public_id.encode()).hexdigest():
            raise ValueError("Fixture browser identity changed")
        before = self._snapshot()
        if before["pendingEffectiveDate"] is not None or before["customCount"] != 1 or before["systemSelectionCount"] != 0:
            raise ValueError("External settings/keyword changes detected")
        if before["profileState"] not in {"CALIBRATING", "ACTIVE"}:
            raise ValueError("Prepared profile was reset or changed outside this fixture")
        zone, learning_date = self._case_time(position, now or datetime.now(timezone.utc))
        collision = self.query("SELECT COUNT(*) FROM language_learning_practice_set WHERE user_id=2 AND domain='READING' AND mode="
                               + _sql(case.mode) + " AND learning_date=" + _sql(learning_date) + ";")
        if collision.strip() != "0":
            raise ValueError("Existing case date/mode must not be reseeded or regenerated")
        raw_scores = self.query("SELECT JSON_ARRAYAGG(official_score) FROM (SELECT official_score FROM language_learning_practice_set "
                                "WHERE user_id=2 AND domain='READING' AND status='COMPLETED' AND mode=" + _sql(case.mode)
                                + " ORDER BY learning_date DESC,id DESC LIMIT 5) recent;")
        # mysql --batch prints SQL NULL as uppercase NULL, not JSON's null.
        scores = [] if raw_scores.strip() == "NULL" else json.loads(raw_scores) or []
        score, adjustment = score_for_band(case.band, scores)
        case_report = {"caseId": case_id, "status": "PREPARING", "learningDate": learning_date, "timezone": zone,
                       "baseScoreFixture": score, "actualPriorCompletedScores": scores, "historyBandAdjustment": adjustment,
                       "expectedBand": case.band, "keyword": case.keyword, "before": before}
        report["cases"].append(case_report)
        report["status"] = "PREPARING_CASE"
        bootstrap._write_json(self.path, report)
        self._transaction(
            "UPDATE language_learning_user_setting s JOIN user u ON u.id=s.user_id SET s.timezone=" + _sql(zone)
            + ",s.updated_at=NOW(6) WHERE " + self.identity + " AND s.timezone=" + _sql(before["timezone"])
            + " AND s.pending_effective_date IS NULL; SET @qa_ok=(@qa_ok AND ROW_COUNT()=1); "
            "UPDATE language_learning_profile p JOIN user u ON u.id=p.user_id SET p.base_level_score=" + _sql(score)
            + ",p.updated_at=NOW(6) WHERE " + self.identity + " AND @qa_ok=1 AND p.id=" + str(before["profileId"])
            + " AND p.state=" + _sql(before["profileState"]) + " AND p.base_level_score <=> " + _sql(before["baseScore"])
            + "; SET @qa_ok=(@qa_ok AND ROW_COUNT()=1); "
            "UPDATE language_learning_custom_keyword k JOIN user u ON u.id=k.user_id SET k.text=" + _sql(case.keyword)
            + ",k.normalized_text=" + _sql(case.keyword) + ",k.canonical_key=" + _sql(case.keyword)
            + ",k.updated_at=NOW(6) WHERE " + self.identity + " AND @qa_ok=1 AND k.id=" + str(report["keywordId"])
            + " AND k.text=" + _sql(report["currentKeyword"]) + " AND k.created_by='QA_FIXTURE' AND k.active=1"
            + " AND k.pending_effective_date IS NULL; SET @qa_ok=(@qa_ok AND ROW_COUNT()=1);"
        )
        case_report.update(status="PREPARED", after=self._snapshot())
        report["currentKeyword"] = case.keyword
        report["status"] = "INITIALIZED"
        bootstrap._write_json(self.path, report)
        return case_report


def prepare_reading_fixture(client: FixtureClient, case: ReadingCase, directory: Path) -> dict[str, Any]:
    """Explicit caller-owned fixture execution; return the verified expected date."""
    if case not in NEW_RUN_CASES:
        raise ValueError("Only the declared campaign cases are admitted")
    directory.mkdir(parents=True, exist_ok=True)
    fixture = ReadingFixture(client, directory / "manifest.json")
    if not fixture.path.exists():
        fixture.initialize()
    return fixture.prepare_case(case.case_id)


if __name__ == "__main__":
    print(json.dumps(dry_run_plan(), ensure_ascii=False, indent=2))
