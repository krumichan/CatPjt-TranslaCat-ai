from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

import pytest

from scripts.qa_campaign_reading import NEW_RUN_CASES
from scripts.qa_campaign_reading_fixture import (
    CASE_ORDER, FOLLOWUP_CASE_ORDER, FOLLOWUP_ZONE, ReadingFixture, dry_run_plan, fixture_date, score_for_band,
)


class Client:
    user_id = 2
    browser_public_id = "synthetic-google-id"
    manifest: dict[str, Any] = {}
    private: dict[str, Any] = {}

    def call(self, action: str, *, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        raise AssertionError("Configured offline fixture must not make HTTP calls")


class SqlFake:
    def __init__(self):
        self.sql: list[str] = []
        self.snapshot: dict[str, Any] = {
            "userId": 2, "googleIdentityMatches": 1, "timezone": "Asia/Tokyo", "origin": "ko", "learning": "ja",
            "pendingEffectiveDate": None, "profileId": 8, "profileState": "LEVEL_TEST_REQUIRED", "baseScore": None,
            "calibrationStartedDate": None, "customCount": 0, "systemSelectionCount": 0,
        }
        self.collision = False
        self.cas_ok = True
        self.scores: list[float | None] = []
        self.sql_null_history = False

    def __call__(self, sql: str) -> str:
        self.sql.append(sql)
        if sql.startswith("SELECT JSON_OBJECT"):
            return json.dumps(self.snapshot)
        if sql.startswith("SELECT id FROM language_learning_custom_keyword"):
            return "21"
        if sql.startswith("SELECT COUNT(*) FROM language_learning_practice_set"):
            return "1" if self.collision else "0"
        if sql.startswith("SELECT JSON_ARRAYAGG"):
            return "NULL" if self.sql_null_history else json.dumps(self.scores)
        if sql.startswith("START TRANSACTION"):
            if not self.cas_ok:
                return "0"
            if "INSERT INTO language_learning_custom_keyword" in sql:
                self.snapshot.update(profileState="CALIBRATING", baseScore=30.0, customCount=1)
            return "1"
        raise AssertionError("Unexpected SQL")


def test_dry_plan_needs_no_client_database_or_provider():
    plan = dry_run_plan()
    assert plan["databaseCalls"] == plan["providerCalls"] == 0
    assert [row["caseId"] for row in plan["cases"]] == [NEW_RUN_CASES[index].case_id for index in (3, 1, 2, 0, 4)]
    assert plan["userId"] == 2


@pytest.mark.parametrize("hour", range(24))
def test_timezone_fixture_dates_are_distinct_for_every_utc_hour(hour: int):
    now = datetime(2026, 9, 19, hour, tzinfo=timezone.utc)
    assert fixture_date(0, now) < fixture_date(3, now)


@pytest.mark.parametrize("target,scores,expected", [
    (1, [], (30.0, 0)), (3, [], (60.0, 0)), (3, [100.0], (50.0, 1)),
    (5, [100.0], (75.0, 1)), (3, [20.0, 40.0], (75.0, -1)), (3, [None, 70.0], (60.0, 0)),
])
def test_profile_fixture_preserves_actual_recent_history_adjustment(target: int, scores: list[float | None], expected: tuple[float, int]):
    assert score_for_band(target, scores) == expected


def test_cannot_force_b1_after_perfect_completed_history():
    with pytest.raises(ValueError, match="impossible"):
        score_for_band(1, [100.0])


def test_initialization_and_case_only_mutate_three_owned_fixture_tables(tmp_path: Path):
    query = SqlFake()
    fixture = ReadingFixture(Client(), tmp_path / "fixture.json", query)
    initialized = fixture.initialize()
    assert initialized["realLevelTestExecuted"] is False
    assert initialized["keywordId"] == 21
    assert initialized["before"]["customCount"] == 0
    prepared = fixture.prepare_case(NEW_RUN_CASES[3].case_id, now=datetime(2026, 9, 19, 5, tzinfo=timezone.utc))
    assert prepared["learningDate"] == "2026-09-18"
    assert prepared["expectedBand"] == 1
    transactions = [sql for sql in query.sql if sql.startswith("START TRANSACTION")]
    assert len(transactions) == 2
    for sql in transactions:
        assert "u.id=2 AND u.social_type='GOOGLE' AND u.public_id=" in sql
        assert "IF(@qa_ok=1,'COMMIT','ROLLBACK')" in sql
        assert "DELETE" not in sql and "TRUNCATE" not in sql
        for table in ("practice_set", "practice_question", "practice_attempt", "activity"):
            assert "UPDATE language_learning_" + table not in sql
            assert "INSERT INTO language_learning_" + table not in sql
    assert "p.base_level_score <=> 30.0" in transactions[1]
    assert "k.id=21" in transactions[1]
    with pytest.raises(ValueError, match="Fixed case order"):
        fixture.prepare_case(NEW_RUN_CASES[3].case_id)


def test_followup_three_cases_use_actual_learning_date_without_timezone_shift(tmp_path: Path):
    query = SqlFake()
    fixture = ReadingFixture(Client(), tmp_path / "followup.json", query,
                             case_order=FOLLOWUP_CASE_ORDER, fixed_zone=FOLLOWUP_ZONE)
    initialized = fixture.initialize()
    assert initialized["caseOrder"] == [3, 2, 4]
    assert initialized["fixedZone"] == "Asia/Seoul"
    assert initialized["currentKeyword"] == "買い物"
    transaction = next(sql for sql in query.sql if "INSERT INTO language_learning_custom_keyword" in sql)
    assert "DATE(UTC_TIMESTAMP()-INTERVAL 12 HOUR)" not in transaction
    now = datetime(2026, 9, 20, 7, tzinfo=timezone.utc)
    for index, case_index in enumerate(FOLLOWUP_CASE_ORDER):
        prepared = fixture.prepare_case(NEW_RUN_CASES[case_index].case_id, now=now)
        assert prepared["learningDate"] == "2026-09-20"
        assert prepared["timezone"] == FOLLOWUP_ZONE
        assert prepared["expectedBand"] == (1, 3, 5)[index]
    with pytest.raises(ValueError, match="not ready"):
        fixture.prepare_case(NEW_RUN_CASES[4].case_id, now=now)


def test_followup_fixture_refuses_changed_case_order_or_zone(tmp_path: Path):
    query = SqlFake()
    path = tmp_path / "followup.json"
    ReadingFixture(Client(), path, query, case_order=FOLLOWUP_CASE_ORDER, fixed_zone=FOLLOWUP_ZONE).initialize()
    with pytest.raises(ValueError, match="not ready"):
        ReadingFixture(Client(), path, query, case_order=(3, 4, 2), fixed_zone=FOLLOWUP_ZONE).prepare_case(
            NEW_RUN_CASES[3].case_id)
    with pytest.raises(ValueError, match="not ready"):
        ReadingFixture(Client(), path, query, case_order=FOLLOWUP_CASE_ORDER, fixed_zone="Asia/Tokyo").prepare_case(
            NEW_RUN_CASES[3].case_id)


@pytest.mark.parametrize("field,value", [("customCount", 1), ("systemSelectionCount", 1), ("baseScore", 75), ("profileState", "ACTIVE")])
def test_initialization_refuses_existing_user_data(tmp_path: Path, field: str, value: Any):
    query = SqlFake()
    query.snapshot[field] = value
    with pytest.raises(ValueError):
        ReadingFixture(Client(), tmp_path / "fixture.json", query).initialize()
    assert not any(sql.startswith("START TRANSACTION") for sql in query.sql)


def test_case_collision_refuses_fixture_reseed_without_writes(tmp_path: Path):
    query = SqlFake()
    fixture = ReadingFixture(Client(), tmp_path / "fixture.json", query)
    fixture.initialize()
    query.sql.clear()
    query.collision = True
    with pytest.raises(ValueError, match="must not be reseeded"):
        fixture.prepare_case(NEW_RUN_CASES[CASE_ORDER[0]].case_id)
    assert not any(sql.startswith("START TRANSACTION") for sql in query.sql)


def test_mysql_uppercase_sql_null_history_means_no_completed_scores(tmp_path: Path):
    query = SqlFake()
    query.sql_null_history = True
    fixture = ReadingFixture(Client(), tmp_path / "fixture.json", query)
    fixture.initialize()
    prepared = fixture.prepare_case(NEW_RUN_CASES[3].case_id)
    assert prepared["actualPriorCompletedScores"] == []
    assert prepared["baseScoreFixture"] == 30.0
    assert prepared["historyBandAdjustment"] == 0


def test_failed_cas_retains_partial_manifest_and_blocks_later_case(tmp_path: Path):
    query = SqlFake()
    fixture = ReadingFixture(Client(), tmp_path / "fixture.json", query)
    fixture.initialize()
    query.cas_ok = False
    with pytest.raises(ValueError, match="rolled back"):
        fixture.prepare_case(NEW_RUN_CASES[3].case_id)
    artifact = json.loads(fixture.path.read_text(encoding="utf-8"))
    assert artifact["status"] == "PREPARING_CASE" and artifact["cases"][0]["status"] == "PREPARING"
    with pytest.raises(ValueError, match="not ready"):
        fixture.prepare_case(NEW_RUN_CASES[1].case_id)


@pytest.mark.parametrize("user_id", [True, 1, 3])
def test_fixture_requires_exact_google_browser_user2(tmp_path: Path, user_id: Any):
    client = Client()
    client.user_id = user_id
    with pytest.raises(ValueError, match="Google QA user2"):
        ReadingFixture(client, tmp_path / "fixture.json", lambda _: "")
