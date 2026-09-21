"""Prepare the three fixed Reading follow-up cases for the normal Google QA user.

Only this run's isolated database may be changed. This seeds synthetic profile
prerequisites, never a generated question, answer, or historical learning score.
The browser's normal Google session remains responsible for public API calls.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import qa_campaign_integration as bootstrap
from scripts.qa_campaign_reading_fixture import (
    FOLLOWUP_CASE_ORDER, FOLLOWUP_ZONE, ReadingFixture,
)


class _BrowserUser:
    def __init__(self, manifest: dict[str, Any], private: dict[str, str]):
        self.user_id = 2
        self.manifest = manifest
        self.private = private
        identity = bootstrap._qa_mysql_query(
            manifest, private,
            "SELECT public_id FROM user WHERE id=2 AND social_type='GOOGLE';",
        )
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,20}", identity):
            raise ValueError("Exact normal Google QA user2 is absent")
        self.browser_public_id = identity

    def call(self, action: str, *, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        raise ValueError("Browser user settings must be saved through normal UI before fixture seeding")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("initialize", "prepare-case", "read-set-status"))
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-id", type=str)
    parser.add_argument("--set-id", type=int)
    args = parser.parse_args()
    environment = args.environment.resolve()
    output = args.output.resolve()
    if output != environment.parent:
        raise ValueError("Fixture output must be this owned campaign directory")
    manifest = bootstrap._owned_manifest(environment)
    if manifest["campaign"] != "rv-20260920":
        raise ValueError("This tool is fixed to the approved follow-up campaign")
    private = json.loads((environment / "secrets.private.json").read_text(encoding="utf-8"))
    client = _BrowserUser(manifest, private)
    fixture = ReadingFixture(
        client, output / "reading-followup-fixture.json",
        case_order=FOLLOWUP_CASE_ORDER, fixed_zone=FOLLOWUP_ZONE,
    )
    if args.action == "read-set-status":
        if type(args.set_id) is not int or args.set_id <= 0:
            parser.error("--set-id must identify the exact positive QA set")
        row = bootstrap._qa_mysql_query(manifest, private,
            "SELECT s.id,s.status,s.generation_status,s.generation_retry_count,"
            "(SELECT COUNT(*) FROM language_learning_practice_question q WHERE q.practice_set_id=s.id) "
            "FROM language_learning_practice_set s JOIN user u ON u.id=s.user_id "
            "WHERE s.id=" + str(args.set_id) + " AND u.id=2 AND u.social_type='GOOGLE' AND u.public_id=CONVERT(0x"
            + client.browser_public_id.encode().hex() + " USING utf8mb4);"
        )
        values = row.split("\t")
        if len(values) != 5:
            raise ValueError("Exact owned Google QA Reading set not found")
        print(json.dumps({"setId": int(values[0]), "status": values[1],
                          "generationStatus": values[2], "generationRetryCount": values[3],
                          "storedQuestionRows": int(values[4])}))
        return
    if args.set_id is not None:
        parser.error("--set-id is only valid for read-set-status")
    if args.action == "initialize":
        if args.case_id is not None:
            parser.error("--case-id is only valid for prepare-case")
        result = fixture.initialize()
    else:
        if args.case_id is None:
            parser.error("--case-id is required for prepare-case")
        result = fixture.prepare_case(args.case_id)
    print(json.dumps({key: result.get(key) for key in (
        "status", "caseId", "learningDate", "timezone", "expectedBand", "keyword",
        "syntheticLevelTestFixture", "realLevelTestExecuted",
    )}, ensure_ascii=True))


if __name__ == "__main__":
    main()
