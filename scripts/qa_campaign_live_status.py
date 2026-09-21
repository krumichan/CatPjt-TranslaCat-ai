"""Read-only status for this campaign's isolated browser-user QA sets.

Never prints credentials, learner answers, source text, or audio object keys.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import qa_campaign_integration as bootstrap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--set-id", type=int, required=True)
    args = parser.parse_args()
    if type(args.set_id) is not int or args.set_id <= 0:
        parser.error("--set-id must be positive")
    environment = args.environment.resolve()
    manifest = bootstrap._owned_manifest(environment)
    if manifest["campaign"] != "rv-20260920":
        raise ValueError("This reader is limited to the owned follow-up campaign")
    private = json.loads((environment / "secrets.private.json").read_text(encoding="utf-8"))
    set_id = args.set_id
    owner = bootstrap._qa_mysql_query(
        manifest, private,
        "SELECT COUNT(*) FROM language_learning_listening_daily_set s "
        "JOIN user u ON u.id=s.user_id WHERE s.id=" + str(set_id)
        + " AND u.id=2 AND u.social_type='GOOGLE';",
    )
    if owner.strip() != "1":
        raise ValueError("The exact Listening set does not belong to the browser QA user")
    set_row = bootstrap._qa_mysql_query(
        manifest, private,
        "SELECT id,learning_mode,status FROM language_learning_listening_daily_set "
        "WHERE id=" + str(set_id) + ";",
    ).split("\t")
    rows = bootstrap._qa_mysql_query(
        manifest, private,
        "SELECT i.item_index,i.status,COALESCE(CAST(i.audio_duration_ms AS CHAR),'NONE'),"
        "COALESCE(i.audio_content_type,'NONE'),COALESCE(i.audio_checksum,'NONE') "
        "FROM language_learning_listening_item i WHERE i.daily_set_id="
        + str(set_id) + " ORDER BY i.item_index,i.replacement_sequence;",
    )
    items = []
    for row in rows.splitlines():
        index, status, duration, mime, checksum = row.split("\t")
        items.append({"index": int(index), "status": status,
                      "audioDurationMs": None if duration == "NONE" else int(duration),
                      "audioContentType": None if mime == "NONE" else mime,
                      "audioChecksumPresent": checksum != "NONE"})
    response_rows = bootstrap._qa_mysql_query(
        manifest, private,
        "SELECT i.item_index,r.task_type,r.status,r.automatic_retry_count,"
        "COALESCE(r.evaluation_error_code,'NONE') "
        "FROM language_learning_listening_task_response r "
        "JOIN language_learning_listening_item_attempt a ON a.id=r.attempt_id "
        "JOIN language_learning_listening_item i ON i.id=a.item_id "
        "WHERE i.daily_set_id=" + str(set_id) + " ORDER BY i.item_index,r.task_type;",
    )
    responses = []
    for row in response_rows.splitlines():
        index, task, status, retries, error = row.split("\t")
        responses.append({"index": int(index), "task": task, "status": status,
                          "automaticRetryCount": int(retries),
                          "safeErrorCode": None if error == "NONE" else error})
    print(json.dumps({"setId": int(set_row[0]), "mode": set_row[1],
                      "status": set_row[2], "items": items,
                      "responses": responses}, ensure_ascii=False))


if __name__ == "__main__":
    main()
