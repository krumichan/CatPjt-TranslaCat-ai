"""Run one missing Listening matrix case for an exact owned QA user.

The existing BE workflow still performs normal authenticated HTTP, WAV frame
validation, five answer/evaluation submissions, and DB/API reconciliation.
This wrapper only selects an isolated QA identity with no set in that mode
today. It never changes the production daily-set reuse policy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import qa_campaign_integration as bootstrap
from scripts.qa_campaign_be_api import Client
from scripts.qa_campaign_be_workflows import listening, listening_case

USER_CASES = {
    "local-user2": {
        ("DICTATION", "EASY"), ("COMPREHENSION", "EASY"), ("SUMMARY", "MY_LEVEL"),
    },
    "google-user3": {
        ("DICTATION", "EASY"), ("DICTATION", "CHALLENGE"),
        ("SUMMARY", "CHALLENGE"),
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--qa-user", choices=sorted(USER_CASES), required=True)
    parser.add_argument("--browser-session", type=Path)
    parser.add_argument("--mode", choices=["DICTATION", "COMPREHENSION", "SUMMARY"], required=True)
    parser.add_argument("--difficulty", choices=["EASY", "MY_LEVEL", "CHALLENGE"], required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if not args.live or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", args.run_label):
        parser.error("Explicit live opt-in and safe run label required")
    if (args.mode, args.difficulty) not in USER_CASES[args.qa_user]:
        parser.error("Only pre-fixed missing mode/difficulty cases are allowed")
    directory = args.directory.resolve()
    manifest = bootstrap._owned_manifest(directory)
    if manifest["campaign"] != "openai-speech-campaign-20260920":
        raise ValueError("Wrong owned QA campaign")
    output = directory / "be-api-runs" / args.run_label
    case = listening_case(args.mode, args.difficulty, args.case_id)
    # The atomic checkpoint writer adds a random suffix to this temporary name.
    # Refuse an unsafe Windows path before making any authenticated HTTP call.
    if len(str(output / f".listening-{case}-workflow.json-12345678")) >= 260:
        parser.error("Shorten --run-label and --case-id for the QA checkpoint path")
    if output.exists():
        raise ValueError("Run evidence already exists; do not replay a started case")
    output.mkdir(parents=True)
    private = json.loads((directory / "secrets.private.json").read_text(encoding="utf-8"))

    if args.qa_user == "local-user2":
        if args.browser_session is not None:
            parser.error("Local QA account does not use a Google browser session")
        email = bootstrap._qa_mysql_query(manifest, private,
            "SELECT email FROM user WHERE id=2 AND social_type='LOCAL' "
            "AND email LIKE 'qa-%@example.invalid';")
        if not email or "\n" in email:
            raise ValueError("Exact synthetic user2 ownership not proven")
        status, response = bootstrap._qa_request("/api/v1/auth/login", {
            "email": email, "password": private["QA_APP_PASSWORD"],
        })
        token = (response.get("body") or {}).get("accessToken")
        if status != 200 or not isinstance(token, str) or not token:
            raise ValueError("Normal synthetic QA login failed")
        client = Client(directory, output_directory=output)
        # The Client's default identity is the original synthetic account.
        # Replace it only after exact DB ownership and normal login checks.
        client.user_id = 2
        client.private["QA_APP_ACCESS_TOKEN"] = token
    else:
        if args.browser_session is None:
            parser.error("Exact normal Google browser session descriptor required")
        client = Client(directory, browser_session=args.browser_session.resolve(), output_directory=output)
        if client.user_id != 3 or client.browser_public_id != "TC-GA5T-4LRB":
            raise ValueError("Expected isolated QA Google user3")

    existing = bootstrap._qa_mysql_query(manifest, private,
        "SELECT COUNT(*) FROM language_learning_listening_daily_set "
        f"WHERE user_id={client.user_id} AND learning_mode='{args.mode}' AND learning_date=CURRENT_DATE();")
    if existing != "0":
        raise ValueError("This owned QA user already has today's mode; never reuse it as a new case")
    report = listening(client, args.mode, difficulty=args.difficulty, case_id=args.case_id)
    print(json.dumps({"qaUserId": client.user_id, "mode": args.mode, "difficulty": args.difficulty,
                      "status": report["status"], "setId": report.get("dailySetId"),
                      "items": len(report.get("items", []))}, ensure_ascii=True))


if __name__ == "__main__":
    main()
