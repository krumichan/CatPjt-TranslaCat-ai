"""One owned-QA synthetic user and the last fixed Listening matrix cell.

Registration and settings use the normal BE HTTP endpoints. The existing QA
registration-password fixture is repaired for this exact newly created row
only; no production authentication policy is changed or bypassed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import qa_campaign_integration as bootstrap
from scripts.qa_campaign_be_api import Client
from scripts.qa_campaign_be_workflows import listening


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if not args.live:
        parser.error("Explicit isolated-QA live opt-in required")
    directory = args.directory.resolve()
    manifest = bootstrap._owned_manifest(directory)
    if manifest["campaign"] != "openai-speech-campaign-20260920":
        raise ValueError("Wrong QA campaign")
    output = directory / "be-api-runs" / "lu4dc1"
    if output.exists():
        raise ValueError("Prior user4 evidence exists; never replay registration or generation")
    output.mkdir(parents=True)
    artifact = output / "one-shot-account-and-listening.json"
    report: dict = {"campaign": manifest["campaign"], "status": "STARTED",
                    "syntheticUser": True, "mode": "DICTATION", "difficulty": "CHALLENGE"}
    bootstrap._write_json(artifact, report)
    private = json.loads((directory / "secrets.private.json").read_text(encoding="utf-8"))
    email = f"qa-listening-challenge-{manifest['campaign']}@example.invalid"
    email_hex = email.encode("utf-8").hex()
    password = private["QA_APP_PASSWORD"]
    def query(sql: str) -> str:
        return bootstrap._qa_mysql_query(manifest, private, sql)
    if query(f"SELECT COUNT(*) FROM user WHERE email=CONVERT(0x{email_hex} USING utf8mb4);") != "0":
        raise ValueError("Synthetic QA email already exists")
    registered_status, registered = bootstrap._qa_request("/api/v1/auth/register", {
        "email": email, "password": password, "username": "Isolated Listening QA",
    })
    identity = (registered.get("body") or {}).get("id")
    report.update({"registerHttpStatus": registered_status, "userId": identity,
                   "status": "REGISTERED_PENDING_EXACT_FIXTURE"})
    bootstrap._write_json(artifact, report)
    if registered_status != 200 or type(identity) is not int or identity <= 0:
        raise ValueError("Normal QA registration failed")
    raw_hex = password.encode("utf-8").hex()
    predicate = (f"id={identity} AND email=CONVERT(0x{email_hex} USING utf8mb4) "
                 f"AND password=CONVERT(0x{raw_hex} USING utf8mb4) AND social_type='LOCAL'")
    if query(f"SELECT COUNT(*) FROM user WHERE {predicate};") != "1":
        raise ValueError("Exact newly registered raw-password fixture not found")
    command = manifest["beProcess"]["command"]
    java = Path(command[0])
    jar = Path(command[command.index("-jar") + 1])
    encoded = bootstrap._bcrypt_fixture_hash(output, jar, java, password)
    encoded_hex = encoded.encode("utf-8").hex()
    if query("UPDATE user SET password=CONVERT(0x" + encoded_hex
             + f" USING utf8mb4) WHERE {predicate}; SELECT ROW_COUNT();") != "1":
        raise ValueError("Exact synthetic QA auth fixture update failed")
    report["status"] = "AUTH_FIXTURE_UPDATED"
    bootstrap._write_json(artifact, report)
    login_status, login = bootstrap._qa_request("/api/v1/auth/login", {
        "email": email, "password": password,
    })
    token = (login.get("body") or {}).get("accessToken")
    report.update({"loginHttpStatus": login_status, "normalLoginSucceeded": bool(token)})
    bootstrap._write_json(artifact, report)
    if login_status != 200 or not isinstance(token, str) or not token:
        raise ValueError("Normal synthetic QA login failed")

    client = Client(directory, output_directory=output)
    client.user_id = identity
    client.private["QA_APP_EMAIL"] = email
    client.private["QA_APP_ACCESS_TOKEN"] = token
    profile = client.seed_profile()
    report["profileFixture"] = profile["status"]
    bootstrap._write_json(artifact, report)

    settings = {
        "originLanguage": "ko", "learningLanguage": "ja", "timezone": "Asia/Tokyo",
        "dailySentenceCount": 5, "dailySpeakingGoalMinutes": 5,
        "speakingVoiceId": "marin", "speakingPlaybackSpeed": "NORMAL",
        "dailyListeningGoalCount": 5, "defaultListeningTaskTypes": ["SUMMARY"],
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{manifest['ports']['be']}/api/v1/language-learning/settings",
        data=json.dumps(settings).encode("utf-8"), method="PATCH",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), bootstrap._NoRedirect())
    try:
        response = opener.open(request, timeout=20)
    except urllib.error.HTTPError as exc:
        response = exc
    with response:
        setting_status = response.status
        response.read()
    report.update({"settingsHttpStatus": setting_status, "status": "SETTINGS_SUBMITTED"})
    bootstrap._write_json(artifact, report)
    if setting_status != 200:
        raise ValueError("Normal synthetic QA settings update failed")
    if query("SELECT COUNT(*) FROM language_learning_listening_daily_set "
             f"WHERE user_id={identity} AND learning_mode='DICTATION';") != "0":
        raise ValueError("New synthetic user unexpectedly has a DICTATION set")
    result = listening(client, "DICTATION", difficulty="CHALLENGE", case_id="u4dc1")
    report.update({"status": result["status"], "dailySetId": result.get("dailySetId"),
                   "itemCount": len(result.get("items", []))})
    bootstrap._write_json(artifact, report)
    print(json.dumps({"qaUserId": identity, "mode": "DICTATION",
                      "difficulty": "CHALLENGE", "status": result["status"],
                      "setId": result.get("dailySetId"), "items": len(result.get("items", []))}))


if __name__ == "__main__":
    main()
