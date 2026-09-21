"""Continue only the exact isolated QA user4 after its long-email login 500.

The original registration/auth failure record is immutable. Shortening this
new synthetic user's email is a QA fixture adjustment, not an auth bypass or
a production schema/policy change. Login and settings still use normal HTTP.
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
    original = json.loads((output / "one-shot-account-and-listening.json").read_text(encoding="utf-8"))
    if (original.get("campaign") != manifest["campaign"] or original.get("userId") != 4
            or original.get("registerHttpStatus") != 200 or original.get("loginHttpStatus") != 500
            or original.get("status") != "AUTH_FIXTURE_UPDATED"):
        raise ValueError("Exact first-attempt login failure evidence is required")
    artifact = output / "resume-after-email-length.json"
    if artifact.exists():
        raise ValueError("QA user4 continuation already started; never replay")
    private = json.loads((directory / "secrets.private.json").read_text(encoding="utf-8"))
    old_email = f"qa-listening-challenge-{manifest['campaign']}@example.invalid"
    new_email = "qa-l4@example.invalid"
    old_hex, new_hex = old_email.encode().hex(), new_email.encode().hex()

    def query(sql: str) -> str:
        return bootstrap._qa_mysql_query(manifest, private, sql)

    report: dict = {"status": "STARTED", "userId": 4, "syntheticUser": True,
                    "preservedOriginalFailure": str(output / "one-shot-account-and-listening.json"),
                    "emailChange": "exact new QA row only; no original user or schema change"}
    bootstrap._write_json(artifact, report)
    predicate = f"id=4 AND social_type='LOCAL' AND email=CONVERT(0x{old_hex} USING utf8mb4)"
    if query(f"SELECT COUNT(*) FROM user WHERE {predicate};") != "1":
        raise ValueError("Exact new QA row is not present")
    if query(f"SELECT COUNT(*) FROM user WHERE email=CONVERT(0x{new_hex} USING utf8mb4);") != "0":
        raise ValueError("Short QA identity already belongs to another user")
    if query("SELECT COUNT(*) FROM language_learning_listening_daily_set WHERE user_id=4;") != "0":
        raise ValueError("New QA user has unexpectedly created a Listening set")
    if query("UPDATE user SET email=CONVERT(0x" + new_hex
             + f" USING utf8mb4) WHERE {predicate}; SELECT ROW_COUNT();") != "1":
        raise ValueError("Exact QA email fixture update failed")
    report["status"] = "EXACT_QA_EMAIL_SHORTENED"
    bootstrap._write_json(artifact, report)

    login_status, login = bootstrap._qa_request("/api/v1/auth/login", {
        "email": new_email, "password": private["QA_APP_PASSWORD"],
    })
    token = (login.get("body") or {}).get("accessToken")
    report.update({"loginHttpStatus": login_status, "normalLoginSucceeded": bool(token)})
    bootstrap._write_json(artifact, report)
    if login_status != 200 or not isinstance(token, str) or not token:
        raise ValueError("Normal short-email QA login failed; no generation")
    client = Client(directory, output_directory=output)
    client.user_id = 4
    client.private["QA_APP_EMAIL"] = new_email
    client.private["QA_APP_ACCESS_TOKEN"] = token
    report["profileFixture"] = client.seed_profile()["status"]
    bootstrap._write_json(artifact, report)

    settings = {"originLanguage": "ko", "learningLanguage": "ja", "timezone": "Asia/Tokyo",
                "dailySentenceCount": 5, "dailySpeakingGoalMinutes": 5,
                "speakingVoiceId": "marin", "speakingPlaybackSpeed": "NORMAL",
                "dailyListeningGoalCount": 5, "defaultListeningTaskTypes": ["SUMMARY"]}
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
        raise ValueError("Normal QA settings PATCH failed; no generation")
    if query("SELECT COUNT(*) FROM language_learning_listening_daily_set "
             "WHERE user_id=4 AND learning_mode='DICTATION';") != "0":
        raise ValueError("New QA user unexpectedly has a DICTATION set")
    result = listening(client, "DICTATION", difficulty="CHALLENGE", case_id="u4dc1")
    report.update({"status": result["status"], "dailySetId": result.get("dailySetId"),
                   "itemCount": len(result.get("items", []))})
    bootstrap._write_json(artifact, report)
    print(json.dumps({"qaUserId": 4, "mode": "DICTATION", "difficulty": "CHALLENGE",
                      "status": result["status"], "setId": result.get("dailySetId"),
                      "items": len(result.get("items", []))}))


if __name__ == "__main__":
    main()
