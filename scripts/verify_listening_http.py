"""로컬 합성 계정과 실제 Listening HTTP/DB 업무 경로 검증. 자격증명은 출력하지 않는다."""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import io
import json
import time
import uuid
import wave

import httpx


def token(user_id: int) -> str:
    def encode(value: object) -> str:
        return (
            base64.urlsafe_b64encode(json.dumps(value, separators=(",", ":")).encode())
            .decode()
            .rstrip("=")
        )

    now = int(time.time())
    encoded = (
        encode({"alg": "HS256", "typ": "JWT"})
        + "."
        + encode(
            {
                "iss": "translacat-be",
                "aud": "translacat-ll",
                "sub": str(user_id),
                "service": "translacat-be",
                "tokenUse": "ll-internal",
                "roles": ["USER"],
                "iat": now,
                "exp": now + 60,
            }
        )
    )
    signature = (
        base64.urlsafe_b64encode(
            hmac.new(bytes(range(32)), encoded.encode(), hashlib.sha256).digest()
        )
        .decode()
        .rstrip("=")
    )
    return encoded + "." + signature


def wave_bytes() -> bytes:
    stream = io.BytesIO()
    with wave.open(stream, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\xe8\x03" * 16000 * 10)
    return stream.getvalue()


def main(via_be: bool = False) -> None:
    started = time.monotonic()
    checks = 0
    with httpx.Client(base_url="http://127.0.0.1:18767/api/v1", timeout=15) as be:
        for mode in ("DICTATION", "COMPREHENSION", "SUMMARY"):
            # 준비: BE의 실제 가입·로그인·설정 경로로 고유한 합성 계정만 생성한다.
            identity = uuid.uuid4().hex
            account = {
                "email": f"listening-{identity}@example.test",
                "password": "Synthetic-" + uuid.uuid4().hex,
            }
            registration = be.post(
                "/auth/register", json={**account, "username": "Synthetic Listening"}
            )
            registration.raise_for_status()
            user_id = registration.json()["body"]["id"]
            login = be.post("/auth/login", json=account)
            login.raise_for_status()
            settings = be.patch(
                "/language-learning/settings",
                headers={"Authorization": "Bearer " + login.json()["body"]["accessToken"]},
                json={
                    "originLanguage": "ko",
                    "learningLanguage": "ja",
                    "timezone": "Asia/Seoul",
                    "dailyListeningGoalCount": 1,
                },
            )
            settings.raise_for_status()
            with httpx.Client(
                base_url=(
                    "http://127.0.0.1:18767/api/v1/language-learning/listening"
                    if via_be
                    else "http://127.0.0.1:18766/internal/v1/language-learning/listening"
                ),
                timeout=20,
                headers={
                    "Authorization": "Bearer "
                    + (login.json()["body"]["accessToken"] if via_be else token(user_id))
                },
            ) as ll:

                def api(
                    method: str,
                    path: str,
                    body: object = None,
                    status: int = 200,
                    current_user: int = user_id,
                    current_mode: str = mode,
                ):
                    nonlocal checks
                    if not via_be:
                        ll.headers["Authorization"] = "Bearer " + token(current_user)
                    if via_be and path.endswith("/audio-upload"):
                        response = ll.post(
                            path,
                            data={"durationMs": str(body["durationMs"])},
                            files={
                                "audio": (
                                    "synthetic.wav",
                                    base64.b64decode(body["audioBase64"]),
                                    body["contentType"],
                                ),
                            },
                        )
                    else:
                        response = ll.request(method, path, json=body)
                    if response.status_code != status:
                        error = response.json().get("body", {}) if via_be else response.json()
                        code = error.get("errorCode", error.get("code", "UNKNOWN"))
                        raise AssertionError(
                            f"{current_mode} {method} {path}: "
                            f"status={response.status_code} code={code}"
                        )
                    checks += 1
                    return response.json().get("body") if via_be else response.json()

                # 실행: 생성·TTS worker의 실제 HTTP 호출과 DB 게시를 조회한다.
                if via_be and mode == "DICTATION":
                    for path, expected in [
                        ("/daily-sets/-9007199254740000", "LANGUAGE_LEARNING_DAILY_SET_NOT_FOUND"),
                        ("/sessions/-9007199254740000", "SESSION_NOT_FOUND"),
                        (
                            "/items/-9007199254740000/audio",
                            "LANGUAGE_LEARNING_DAILY_ITEM_NOT_FOUND",
                        ),
                        ("/responses/-9007199254740000/audio", "LISTENING_AUDIO_INVALID"),
                    ]:
                        failure = api("GET", path, status=400)
                        actual = failure.get("code", failure.get("errorCode"))
                        assert actual == expected, f"{path}: expected={expected} actual={actual}"
                created = api(
                    "POST",
                    "/daily-sets",
                    {"learningMode": mode, "difficulty": "MY_LEVEL", "itemCount": 1},
                )
                set_id = created["dailySetId"]
                assert set_id < 0
                until = time.monotonic() + 45
                while created["status"] not in ("READY", "FAILED") and time.monotonic() < until:
                    time.sleep(0.15)
                    created = api("GET", f"/daily-sets/{set_id}")
                assert created["status"] == "READY", (
                    f"{mode} generation {created['status']} / {created['failureReason']}"
                )
                item_id = created["items"][0]["itemId"]
                audio = ll.get(f"/items/{item_id}/audio")
                assert (
                    audio.status_code == 200
                    and audio.headers["content-type"].startswith("audio/")
                    and len(audio.content) > 44
                )
                request = {"dailySetId": set_id, "idempotencyKey": "synthetic-listening-session"}
                session = api("POST", "/sessions", request)
                assert api("POST", "/sessions", request)["sessionId"] == session["sessionId"]
                session_id = session["sessionId"]
                attempt_id = session["attempts"][0]["attemptId"]
                hidden = api("GET", f"/sessions/{session_id}/items/{item_id}")
                assert (
                    hidden["sourceText"] is None
                    and hidden["correctOptionKey"] is None
                    and hidden["referenceMeanings"] == []
                )
                tasks = session["selectedTaskTypes"]
                for task in tasks:
                    answer = (
                        "B"
                        if task == "COMPREHENSION"
                        else "京都で静かな寺を見学したいです。"
                        if task == "DICTATION"
                        else "합성 의미 답변"
                    )
                    api(
                        "POST",
                        f"/attempts/{attempt_id}/responses/{task}",
                        {"answer": answer, "assistanceUsage": []},
                    )
                api("POST", f"/attempts/{attempt_id}/submit", {"actualDurationMs": 1500})
                api("POST", f"/attempts/{attempt_id}/submit", {"actualDurationMs": 9000})
                api(
                    "POST",
                    f"/attempts/{attempt_id}/responses/{tasks[0]}",
                    {"answer": "late overwrite"},
                    400,
                )
                until = time.monotonic() + 40
                while session["status"] != "COMPLETED" and time.monotonic() < until:
                    time.sleep(0.15)
                    session = api("GET", f"/sessions/{session_id}")

                # 검증: 공식 완료·누적 시간·원본 답변과 모드별 업무 평가를 확인한다.
                assert (
                    session["status"] == "COMPLETED"
                    and session["completedItemCount"] == 1
                    and session["actualDurationMs"] == 1500
                )
                assert all(
                    value["status"] == "EVALUATED"
                    for value in session["attempts"][0]["tasks"]
                    if value["taskType"] in tasks
                )
                revealed = api("GET", f"/sessions/{session_id}/items/{item_id}")
                assert revealed["sourceText"] is not None
                if via_be:
                    # 검증: 공통 이력·대시보드도 LL 원본 평가와 진행 수를 사용한다.
                    headers = {"Authorization": "Bearer " + login.json()["body"]["accessToken"]}
                    history_response = be.get(
                        "/language-learning/history?source=LISTENING", headers=headers
                    )
                    history_response.raise_for_status()
                    history = history_response.json()["body"]
                    entry = next(
                        value
                        for value in history
                        if value["activityId"] == f"LISTENING:{session_id}"
                    )
                    assert entry["completionStatus"] == "COMPLETED"
                    detail_response = be.get(
                        f"/language-learning/history/LISTENING:{session_id}", headers=headers
                    )
                    detail_response.raise_for_status()
                    detail = detail_response.json()["body"]["detail"]
                    assert detail["session"]["sessionId"] == session_id
                    assert detail["attempts"][0]["sourceText"] == revealed["sourceText"]
                    dashboard_response = be.get(
                        "/language-learning/dashboard?source=LISTENING", headers=headers
                    )
                    dashboard_response.raise_for_status()
                    dashboard = dashboard_response.json()["body"]
                    assert (
                        dashboard["activityPerformance"]["listening"]["today"]["completed"]
                        == 1
                    )
                    assert dashboard["trends"]["listeningTasks"]
                    checks += 3
                print(
                    f"PASS mode={mode} viaBE={via_be}: actual LL/Python/synthetic-provider/DB, "
                    "hidden answer, duplicate, late answer protection"
                )

                if mode == "DICTATION":
                    # 실행: Repeat 연습의 업로드·재녹음·STT·신고 보존을 검사한다.
                    practice = api(
                        "POST",
                        f"/sessions/{session_id}/items/{item_id}/practice-attempts",
                        {
                            "selectedTaskTypes": ["REPEAT_AFTER_AUDIO"],
                            "idempotencyKey": "synthetic-repeat",
                        },
                    )
                    repeat_id = practice["attemptId"]
                    upload = {
                        "audioBase64": base64.b64encode(wave_bytes()).decode(),
                        "contentType": "audio/wav",
                        "durationMs": 10000,
                    }
                    first = api("POST", f"/attempts/{repeat_id}/audio-upload", upload)
                    second = api("POST", f"/attempts/{repeat_id}/audio-upload", upload)
                    assert first["rerecordCount"] == 0 and second["rerecordCount"] == 1
                    api("POST", f"/attempts/{repeat_id}/submit", {"actualDurationMs": 10000})
                    until = time.monotonic() + 40
                    while time.monotonic() < until:
                        result = api("GET", f"/sessions/{session_id}")
                        repeated = next(
                            value for value in result["attempts"] if value["attemptId"] == repeat_id
                        )
                        if repeated["status"] in ("EVALUATED", "NOT_EVALUABLE"):
                            break
                        time.sleep(0.15)
                    assert result["completedItemCount"] == 1 and result["actualDurationMs"] == 1500
                    response_id = second["taskResponseId"]
                    report_request = {
                        "reasonCode": "SYNTHETIC",
                        "consentToRetainAudio": False,
                        "idempotencyKey": "synthetic-report",
                    }
                    report = api("POST", f"/responses/{response_id}/reports", report_request)
                    assert report["audioRetentionUntil"] is None
                    assert (
                        api("POST", f"/responses/{response_id}/reports", report_request)["reportId"]
                        == report["reportId"]
                    )
                    assert ll.get(f"/responses/{response_id}/audio").content == wave_bytes()
                    print(
                        "PASS Repeat practice: actual audio/STT, rerecord, report idempotency, "
                        "official progress unchanged"
                    )
    print(
        f"Listening HTTP checks={checks} elapsedSeconds={time.monotonic() - started:.2f} "
        "paidCalls=0"
    )


if __name__ == "__main__":
    arguments = argparse.ArgumentParser()
    arguments.add_argument("--via-be", action="store_true")
    main(arguments.parse_args().via_be)
