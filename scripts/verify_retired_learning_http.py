"""초기화한 로컬 테스트 DB의 이전 ID와 폐기 HTTP 경계를 실제 서버로 검사한다."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

import httpx

from scripts.verify_listening_http import token
from scripts.verify_overview_http import external_token

ROOT = Path(__file__).resolve().parents[2]
LL = ROOT / "CatPjt-TranslaCat-ll"
PLAN = LL / ".tmp_ktor_final_cleanup/data-reset-plan.json"
TABLES = {
    "language_learning_daily_set": "writing",
    "language_learning_practice_set": "practice",
    "language_learning_listening_set": "listening",
    "language_learning_speaking_session": "speaking",
    "language_learning_level_test_session": "level",
}


class Scratch:
    def __init__(self, plan: dict):
        # 이전 초기화 증거와 같은 loopback 컨테이너·catalog만 읽는다.
        self.container = plan["container"]
        self.core = plan["coreCatalog"]
        self.learning = plan["llCatalog"]
        assert self.container == "9d6e91ffcea2" and plan["host"] == "127.0.0.1:33316"
        assert self.core == "translacat_be_it_live_7cb72f72"
        assert self.learning == "translacat_ll_it_live_7cb72f72"
        inspected = json.loads(
            subprocess.check_output(["docker", "inspect", self.container], text=True)
        )[0]
        assert any(
            row["HostPort"] == "33316" for row in inspected["NetworkSettings"]["Ports"]["3306/tcp"]
        )
        self.password = next(
            row.split("=", 1)[1]
            for row in inspected["Config"]["Env"]
            if row.startswith("MYSQL_ROOT_PASSWORD=")
        )

    def select(self, query: str) -> list[str]:
        assert query.lstrip().upper().startswith("SELECT ")
        result = subprocess.run(
            [
                "docker",
                "exec",
                "-i",
                "-e",
                "MYSQL_PWD=" + self.password,
                self.container,
                "mysql",
                "-uroot",
                "--batch",
                "--skip-column-names",
                self.core,
            ],
            input=query,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, "격리 DB 읽기 실패"
        return result.stdout.splitlines()


def body_code(response: httpx.Response, external: bool) -> str | None:
    value = response.json()
    if external:
        value = value.get("body", {})
    return value.get("errorCode", value.get("code"))


def main(browser: bool) -> None:
    started = time.monotonic()
    plan = json.loads(PLAN.read_text(encoding="utf-8-sig"))
    db = Scratch(plan)
    references = []
    for row in plan["oldReferences"]:
        table, identifier, owner = row.split("\t")
        assert table in TABLES
        references.append((table, int(identifier), int(owner)))
    assert len(references) == 10
    owners = sorted({owner for _, _, owner in references})
    account_rows = db.select(
        f"SELECT id,email FROM {db.core}.user WHERE id IN ({','.join(map(str, owners))}) "
        "AND email LIKE '%@example.test'"
    )
    emails = {int(row.split("\t")[0]): row.split("\t")[1] for row in account_rows}
    assert set(emails) == set(owners), "합성 소유자 계정만 검증할 수 있습니다."
    counts = {
        "http": 0,
        "staleReads": 0,
        "lateMutations": 0,
        "retiredRoutes": 0,
        "highWaterTables": 0,
        "invalidPositiveIds": 0,
    }

    with httpx.Client(timeout=20) as client:
        # 준비: 현재 BE 가입·로그인 경로로 신규 합성 계정을 만든다. 원래 계정은 읽기만 한다.
        identity = uuid.uuid4().hex
        credentials = {
            "email": f"rb-{identity[:16]}@example.test",
            "password": "Synthetic-" + uuid.uuid4().hex,
        }
        registered = client.post(
            "http://127.0.0.1:18767/api/v1/auth/register",
            json={**credentials, "username": "Synthetic Reset"},
        )
        assert registered.status_code == 200, "합성 계정 가입 실패"
        fresh = registered.json()["body"]
        login = client.post("http://127.0.0.1:18767/api/v1/auth/login", json=credentials)
        assert login.status_code == 200, (
            f"합성 계정 로그인 실패: HTTP {login.status_code}, code={body_code(login, True)}"
        )
        session = login.json()["body"]
        fresh_user = int(fresh["id"])
        counts["http"] += 2
        configured = client.patch(
            "http://127.0.0.1:18767/api/v1/language-learning/settings",
            headers={"Authorization": "Bearer " + session["accessToken"]},
            json={"originLanguage": "ko", "learningLanguage": "en", "timezone": "Asia/Seoul"},
        )
        assert configured.status_code == 200, "신규 합성 계정의 현재 설정 경로 실패"
        counts["http"] += 1

        def check(
            method: str,
            path: str,
            user: int,
            status: int,
            code: str | None = None,
            external: bool = True,
            payload: dict | None = None,
        ) -> None:
            prefix = (
                "http://127.0.0.1:18767/api/v1"
                if external
                else "http://127.0.0.1:18766/internal/v1"
            )
            bearer = (
                (
                    session["accessToken"]
                    if user == fresh_user
                    else external_token(user, emails[user])
                )
                if external
                else token(user)
            )
            response = client.request(
                method, prefix + path, headers={"Authorization": "Bearer " + bearer}, json=payload
            )
            counts["http"] += 1
            assert response.status_code == status, (
                f"{method} {path.split('?')[0]}: HTTP {response.status_code}, expected {status}"
            )
            if code:
                assert body_code(response, external) == code, f"{method}: 오류 코드 계약 불일치"

        def business_state() -> dict[str, list[str]]:
            identifiers = ",".join(map(str, owners + [fresh_user]))
            return {
                table: db.select(
                    f"SELECT COUNT(*),COALESCE(SUM(id),0) FROM {db.learning}.{table} "
                    f"WHERE user_id IN ({identifiers})"
                )
                for table in TABLES
            }

        before_business = business_state()

        # 실행: 이전 소유자와 신규 사용자로 과거 ID를 조회·재시도한다.
        # 과거 자료가 새 학습 행으로 연결되는지 확인한다.
        for table, identifier, owner in references:
            feature = TABLES[table]
            public_id = identifier if feature == "level" else -identifier
            assert db.select(
                f"SELECT COUNT(*) FROM {db.learning}.{table} WHERE id={identifier}"
            ) == ["0"]
            if feature != "level":
                positive_path = {
                    "writing": f"history/WRITING:{identifier}",
                    "practice": f"practice/sets/{identifier}",
                    "listening": f"listening/daily-sets/{identifier}",
                    "speaking": f"speaking/sessions/{identifier}",
                }[feature]
                check(
                    "GET", "/language-learning/" + positive_path, owner, 400, "LEARNING_ID_INVALID"
                )
                counts["invalidPositiveIds"] += 1
            for user in (owner, fresh_user):
                if feature == "writing":
                    check(
                        "GET",
                        f"/language-learning/history/WRITING:{public_id}",
                        user,
                        400,
                        "LANGUAGE_LEARNING_DAILY_SET_NOT_FOUND",
                    )
                    for action in ("retry-generation", "regenerate"):
                        check(
                            "POST",
                            f"/language-learning/writing/daily/{public_id}/{action}",
                            user,
                            400,
                            "LANGUAGE_LEARNING_DAILY_SET_NOT_FOUND",
                        )
                        counts["lateMutations"] += 1
                elif feature in ("practice", "listening"):
                    route = "practice/sets" if feature == "practice" else "listening/daily-sets"
                    check(
                        "GET",
                        f"/language-learning/{route}/{public_id}",
                        user,
                        400,
                        "LANGUAGE_LEARNING_DAILY_SET_NOT_FOUND",
                    )
                    check(
                        "POST",
                        f"/language-learning/{route}/{public_id}/retry-generation",
                        user,
                        400,
                        "LANGUAGE_LEARNING_DAILY_SET_NOT_FOUND",
                    )
                    counts["lateMutations"] += 1
                elif feature == "speaking":
                    check(
                        "GET",
                        f"/language-learning/speaking/sessions/{public_id}",
                        user,
                        400,
                        "SESSION_NOT_FOUND",
                    )
                else:
                    check(
                        "GET",
                        f"/language-learning/level-test/sessions/{public_id}",
                        user,
                        404,
                        "LANGUAGE_LEARNING_LEVEL_TEST_NOT_FOUND",
                    )
                counts["staleReads"] += 1
        assert business_state() == before_business, (
            "이전 요청이 대상 사용자의 현재 학습 행을 변경했습니다."
        )

        # 검증: 폐기 서비스는 인증된 요청에도 404이고 현재 조회와 초기 상태는 정상이다.
        for method, path in (
            ("POST", "/service/language-learning/growth/commands"),
            ("POST", "/service/language-learning/results"),
            ("POST", "/language-learning/settings/listening-selection"),
            ("GET", "/language-learning/level-test/baseline"),
            ("GET", "/language-learning/level-test/completions"),
            ("GET", "/service/language-learning/settings/users/123"),
            ("GET", "/service/language-learning/settings/admin"),
            ("GET", "/service/language-learning/settings/language-pairs"),
        ):
            check(
                method,
                path,
                fresh_user,
                404,
                external=False,
                payload={} if method == "POST" else None,
            )
            counts["retiredRoutes"] += 1
        check("GET", "/language-learning/level-test/status", fresh_user, 200)
        check(
            "POST",
            "/language-learning/growth/snapshot",
            fresh_user,
            200,
            external=False,
            payload={"masteryKeys": None},
        )
        check(
            "GET",
            "/language-learning/growth/activities?from=2026-09-27&to=2026-09-27",
            fresh_user,
            200,
            external=False,
        )
        for table, identifier, _ in references:
            assert db.select(
                f"SELECT COUNT(*) FROM {db.learning}.{table} WHERE id={identifier}"
            ) == ["0"]
        for old in plan["autoIncrement"]:
            if old["catalog"] != db.learning or old["nextId"] <= 0:
                continue
            table = old["table"]
            assert table.startswith("language_learning_") and table.replace("_", "").isalnum()
            value = db.select(
                "SELECT AUTO_INCREMENT FROM information_schema.TABLES "
                f"WHERE TABLE_SCHEMA='{db.learning}' AND TABLE_NAME='{table}'"
            )
            assert len(value) == 1 and int(value[0]) >= old["nextId"], (
                "현재 AUTO_INCREMENT 상한이 이전보다 작습니다."
            )
            counts["highWaterTables"] += 1

        # 선택한 브라우저 검사는 토큰을 환경으로만 전달하며 trace/video/스크린샷에 기록하지 않는다.
        if browser:
            stats_path = (
                Path(__file__).resolve().parents[1] / ".tmp_ktor_m0/synthetic_execution_stats.json"
            )
            before_stats = json.loads(stats_path.read_text(encoding="utf-8"))
            assert os.environ.get("NEXTAUTH_SECRET"), (
                "브라우저 검사는 현재 FE의 NEXTAUTH_SECRET 환경이 필요합니다."
            )
            old_level = next(
                identifier for table, identifier, _ in references if TABLES[table] == "level"
            )
            old_writing = next(
                -identifier for table, identifier, _ in references if TABLES[table] == "writing"
            )
            # root HTTP 검사가 이미 생성한 미답변 세트만 재사용한다.
            # 현재 조회로 새로운 출제를 요청하지 않는다.
            writing_floor = next(
                row["nextId"]
                for row in plan["autoIncrement"]
                if row["catalog"] == db.learning and row["table"] == "language_learning_daily_set"
            )
            ready = db.select(
                "SELECT s.id,s.user_id,s.writing_type,u.email,u.public_id "
                f"FROM {db.learning}.language_learning_daily_set s "
                f"JOIN {db.core}.user u ON u.id=s.user_id "
                f"WHERE s.id>={writing_floor} AND s.status IN ('READY','PARTIAL') "
                "AND u.email LIKE '%@example.test' "
                "AND s.learning_date=DATE(UTC_TIMESTAMP()+INTERVAL 9 HOUR) "
                f"AND EXISTS(SELECT 1 FROM {db.learning}.language_learning_daily_item i "
                "WHERE i.daily_set_id=s.id "
                f"AND NOT EXISTS(SELECT 1 FROM {db.learning}.language_learning_writing_answer a "
                "WHERE a.daily_item_id=i.id)) ORDER BY s.id DESC LIMIT 1"
            )
            assert ready, "실제 HTTP로 생성한 현재 미답변 READY/PARTIAL Writing 문항이 필요합니다."
            current_id, current_user, current_type, current_email, current_public = ready[0].split(
                "\t"
            )
            writing_token = external_token(int(current_user), current_email)
            existing = client.get(
                "http://127.0.0.1:18767/api/v1/language-learning/writing/daily",
                params={"writingType": current_type},
                headers={"Authorization": "Bearer " + writing_token},
            )
            counts["http"] += 1
            assert existing.status_code == 200, "이미 존재하는 현재 Writing 조회 실패"
            current = existing.json()["body"]
            assert current["dailySetId"] == -int(current_id) != old_writing
            unanswered = [
                item
                for item in current["items"]
                if item["canSubmit"] and not item["attempts"]
            ]
            assert unanswered, "현재 세트에 아직 답변하지 않은 제출 가능 문항이 없습니다."
            env = os.environ | {
                "E2E_RESET_ACCESS_TOKEN": session["accessToken"],
                "E2E_RESET_REFRESH_TOKEN": session["refreshToken"],
                "E2E_RESET_PUBLIC_ID": session["publicId"],
                "E2E_RESET_LEVEL_ID": str(old_level),
                "E2E_RESET_WRITING_ID": str(old_writing),
                "E2E_API_BASE_URL": "http://127.0.0.1:18767/api/v1",
                "E2E_BASE_URL": "http://localhost:3000",
                "PLAYWRIGHT_SKIP_WEB_SERVER": "1",
                "E2E_RESET_CURRENT_ACCESS_TOKEN": writing_token,
                "E2E_RESET_CURRENT_PUBLIC_ID": current_public,
                "E2E_RESET_CURRENT_SET_ID": str(current["dailySetId"]),
                "E2E_RESET_CURRENT_WRITING_TYPE": current_type,
                "E2E_RESET_CURRENT_ITEM_IDS": json.dumps(
                    [item["itemId"] for item in current["items"]]
                ),
                "E2E_RESET_UNANSWERED_ITEMS": json.dumps(
                    [{"itemId": item["itemId"], "order": item["order"]} for item in unanswered]
                ),
                "E2E_RESET_CURRENT_LEARNING_DATE": current["learningDate"],
            }
            result = subprocess.run(
                [
                    r"C:\nvm4w\nodejs\npm.cmd",
                    "exec",
                    "playwright",
                    "test",
                    "e2e/integration/learning-reset-boundary.spec.ts",
                    "--project=integration-chromium",
                    "--reporter=list",
                ],
                cwd=ROOT / "CatPjt-TranslaCat-fe",
                env=env,
                check=False,
            )
            assert result.returncode == 0, "이전 URL 브라우저 경계 검사 실패"
            after_stats = json.loads(stats_path.read_text(encoding="utf-8"))
            assert after_stats["processId"] == before_stats["processId"], (
                "Provider 프로세스가 바뀌었습니다."
            )
            assert after_stats["modelCalls"] == before_stats["modelCalls"], (
                "기존 세트 브라우저 조회에서 모델 호출이 추가됐습니다."
            )
            counts["browserModelCalls"] = after_stats["modelCalls"] - before_stats["modelCalls"]
        report = counts | {
            "elapsedSeconds": round(time.monotonic() - started, 2),
            "browser": "PASS" if browser else "NOT_RUN",
            "paidCalls": 0,
            "syntheticEmailLength": len(credentials["email"]),
            "refreshTokenLength": len(session["refreshToken"]),
        }
        output = LL / ".tmp_ktor_final_cleanup/retired-learning-http-result.json"
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--browser", action="store_true")
    main(parser.parse_args().browser)
