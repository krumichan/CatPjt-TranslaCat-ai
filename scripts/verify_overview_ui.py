"""현재 scratch DB의 합성 활동으로 기존 실제 Overview UI 검사 네 개를 실행한다."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from scripts.verify_overview_http import CONTAINER, CORE, LEARNING, external_token

FEATURES = {
    "WRITING": ("language_learning_daily_set", "1=1"),
    "READING": ("language_learning_practice_set", "s.domain='READING'"),
    "VOCABULARY": ("language_learning_practice_set", "s.domain='VOCABULARY'"),
    "LISTENING": ("language_learning_listening_session", "1=1"),
    "SPEAKING": ("language_learning_speaking_session", "1=1"),
}


def scratch_environment() -> dict[str, str]:
    # 소유 컨테이너·포트·catalog를 고정하고 실제 비밀번호는 메모리에만 보관한다.
    if (CORE, LEARNING) != (
        "translacat_be_it_live_7cb72f72",
        "translacat_ll_it_live_7cb72f72",
    ):
        raise RuntimeError("OVERVIEW_UI_SCRATCH_CATALOG_MISMATCH")
    result = subprocess.run(
        ["docker", "inspect", CONTAINER],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    if result.returncode:
        raise RuntimeError("OVERVIEW_UI_DOCKER_INSPECT_FAILED")
    inspected = json.loads(result.stdout)[0]
    bindings = inspected["NetworkSettings"]["Ports"].get("3306/tcp") or []
    if (
        not inspected["Id"].startswith(CONTAINER)
        or not inspected["State"]["Running"]
        or not any(row["HostPort"] == "33316" for row in bindings)
    ):
        raise RuntimeError("OVERVIEW_UI_SCRATCH_CONTAINER_MISMATCH")
    password = next(
        (
            row.split("=", 1)[1]
            for row in inspected["Config"]["Env"]
            if row.startswith("MYSQL_ROOT_PASSWORD=")
        ),
        None,
    )
    if not password:
        raise RuntimeError("OVERVIEW_UI_SCRATCH_CREDENTIAL_MISSING")
    return {**os.environ, "MYSQL_PWD": password}


def select_users(environment: dict[str, str]) -> list[dict[str, str]]:
    # 초기화 전 baseline 파일 대신 현재 native 활동·실제 기능 행을 함께 확인한다.
    # 외부 화면의 설정/Level 선행조건을 갖춘 합성 계정 네 개만 읽고 DB는 수정하지 않는다.
    users: list[dict[str, str]] = []
    selected_ids: list[int] = []
    groups = (("WRITING",), ("READING", "VOCABULARY"), ("LISTENING",), ("SPEAKING",))
    for choices in groups:
        for source in choices:
            table, condition = FEATURES[source]
            excluded = ",".join(str(value) for value in selected_ids) or "0"
            activity = (
                f"EXISTS (SELECT 1 FROM {LEARNING}.language_learning_activity a "
                f"WHERE a.user_id=u.id AND a.source='{source}' "
                "AND a.learning_date BETWEEN UTC_DATE()-INTERVAL 28 DAY "
                "AND UTC_DATE()+INTERVAL 1 DAY)"
            )
            if source == "WRITING":
                # Writing 이력은 공통 Activity 대신 세트·당일 답변·DAILY 성공 평가를 직접 읽는다.
                # 같은 사용자 소유의 완료 세트와 연결된 성공 평가까지 요구한다.
                activity = (
                    f"EXISTS (SELECT 1 FROM {LEARNING}.language_learning_daily_set ws "
                    f"JOIN {LEARNING}.language_learning_daily_item wi ON wi.daily_set_id=ws.id "
                    f"JOIN {LEARNING}.language_learning_writing_answer wa "
                    "ON wa.daily_item_id=wi.id "
                    f"JOIN {LEARNING}.language_learning_writing_evaluation we "
                    "ON we.answer_id=wa.id "
                    "WHERE ws.user_id=u.id AND wi.user_id=u.id AND wa.user_id=u.id "
                    "AND we.user_id=u.id AND ws.status='COMPLETED' AND we.status='SUCCESS' "
                    "AND we.evaluation_context='DAILY' AND wa.attempt_date=ws.learning_date "
                    "AND ws.learning_date BETWEEN UTC_DATE()-INTERVAL 28 DAY "
                    "AND UTC_DATE()+INTERVAL 1 DAY)"
                )
            sql = (
                f"SELECT u.id,u.email,u.public_id FROM {CORE}.user u "
                f"JOIN {LEARNING}.language_learning_user_setting cfg ON cfg.user_id=u.id "
                f"JOIN {LEARNING}.language_learning_level_test_baseline b ON b.user_id=u.id "
                "WHERE u.email LIKE '%@example.test' AND u.public_id IS NOT NULL "
                "AND cfg.origin_language IS NOT NULL AND cfg.learning_language IS NOT NULL "
                f"AND u.id NOT IN ({excluded}) "
                f"AND EXISTS (SELECT 1 FROM {LEARNING}.{table} s "
                f"WHERE s.user_id=u.id AND {condition}) "
                f"AND {activity} "
                "ORDER BY u.id DESC LIMIT 1"
            )
            result = subprocess.run(
                [
                    "docker", "exec", "-i", "-e", "MYSQL_PWD", CONTAINER,
                    "mysql", "-uroot", "--batch", "--skip-column-names", CORE,
                ],
                input=sql,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
                timeout=15,
            )
            if result.returncode:
                raise RuntimeError(f"OVERVIEW_UI_SCRATCH_QUERY_FAILED_{source}")
            if not result.stdout.strip():
                continue
            raw_id, email, public_id = result.stdout.strip().split("\t")
            user_id = int(raw_id)
            if user_id <= 0 or not email.endswith("@example.test") or not public_id:
                raise RuntimeError("OVERVIEW_UI_SYNTHETIC_ACCOUNT_INVALID")
            selected_ids.append(user_id)
            users.append(
                {
                    "source": source,
                    "publicId": public_id,
                    "accessToken": external_token(user_id, email),
                }
            )
            break
        else:
            raise RuntimeError(f"OVERVIEW_UI_CURRENT_ACTIVITY_MISSING_{choices[0]}")
    return users


def main() -> int:
    started = time.monotonic()
    users = select_users(scratch_environment())

    # 기존 FE 인증 환경만 전달하고 업무 API·모델 출력·Provider 제어를 가로채지 않는다.
    environment = {
        **os.environ,
        "E2E_OVERVIEW_USERS": json.dumps(users),
        "NEXTAUTH_SECRET": "synthetic-cutover-nextauth-secret-20260926",
        "NEXTAUTH_URL": "http://localhost:3000",
        "E2E_BASE_URL": "http://localhost:3000",
        "NEXT_PUBLIC_API_URL": "http://127.0.0.1:18767/api/v1",
        "E2E_API_BASE_URL": "http://127.0.0.1:18767/api/v1",
        "PLAYWRIGHT_SKIP_WEB_SERVER": "0",
    }
    workspace = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            r"C:\nvm4w\nodejs\node.exe",
            r"node_modules\@playwright\test\cli.js",
            "test", "e2e/integration/overview-cutover.spec.ts",
            "--project=integration-chromium", "--reporter=list", "--workers=1",
            "--retries=0", "--trace=off",
        ],
        cwd=workspace / "CatPjt-TranslaCat-fe",
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=300,
    )

    # 토큰·계정·브라우저 원문 로그 대신 검사 범위, 종료 코드와 시간만 출력한다.
    print(
        f"Overview UI accounts={len(users)} sources={','.join(row['source'] for row in users)} "
        f"exit={result.returncode} elapsedSeconds={time.monotonic() - started:.2f}"
    )
    return result.returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.TimeoutExpired, ValueError, KeyError, OSError) as error:
        # 하위 프로세스 인자·응답 본문을 traceback으로 노출하지 않는다.
        code = str(error) if isinstance(error, RuntimeError) else type(error).__name__
        print(f"Overview UI preparation/execution failed: {code}")
        raise SystemExit(1) from None
