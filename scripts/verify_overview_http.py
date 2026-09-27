"""격리된 로컬 MySQL 합성 계정으로 기존 BE/신규 LL 공통 조회 결과를 대조한다."""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import subprocess
import time
from pathlib import Path

import httpx

from scripts.verify_listening_http import token

CONTAINER = "9d6e91ffcea2"
CORE = "translacat_be_it_live_7cb72f72"
LEARNING = "translacat_ll_it_live_7cb72f72"
BASELINE = (
    Path(__file__).resolve().parents[2]
    / "CatPjt-TranslaCat-ll/build/overview-runtime-baseline.json"
)


def accounts(repair_prerequisite: bool = False) -> list[tuple[int, str]]:
    # 명시된 테스트 컨테이너와 scratch catalog만 사용한다. 비밀번호는 메모리에만 둔다.
    inspected = json.loads(subprocess.check_output(["docker", "inspect", CONTAINER], text=True))[0]
    binding = inspected["NetworkSettings"]["Ports"]["3306/tcp"]
    assert any(row["HostPort"] == "33316" for row in binding)
    password = next(
        row.split("=", 1)[1]
        for row in inspected["Config"]["Env"]
        if row.startswith("MYSQL_ROOT_PASSWORD=")
    )
    selected: dict[int, str] = {}
    for feature in ("writing", "practice", "listening", "speaking", "level_test"):
        table = {
            "writing": "language_learning_daily_set",
            "practice": "language_learning_practice_set",
            "listening": "language_learning_listening_set",
            "speaking": "language_learning_speaking_session",
            "level_test": "language_learning_level_test_session",
        }[feature]
        sql = (
            f"SELECT DISTINCT u.id,u.email FROM {CORE}.user u "
            f"JOIN {LEARNING}.{table} s ON s.user_id=u.id "
            "WHERE u.email LIKE '%@example.test' ORDER BY u.id DESC LIMIT 2"
        )
        result = subprocess.run(
            [
                "docker",
                "exec",
                "-i",
                "-e",
                "MYSQL_PWD=" + password,
                CONTAINER,
                "mysql",
                "-uroot",
                "--batch",
                "--skip-column-names",
                CORE,
            ],
            input=sql,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode:
            raise AssertionError(f"scratch query failed for {feature}: exit={result.returncode}")
        for line in result.stdout.splitlines():
            user, email = line.split("\t", 1)
            selected[int(user)] = email
    assert selected
    if repair_prerequisite:
        # 이전 harness의 B3 오기를 명시된 합성 prerequisite 행에만 한정해 수정한다.
        # 실제 학습 결과·기존 일반 데이터·migration은 대상이 아니다.
        identifiers = ",".join(str(user) for user in selected)
        guard = (
            f"s.user_id IN ({identifiers}) AND s.idempotency_key LIKE 'synthetic-%' "
            "AND s.status='COMPLETED' AND s.base_level_score=60 AND s.proficiency_band='B3'"
        )
        sql = (
            f"UPDATE {LEARNING}.language_learning_level_test_baseline b "
            f"JOIN {LEARNING}.language_learning_level_test_session s ON s.id=b.session_id "
            f"SET b.proficiency_band='INTERMEDIATE' WHERE {guard} "
            "AND b.base_level_score=60 AND b.proficiency_band='B3'; "
            f"UPDATE {LEARNING}.language_learning_level_test_session s "
            f"SET s.proficiency_band='INTERMEDIATE' WHERE {guard};"
        )
        result = subprocess.run(
            [
                "docker",
                "exec",
                "-i",
                "-e",
                "MYSQL_PWD=" + password,
                CONTAINER,
                "mysql",
                "-uroot",
                "--batch",
                "--skip-column-names",
                CORE,
            ],
            input=sql,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, "synthetic prerequisite fixture correction failed"
    return list(selected.items())


def external_token(user: int, email: str) -> str:
    def encode(value: object) -> str:
        return (
            base64.urlsafe_b64encode(json.dumps(value, separators=(",", ":")).encode())
            .decode()
            .rstrip("=")
        )

    now = int(time.time())
    unsigned = (
        encode({"alg": "HS512", "typ": "JWT"})
        + "."
        + encode({"sub": email, "id": user, "iat": now, "exp": now + 120})
    )
    signature = (
        base64.urlsafe_b64encode(
            hmac.new(bytes(range(64)), unsigned.encode(), hashlib.sha512).digest()
        )
        .decode()
        .rstrip("=")
    )
    return unsigned + "." + signature


def difference(left: object, right: object, path: str = "$") -> str | None:
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return path + ".keys"
        return next(
            (
                found
                for key in left
                if (found := difference(left[key], right[key], path + "." + key))
            ),
            None,
        )
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return path + ".length"
        return next(
            (
                found
                for i, (a, b) in enumerate(zip(left, right, strict=True))
                if (found := difference(a, b, f"{path}[{i}]"))
            ),
            None,
        )
    return None if left == right else path


def verify_boundaries(client: httpx.Client, user: int, email: str) -> int:
    # 잘못된 조회 조건과 공개 ID 형식 오류가 BE를 지나도 LL 상태·코드를 유지하는지 확인한다.
    cases = [
        ("dashboard?source=INVALID", 400, "LANGUAGE_LEARNING_DASHBOARD_SOURCE_INVALID"),
        (
            "dashboard?source=WRITING&taskType=DICTATION",
            400,
            "LANGUAGE_LEARNING_DASHBOARD_SOURCE_INVALID",
        ),
        (
            "dashboard?from=2026-09-27&to=2026-09-26",
            400,
            "LISTENING_INVALID_STATE",
        ),
        ("history/invalid", 400, "LANGUAGE_LEARNING_DAILY_SET_NOT_FOUND"),
        ("history/WRITING:1", 400, "LEARNING_ID_INVALID"),
    ]
    for path, status, code in cases:
        external = client.get(
            "http://127.0.0.1:18767/api/v1/language-learning/" + path,
            headers={"Authorization": "Bearer " + external_token(user, email)},
        )
        internal = client.get(
            "http://127.0.0.1:18766/internal/v1/language-learning/overview/" + path,
            headers={"Authorization": "Bearer " + token(user)},
        )

        # 본문·계정·토큰을 출력하지 않고 계약을 벗어난 필드만 진단한다.
        assert external.status_code == internal.status_code == status, (
            f"boundary {path}: expected={status} BE={external.status_code} "
            f"LL={internal.status_code}"
        )
        # 기존 Listening 계열 오류는 공통 ErrorDto와 다른 code 필드 포장을 사용한다.
        external_body = external.json().get("body", {})
        code_field = "code" if code.startswith("LISTENING_") else "errorCode"
        assert external_body.get(code_field) == code, f"boundary {path}: BE errorCode mismatch"
        assert internal.json().get("code") == code, f"boundary {path}: LL code mismatch"
    return len(cases) * 2


def main(
    cutover: bool, repair_prerequisite: bool, boundaries: bool = False, current: bool = False
) -> None:
    if current and cutover:
        raise ValueError("Choose current routing or saved original-policy comparison")
    started = time.monotonic()
    count = 0
    fixtures = []
    saved = json.loads(BASELINE.read_text(encoding="utf-8")) if cutover else None
    selected = (
        [(row["user"], row["email"]) for row in saved["accounts"]]
        if saved
        else accounts(repair_prerequisite)
    )
    paths = [
        "dashboard?source=" + source
        for source in ("ALL", "WRITING", "SPEAKING", "LISTENING", "READING", "VOCABULARY")
    ]
    paths += [
        "dashboard?source=LISTENING&taskType=DICTATION",
        "history?period=30d",
        "history?source=LISTENING&period=30d&status=COMPLETED",
    ]
    with httpx.Client(timeout=25) as client:
        for user, email in selected:
            assert email.endswith("@example.test")
            for path in paths:
                external = client.get(
                    "http://127.0.0.1:18767/api/v1/language-learning/" + path,
                    headers={"Authorization": "Bearer " + external_token(user, email)},
                )
                internal = client.get(
                    "http://127.0.0.1:18766/internal/v1/language-learning/overview/" + path,
                    headers={"Authorization": "Bearer " + token(user)},
                )
                count += 2
                assert external.status_code == internal.status_code == 200, (
                    f"{path} status BE={external.status_code} LL={internal.status_code} "
                    f"code={external.json().get('body', {}).get('errorCode')}"
                )
                a, b = external.json()["body"], internal.json()
                changed = difference(a, b)
                assert changed is None, f"{path}: original BE/new LL mismatch at {changed}"
                fixtures.append({"user": user, "path": path, "body": a})

        # 삭제 후에는 전환 전 공통 결과도 다시 대조해 양쪽이 같이 달라지는 회귀를 잡는다.
        if saved:
            changed = difference(saved["fixtures"], fixtures)
            assert changed is None, f"saved original baseline mismatch at {changed}"
        elif not current:
            BASELINE.write_text(
                json.dumps(
                    {
                        "accounts": [{"user": u, "email": e} for u, e in selected],
                        "fixtures": fixtures,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        if boundaries:
            count += verify_boundaries(client, *selected[0])
    print(
        f"Overview parity accounts={len(selected)} checks={count} "
        f"elapsedSeconds={time.monotonic() - started:.2f} paidCalls=0 "
        f"cutover={cutover} currentRouting={current}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutover", action="store_true")
    parser.add_argument("--repair-test-prerequisite", action="store_true")
    parser.add_argument("--boundaries", action="store_true")
    # 초기화 뒤 현행 전달 경로만 비교할 때 이전 정책 기준 파일은 덮어쓰지 않는다.
    parser.add_argument("--current", action="store_true")
    options = parser.parse_args()
    main(options.cutover, options.repair_test_prerequisite, options.boundaries, options.current)
