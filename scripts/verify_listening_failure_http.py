"""실제 BE/LL/범용 Python/DB 경로의 Listening 실패 보존 검사. 모델 출력만 고정한다."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

CONTROL = (
    Path(__file__).resolve().parents[2]
    / ".codex-workspace/verification/ai/runtime/listening-control.json"
)


class ListeningClient:
    def __init__(self) -> None:
        self.count = 0
        self.client = httpx.Client(base_url="http://127.0.0.1:18767/api/v1", timeout=20)
        identity = uuid.uuid4().hex
        account = {
            "email": f"lf-{identity}@example.test",
            "password": "Synthetic-" + uuid.uuid4().hex,
        }
        self.call("POST", "/auth/register", {**account, "username": "Synthetic Listening"})
        login = self.call("POST", "/auth/login", account)
        self.client.headers["Authorization"] = "Bearer " + login["accessToken"]
        self.call(
            "PATCH",
            "/language-learning/settings",
            {
                "originLanguage": "ko",
                "learningLanguage": "ja",
                "timezone": "Asia/Seoul",
                "dailyListeningGoalCount": 1,
            },
        )

    def call(self, method: str, path: str, body: object = None) -> Any:
        self.count += 1
        response = self.client.request(method, path, json=body)
        if response.status_code != 200:
            raise AssertionError(f"HTTP {method} {path}: {response.status_code}")
        return response.json()["body"]

    def learning(self, method: str, path: str, body: object = None) -> Any:
        return self.call(method, "/language-learning/listening" + path, body)

    def wait(self, path: str, condition: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
        end = time.monotonic() + 50
        while time.monotonic() < end:
            value = self.learning("GET", path)
            if condition(value):
                return value
            time.sleep(0.2)
        raise AssertionError(f"Listening state deadline: {path}")


def control(failure: str | None) -> None:
    CONTROL.write_text(json.dumps({"failure": failure}), encoding="utf-8")


def main() -> None:
    if CONTROL.exists():
        raise RuntimeError("Existing Listening fixture control must be resolved before this test")
    started = time.monotonic()
    clients: list[ListeningClient] = []
    try:
        # 준비 및 실행: 잘못된 생성 schema는 성공 데이터로 정정하지 않고 FAILED로 남긴다.
        control("SCHEMA_INVALID")
        failed = ListeningClient()
        clients.append(failed)
        created = failed.learning("POST", "/daily-sets", {"itemCount": 1})
        path = f"/daily-sets/{created['dailySetId']}"
        state = failed.wait(path, lambda value: value["status"] == "FAILED")
        assert state["items"] == [] and state["failureReason"]

        # 실행 및 검증: 기존 수동 재시도 한도 안에서 같은 세트를 복구한다.
        control(None)
        retried = failed.learning("POST", path + "/retry-generation")
        assert retried["dailySetId"] == created["dailySetId"]
        ready = failed.wait(path, lambda value: value["status"] == "READY")
        assert len(ready["items"]) == 1

        # 준비: 평가 실패를 주입해도 다른 Task의 원본 평가와 저장한 답변은 보존해야 한다.
        session = failed.learning(
            "POST",
            "/sessions",
            {"dailySetId": created["dailySetId"], "idempotencyKey": uuid.uuid4().hex},
        )
        attempt = session["attempts"][0]
        attempt_id = attempt["attemptId"]
        for task, answer in (
            ("DICTATION", "京都で静かな寺を見学したいです。"),
            ("INTERPRETATION", "합성 의미 답변"),
        ):
            failed.learning("POST", f"/attempts/{attempt_id}/responses/{task}", {"answer": answer})
        control("SCHEMA_INVALID")
        failed.learning("POST", f"/attempts/{attempt_id}/submit", {"actualDurationMs": 1200})
        session_path = f"/sessions/{session['sessionId']}"
        evaluated = failed.wait(
            session_path,
            lambda value: any(
                task["status"] == "EVALUATION_FAILED" for task in value["attempts"][0]["tasks"]
            ),
        )
        before = {task["taskType"]: task for task in evaluated["attempts"][0]["tasks"]}
        assert before["DICTATION"]["status"] == "EVALUATED"
        assert before["INTERPRETATION"]["evaluationErrorCode"]

        # 실행 및 검증: 수동 재평가는 실패한 Task만 교체하고 성공한 원본 결과를 유지한다.
        control(None)
        failed.learning(
            "POST", f"/attempts/{attempt_id}/retry-evaluation", {"taskType": "INTERPRETATION"}
        )
        completed = failed.wait(session_path, lambda value: value["status"] == "COMPLETED")
        after = {task["taskType"]: task for task in completed["attempts"][0]["tasks"]}
        assert after["DICTATION"] == before["DICTATION"]
        assert after["INTERPRETATION"]["answerText"] == before["INTERPRETATION"]["answerText"]
        assert after["INTERPRETATION"]["status"] == "EVALUATED"
        assert completed["completedItemCount"] == 1 and completed["actualDurationMs"] == 1200
        print(
            "PASS actual BE generation/evaluation failure, bounded manual recovery, "
            "saved answer/result protection"
        )

        # 준비 및 실행: 길이 검증에 실패한 오디오의 문항은 재생성 실패로 삭제하지 않는다.
        control("TTS_SHORT")
        partial = ListeningClient()
        clients.append(partial)
        created = partial.learning("POST", "/daily-sets", {"itemCount": 1})
        path = f"/daily-sets/{created['dailySetId']}"
        state = partial.wait(
            path,
            lambda value: bool(value["items"]) and value["items"][0]["status"] == "NOT_EVALUABLE",
        )
        old_items = state["items"]
        assert not old_items[0]["ttsRetryAllowed"]
        control("SCHEMA_INVALID")
        partial.learning("POST", path + "/retry-generation")
        failed_again = partial.wait(
            path, lambda value: not value["generationInProgress"] and bool(value["failureReason"])
        )

        # 검증: 물리 문항과 기존 상태·오디오 검증 결과가 그대로 남는다.
        assert failed_again["items"] == old_items
        assert failed_again["physicalItemCount"] == state["physicalItemCount"]
        print(
            "PASS actual BE regeneration failure preserves original item "
            "and failed audio validation"
        )
        print(
            f"Listening failure HTTP checks={sum(client.count for client in clients)} "
            f"elapsedSeconds={time.monotonic() - started:.2f} paidCalls=0"
        )
    finally:
        for client in clients:
            client.client.close()
        CONTROL.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
