"""명시적 로컬 Listening 이행 검사용 모델 출력과 기술 오디오 길이 fixture."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from app.ai.ports import StructuredGenerationResult

_ROOT = Path(__file__).resolve().parents[2] / "CatPjt-TranslaCat-ll"
_CONTROL = Path(__file__).resolve().parents[1] / ".tmp_ktor_m0/listening-control.json"
_SOURCES = (
    "京都で静かな寺を見学したいです。",
    "旅行の前にホテルを予約しておきました。",
)


def _require_marker() -> None:
    if os.environ.get("TRANSLACAT_TEST_MODEL_EXECUTION") != "1":
        raise RuntimeError("Listening fixture requires explicit local test marker")


def _control() -> dict[str, Any]:
    return json.loads(_CONTROL.read_text(encoding="utf-8")) if _CONTROL.exists() else {}


def listening_fixture_duration(text: str) -> float | None:
    # 등록한 합성 문장의 hash만 변경한다. 다른 기능의 오디오 fixture에는 관여하지 않는다.
    key = hashlib.sha256(text.encode("utf-8")).hexdigest()
    known = {hashlib.sha256(value.encode("utf-8")).hexdigest() for value in _SOURCES}
    if key not in known:
        return None
    _require_marker()
    return 4.0 if _control().get("failure") == "TTS_SHORT" else 10.0


async def execute_listening_fixture(kwargs: dict[str, Any]) -> StructuredGenerationResult | None:
    name = str(kwargs.get("schema_name_value", ""))
    stages = {
        "ListeningGenerationPayload": "generation",
        "ListeningINTERPRETATION": "interpretation",
        "ListeningSUMMARY": "summary",
        "RecommendationExplanationPayload": "explanation",
    }
    stage = stages.get(name)
    if stage is None:
        return None
    _require_marker()

    # 업무 서비스는 import하지 않고 추출한 prompt와 고정 Provider 출력만 사용한다.
    expected = (_ROOT / f"src/main/resources/listening/{stage}-system-prompt.txt").read_text(
        encoding="utf-8"
    )
    if kwargs.get("instructions") != expected:
        raise ValueError("Listening instruction contract mismatch")
    messages = kwargs.get("messages")
    if not isinstance(messages, list) or len(messages) != 1 or messages[0].get("role") != "user":
        raise ValueError("Listening message contract mismatch")
    control = _control()
    if stage == "generation" and control.get("failure") == "HOLD_GENERATION":
        marker = _CONTROL.with_name("listening-hold.json")
        marker.write_text('{"stage":"generation"}', encoding="utf-8")
        # 재시작 검사는 외부 실행 중인 상태만 지연한다. 고정 출력과 LL 판정은 그대로 실행한다.
        while _control().get("failure") == "HOLD_GENERATION":
            await asyncio.sleep(0.1)
    if control.get("failure") == "PROVIDER_RATE_LIMITED":
        error = RuntimeError("Synthetic Listening cooldown")
        error.status_code = 429
        error.retry_after_seconds = 2
        raise error
    if control.get("failure") == "SCHEMA_INVALID":
        return StructuredGenerationResult({}, 1, 1, "test-provider", "synthetic-listening-invalid")

    if stage == "explanation":
        output = {
            "explanation": "최근 합성 학습 근거에서 이 항목을 더 연습할 수 있습니다.",
            "ctaLabel": "학습 시작",
        }
    elif stage == "generation":
        fixture = json.loads(
            (
                _ROOT / "src/test/resources/contracts/listening-generation-python-golden.json"
            ).read_text(encoding="utf-8")
        )
        payload = json.loads(messages[0]["content"].split("\n\n", 1)[1])
        mode = payload["setContext"]["learningMode"]
        if mode != "DICTATION":
            variants = json.loads(
                (
                    _ROOT / "src/test/resources/contracts/listening-modes-python-golden.json"
                ).read_text(encoding="utf-8")
            )
            fixture = next(value for value in variants["generation"] if value["mode"] == mode)
        # 테스트 Provider의 band별 고정 variant 선택이며 LL 채택 정책과 저장 경로는 그대로 실행한다.
        band = (
            payload["languageComplexity"]["targetComplexityBand"]
            or payload["languageComplexity"]["baseComplexityBand"]
        )
        output = fixture["response"]
        for item in output["items"]:
            item["languageComplexityBand"] = band
    else:
        fixtures = json.loads(
            (
                _ROOT / "src/test/resources/contracts/listening-semantic-python-golden.json"
            ).read_text(encoding="utf-8")
        )
        task = "INTERPRETATION" if stage == "interpretation" else "SUMMARY"
        output = next(value["response"] for value in fixtures if value["taskType"] == task)
        if task == "INTERPRETATION":
            request = json.loads(messages[0]["content"].split("\n\n", 1)[1])
            variants = json.loads(
                (
                    _ROOT / "src/test/resources/contracts/listening-modes-python-golden.json"
                ).read_text(encoding="utf-8")
            )
            variant = next(
                (
                    value
                    for value in variants["interpretation"]
                    if value["source"] == request["sourceText"]
                ),
                None,
            )
            if variant is not None:
                output = variant["response"]
    return StructuredGenerationResult(output, 13, 21, "test-provider", "synthetic-listening")
