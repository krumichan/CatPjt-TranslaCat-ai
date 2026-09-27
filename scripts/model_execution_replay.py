"""이행 전 합성 Provider로 검증한 HTTP 입력/출력을 업무 모듈 없이 재생한다."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.ai.ports import StructuredGenerationResult
from app.ai.providers.openai.response import OpenAIProviderResponseError
from app.ai.providers.openai.schema import OpenAISchemaConfigurationError

_CORRELATION = re.compile(
    r"(?<![0-9a-f])[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(?![0-9a-f])"
    r"|(?<![0-9a-f])[0-9a-f]{32}(?![0-9a-f])"
    r"|daily-(?:generate|regen)-[0-9]+-[0-9]+"
)


def _canonical(request: dict[str, Any]) -> tuple[str, list[str]]:
    # 남은 기한과 실행마다 바뀌는 상관 ID만 제외하며 schema·메시지·모델 요구는 그대로 비교한다.
    value = {key: item for key, item in request.items() if key != "remaining_milliseconds"}
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         default=lambda item: item.value)
    identifiers: list[str] = []

    def replace(match: re.Match[str]) -> str:
        identifier = match.group()
        if identifier not in identifiers:
            identifiers.append(identifier)
        return f"<correlation-{identifiers.index(identifier)}>"

    return _CORRELATION.sub(replace, encoded), identifiers


@lru_cache(maxsize=1)
def _fixtures() -> dict[str, tuple[dict[str, Any], list[str]]]:
    path = Path(__file__).resolve().parents[1] / "tests/fixtures/model_execution_replay.jsonl"
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        key, identifiers = _canonical(record["request"])
        if key in result and result[key][0] != record["result"]:
            raise ValueError("Conflicting synthetic replay fixture")
        result[key] = record["result"], identifiers
    return result


def execute_replay(kwargs: dict[str, Any]) -> StructuredGenerationResult | None:
    key, identifiers = _canonical(kwargs)
    fixture = _fixtures().get(key)
    if fixture is None:
        return None
    result, recorded_identifiers = fixture

    # 고정 거부/Schema 오류도 원래 기술 분류로 재생한다. 응답을 정상 데이터로 보정하지 않는다.
    if result.get("errorType") == "OpenAIProviderResponseError":
        raise OpenAIProviderResponseError(
            "synthetic provider error", reason_code=result["reasonCode"],
            retryable=result["retryable"],
        )
    if result.get("errorType") == "OpenAISchemaConfigurationError":
        raise OpenAISchemaConfigurationError("synthetic schema rejection")
    encoded = json.dumps(result["data"], ensure_ascii=False)
    mapping = dict(zip(recorded_identifiers, identifiers, strict=True))
    encoded = _CORRELATION.sub(lambda match: mapping.get(match.group(), match.group()), encoded)
    return StructuredGenerationResult(
        json.loads(encoded), 7, 2, "test-provider", "synthetic-replay",
    )
