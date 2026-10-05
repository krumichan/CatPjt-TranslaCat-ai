"""Exact synthetic replay must reject contract drift and preserve classified failures."""

import copy
import json
from pathlib import Path

import pytest

from app.ai.providers.openai.response import OpenAIProviderResponseError
from app.ai.providers.openai.schema import OpenAISchemaConfigurationError
from scripts.model_execution_replay import execute_replay

RECORDS = [
    json.loads(line)
    for line in (Path(__file__).parent / "fixtures/model_execution_replay.jsonl")
    .read_text(encoding="utf-8")
    .splitlines()
]


@pytest.mark.parametrize("record", RECORDS, ids=[f"fixture-{i + 1}" for i in range(len(RECORDS))])
def test_replay_preserves_saved_result_or_provider_failure(record):
    # 준비: 재생 요청과 기대값은 별도 저장된 합성 Provider 결과를 사용한다.
    request = copy.deepcopy(record["request"])
    expected = record["result"]

    # 실행 / 검증: 원래 오류 분류를 성공 데이터로 바꾸지 않는다.
    if expected.get("errorType") == "OpenAIProviderResponseError":
        with pytest.raises(OpenAIProviderResponseError) as raised:
            execute_replay(request)
        assert raised.value.reason_code == expected["reasonCode"]
        assert raised.value.retryable == expected["retryable"]
    elif expected.get("errorType") == "OpenAISchemaConfigurationError":
        with pytest.raises(OpenAISchemaConfigurationError):
            execute_replay(request)
    else:
        actual = execute_replay(request)
        assert actual is not None
        assert actual.data == expected["data"]


@pytest.mark.parametrize(
    "field",
    [
        "instructions",
        "messages",
        "response_schema",
        "strict",
        "tier",
        "reasoning_effort",
        "verbosity",
        "max_output_tokens",
        "schema_name_value",
    ],
)
def test_replay_rejects_unrecorded_contract_changes(field):
    # 준비: 정상 fixture에 기록되지 않은 의미 변경을 한 필드씩 주입한다.
    request = copy.deepcopy(
        next(row["request"] for row in RECORDS if "errorType" not in row["result"])
    )
    if field == "messages":
        request[field] = [
            *request[field],
            {"role": "user", "content": "unrecorded-contract-sentinel"},
        ]
    elif field == "response_schema":
        request[field] = {
            **(request.get(field) or {}),
            "description": "unrecorded-contract-sentinel",
        }
    elif field == "strict":
        request[field] = not request.get(field, False)
    elif field == "max_output_tokens":
        request[field] = request[field] + 1
    else:
        request[field] = str(request.get(field, "")) + " unrecorded-contract-sentinel"

    # 실행 / 검증: wildcard나 fallback으로 기록되지 않은 요청을 승인하지 않는다.
    assert execute_replay(request) is None


def test_replay_only_ignores_remaining_deadline_for_same_contract():
    # 준비: 같은 계약의 현재 남은 시간은 매 실행마다 달라질 수 있다.
    row = next(record for record in RECORDS if "errorType" not in record["result"])
    request = copy.deepcopy(row["request"])
    request["remaining_milliseconds"] = 12345

    # 실행 / 검증: 이 허용된 차이는 저장된 합성 결과를 그대로 반환한다.
    actual = execute_replay(request)
    assert actual is not None
    assert actual.data == row["result"]["data"]
