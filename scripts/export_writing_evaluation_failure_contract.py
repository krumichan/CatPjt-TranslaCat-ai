"""퇴역 보존본의 평가 호출 메서드를 그대로 실행하여 실패 경계만 합성 golden으로 기록한다."""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import httpx
from fastapi import HTTPException
from openai import APIConnectionError, APITimeoutError
from pydantic import BaseModel, ValidationError

from app.ai.providers.openai.response import OpenAIProviderResponseError
from app.ai.providers.openai.schema import OpenAISchemaConfigurationError


async def main() -> None:
    # 준비: 원본 메서드의 AST만 로드한다. 퇴역 package를 런타임 app에 되살리지 않는다.
    ll = Path(__file__).resolve().parents[2] / "CatPjt-TranslaCat-ll"
    source = ll.parent / (
        ".codex-workspace/verification/ll/runtime/writing-python-retirement-20260926/"
        "app/features/language_learning/writing/service.py"
    )
    text = source.read_text(encoding="utf-8")
    tree = ast.parse(text)
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and any(
            isinstance(member, ast.AsyncFunctionDef) and member.name == "_call_and_validate"
            for member in node.body
        )
    )
    method = next(
        node
        for node in owner.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_call_and_validate"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            method,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    logger = logging.getLogger("synthetic-retired-writing")
    logger.disabled = True
    namespace = {
        "asyncio": asyncio,
        "HTTPException": HTTPException,
        "ValidationError": ValidationError,
        "logger": logger,
    }
    exec(compile(module, str(source), "exec"), namespace)
    execute = namespace["_call_and_validate"]
    request = httpx.Request("POST", "http://127.0.0.1/synthetic")

    class Payload(BaseModel):
        synthetic_value: int

    class StatusFailure(RuntimeError):
        status_code = 400

    cases = [
        (
            "refusal",
            lambda: OpenAIProviderResponseError(
                "synthetic", reason_code="REFUSAL", retryable=False
            ),
            "REFUSAL",
            None,
            None,
        ),
        (
            "token_limit",
            lambda: OpenAIProviderResponseError(
                "synthetic", reason_code="OUTPUT_TOKEN_LIMIT", retryable=False
            ),
            "OUTPUT_TOKEN_LIMIT",
            None,
            None,
        ),
        ("configuration", StatusFailure, "PROVIDER_CONFIGURATION_ERROR", "HTTP_STATUS", 400),
        (
            "sdk_timeout",
            lambda: APITimeoutError(request=request),
            "PROVIDER_TIMEOUT",
            "SDK_TIMEOUT",
            None,
        ),
        (
            "sdk_connection",
            lambda: APIConnectionError(request=request),
            "PROVIDER_UNAVAILABLE",
            "SDK_CONNECTION",
            None,
        ),
        ("timeout", TimeoutError, "PROVIDER_TIMEOUT", "TIMEOUT", None),
        (
            "schema_configuration",
            lambda: OpenAISchemaConfigurationError("synthetic"),
            "EXECUTION_SCHEMA_INVALID",
            None,
            None,
        ),
        ("invalid_schema", None, "EVALUATION_SCHEMA_INVALID", None, None),
        ("non_object", None, "EVALUATION_SCHEMA_INVALID", None, None),
        ("unknown", RuntimeError, "PROVIDER_EXECUTION_FAILED", None, None),
        (
            "http_exception",
            lambda: HTTPException(503, detail="synthetic"),
            "HTTP_EXCEPTION",
            "HTTP_STATUS",
            503,
        ),
    ]
    results = []
    for name, create, code, kind, provider_status in cases:
        calls = 0

        class Provider:
            # 각 사례의 예외 생성기와 응답 유형을 기본값으로 고정한다.
            async def call(self, *, _create=create, _name=name, **kwargs):
                nonlocal calls
                calls += 1
                if _create:
                    raise _create()
                return [] if _name == "non_object" else {"synthetic_value": None}

        # 실행: 기존 총 2회 한도와 catch 순서를 변경하지 않고 실제 예외를 통과시킨다.
        try:
            await execute(
                SimpleNamespace(provider=Provider()),
                request_id="synthetic",
                operation="writing evaluation",
                type_name="LANGUAGE_LEARNING_WRITING_EVALUATION",
                prompt="Synthetic prompt",
                schema={},
                model_type=Payload,
                timeout_seconds=30,
                max_retries=1,
            )
            raise AssertionError("Synthetic failure must fail")
        except HTTPException as failure:
            results.append(
                {
                    "name": name,
                    "code": code,
                    "failureKind": kind,
                    "providerStatus": provider_status,
                    "calls": calls,
                    "httpStatus": failure.status_code,
                }
            )

    # 검증 자료는 합성 오류 코드·횟수·원본 checksum만 포함한다.
    target = ll / "src/test/resources/contracts/writing-evaluation-failure-python-golden.json"
    target.write_text(
        json.dumps(
            {"sourceSha256": hashlib.sha256(text.encode()).hexdigest(), "cases": results},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"synthetic_writing_evaluation_failure_cases={len(results)}")


if __name__ == "__main__":
    asyncio.run(main())
