"""업무 정책을 담지 않는 인증된 범용 모델 실행 HTTP 경계."""
from __future__ import annotations

import asyncio
import json
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from openai import APIConnectionError, APITimeoutError
from pydantic import ValidationError

from app.ai.model_policy import AiModelTier
from app.ai.providers.openai.client import OpenAIService
from app.ai.providers.openai.response import OpenAIProviderResponseError
from app.ai.providers.openai.schema import OpenAISchemaConfigurationError
from app.api.dependencies import get_model_execution_provider
from app.common.provider_failure import provider_failure_origin
from app.common.provider_retry import retry_after_details
from app.schemas.model_execution import ModelExecutionRequest, ModelExecutionResponse

router = APIRouter(prefix="/model", tags=["Internal Model Execution"])
_MAX_REQUEST_BYTES = 500_000


@router.post("/execute", response_model=ModelExecutionResponse)
async def execute_model(
    request: Request,
    provider: Annotated[OpenAIService, Depends(get_model_execution_provider)],
) -> ModelExecutionResponse:
    # FastAPI의 기본 422는 입력값을 그대로 포함할 수 있으므로 이 경계에서는 고정 코드만 돌려준다.
    raw = bytearray()
    async for chunk in request.stream():
        if len(raw) + len(chunk) > _MAX_REQUEST_BYTES:
            raise HTTPException(
                413, detail={"code": "EXECUTION_REQUEST_TOO_LARGE", "retryable": False},
            )
        raw.extend(chunk)
    try:
        command = ModelExecutionRequest.model_validate(json.loads(raw))
    except (ValidationError, ValueError, TypeError, UnicodeDecodeError, RecursionError):
        raise HTTPException(
            422, detail={"code": "EXECUTION_REQUEST_INVALID", "retryable": False},
        ) from None

    try:
        async with asyncio.timeout(command.remaining_milliseconds / 1000.0):
            result = await provider.execute_explicit(
                instructions=command.instructions,
                messages=[message.model_dump() for message in command.messages],
                tier=AiModelTier(command.tier.value),
                reasoning_effort=command.reasoning_effort,
                verbosity=command.verbosity,
                max_output_tokens=command.max_output_tokens,
                remaining_milliseconds=command.remaining_milliseconds,
                response_schema=command.response_schema,
                schema_name_value=command.schema_name,
                strict=command.strict,
            )
    except OpenAISchemaConfigurationError:
        raise HTTPException(
            422, detail={"code": "EXECUTION_SCHEMA_INVALID", "retryable": False},
        ) from None
    except OpenAIProviderResponseError as error:
        # 응답 래퍼의 ValueError 상속은 Provider 값 오류가 아니다. 원래 reason code를 유지한다.
        signal = {key: value for key, value in provider_failure_origin(error).items()
                  if key == "failureSignal"}
        raise HTTPException(
            502, detail={"code": error.reason_code, "retryable": error.retryable,
                         **signal},
        ) from None
    except (TimeoutError, APITimeoutError, httpx.TimeoutException) as error:
        raise HTTPException(
            504, detail={"code": "PROVIDER_TIMEOUT", "retryable": True,
                         **provider_failure_origin(error)},
        ) from None
    except (ConnectionError, APIConnectionError, httpx.TransportError) as error:
        raise HTTPException(
            503, detail={"code": "PROVIDER_UNAVAILABLE", "retryable": True,
                         **provider_failure_origin(error)},
        ) from None
    except Exception as error:
        # SDK 상태는 기술 분류로 전달한다. 업무 단계의 재시도·거부 판정은 LL에서 수행한다.
        status = getattr(error, "status_code", None)
        if status in {408, 409, 429} or isinstance(status, int) and status >= 500:
            raise HTTPException(
                503, detail={
                    "code": "PROVIDER_UNAVAILABLE", "retryable": True,
                    "failureKind": "HTTP_STATUS", "providerStatus": status,
                    **provider_failure_origin(error),
                    **retry_after_details(error),
                },
            ) from None
        if isinstance(status, int) and 400 <= status < 500:
            raise HTTPException(
                502, detail={
                    "code": "PROVIDER_CONFIGURATION_ERROR", "retryable": False,
                    "failureKind": "HTTP_STATUS", "providerStatus": status,
                    **provider_failure_origin(error),
                },
            ) from None
        raise HTTPException(
            502, detail={"code": "PROVIDER_EXECUTION_FAILED", "retryable": False,
                         **provider_failure_origin(error)},
        ) from None

    return ModelExecutionResponse(
        output=result.data,
        inputTokens=result.input_tokens,
        outputTokens=result.output_tokens,
        provider=result.provider,
        model=result.model,
    )
