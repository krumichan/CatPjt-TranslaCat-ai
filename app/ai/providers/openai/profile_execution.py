"""명시 allowlist를 사용하는 OpenAI 1회 실행과 본문 없는 기술 진단."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any, cast

from openai.types.responses.response_input_param import ResponseInputParam
from openai.types.responses.response_text_config_param import ResponseTextConfigParam
from openai.types.shared_params.reasoning import Reasoning

from app.ai.providers.openai.client import OpenAIService
from app.ai.providers.openai.execution_profiles import EXECUTION_PROFILES
from app.ai.providers.openai.progressive_fields import ProgressiveFieldError, ProgressiveFields
from app.ai.providers.openai.response import OpenAIProviderResponseError, decode_response
from app.ai.providers.openai.schema import build_explicit_text_config
from app.schemas.model_profile_execution import ModelProfileExecutionRequest

SAFE_RATE_HEADERS = frozenset(
    {
        "x-ratelimit-limit-requests",
        "x-ratelimit-limit-tokens",
        "x-ratelimit-remaining-requests",
        "x-ratelimit-remaining-tokens",
        "x-ratelimit-reset-requests",
        "x-ratelimit-reset-tokens",
        "retry-after",
    }
)


def decode_profile_response(response: Any, *, structured: bool) -> Any:
    # 기존 LL/CHAT decoder 계약은 유지한다. 별도 profile의 JSON만 중복 키를 엄격히 거부한다.
    text = decode_response(response, structured=False)
    if not structured:
        return text

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise OpenAIProviderResponseError(
                    "Duplicate JSON object key", reason_code="DUPLICATE_JSON_KEY", retryable=False
                )
            result[key] = value
        return result

    def invalid_constant(_value):
        raise ValueError("Non-JSON numeric constant")

    try:
        return json.loads(text, object_pairs_hook=unique_object, parse_constant=invalid_constant)
    except (json.JSONDecodeError, ValueError):
        raise OpenAIProviderResponseError(
            "Invalid structured JSON", reason_code="JSON_INVALID", retryable=False
        ) from None


class ProfileExecutor:
    def __init__(self, service: OpenAIService):
        self.service = service

    async def stream(
        self, *, command: ModelProfileExecutionRequest, parent_trace_id: str | None
    ) -> AsyncIterator[dict[str, Any]]:
        """KEYED 응답 한 번에서 닫힌 필드를 순서대로 내보내고 최종 응답을 재검증한다."""
        profile = EXECUTION_PROFILES[command.profile_id]
        schema = command.response_schema or {}
        properties = schema.get("properties")
        required = schema.get("required")
        if (
            not command.strict
            or schema.get("type") != "object"
            or not isinstance(properties, dict)
            or not isinstance(required, list)
            or list(properties) != required
            or not required
            or schema.get("additionalProperties") is not False
        ):
            raise ProgressiveFieldError("PROGRESSIVE_SCHEMA_INVALID")
        parser = ProgressiveFields(required, recover_holes=True)
        text_config = build_explicit_text_config(
            schema=command.response_schema,
            schema_name_value=command.schema_name,
            strict=command.strict,
            verbosity=profile.verbosity,
        )
        headers = {"X-Client-Request-Id": command.trace_id}
        if parent_trace_id:
            headers["X-Novel-Trace-Id"] = parent_trace_id
        started = time.monotonic()
        first_token_ms = None
        completed = None
        stream = await self.service.client.with_options(
            timeout=command.remaining_milliseconds / 1000,
            max_retries=0,
        ).responses.create(
            model=profile.model,
            instructions=command.instructions,
            input=cast(ResponseInputParam, [message.model_dump() for message in command.messages]),
            reasoning=cast(Reasoning, {"effort": profile.effort}),
            max_output_tokens=command.max_output_tokens,
            text=cast(ResponseTextConfigParam, text_config),
            store=False,
            service_tier="default",
            extra_headers=headers,
            stream=True,
        )
        try:
            async for event in stream:
                if event.type == "response.output_text.delta":
                    if first_token_ms is None:
                        first_token_ms = (time.monotonic() - started) * 1000
                    failure = None
                    try:
                        fields = parser.feed(event.delta)
                    except ProgressiveFieldError as error:
                        fields = parser.drain_salvage()
                        failure = error
                    for key, value in fields:
                        yield {
                            "type": "field",
                            "key": key,
                            "value": value,
                            "providerElapsedMs": (time.monotonic() - started) * 1000,
                        }
                    for key, code in parser.drain_failures():
                        yield {"type": "field-error", "key": key, "code": code}
                    if failure is not None:
                        raise failure
                elif event.type == "response.completed":
                    completed = event.response
                elif event.type in {"response.failed", "response.incomplete", "error"}:
                    raise OpenAIProviderResponseError(
                        "Streaming response did not complete",
                        reason_code="PROGRESSIVE_PROVIDER_INCOMPLETE",
                        retryable=False,
                    )
            if completed is None:
                raise OpenAIProviderResponseError(
                    "Streaming response missing completion",
                    reason_code="PROGRESSIVE_PROVIDER_INCOMPLETE",
                    retryable=False,
                )
            output = decode_profile_response(completed, structured=True)
            parser.complete(output)
            usage = completed.usage
            result = {
                "output": output,
                "inputTokens": usage.input_tokens if usage else 0,
                "outputTokens": usage.output_tokens if usage else 0,
                "provider": "openai",
                "model": completed.model,
                "providerCalls": 1,
                "profileId": command.profile_id,
                "profileVersion": profile.version,
                "reasoningEffort": profile.effort,
                "serviceTier": completed.service_tier or profile.service_tier,
                "cachedInputTokens": usage.input_tokens_details.cached_tokens if usage else 0,
                "reasoningTokens": usage.output_tokens_details.reasoning_tokens if usage else 0,
                "metadata": {
                    "traceId": command.trace_id,
                    "parentTraceId": parent_trace_id,
                    "providerRequestId": getattr(completed, "_request_id", None),
                    "providerMs": (time.monotonic() - started) * 1000,
                    "providerFirstTokenMs": first_token_ms,
                },
            }
            yield {"type": "result", "result": result, "_rawResponse": completed}
        finally:
            await stream.close()

    async def execute(
        self, *, command: ModelProfileExecutionRequest, parent_trace_id: str | None
    ) -> dict:
        profile = EXECUTION_PROFILES[command.profile_id]
        text_config = build_explicit_text_config(
            schema=command.response_schema,
            schema_name_value=command.schema_name,
            strict=command.strict,
            verbosity=profile.verbosity,
        )
        headers = {"X-Client-Request-Id": command.trace_id}
        if parent_trace_id:
            headers["X-Novel-Trace-Id"] = parent_trace_id

        # SDK 재시도는 0으로 유지하고 요청별 남은 deadline 안에서 한 번만 전송한다.
        started = time.monotonic()
        raw = await self.service.client.with_options(
            timeout=command.remaining_milliseconds / 1000,
            max_retries=0,
        ).responses.with_raw_response.create(
            model=profile.model,
            instructions=command.instructions,
            input=cast(ResponseInputParam, [message.model_dump() for message in command.messages]),
            reasoning=cast(Reasoning, {"effort": profile.effort}),
            max_output_tokens=command.max_output_tokens,
            text=cast(ResponseTextConfigParam, text_config),
            store=False,
            service_tier="default",
            extra_headers=headers,
        )
        response = raw.parse()
        provider_ms = (time.monotonic() - started) * 1000

        # 형식/거부 판정은 기존 decoder를 사용한다. 업무 정렬/누락 검사는 호출자가 소유한다.
        decoding = time.monotonic()
        output = decode_profile_response(response, structured=command.response_schema is not None)
        usage = response.usage
        return {
            "output": output,
            "inputTokens": usage.input_tokens if usage else 0,
            "outputTokens": usage.output_tokens if usage else 0,
            "provider": "openai",
            "model": response.model,
            "providerCalls": 1,
            "profileId": command.profile_id,
            "profileVersion": profile.version,
            "reasoningEffort": profile.effort,
            "serviceTier": response.service_tier or profile.service_tier,
            "cachedInputTokens": usage.input_tokens_details.cached_tokens if usage else 0,
            "reasoningTokens": usage.output_tokens_details.reasoning_tokens if usage else 0,
            "metadata": {
                "traceId": command.trace_id,
                "parentTraceId": parent_trace_id,
                "providerRequestId": raw.headers.get("x-request-id"),
                "providerMs": provider_ms,
                "decodeMs": (time.monotonic() - decoding) * 1000,
                "providerFirstTokenMs": None,
                "connectionMs": None,
                "rateLimits": {
                    key: raw.headers[key] for key in SAFE_RATE_HEADERS if key in raw.headers
                },
            },
        }
