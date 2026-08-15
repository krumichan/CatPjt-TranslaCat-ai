from __future__ import annotations

import asyncio
import time

from pydantic import ValidationError

from app.ai.ports import StructuredTextGenerationProvider
from app.core.config import settings
from app.features.language_learning.speaking.errors import SpeakingStageException
from app.features.language_learning.speaking.idempotency import InMemoryIdempotencyStore
from app.features.language_learning.speaking.policy import SPEAKING_CONVERSATION_PROMPT_VERSION
from app.features.language_learning.speaking.prompts import build_assistance_prompt
from app.features.language_learning.speaking.provider_error import map_provider_exception
from app.features.language_learning.speaking.retry import run_with_stage_retry
from app.schemas.language_learning_speaking import (
    AssistancePayload,
    AssistanceRequest,
    AssistanceResponse,
    AssistanceType,
    SpeakingErrorCode,
    SpeakingStage,
    SpeakingUsage,
    StageUsage,
)


class SpeakingAssistanceService:
    TYPE_NAME = "LANGUAGE_LEARNING_SPEAKING_ASSISTANCE"

    def __init__(
        self,
        provider: StructuredTextGenerationProvider,
        *,
        timeout_seconds: float | None = None,
        automatic_retries: int | None = None,
        idempotency_store: InMemoryIdempotencyStore[AssistanceResponse] | None = None,
    ) -> None:
        self.provider = provider
        self.timeout_seconds = (
            timeout_seconds or settings.AI_SPEAKING_CONVERSATION_TIMEOUT_SECONDS
        )
        self.automatic_retries = (
            settings.AI_SPEAKING_AUTOMATIC_RETRY_LIMIT
            if automatic_retries is None
            else automatic_retries
        )
        self.idempotency_store = idempotency_store or InMemoryIdempotencyStore()

    async def generate(self, request: AssistanceRequest) -> AssistanceResponse:
        if request.assistance_type not in {
            AssistanceType.HINT,
            AssistanceType.TRANSLATION,
            AssistanceType.SAMPLE_ANSWER,
        }:
            raise SpeakingStageException(
                code=SpeakingErrorCode.INVALID_RESPONSE_SCHEMA,
                stage=SpeakingStage.CONVERSATION,
                message="AI 생성이 필요한 Speaking Assistance Type이 아닙니다.",
                retryable=False,
            )

        response, replayed = await self.idempotency_store.execute(
            request.idempotency_key,
            lambda: self._generate_once(request),
        )
        result = response.model_copy(deep=True)
        result.idempotent_replay = replayed
        return result

    async def _generate_once(self, request: AssistanceRequest) -> AssistanceResponse:
        prompt = build_assistance_prompt(request)
        schema = AssistancePayload.model_json_schema()
        started = time.perf_counter()

        async def operation():
            try:
                result = await asyncio.wait_for(
                    self.provider.call_with_metadata(
                        type_name=self.TYPE_NAME,
                        data=prompt,
                        schema=schema,
                    ),
                    timeout=self.timeout_seconds,
                )
                if not isinstance(result.data, dict):
                    raise ValueError("structured assistance response must be object")
                payload = AssistancePayload.model_validate(result.data)
                if payload.type != request.assistance_type:
                    raise ValueError("assistance type mismatch")
                return result, payload
            except (TimeoutError, asyncio.TimeoutError) as exc:
                raise SpeakingStageException(
                    code=SpeakingErrorCode.PROVIDER_TIMEOUT,
                    stage=SpeakingStage.CONVERSATION,
                    message="Speaking Assistance Provider 응답 시간이 초과되었습니다.",
                    retryable=True,
                ) from exc
            except (ValidationError, ValueError) as exc:
                raise SpeakingStageException(
                    code=SpeakingErrorCode.INVALID_RESPONSE_SCHEMA,
                    stage=SpeakingStage.CONVERSATION,
                    message="Speaking Assistance 응답 Schema가 유효하지 않습니다.",
                    retryable=True,
                ) from exc
            except SpeakingStageException:
                raise
            except Exception as exc:
                raise map_provider_exception(
                    exc,
                    stage=SpeakingStage.CONVERSATION,
                    fallback_code=SpeakingErrorCode.CONVERSATION_GENERATION_FAILED,
                    fallback_message="Speaking Assistance 생성에 실패했습니다.",
                ) from exc

        result, payload = await run_with_stage_retry(
            operation,
            max_retries=self.automatic_retries,
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return AssistanceResponse(
            request_id=request.request_id,
            session_id=request.session_id,
            turn_index=request.turn_index,
            type=payload.type,
            content=payload.content,
            usage=SpeakingUsage(
                conversation=StageUsage(
                    latency_ms=elapsed_ms,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    provider=result.provider,
                    model=result.model,
                    prompt_version=SPEAKING_CONVERSATION_PROMPT_VERSION,
                )
            ),
        )
